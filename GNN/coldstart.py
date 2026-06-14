"""
Cold-start station inductive split.

We hold out ~15% of stations from training entirely: no event whose
found_at / lost_at / pickup is in the held-out set is ever seen during
training, and the heterogeneous graph contains no bipartite (found/lost/
pickup) edges incident to those stations. Metro adjacency is intact.

We then evaluate on cold-test = events whose found_at is held out. The
model must build a representation of an input station it has never seen,
purely from station features + metro neighbors (i.e. message passing).

Comparisons:
  - full GNN (n_layers=2)
  - no_mp (n_layers=0): station = feature-projected; no message passing
  - same_station baseline: predict pickup = found_at
  - mode_lookup baseline: per-(context, found_at) empirical mode w/ smoothing
"""
import os, json, time, copy, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from events import load_events, FoundEvent
from graph_build import build_hetero_graph, events_to_tensors, graph_to_device
from model import HetGNN
from decision import build_dist_matrix, recommend_storage
from network import STATION_LIST, STATION_INDEX, N_STN
from contexts import N_CONTEXTS
from train import evaluate, evaluate_baselines, _device


def split_coldstart(events, n_unseen=24, seed=42, val_frac=0.10):
    """Pick `n_unseen` stations uniformly at random. Cold = any event that
    touches an unseen station as found_at OR lost_at. Warm = the rest, then
    shuffled and split into train + warm-val.
    """
    rng = random.Random(seed)
    stations = list(STATION_LIST)
    rng.shuffle(stations)
    unseen = set(stations[:n_unseen])
    seen = set(stations[n_unseen:])

    def touches(e):
        if e.found_at in unseen: return True
        if e.lost_at and e.lost_at in unseen: return True
        return False

    cold = [e for e in events if touches(e)]
    warm = [e for e in events if not touches(e)]
    rng.shuffle(warm)
    n_val = int(len(warm) * val_frac)
    val = warm[:n_val]
    train = warm[n_val:]
    return train, val, cold, unseen


def train_model(name, train_ev, val_ev, *, n_layers=2, epochs=40, batch_size=512,
                hidden=64, dropout=0.1, lr=1e-3, weight_decay=1e-5,
                patience=8, seed=0, device=None):
    device = device or _device()
    print(f"\n========== train {name}  (n_layers={n_layers}) ==========")
    torch.manual_seed(seed); np.random.seed(seed)

    # Graph built from train events ONLY — no bipartite edges incident to
    # cold stations exist. Metro edges include them (network is fixed).
    graph = build_hetero_graph(train_ev, include_picked_up=True)
    graph = graph_to_device(graph, device)
    print(f"  metro={graph['metro_edge_index'].shape[1]}  "
          f"found={graph['found_at_edge_index'].shape[1]}  "
          f"lost={graph['lost_at_edge_index'].shape[1]}  "
          f"pickup={graph['picked_up_edge_index'].shape[1]}")

    model = HetGNN(
        station_feat_dim=graph["station_feats"].shape[1],
        n_contexts=N_CONTEXTS,
        hidden=hidden, n_layers=n_layers, dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    dist_matrix = build_dist_matrix().to(device)
    train_t = events_to_tensors(train_ev)
    val_t   = events_to_tensors(val_ev)
    for d in (train_t, val_t):
        for k in d: d[k] = d[k].to(device)

    best_hops = float("inf"); best_state = None
    no_improve = 0
    n_train = train_t["pickup"].shape[0]
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0; n = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            h_st, h_ctx = model.encode(graph)
            logits = model.score(h_ctx[train_t["ctx"][idx]], h_st,
                                 found_st_idx=train_t["found_st"][idx],
                                 lost_st_idx=train_t["lost_st"][idx])
            loss = (F.cross_entropy(logits, train_t["pickup"][idx], reduction="none")
                    * train_t["weight"][idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * idx.shape[0]; n += idx.shape[0]
        train_loss = total_loss / max(n, 1)

        val = evaluate(model, graph, val_t, dist_matrix, device)
        dt = time.time() - t0; t0 = time.time()
        print(f"  ep {epoch:2d} {dt:5.1f}s tl={train_loss:.4f} vl={val['loss']:.4f} "
              f"t1={val['top1']:.3f} E[h]={val['expected_hops']:.2f}")

        if val["expected_hops"] < best_hops - 1e-4:
            best_hops = val["expected_hops"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stop @ ep {epoch}")
                break

    model.load_state_dict(best_state)
    return model, graph, dist_matrix, n_params


def main():
    import os
    data_path = os.environ.get("ABLATE_DATA", "artifacts/data/synth_v2.jsonl")
    out_dir = Path(os.environ.get("ABLATE_OUT", "artifacts/coldstart"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device()
    print(f"Device: {device}")

    events = load_events(data_path)
    train_ev, val_ev, cold_ev, unseen = split_coldstart(
        events, n_unseen=24, seed=42, val_frac=0.10
    )
    print(f"Stations: {N_STN} total, {len(unseen)} held out (cold)")
    print(f"Events: train={len(train_ev)}  warm-val={len(val_ev)}  cold={len(cold_ev)}")
    print(f"Sample of held-out stations: {sorted(list(unseen))[:8]} ...")

    # Train both models on the WARM event set
    full, graph_full, dist, n_params_full = train_model(
        "full", train_ev, val_ev, n_layers=2, device=device, seed=0,
    )
    no_mp, graph_nomp, _, n_params_nomp = train_model(
        "no_mp", train_ev, val_ev, n_layers=0, device=device, seed=0,
    )

    # Tensorise both eval sets
    warm_t = events_to_tensors(val_ev)
    cold_t = events_to_tensors(cold_ev)
    for d in (warm_t, cold_t):
        for k in d: d[k] = d[k].to(device)

    def _evaluate_extras(eval_t):
        """Coverage and mean number of unseen stations in the input."""
        n = eval_t["pickup"].shape[0]
        return {"n_events": n}

    print("\n=== Warm-val (sanity check: in-distribution events) ===")
    w_full = evaluate(full,  graph_full, warm_t, dist, device)
    w_nomp = evaluate(no_mp, graph_nomp, warm_t, dist, device)
    for label, m in (("full", w_full), ("no_mp", w_nomp)):
        print(f"  {label:6s} top1={m['top1']:.3f} top3={m['top3']:.3f} "
              f"top5={m['top5']:.3f} E[hops]={m['expected_hops']:.2f}")

    print("\n=== COLD-TEST (found_at or lost_at ∈ unseen stations) ===")
    c_full = evaluate(full,  graph_full, cold_t, dist, device)
    c_nomp = evaluate(no_mp, graph_nomp, cold_t, dist, device)
    for label, m in (("full", c_full), ("no_mp", c_nomp)):
        print(f"  {label:6s} top1={m['top1']:.3f} top3={m['top3']:.3f} "
              f"top5={m['top5']:.3f} E[hops]={m['expected_hops']:.2f} "
              f"cov={m['coverage']:.2f}")

    # Baselines on cold set
    bl = evaluate_baselines(train_ev, cold_ev, dist, device)
    print("\n=== Baselines on cold-test ===")
    for name, m in bl.items():
        print(f"  {name:14s}  E[hops]={m['expected_hops']:.2f} cov={m['coverage']:.2f}")

    summary = {
        "n_unseen_stations": len(unseen),
        "unseen_stations": sorted(unseen),
        "n_train_events": len(train_ev),
        "n_warm_val": len(val_ev),
        "n_cold": len(cold_ev),
        "warm_val": {"full": w_full, "no_mp": w_nomp},
        "cold_test": {
            "full": c_full,
            "no_mp": c_nomp,
            "baselines": bl,
        },
        "n_params_full": n_params_full,
        "n_params_no_mp": n_params_nomp,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Headline summary
    print("\n=== HEADLINE ===")
    print(f"{'           ':14s}  {'warm-val':>10s}  {'cold-test':>10s}")
    print(f"{'full GNN':14s}  E[h]={w_full['expected_hops']:>5.2f}  "
          f"E[h]={c_full['expected_hops']:>5.2f}")
    print(f"{'no-MP (feat)':14s}  E[h]={w_nomp['expected_hops']:>5.2f}  "
          f"E[h]={c_nomp['expected_hops']:>5.2f}")
    print(f"{'mode_lookup':14s}  {'—':>10s}  E[h]={bl['mode_lookup']['expected_hops']:>5.2f}")
    print(f"{'same_station':14s}  {'—':>10s}  E[h]={bl['same_station']['expected_hops']:>5.2f}")
    print(f"\nFull GNN advantage on cold-test:")
    print(f"  vs no_mp:       {c_nomp['expected_hops'] - c_full['expected_hops']:+.2f} hops")
    print(f"  vs mode_lookup: {bl['mode_lookup']['expected_hops'] - c_full['expected_hops']:+.2f} hops")
    print(f"  vs same_st:     {bl['same_station']['expected_hops'] - c_full['expected_hops']:+.2f} hops")


if __name__ == "__main__":
    main()
