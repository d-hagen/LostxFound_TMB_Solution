"""
Ablation runner. Trains 3 variants of HetGNN and writes a comparison JSON.

  A. no_metro   : zero out the metro relation (station-station edges) before
                  message passing — tests whether physical topology matters.
  B. no_ctx_mp  : skip context-side aggregation in every layer — contexts
                  remain at the initial learned embedding lookup, no
                  message passing. Tests whether bipartite GNN structure
                  contributes beyond a plain embedding table.
  C. no_mp      : zero layers — both stations (linear-projected features) and
                  contexts (embedding table) are scored bilinearly with no
                  message passing at all. Pure embedding baseline.

Each variant uses the same data split / seed as the full model. The full-model
numbers are read from artifacts/models/dev/metadata.json (already trained).
"""
import os, json, time, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from events import load_events, split_events
from graph_build import build_hetero_graph, events_to_tensors, graph_to_device
from model import HetGNN, HetGNNLayer
from decision import build_dist_matrix, recommend_storage
from network import N_STN
from contexts import N_CONTEXTS
from train import evaluate, evaluate_baselines, _device


# ── ablation B: monkey-patch context-side aggregation off ────────────────────
_orig_layer_forward = HetGNNLayer.forward

def _layer_forward_no_ctx_mp(self, h_st, h_ctx, g):
    # Run the original, but discard the context update — contexts stay static
    # at whatever the embedding-table lookup produced before layer 1.
    new_st, _ = _orig_layer_forward(self, h_st, h_ctx, g)
    return new_st, h_ctx


def _drop_metro_edges(graph):
    g2 = dict(graph)
    g2["metro_edge_index"] = torch.zeros(2, 0, dtype=torch.long,
                                          device=graph["metro_edge_index"].device)
    g2["metro_weight"] = torch.zeros(0, dtype=torch.float32,
                                      device=graph["metro_weight"].device)
    return g2


def _drop_bipartite_edges(graph):
    """Drop found_at, lost_at and picked_up edges — leave only metro adjacency."""
    g2 = dict(graph)
    dev = graph["metro_edge_index"].device
    for k in ("found_at", "lost_at", "picked_up"):
        g2[f"{k}_edge_index"] = torch.zeros(2, 0, dtype=torch.long, device=dev)
        g2[f"{k}_weight"]     = torch.zeros(0, dtype=torch.float32, device=dev)
    return g2


def _drop_lost_edges(graph):
    """Drop only the lost_at edges (leave found/picked-up intact)."""
    g2 = dict(graph)
    dev = graph["metro_edge_index"].device
    g2["lost_at_edge_index"] = torch.zeros(2, 0, dtype=torch.long, device=dev)
    g2["lost_at_weight"]     = torch.zeros(0, dtype=torch.float32, device=dev)
    return g2


def run_variant(name, data_path, *, drop_metro=False, drop_bipartite=False,
                drop_lost_edges=False, mask_lost_input=False,
                no_ctx_mp=False, n_layers=2, epochs=40, batch_size=512,
                hidden=64, dropout=0.1, lr=1e-3, weight_decay=1e-5,
                patience=8, seed=0, out_dir=None):
    device = _device()
    print(f"\n========== {name}  (device={device}) ==========")
    print(f"  drop_metro={drop_metro}  drop_bipartite={drop_bipartite}  "
          f"drop_lost_edges={drop_lost_edges}  mask_lost_input={mask_lost_input}  "
          f"no_ctx_mp={no_ctx_mp}  n_layers={n_layers}")

    torch.manual_seed(seed); np.random.seed(seed)
    events = load_events(data_path)
    train_ev, val_ev, test_ev = split_events(events, val_frac=0.1, test_frac=0.1, seed=seed)

    graph = build_hetero_graph(train_ev, include_picked_up=True)
    graph = graph_to_device(graph, device)
    if drop_metro:
        graph = _drop_metro_edges(graph)
        print(f"  metro edges dropped → 0")
    if drop_bipartite:
        graph = _drop_bipartite_edges(graph)
        print(f"  bipartite (found+lost+pickup) edges dropped → 0")
    if drop_lost_edges:
        graph = _drop_lost_edges(graph)
        print(f"  lost_at edges dropped → 0")
    print(f"  metro={graph['metro_edge_index'].shape[1]}  "
          f"found={graph['found_at_edge_index'].shape[1]}  "
          f"lost={graph['lost_at_edge_index'].shape[1]}  "
          f"pickup={graph['picked_up_edge_index'].shape[1]}")

    # Apply ablation B patch (and remember to restore after)
    if no_ctx_mp:
        HetGNNLayer.forward = _layer_forward_no_ctx_mp
    else:
        HetGNNLayer.forward = _orig_layer_forward

    try:
        model = HetGNN(
            station_feat_dim=graph["station_feats"].shape[1],
            n_contexts=N_CONTEXTS,
            hidden=hidden, n_layers=n_layers, dropout=dropout,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model params: {n_params:,}")

        dist_matrix = build_dist_matrix().to(device)
        train_t = events_to_tensors(train_ev)
        val_t   = events_to_tensors(val_ev)
        test_t  = events_to_tensors(test_ev)
        for d in (train_t, val_t, test_t):
            for k in d: d[k] = d[k].to(device)

        best_hops = float("inf"); best_state = None; best_val = None
        no_improve = 0
        n_train = train_t["pickup"].shape[0]
        t0 = time.time()

        # If we are ablating the lost-station INPUT, replace the lost_st tensor
        # with found_st (model then ignores the lost signal entirely).
        if mask_lost_input:
            for d in (train_t, val_t, test_t):
                d["lost_st"] = d["found_st"].clone()

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
                  f"t1={val['top1']:.3f} t3={val['top3']:.3f} E[h]={val['expected_hops']:.2f}")

            if val["expected_hops"] < best_hops - 1e-4:
                best_hops = val["expected_hops"]
                best_val = val
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  early stop @ ep {epoch}")
                    break

        # Final test on best checkpoint
        model.load_state_dict(best_state)
        test = evaluate(model, graph, test_t, dist_matrix, device)
        print(f"  TEST top1={test['top1']:.3f} top3={test['top3']:.3f} "
              f"top5={test['top5']:.3f} E[hops]={test['expected_hops']:.2f}")

        result = {
            "name": name,
            "config": {"drop_metro": drop_metro, "no_ctx_mp": no_ctx_mp,
                       "n_layers": n_layers, "hidden": hidden, "seed": seed},
            "test_metrics": test,
            "best_val_metrics": best_val,
            "n_params": n_params,
        }

        if out_dir:
            out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "metadata.json").write_text(json.dumps(result, indent=2))

        return result
    finally:
        # always restore original layer behavior so subsequent runs aren't polluted
        HetGNNLayer.forward = _orig_layer_forward


def main():
    data_path = os.environ.get("ABLATE_DATA", "artifacts/data/synth_v2.jsonl")
    base_out = Path(os.environ.get("ABLATE_OUT", "artifacts/models_v2"))
    seed = 0
    common = dict(epochs=40, batch_size=512, hidden=64, dropout=0.1,
                  lr=1e-3, weight_decay=1e-5, patience=8, seed=seed)

    variants = [
        ("full",                  dict(n_layers=2)),
        ("no_metro",              dict(drop_metro=True,  n_layers=2)),
        ("no_ctx_mp",             dict(no_ctx_mp=True,   n_layers=2)),
        ("no_mp",                 dict(n_layers=0)),
        ("only_metro",            dict(drop_bipartite=True, n_layers=2)),
        ("no_lost_edges",         dict(drop_lost_edges=True, n_layers=2)),
        ("no_lost_input",         dict(mask_lost_input=True, n_layers=2)),
        ("no_lost_edges_or_in",   dict(drop_lost_edges=True, mask_lost_input=True, n_layers=2)),
    ]

    results = {}
    for name, cfg in variants:
        out = base_out / name
        r = run_variant(name, data_path, out_dir=out, **{**common, **cfg})
        results[name] = r

    # Baselines: run once against the full graph
    from events import load_events, split_events
    from decision import build_dist_matrix
    device = _device()
    events = load_events(data_path)
    train_ev, _, test_ev = split_events(events, val_frac=0.1, test_frac=0.1, seed=seed)
    dist_matrix = build_dist_matrix().to(device)
    bl = evaluate_baselines(train_ev, test_ev, dist_matrix, device)
    results["_baselines"] = bl

    out_path = base_out / "ablation_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # Print comparison table
    print("\n=== Comparison Table (v2 data with lost_at) ===")
    hdr = f"{'variant':22s} {'params':>9s} {'top1':>6s} {'top3':>6s} {'top5':>6s} {'E[hops]':>8s} {'cov':>5s}"
    print(hdr); print("-" * len(hdr))
    for k, r in results.items():
        if k == "_baselines": continue
        t = r["test_metrics"]
        print(f"{k:22s} {r['n_params']:>9,} {t['top1']:>6.3f} {t['top3']:>6.3f} "
              f"{t['top5']:>6.3f} {t['expected_hops']:>8.2f} {t['coverage']:>5.2f}")
    for name, m in bl.items():
        print(f"{'(baseline) '+name:22s} {'-':>9s} {'-':>6s} {'-':>6s} {'-':>6s} "
              f"{m['expected_hops']:>8.2f} {m['coverage']:>5.2f}")


if __name__ == "__main__":
    main()
