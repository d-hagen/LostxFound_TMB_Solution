"""
Train the heterogeneous GNN for picked_up_at link prediction.

Protocol (inductive over events, transductive over the graph):
  - Split events into train / val / test.
  - Build the heterogeneous graph from TRAIN events only — val/test
    picked_up_at edges never appear in the message-passing substrate.
  - Train: for each minibatch of training events, encode the graph,
    score (context, station) for the batch's contexts, CE loss against
    the observed pickup.
  - Eval: same forward pass; metrics on held-out events.

This is exactly the classical link-prediction setup: predict missing
edges of a relation in a heterogeneous graph.
"""
import os, json, argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from events import load_events, split_events
from graph_build import build_hetero_graph, events_to_tensors, graph_to_device
from model import HetGNN
from decision import build_dist_matrix, recommend_storage
from network import STATION_LIST, STATION_INDEX, STATION_ZONE, ZONE_LIST, N_STN
from contexts import N_CONTEXTS


def _device():
    override = os.environ.get("GNN_DEVICE")
    if override in ("cpu", "cuda", "mps"):
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, graph, eval_t, dist_matrix, device):
    """Return metrics dict. Single forward pass over the graph."""
    model.eval()
    with torch.no_grad():
        h_st, h_ctx = model.encode(graph)
        logits = model.score(h_ctx[eval_t["ctx"]], h_st,
                             found_st_idx=eval_t["found_st"],
                             lost_st_idx=eval_t.get("lost_st"))
        probs = F.softmax(logits, dim=-1)
        loss = (F.cross_entropy(logits, eval_t["pickup"], reduction="none") * eval_t["weight"]).sum().item()

        preds5 = torch.topk(probs, k=5, dim=-1).indices
        tgt = eval_t["pickup"].unsqueeze(1)
        n = eval_t["pickup"].shape[0]
        top1 = (preds5[:, :1] == tgt).any(dim=1).sum().item() / max(n, 1)
        top3 = (preds5[:, :3] == tgt).any(dim=1).sum().item() / max(n, 1)
        top5 = (preds5[:, :5] == tgt).any(dim=1).sum().item() / max(n, 1)

        # decision rule → realised pickup hops
        _, idx = recommend_storage(probs, dist_matrix, top_k=1)
        s_star = idx[:, 0]
        hops = dist_matrix[eval_t["pickup"], s_star]
        chosen = set(s_star.cpu().tolist())
        # per-zone breakdown
        st_zone = torch.tensor(
            [ZONE_LIST.index(STATION_ZONE[STATION_LIST[i]]) for i in range(N_STN)],
            dtype=torch.long, device=device,
        )
        zone_hops = {}
        pickup_zones = st_zone[eval_t["pickup"]]
        for zi, z in enumerate(ZONE_LIST):
            m = pickup_zones == zi
            zone_hops[z] = float(hops[m].mean().item()) if m.any() else 0.0

    return {
        "loss": loss / max(n, 1),
        "top1": top1, "top3": top3, "top5": top5,
        "expected_hops": float(hops.mean().item()),
        "coverage": len(chosen) / N_STN,
        "zone_hops": zone_hops,
    }


def evaluate_baselines(train_events, eval_events, dist_matrix, device):
    """Two baselines:
       - same_station: predict pickup = found_at
       - mode_lookup: per (context, found_at) most common pickup (Laplace smoothed)
    """
    from collections import defaultdict, Counter
    # Train mode lookup
    table = defaultdict(Counter)
    fallback = Counter()
    for e in train_events:
        if not e.pickup:
            continue
        table[(e.context_id, STATION_INDEX[e.found_at])][STATION_INDEX[e.pickup]] += e.weight
        fallback[STATION_INDEX[e.pickup]] += e.weight

    def _proba_lookup():
        p = torch.full((len(eval_events), N_STN), 1e-3, device=device)
        for i, e in enumerate(eval_events):
            key = (e.context_id, STATION_INDEX[e.found_at])
            counts = table.get(key) or fallback
            for s, c in counts.items():
                p[i, s] += c
        return p / p.sum(dim=1, keepdim=True)

    def _proba_same():
        p = torch.zeros(len(eval_events), N_STN, device=device)
        for i, e in enumerate(eval_events):
            p[i, STATION_INDEX[e.found_at]] = 1.0
        return p

    pickup_idx = torch.tensor([STATION_INDEX[e.pickup] for e in eval_events],
                              dtype=torch.long, device=device)
    out = {}
    for name, fn in [("same_station", _proba_same), ("mode_lookup", _proba_lookup)]:
        p = fn()
        _, idx = recommend_storage(p, dist_matrix, top_k=1)
        hops = dist_matrix[pickup_idx, idx[:, 0]]
        out[name] = {"expected_hops": float(hops.mean().item()),
                     "coverage": len(set(idx[:, 0].cpu().tolist())) / N_STN}
    return out


def train(data_path, output_dir, hidden=64, layers=2, lr=1e-3, weight_decay=1e-5,
          epochs=40, batch_size=512, dropout=0.1, seed=0, patience=8):
    device = _device()
    print(f"Device: {device}")
    torch.manual_seed(seed); np.random.seed(seed)

    events = load_events(data_path)
    train_ev, val_ev, test_ev = split_events(events, val_frac=0.1, test_frac=0.1, seed=seed)
    print(f"Events: train={len(train_ev)}  val={len(val_ev)}  test={len(test_ev)}")

    graph = build_hetero_graph(train_ev, include_picked_up=True)
    print(f"Graph: {graph['n_stations']} stations + {graph['n_contexts']} contexts")
    print(f"  metro edges      : {graph['metro_edge_index'].shape[1]:>6d}")
    print(f"  found_at edges   : {graph['found_at_edge_index'].shape[1]:>6d}")
    print(f"  picked_up edges  : {graph['picked_up_edge_index'].shape[1]:>6d}")

    graph = graph_to_device(graph, device)

    model = HetGNN(
        station_feat_dim=graph["station_feats"].shape[1],
        n_contexts=N_CONTEXTS,
        hidden=hidden, n_layers=layers, dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    dist_matrix = build_dist_matrix().to(device)

    train_t = events_to_tensors(train_ev)
    val_t   = events_to_tensors(val_ev)
    test_t  = events_to_tensors(test_ev)
    for d in (train_t, val_t, test_t):
        for k in d: d[k] = d[k].to(device)

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    best_hops = float("inf"); best_epoch = -1; no_improve = 0
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
        print(f"epoch {epoch:2d}  {dt:5.1f}s  train_loss={train_loss:.4f}  "
              f"val_loss={val['loss']:.4f}  top1={val['top1']:.3f}  top3={val['top3']:.3f}  "
              f"E[hops]={val['expected_hops']:.2f}  cov={val['coverage']:.2f}")

        if val["expected_hops"] < best_hops - 1e-4:
            best_hops = val["expected_hops"]; best_epoch = epoch; no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "hidden": hidden, "layers": layers, "dropout": dropout,
                    "station_feat_dim": graph["station_feats"].shape[1],
                    "n_contexts": N_CONTEXTS, "n_stations": N_STN,
                },
                "val_metrics": val,
                "seed": seed,
            }, output_dir / "model.pt")
            # Also persist the training-graph edges so demo / aggregate can
            # reconstruct the same encoding state at inference time.
            torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in graph.items()}, output_dir / "graph.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop @ epoch {epoch} (best {best_epoch})")
                break

    # Final test
    ckpt = torch.load(output_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test = evaluate(model, graph, test_t, dist_matrix, device)
    print(f"\nTEST  top1={test['top1']:.3f}  top5={test['top5']:.3f}  "
          f"E[hops]={test['expected_hops']:.2f}  cov={test['coverage']:.2f}")

    print("\n=== Baselines (on test) ===")
    bl = evaluate_baselines(train_ev, test_ev, dist_matrix, device)
    for name, m in bl.items():
        print(f"  {name:14s}  E[hops]={m['expected_hops']:.2f}  cov={m['coverage']:.2f}")

    delta = test["expected_hops"] - min(m["expected_hops"] for m in bl.values())
    print(f"\nGNN vs best baseline E[hops]: {delta:+.2f}  "
          f"({'WIN' if delta < 0 else 'LOSS — investigate'})")

    meta = {
        "data_path": str(data_path),
        "config": ckpt["config"],
        "test_metrics": test,
        "baselines": bl,
        "seed": seed,
        "n_params": n_params,
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved -> {output_dir}/model.pt + graph.pt + metadata.json")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="artifacts/models/dev")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    train(
        data_path=args.data, output_dir=args.out,
        hidden=args.hidden, layers=args.layers,
        lr=args.lr, dropout=args.dropout,
        epochs=args.epochs, batch_size=args.batch,
        seed=args.seed, patience=args.patience,
    )
