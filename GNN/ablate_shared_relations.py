"""
Ablation: collapse the four per-relation weight matrices into a single
shared one — the R-GCN -> vanilla GCN downgrade. The graph structure is
unchanged; only the relational distinction between metro / found / lost /
picked-up is removed at the parameter level.

Compares against the full R-GCN model trained on the same v2 data.
"""
import json, time, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from events import load_events, split_events
from graph_build import build_hetero_graph, events_to_tensors, graph_to_device
from model import HetGNN, HetGNNLayer, _scatter_sum
from decision import build_dist_matrix, recommend_storage
from network import N_STN
from contexts import N_CONTEXTS
from train import evaluate, _device


# ── Build a HetGNNLayer where all 7 relation linears are the SAME tied module
class SharedHetGNNLayer(HetGNNLayer):
    def __init__(self, dim, dropout=0.1):
        super().__init__(dim, dropout)
        # Tie all 7 per-relation linears (4 fwd + 3 rev) to a single shared
        # Linear instance. Self-loops keep their own matrices so that the
        # "self" component remains learnable separately — without this the
        # model can't distinguish a node from the sum of its neighbors.
        shared = nn.Linear(dim, dim)
        self.lin_metro     = shared
        self.lin_found_fwd = shared
        self.lin_found_rev = shared
        self.lin_lost_fwd  = shared
        self.lin_lost_rev  = shared
        self.lin_pick_fwd  = shared
        self.lin_pick_rev  = shared


class SharedHetGNN(HetGNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace each layer with the shared variant
        hidden = kwargs.get("hidden", 64)
        n_layers = kwargs.get("n_layers", 2)
        dropout = kwargs.get("dropout", 0.1)
        self.layers = nn.ModuleList([SharedHetGNNLayer(hidden, dropout)
                                     for _ in range(n_layers)])


def train_and_eval(name, ModelClass, *, epochs=40, batch_size=512,
                   hidden=64, n_layers=2, dropout=0.1, lr=1e-3,
                   weight_decay=1e-5, patience=8, seed=0,
                   data_path="artifacts/data/synth_v2.jsonl"):
    device = _device()
    print(f"\n========== {name}  (device={device}) ==========")
    torch.manual_seed(seed); np.random.seed(seed)

    events = load_events(data_path)
    train_ev, val_ev, test_ev = split_events(events, val_frac=0.1, test_frac=0.1, seed=seed)
    graph = build_hetero_graph(train_ev, include_picked_up=True)
    graph = graph_to_device(graph, device)

    model = ModelClass(
        station_feat_dim=graph["station_feats"].shape[1],
        n_contexts=N_CONTEXTS,
        hidden=hidden, n_layers=n_layers, dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Count effective parameters (deduplicated by id)
    seen, n_params = set(), 0
    for p in model.parameters():
        if id(p) in seen: continue
        seen.add(id(p))
        n_params += p.numel()
    print(f"  effective params: {n_params:,}")

    dist_matrix = build_dist_matrix().to(device)
    train_t = events_to_tensors(train_ev)
    val_t   = events_to_tensors(val_ev)
    test_t  = events_to_tensors(test_ev)
    for d in (train_t, val_t, test_t):
        for k in d: d[k] = d[k].to(device)

    best_hops = float("inf"); best_state = None; bad = 0
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
            best_state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop @ ep {epoch}")
                break

    model.load_state_dict(best_state)
    test = evaluate(model, graph, test_t, dist_matrix, device)
    print(f"  TEST top1={test['top1']:.3f} top3={test['top3']:.3f} "
          f"top5={test['top5']:.3f} E[hops]={test['expected_hops']:.2f}")
    return {"name": name, "test": test, "n_params": n_params}


def main():
    import os
    data_path = os.environ.get("ABLATE_DATA", "artifacts/data/synth_v2.jsonl")
    out_dir = Path(os.environ.get("ABLATE_OUT", "artifacts/models_v2"))
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results["full"]   = train_and_eval("full (R-GCN, per-relation weights)",
                                        HetGNN, data_path=data_path)
    results["shared"] = train_and_eval("shared (vanilla-GCN, one shared weight)",
                                        SharedHetGNN, data_path=data_path)

    out = out_dir / "shared_relations.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    print("\n=== Per-relation vs shared weights ===")
    hdr = f"{'variant':38s}  {'params':>9s}  {'top1':>6s}  {'top5':>6s}  {'E[hops]':>8s}"
    print(hdr); print("-" * len(hdr))
    for k in ("full", "shared"):
        r = results[k]; t = r["test"]
        print(f"{r['name']:38s}  {r['n_params']:>9,}  {t['top1']:>6.3f}  "
              f"{t['top5']:>6.3f}  {t['expected_hops']:>8.2f}")


if __name__ == "__main__":
    main()
