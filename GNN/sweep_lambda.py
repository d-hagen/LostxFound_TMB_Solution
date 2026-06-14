"""
Sweep lambda in the combined decision rule:
    s* = argmin_s [ E[hops](s) + lambda * move_cost(found, s) ]

For each lambda, evaluate the trained v2 full model on the test set and
report passenger-side E[hops] and operator-side movement cost.

Movement cost defaults: 0.05 per same-line stop, +1.0 per transfer.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from events import load_events, split_events
from graph_build import build_hetero_graph, events_to_tensors, graph_to_device
from model import HetGNN
from decision import (
    build_dist_matrix, build_move_cost_matrix, recommend_storage_combined,
)
from network import N_STN
from contexts import N_CONTEXTS
from train import _device


def main():
    data_path = "artifacts/data/synth_v2.jsonl"
    ckpt_path = Path("artifacts/models_v2/full/metadata.json").parent
    device = _device()
    print(f"Device: {device}")

    # Load events and rebuild same split as the trained model
    events = load_events(data_path)
    train_ev, val_ev, test_ev = split_events(events, val_frac=0.1, test_frac=0.1, seed=0)
    graph = build_hetero_graph(train_ev, include_picked_up=True)
    graph = graph_to_device(graph, device)

    # Reconstruct model and load checkpoint
    meta_path = Path("artifacts/models_v2/full/metadata.json")
    cfg = json.loads(meta_path.read_text())["config"]
    model = HetGNN(
        station_feat_dim=graph["station_feats"].shape[1],
        n_contexts=N_CONTEXTS,
        hidden=cfg["hidden"], n_layers=cfg["n_layers"], dropout=0.1,
    ).to(device)
    # The full-model checkpoint isn't saved separately by ablate.py — retrain
    # quickly here on the same split so we have a model in memory.
    # (For a real run you'd persist the state_dict; this is a small re-train.)
    print("Re-training v2 full model briefly to obtain weights...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_t = events_to_tensors(train_ev)
    val_t   = events_to_tensors(val_ev)
    test_t  = events_to_tensors(test_ev)
    for d in (train_t, val_t, test_t):
        for k in d: d[k] = d[k].to(device)

    n_train = train_t["pickup"].shape[0]
    import copy
    best_val = float("inf"); best_state = None; bad = 0
    dist_matrix = build_dist_matrix().to(device)

    for epoch in range(40):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, 512):
            idx = perm[i:i + 512]
            h_st, h_ctx = model.encode(graph)
            logits = model.score(h_ctx[train_t["ctx"][idx]], h_st,
                                 found_st_idx=train_t["found_st"][idx],
                                 lost_st_idx=train_t["lost_st"][idx])
            loss = (F.cross_entropy(logits, train_t["pickup"][idx], reduction="none")
                    * train_t["weight"][idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            h_st, h_ctx = model.encode(graph)
            logits = model.score(h_ctx[val_t["ctx"]], h_st,
                                 found_st_idx=val_t["found_st"],
                                 lost_st_idx=val_t["lost_st"])
            vl = F.cross_entropy(logits, val_t["pickup"]).item()
        if vl < best_val - 1e-4:
            best_val = vl; best_state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
            if bad >= 8: break
    model.load_state_dict(best_state)
    print(f"Trained {epoch+1} epochs; best val CE = {best_val:.4f}")

    # Build both cost matrices
    move_cost_matrix = build_move_cost_matrix(step_cost=0.05, transfer_penalty=1.0).to(device)

    # Run sweep
    model.eval()
    with torch.no_grad():
        h_st, h_ctx = model.encode(graph)
        logits = model.score(h_ctx[test_t["ctx"]], h_st,
                             found_st_idx=test_t["found_st"],
                             lost_st_idx=test_t["lost_st"])
        probs = F.softmax(logits, dim=-1)

    lambdas = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    rows = []
    pickup_idx = test_t["pickup"]
    found_idx = test_t["found_st"]
    for lam in lambdas:
        _, idx, E_hops_per_s, move_per_s = recommend_storage_combined(
            probs, dist_matrix, move_cost_matrix,
            found_st_idx=found_idx, lam=lam, top_k=1,
        )
        s_star = idx[:, 0]
        # Realised passenger metric (true pickup → chosen storage)
        realised_hops = dist_matrix[pickup_idx, s_star].mean().item()
        # Realised operator metric (found → chosen storage)
        realised_move = move_cost_matrix[found_idx, s_star].mean().item()
        coverage = len(set(s_star.cpu().tolist())) / N_STN
        rows.append({
            "lambda": lam,
            "E_hops_realised": realised_hops,
            "move_cost_realised": realised_move,
            "coverage": coverage,
        })

    print("\n=== Lambda sweep: passenger vs operator cost ===")
    print(f"{'lambda':>8s}  {'E[hops]':>9s}  {'move_cost':>10s}  {'coverage':>9s}")
    print("-" * 44)
    for r in rows:
        print(f"{r['lambda']:>8.2f}  {r['E_hops_realised']:>9.3f}  "
              f"{r['move_cost_realised']:>10.3f}  {r['coverage']:>9.3f}")

    out = Path("artifacts/models_v2/lambda_sweep.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
