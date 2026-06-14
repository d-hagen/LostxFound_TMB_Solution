"""
Cold-start evaluation of non-GNN ML baselines.

Same cold-start split as coldstart.py (24 of 157 stations held out entirely
from training). Train each baseline on warm events; evaluate on cold-test
(events whose found_at or lost_at is in the held-out set).

Headline: an MLP / LogReg has *no* learned representation for unseen stations
beyond a never-updated one-hot weight. The GNN can build a representation
from those stations' metro neighbors via message passing.
"""
import json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from events import load_events
from decision import build_dist_matrix, recommend_storage
from network import STATION_INDEX, N_STN
from coldstart import split_coldstart
from ml_baselines import featurize, MLP, train_mlp, evaluate_probs


def main():
    import os
    data_path = os.environ.get("ABLATE_DATA", "artifacts/data/synth_v2.jsonl")
    suffix = os.environ.get("ABLATE_TAG", "")
    out_path = Path(f"artifacts/ml_baselines_coldstart{suffix}.json")
    print(f"Data: {data_path}  out: {out_path}")
    device = torch.device("cpu")

    events = load_events(data_path)
    train_ev, val_ev, cold_ev, unseen = split_coldstart(
        events, n_unseen=24, seed=42, val_frac=0.10
    )
    print(f"Stations held out: {len(unseen)}")
    print(f"Events: train={len(train_ev)}  warm-val={len(val_ev)}  cold={len(cold_ev)}")

    Xt_oh, Xt_de, yt = featurize(train_ev)
    Xv_oh, Xv_de, yv = featurize(val_ev)
    Xw_oh, Xw_de, yw = featurize(val_ev)   # warm
    Xc_oh, Xc_de, yc = featurize(cold_ev)  # cold

    dist_matrix = build_dist_matrix().to(device)
    found_cold = np.array([STATION_INDEX[e.found_at] for e in cold_ev])
    found_warm = np.array([STATION_INDEX[e.found_at] for e in val_ev])

    results = {}

    # ── MLP ──
    print("\n[MLP] training on warm only...")
    t0 = time.time()
    mlp, mlp_params = train_mlp(Xt_oh, yt, Xv_oh, yv, device,
                                 hidden=64, n_layers=2, epochs=40, batch=512)
    with torch.no_grad():
        warm_p = F.softmax(mlp(torch.tensor(Xw_oh, device=device)), dim=-1).cpu().numpy()
        cold_p = F.softmax(mlp(torch.tensor(Xc_oh, device=device)), dim=-1).cpu().numpy()
    warm_m = evaluate_probs(warm_p, yw, found_warm, dist_matrix, device)
    cold_m = evaluate_probs(cold_p, yc, found_cold, dist_matrix, device)
    print(f"  warm: top1={warm_m['top1']:.3f} E[hops]={warm_m['expected_hops']:.2f}")
    print(f"  COLD: top1={cold_m['top1']:.3f} E[hops]={cold_m['expected_hops']:.2f}  ({time.time()-t0:.1f}s)")
    results["MLP"] = {"warm": warm_m, "cold": cold_m, "params": mlp_params}

    # ── Logistic Regression ──
    print("\n[LogReg] training on warm only...")
    t0 = time.time()
    lr = LogisticRegression(max_iter=300, solver="lbfgs", C=1.0)
    lr.fit(Xt_oh, yt)
    def _proba(X):
        p = lr.predict_proba(X)
        full = np.full((p.shape[0], N_STN), 1e-9, dtype=np.float32)
        full[:, lr.classes_] = p
        full /= full.sum(axis=1, keepdims=True)
        return full
    warm_p = _proba(Xw_oh); cold_p = _proba(Xc_oh)
    warm_m = evaluate_probs(warm_p, yw, found_warm, dist_matrix, device)
    cold_m = evaluate_probs(cold_p, yc, found_cold, dist_matrix, device)
    print(f"  warm: top1={warm_m['top1']:.3f} E[hops]={warm_m['expected_hops']:.2f}")
    print(f"  COLD: top1={cold_m['top1']:.3f} E[hops]={cold_m['expected_hops']:.2f}  ({time.time()-t0:.1f}s)")
    results["LogReg"] = {"warm": warm_m, "cold": cold_m}

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")

    # Cross-reference with GNN cold-start results
    gnn_cs_path = os.environ.get("GNN_CS_PATH", "artifacts/coldstart/summary.json")
    gnn_cs = json.loads(Path(gnn_cs_path).read_text())
    gnn_full = gnn_cs["cold_test"]["full"]
    gnn_nomp = gnn_cs["cold_test"]["no_mp"]
    bl = gnn_cs["cold_test"]["baselines"]

    print("\n=== Cold-test comparison (events touching unseen stations) ===")
    hdr = f"{'model':22s}  {'top1':>6s}  {'top5':>6s}  {'E[hops]':>8s}"
    print(hdr); print("-" * len(hdr))
    print(f"{'GNN (full)':22s}  {gnn_full['top1']:>6.3f}  "
          f"{gnn_full['top5']:>6.3f}  {gnn_full['expected_hops']:>8.2f}")
    print(f"{'GNN (no message pass.)':22s}  {gnn_nomp['top1']:>6.3f}  "
          f"{gnn_nomp['top5']:>6.3f}  {gnn_nomp['expected_hops']:>8.2f}")
    for name in ("MLP", "LogReg"):
        r = results[name]["cold"]
        print(f"{name:22s}  {r['top1']:>6.3f}  {r['top5']:>6.3f}  "
              f"{r['expected_hops']:>8.2f}")
    print(f"{'baseline: same_station':22s}  {'—':>6s}  {'—':>6s}  "
          f"{bl['same_station']['expected_hops']:>8.2f}")
    print(f"{'baseline: mode_lookup':22s}  {'—':>6s}  {'—':>6s}  "
          f"{bl['mode_lookup']['expected_hops']:>8.2f}")


if __name__ == "__main__":
    main()
