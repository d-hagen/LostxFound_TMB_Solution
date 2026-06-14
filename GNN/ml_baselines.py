"""
Non-GNN ML baselines on the same lost-and-found task. Compares:

  - MLP (PyTorch, same hidden dim as GNN, no graph)
  - Logistic Regression (sklearn)
  - Random Forest (sklearn)
  - k-NN (sklearn)

Same data split as the GNN (seed 0, 80/10/10), same input features
(found_at, lost_at, item, hour, day), same target (pickup over 157 stations),
same evaluation: top-1/3/5 + E[hops] computed with the transfer-aware matrix
and the GNN's decision rule.
"""
import json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from events import load_events, split_events
from decision import build_dist_matrix, recommend_storage
from network import STATION_INDEX, N_STN
from contexts import (
    ITEM_TYPES, ITEM_INDEX, N_ITEMS,
    HOUR_BUCKETS, DAY_BUCKETS, hour_bucket, day_bucket,
)

N_HOURS = HOUR_BUCKETS
N_DAYS  = DAY_BUCKETS


def featurize(events):
    """Return:
       X_onehot : [N, F] one-hot block features (for LogReg, MLP)
       X_dense  : [N, 5] integer-coded (for RF, kNN)
       y        : [N] pickup station index
    """
    n = len(events)
    F_dim = 2 * N_STN + N_ITEMS + N_HOURS + N_DAYS
    X_onehot = np.zeros((n, F_dim), dtype=np.float32)
    X_dense  = np.zeros((n, 5), dtype=np.int32)
    y = np.zeros(n, dtype=np.int64)
    for i, e in enumerate(events):
        fi = STATION_INDEX[e.found_at]
        li = STATION_INDEX[e.lost_at] if e.lost_at else fi
        ii = ITEM_INDEX.get(e.item_type, ITEM_INDEX["other"])
        hi = hour_bucket(e.hour)
        di = day_bucket(e.dow)

        # one-hot block: [found_at | lost_at | item | hour | day]
        X_onehot[i, fi] = 1.0
        X_onehot[i, N_STN + li] = 1.0
        X_onehot[i, 2*N_STN + ii] = 1.0
        X_onehot[i, 2*N_STN + N_ITEMS + hi] = 1.0
        X_onehot[i, 2*N_STN + N_ITEMS + N_HOURS + di] = 1.0

        X_dense[i] = [fi, li, ii, hi, di]
        y[i] = STATION_INDEX[e.pickup] if e.pickup else -1
    return X_onehot, X_dense, y


def evaluate_probs(probs, pickup_idx, found_idx, dist_matrix, device):
    """Compute top-1/3/5 and E[hops] from a [N, N_STN] probability matrix."""
    probs_t = torch.tensor(probs, dtype=torch.float32, device=device)
    target  = torch.tensor(pickup_idx, dtype=torch.long, device=device)
    n = probs_t.shape[0]
    preds5 = torch.topk(probs_t, k=5, dim=-1).indices
    tgt = target.unsqueeze(1)
    top1 = (preds5[:, :1] == tgt).any(dim=1).float().mean().item()
    top3 = (preds5[:, :3] == tgt).any(dim=1).float().mean().item()
    top5 = (preds5[:, :5] == tgt).any(dim=1).float().mean().item()
    _, idx = recommend_storage(probs_t, dist_matrix, top_k=1)
    s_star = idx[:, 0]
    hops = dist_matrix[target, s_star]
    coverage = len(set(s_star.cpu().tolist())) / N_STN
    return {"top1": top1, "top3": top3, "top5": top5,
            "expected_hops": float(hops.mean().item()), "coverage": coverage}


# ── MLP ──────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, n_layers=2, n_classes=N_STN, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, device, hidden=64,
              n_layers=2, epochs=40, batch=512, lr=1e-3, patience=8):
    in_dim = X_train.shape[1]
    model = MLP(in_dim, hidden=hidden, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    Xt = torch.tensor(X_train, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)
    Xv = torch.tensor(X_val, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)
    best = float("inf"); best_state = None; bad = 0
    n = Xt.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            logits = model(Xt[idx])
            loss = F.cross_entropy(logits, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(Xv), yv).item()
        if vl < best - 1e-4:
            best = vl; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def main():
    import os
    data_path = os.environ.get("ABLATE_DATA", "artifacts/data/synth_v2.jsonl")
    out_suffix = os.environ.get("ABLATE_TAG", "")
    out_path = Path(f"artifacts/ml_baselines{out_suffix}.json")
    print(f"Data: {data_path}  out: {out_path}")
    device = torch.device("cpu")

    events = load_events(data_path)
    train_ev, val_ev, test_ev = split_events(events, val_frac=0.1, test_frac=0.1, seed=0)
    print(f"Events: train={len(train_ev)}  val={len(val_ev)}  test={len(test_ev)}")

    Xt_oh, Xt_de, yt = featurize(train_ev)
    Xv_oh, Xv_de, yv = featurize(val_ev)
    Xe_oh, Xe_de, ye = featurize(test_ev)

    dist_matrix = build_dist_matrix().to(device)
    found_test  = np.array([STATION_INDEX[e.found_at] for e in test_ev])

    results = {}

    # ── MLP ──
    print("\n[MLP] training...")
    t0 = time.time()
    mlp, mlp_params = train_mlp(Xt_oh, yt, Xv_oh, yv, device,
                                 hidden=64, n_layers=2, epochs=40, batch=512)
    with torch.no_grad():
        probs = F.softmax(mlp(torch.tensor(Xe_oh, device=device)), dim=-1).cpu().numpy()
    m = evaluate_probs(probs, ye, found_test, dist_matrix, device)
    m["params"] = mlp_params; m["seconds"] = time.time() - t0
    results["MLP"] = m
    print(f"  top1={m['top1']:.3f} top5={m['top5']:.3f} E[hops]={m['expected_hops']:.2f} "
          f"params={mlp_params:,} ({m['seconds']:.1f}s)")

    # ── Logistic Regression ──
    print("\n[LogReg] training...")
    t0 = time.time()
    lr = LogisticRegression(max_iter=300, n_jobs=-1, solver="lbfgs", C=1.0)
    lr.fit(Xt_oh, yt)
    probs = lr.predict_proba(Xe_oh)
    # sklearn may omit classes that have zero training instances — pad to N_STN
    full = np.full((probs.shape[0], N_STN), 1e-9, dtype=np.float32)
    full[:, lr.classes_] = probs
    full /= full.sum(axis=1, keepdims=True)
    m = evaluate_probs(full, ye, found_test, dist_matrix, device)
    m["params"] = int(lr.coef_.size + lr.intercept_.size); m["seconds"] = time.time() - t0
    results["LogReg"] = m
    print(f"  top1={m['top1']:.3f} top5={m['top5']:.3f} E[hops]={m['expected_hops']:.2f} "
          f"params={m['params']:,} ({m['seconds']:.1f}s)")

    # ── Random Forest ──
    print("\n[RandomForest] training (this may take a minute)...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                 n_jobs=-1, random_state=0)
    rf.fit(Xt_de, yt)
    probs = rf.predict_proba(Xe_de)
    full = np.full((probs.shape[0], N_STN), 1e-9, dtype=np.float32)
    full[:, rf.classes_] = probs
    full /= full.sum(axis=1, keepdims=True)
    m = evaluate_probs(full, ye, found_test, dist_matrix, device)
    m["params"] = None  # not directly comparable
    m["seconds"] = time.time() - t0
    results["RandomForest"] = m
    print(f"  top1={m['top1']:.3f} top5={m['top5']:.3f} E[hops]={m['expected_hops']:.2f} "
          f"({m['seconds']:.1f}s)")

    # ── k-NN ──
    print("\n[kNN k=25] training...")
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)
    knn.fit(Xt_de, yt)
    probs = knn.predict_proba(Xe_de)
    full = np.full((probs.shape[0], N_STN), 1e-9, dtype=np.float32)
    full[:, knn.classes_] = probs
    full /= full.sum(axis=1, keepdims=True)
    m = evaluate_probs(full, ye, found_test, dist_matrix, device)
    m["seconds"] = time.time() - t0
    results["kNN"] = m
    print(f"  top1={m['top1']:.3f} top5={m['top5']:.3f} E[hops]={m['expected_hops']:.2f} "
          f"({m['seconds']:.1f}s)")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")

    print("\n=== ML baselines on this dataset ===")
    hdr = f"{'model':18s}  {'top1':>6s}  {'top3':>6s}  {'top5':>6s}  {'E[hops]':>8s}"
    print(hdr); print("-" * len(hdr))
    for name in ("MLP", "RandomForest", "LogReg", "kNN"):
        r = results[name]
        print(f"{name:18s}  {r['top1']:>6.3f}  {r['top3']:>6.3f}  "
              f"{r['top5']:>6.3f}  {r['expected_hops']:>8.2f}")


if __name__ == "__main__":
    main()
