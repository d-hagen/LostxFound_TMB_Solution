"""
Train the GNN destination predictor and compare against baselines.

Reports per-epoch val metrics; saves the best model by expected_hops; runs
final test-set evaluation against same_station / mode_lookup / centroid.

Outputs:
  <output_dir>/model.pt          (state_dict + config + val metrics)
  <output_dir>/metadata.json     (training corpus path, test metrics, baselines)
"""
import sys, os, json, argparse, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from events import load_events, split_events, N_ITEMS
from features import build_static_node_features, build_normalized_adj, batch_events
from model import GNNDestinationPredictor
from decision import recommend_storage, build_dist_matrix
from baselines import SameStationBaseline, ModeLookupBaseline, CentroidBaseline
from network import N_STN, STATION_LIST, STATION_INDEX, STATION_ZONE, ZONE_LIST


def _device():
    """cuda > cpu, overridable via env GNN_DEVICE=cpu|mps|cuda.

    Note: MPS is NOT auto-picked. On Apple Silicon the model is small enough
    that MPS kernel-launch overhead is slower than plain CPU. Opt in via
    GNN_DEVICE=mps if you scale the model up.
    """
    override = os.environ.get("GNN_DEVICE")
    if override in ("cpu", "cuda", "mps"):
        return override
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _precompute_batch_tensors(events, device):
    """Build per-event tensors once, on-device. Per-minibatch becomes index_select."""
    t = batch_events(events)
    return {k: v.to(device) for k, v in t.items()}


def _slice(tensors, idx):
    return {k: v.index_select(0, idx) for k, v in tensors.items()}


def evaluate_model(model, tensors, node_feats, adj, dist_matrix, batch_size=256, device="cpu"):
    """tensors: dict from _precompute_batch_tensors(events, device)."""
    model.eval()
    n_total = tensors["pickup"].shape[0]
    total_loss, n = 0.0, 0
    top1 = top3 = top5 = 0
    total_hops = 0.0
    chosen = set()
    zone_hops = {z: [] for z in ZONE_LIST}

    # Per-station zone-index for vectorized zone bookkeeping
    station_zone_idx = torch.tensor(
        [ZONE_LIST.index(STATION_ZONE[STATION_LIST[i]]) for i in range(N_STN)],
        dtype=torch.long, device=device,
    )

    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            idx = torch.arange(i, min(i + batch_size, n_total), device=device)
            b = _slice(tensors, idx)
            logits = model(node_feats, adj, b["found_idx"], b["item_idx"], b["hour"], b["dow"])
            loss = (F.cross_entropy(logits, b["pickup"], reduction="none") * b["weight"]).sum()
            total_loss += loss.item()
            n += idx.shape[0]

            probs = F.softmax(logits, dim=-1)
            preds5 = torch.topk(probs, k=5, dim=-1).indices
            tgt = b["pickup"].unsqueeze(1)
            top1 += (preds5[:, :1] == tgt).any(dim=1).sum().item()
            top3 += (preds5[:, :3] == tgt).any(dim=1).sum().item()
            top5 += (preds5[:, :5] == tgt).any(dim=1).sum().item()

            _, s_idx = recommend_storage(probs, dist_matrix, top_k=1)
            s_star = s_idx[:, 0]                                  # [B]
            hops = dist_matrix[b["pickup"], s_star]               # [B]
            total_hops += hops.sum().item()
            chosen.update(s_star.cpu().tolist())
            pickup_zones = station_zone_idx[b["pickup"]]          # [B]
            for zi in range(len(ZONE_LIST)):
                mask = pickup_zones == zi
                if mask.any():
                    zone_hops[ZONE_LIST[zi]].extend(hops[mask].cpu().tolist())

    return {
        "loss": total_loss / max(n, 1),
        "top1": top1 / max(n, 1),
        "top3": top3 / max(n, 1),
        "top5": top5 / max(n, 1),
        "expected_hops": total_hops / max(n, 1),
        "coverage": len(chosen) / N_STN,
        "zone_hops": {z: float(np.mean(h)) if h else 0.0 for z, h in zone_hops.items()},
    }


def evaluate_baseline(baseline, events, dist_matrix):
    probs = baseline.predict_proba(events)
    _, idx = recommend_storage(probs, dist_matrix, top_k=1)
    top1 = 0
    total_hops = 0.0
    chosen = set()
    for i, e in enumerate(events):
        pi = STATION_INDEX[e.pickup]
        sj = idx[i, 0].item()
        total_hops += dist_matrix[pi, sj].item()
        chosen.add(sj)
        # Top-1 from the probability mode (not the decision rule)
        top_pred = int(probs[i].argmax().item())
        if top_pred == pi:
            top1 += 1
    return {
        "name": baseline.name,
        "top1": top1 / max(len(events), 1),
        "expected_hops": total_hops / max(len(events), 1),
        "coverage": len(chosen) / N_STN,
    }


def train(data_path, output_dir, hidden=64, layers=3, lr=1e-3, weight_decay=1e-5,
          epochs=30, batch_size=128, dropout=0.1, seed=0, patience=5,
          init_from=None):
    device = _device()
    print(f"Device: {device}")
    torch.manual_seed(seed); np.random.seed(seed)

    events = load_events(data_path)
    train_ev, val_ev, test_ev = split_events(events, val_frac=0.1, test_frac=0.1, seed=seed)
    print(f"Events: train={len(train_ev)} val={len(val_ev)} test={len(test_ev)}")

    node_feats = build_static_node_features().to(device)
    adj = build_normalized_adj().to(device)
    dist_matrix = build_dist_matrix().to(device)

    # If init_from is provided, model architecture is taken from that checkpoint
    # so feature dims match. Otherwise build fresh from current node_feats.
    if init_from:
        init_ckpt = torch.load(init_from, map_location=device, weights_only=False)
        cfg = init_ckpt["config"]
        model = GNNDestinationPredictor(
            n_stations=cfg["n_stations"], node_feat_dim=cfg["node_feat_dim"],
            n_items=cfg["n_items"], hidden=cfg["hidden"],
            n_layers=cfg["layers"], dropout=cfg["dropout"],
        ).to(device)
        model.load_state_dict(init_ckpt["model_state"])
        # Override locals so the metadata reflects the architecture actually trained
        hidden, layers, dropout = cfg["hidden"], cfg["layers"], cfg["dropout"]
        print(f"Initialised from {init_from}  "
              f"(val E[hops] at pretrain end: {init_ckpt['val_metrics']['expected_hops']:.3f})")
    else:
        model = GNNDestinationPredictor(
            n_stations=N_STN, node_feat_dim=node_feats.shape[1], n_items=N_ITEMS,
            hidden=hidden, n_layers=layers, dropout=dropout,
        ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}  lr={lr}")

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    best_val_hops = float("inf"); best_epoch = -1; no_improve = 0

    # Build per-event tensors once on the target device — eliminates the
    # per-minibatch Python loop in batch_events().
    train_t = _precompute_batch_tensors(train_ev, device)
    val_t   = _precompute_batch_tensors(val_ev, device)
    n_train = train_t["pickup"].shape[0]

    epoch_t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0; n = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            b = _slice(train_t, idx)
            logits = model(node_feats, adj, b["found_idx"], b["item_idx"], b["hour"], b["dow"])
            loss = (F.cross_entropy(logits, b["pickup"], reduction="none") * b["weight"]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * idx.shape[0]; n += idx.shape[0]

        train_loss = total_loss / max(n, 1)
        val = evaluate_model(model, val_t, node_feats, adj, dist_matrix, batch_size, device)
        epoch_dt = time.time() - epoch_t0; epoch_t0 = time.time()
        print(f"epoch {epoch:2d}  {epoch_dt:5.1f}s  train_loss={train_loss:.4f}  "
              f"val_loss={val['loss']:.4f}  top1={val['top1']:.3f}  top3={val['top3']:.3f}  "
              f"E[hops]={val['expected_hops']:.2f}  cov={val['coverage']:.2f}")

        if val["expected_hops"] < best_val_hops - 1e-4:
            best_val_hops = val["expected_hops"]; best_epoch = epoch; no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "hidden": hidden, "layers": layers, "dropout": dropout,
                    "node_feat_dim": node_feats.shape[1],
                    "n_items": N_ITEMS, "n_stations": N_STN,
                },
                "val_metrics": val,
                "seed": seed,
            }, output_dir / "model.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop @ epoch {epoch} (best {best_epoch})"); break

    # Final test eval on best checkpoint
    ckpt = torch.load(output_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_t = _precompute_batch_tensors(test_ev, device)
    test = evaluate_model(model, test_t, node_feats, adj, dist_matrix, batch_size, device)
    print(f"\nTEST  top1={test['top1']:.3f}  top5={test['top5']:.3f}  "
          f"E[hops]={test['expected_hops']:.2f}  cov={test['coverage']:.2f}")

    print("\n=== Baselines on test set ===")
    dist_cpu = dist_matrix.cpu()
    baselines = [
        SameStationBaseline().fit(train_ev),
        ModeLookupBaseline().fit(train_ev),
        CentroidBaseline(dist_cpu),
    ]
    baseline_metrics = []
    for bl in baselines:
        m = evaluate_baseline(bl, test_ev, dist_cpu)
        baseline_metrics.append(m)
        print(f"  {m['name']:14s}  top1={m['top1']:.3f}  E[hops]={m['expected_hops']:.2f}  cov={m['coverage']:.2f}")

    delta = test["expected_hops"] - min(m["expected_hops"] for m in baseline_metrics)
    print(f"\nGNN vs best baseline E[hops]: {delta:+.2f}  "
          f"({'WIN' if delta < 0 else 'LOSS — investigate'})")

    metadata = {
        "data_path": str(data_path),
        "config": ckpt["config"],
        "test_metrics": test,
        "baselines": baseline_metrics,
        "seed": seed,
        "n_params": n_params,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved -> {output_dir}/model.pt  +  metadata.json")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="artifacts/models/dev")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--init_from", default=None,
                        help="Checkpoint to initialise weights from (for fine-tuning).")
    args = parser.parse_args()

    train(
        data_path=args.data, output_dir=args.out,
        hidden=args.hidden, layers=args.layers,
        lr=args.lr, dropout=args.dropout,
        epochs=args.epochs, batch_size=args.batch,
        seed=args.seed, patience=args.patience,
        init_from=args.init_from,
    )
