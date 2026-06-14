"""
Train the inductive per-item GNN on synthetic lost-and-found events.

Each training example = one per-item ego-graph. The model never sees a
single shared global graph — true inductive learning over many graphs.

Metrics:
  loss, top-1, top-3 (does the true claim station rank in top-k of the
  item's subgraph nodes?).
"""
from __future__ import annotations

import os
import sys
import json
import random
import argparse
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))
from synthetic import generate_dataset, save_dataset, load_dataset, Event
from subgraph import build_item_graph, collate_batch, NODE_FEAT_DIM, ItemGraph
from model import StorageGNN, per_item_cross_entropy, per_item_softmax


def topk_in_item(probs, batch_idx, offsets, y_local, k: int):
    """Fraction of items whose true claim node is in the top-k by prob."""
    n_items = offsets.shape[0]
    hits = 0
    for b in range(n_items):
        mask = (batch_idx == b)
        p = probs[mask]
        target_local = y_local[b].item()
        topk = torch.topk(p, k=min(k, p.shape[0])).indices.tolist()
        if target_local in topk:
            hits += 1
    return hits / max(n_items, 1)


def evaluate(model, items: list[ItemGraph], batch_size: int = 64, device: str = "cpu"):
    model.eval()
    total_loss, total_n = 0.0, 0
    top1_hits, top3_hits = 0, 0
    n = len(items)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = items[i:i + batch_size]
            batch = collate_batch(chunk)
            for k, v in batch.items():
                batch[k] = v.to(device)
            logits = model(batch["x"], batch["edge_index"],
                           batch_idx=batch["batch_idx"],
                           found_global_idx=batch["found_global"])
            loss = per_item_cross_entropy(logits, batch)
            probs = per_item_softmax(logits, batch["batch_idx"], len(chunk))
            top1_hits += topk_in_item(probs, batch["batch_idx"], batch["offsets"], batch["y_local"], 1) * len(chunk)
            top3_hits += topk_in_item(probs, batch["batch_idx"], batch["offsets"], batch["y_local"], 3) * len(chunk)
            total_loss += loss.item() * len(chunk)
            total_n += len(chunk)
    return {
        "loss":   total_loss / max(total_n, 1),
        "top1":   top1_hits  / max(total_n, 1),
        "top3":   top3_hits  / max(total_n, 1),
        "n":      total_n,
    }


def filter_in_subgraph(events: list[Event], k: int) -> list[ItemGraph]:
    """Build ItemGraphs and drop events whose claim is outside the ego-graph."""
    out = []
    for e in events:
        ig = build_item_graph(e, k=k)
        if ig.y_idx >= 0:
            out.append(ig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",        type=int, default=4000)
    ap.add_argument("--k",        type=int, default=5)
    ap.add_argument("--epochs",   type=int, default=20)
    ap.add_argument("--batch",    type=int, default=64)
    ap.add_argument("--hidden",   type=int, default=64)
    ap.add_argument("--layers",   type=int, default=3)
    ap.add_argument("--lr",       type=float, default=2e-3)
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--out",      type=str, default="artifacts")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    out_dir = Path(__file__).parent / args.out
    out_dir.mkdir(exist_ok=True)

    # ── dataset ─────────────────────────────────────────────────────────
    events = generate_dataset(args.n, seed=args.seed)
    save_dataset(events, out_dir / "events.json")
    items = filter_in_subgraph(events, k=args.k)
    print(f"events: {len(events)}, in-subgraph: {len(items)} ({len(items)/len(events):.1%})")

    rng = random.Random(args.seed)
    rng.shuffle(items)
    n_train = int(0.8 * len(items))
    n_val = int(0.1 * len(items))
    train_items = items[:n_train]
    val_items   = items[n_train:n_train + n_val]
    test_items  = items[n_train + n_val:]
    print(f"split: train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    # ── model ───────────────────────────────────────────────────────────
    model = StorageGNN(NODE_FEAT_DIM, hidden=args.hidden, n_layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    # ── train ───────────────────────────────────────────────────────────
    history = []
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(train_items)
        ep_loss, ep_n = 0.0, 0
        for i in range(0, len(train_items), args.batch):
            chunk = train_items[i:i + args.batch]
            batch = collate_batch(chunk)
            for k, v in batch.items():
                batch[k] = v.to(device)
            logits = model(batch["x"], batch["edge_index"],
                           batch_idx=batch["batch_idx"],
                           found_global_idx=batch["found_global"])
            loss = per_item_cross_entropy(logits, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(chunk)
            ep_n += len(chunk)

        train_loss = ep_loss / max(ep_n, 1)
        val = evaluate(model, val_items, batch_size=args.batch, device=device)
        history.append({"epoch": ep, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        print(f"ep {ep:2d}  train_loss={train_loss:.4f}  "
              f"val_loss={val['loss']:.4f}  top1={val['top1']:.3f}  top3={val['top3']:.3f}")

    # ── final eval ─────────────────────────────────────────────────────
    test = evaluate(model, test_items, batch_size=args.batch, device=device)
    print(f"\nFINAL TEST  loss={test['loss']:.4f}  top1={test['top1']:.3f}  top3={test['top3']:.3f}  (n={test['n']})")

    # baseline: pick a random node in the subgraph
    rand_top1 = sum(1.0 / it.x.shape[0] for it in test_items) / len(test_items)
    print(f"random baseline top1: {rand_top1:.3f}")

    # ── save ────────────────────────────────────────────────────────────
    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
        "node_feat_dim": NODE_FEAT_DIM,
    }, out_dir / "model.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump({"history": history, "test": test, "rand_top1": rand_top1}, f, indent=2)
    print(f"saved → {out_dir}/model.pt, {out_dir}/history.json")


if __name__ == "__main__":
    main()
