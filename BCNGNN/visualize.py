"""
Visualise a single per-item input graph (one training example).

Shows:
  - The full Barcelona network drawn faintly underneath, for context.
  - The k-hop ego-subgraph around the found station, highlighted in colour.
  - Found station: ★ star marker.
  - Claim station (label): ● ringed marker in green.
  - Zone fill colour per node.
  - Side panel: item category, time bucket, zone target, drift profile,
    a heatmap of the feature matrix (N_sub × F).

Run:
    python3 visualize.py              # one random example
    python3 visualize.py --idx 3      # specific seeded example
    python3 visualize.py --category wallet
"""
from __future__ import annotations

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
from metro import ALL_LINES, LINE_COLORS
from network import G, STATION_LIST, STATION_ZONE, TRANSFERS
from network_map import _GEO, _ZONE_COLORS
from synthetic import generate_dataset, CATEGORY_PROFILES, CATEGORIES, TIME_BUCKETS
from subgraph import build_item_graph, NODE_FEAT_DIM


def get_pos():
    return {s: np.array([lon, lat]) for s, (lat, lon) in _GEO.items() if s in STATION_LIST}


def _load_predictions(event, ig, model_path: Path):
    """If a trained checkpoint exists, return a per-node probability vector
    for the subgraph (else None)."""
    if not model_path.exists():
        return None
    import torch
    from model import StorageGNN, per_item_softmax
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = StorageGNN(ckpt["node_feat_dim"], hidden=a["hidden"], n_layers=a["layers"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        logits = model(ig.x, ig.edge_index)
        batch_idx = torch.zeros(logits.shape[0], dtype=torch.long)
        probs = per_item_softmax(logits, batch_idx, n_items=1)
    return probs.numpy()


def render(event, k: int, out_path: str, model_path: Path | None = None):
    pos = get_pos()
    ig = build_item_graph(event, k=k)
    sub_set = set(ig.nodes)
    pred = _load_predictions(event, ig, model_path) if model_path else None

    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(2, 3, width_ratios=[2.4, 2.4, 1.6], height_ratios=[1.05, 1], figure=fig)
    fig.patch.set_facecolor("white")

    ax_map  = fig.add_subplot(gs[:, 0])
    ax_pred = fig.add_subplot(gs[:, 1])
    ax_info = fig.add_subplot(gs[0, 2])
    ax_feat = fig.add_subplot(gs[1, 2])
    ax_map.set_facecolor("#F5F5F0")
    ax_pred.set_facecolor("#F5F5F0")

    # ── faded full network ──────────────────────────────────────────────
    for line_name, stations in ALL_LINES.items():
        color = LINE_COLORS.get(line_name, {}).get("hex", "#888")
        for i in range(len(stations) - 1):
            s1, s2 = stations[i], stations[i + 1]
            if s1 in pos and s2 in pos:
                ax_map.plot([pos[s1][0], pos[s2][0]], [pos[s1][1], pos[s2][1]],
                            color=color, lw=1.6, alpha=0.12, zorder=1,
                            solid_capstyle='round')

    for s in STATION_LIST:
        if s in pos:
            ax_map.scatter(*pos[s], s=12, c='#BBBBBB', alpha=0.35, zorder=2)

    # ── highlighted subgraph edges ──────────────────────────────────────
    drawn = set()
    for line_name, stations in ALL_LINES.items():
        color = LINE_COLORS.get(line_name, {}).get("hex", "#888")
        for i in range(len(stations) - 1):
            s1, s2 = stations[i], stations[i + 1]
            if s1 in sub_set and s2 in sub_set and s1 in pos and s2 in pos:
                key = tuple(sorted((s1, s2)))
                if key in drawn:
                    continue
                drawn.add(key)
                ax_map.plot([pos[s1][0], pos[s2][0]], [pos[s1][1], pos[s2][1]],
                            color=color, lw=3.5, alpha=0.95, zorder=3,
                            solid_capstyle='round')

    # ── highlighted subgraph nodes ──────────────────────────────────────
    for s in ig.nodes:
        if s not in pos:
            continue
        zone = STATION_ZONE.get(s, "residential")
        c = _ZONE_COLORS.get(zone, "#BDC3C7")
        size = 70 if s not in TRANSFERS else 110
        ax_map.scatter(*pos[s], s=size, c=c, edgecolors='#222',
                       linewidths=0.8, zorder=4)

    # found-station marker
    if event.found_station in pos:
        ax_map.scatter(*pos[event.found_station], marker='*', s=550,
                       c='#FFD700', edgecolors='black', linewidths=1.4, zorder=6,
                       label='found at')

    # claim-station marker
    if event.claim_station in pos:
        ax_map.scatter(*pos[event.claim_station], marker='o', s=380,
                       facecolors='none', edgecolors='#1B5E20',
                       linewidths=3.0, zorder=6,
                       label='claim (label)')

    # ── labels on subgraph nodes ────────────────────────────────────────
    for s in ig.nodes:
        if s not in pos:
            continue
        ax_map.annotate(s, pos[s], fontsize=7.5, ha='center', va='bottom',
                        xytext=(0, 6), textcoords='offset points',
                        fontweight='bold', color='#111',
                        bbox=dict(boxstyle='round,pad=0.18', fc='white',
                                  ec='#AAA', alpha=0.85, lw=0.4),
                        zorder=7)

    # zoom map to subgraph bounding box (with padding)
    sub_xy = np.array([pos[s] for s in ig.nodes if s in pos])
    pad_x = 0.012
    pad_y = 0.008
    ax_map.set_xlim(sub_xy[:, 0].min() - pad_x, sub_xy[:, 0].max() + pad_x)
    ax_map.set_ylim(sub_xy[:, 1].min() - pad_y, sub_xy[:, 1].max() + pad_y)

    # legend / title
    target_zone, target_w, locality = CATEGORY_PROFILES[event.category]
    ax_map.set_title(
        f"INPUT: per-item ego-graph (k={k})\n"
        f"category: {event.category}   found: {event.found_station}   "
        f"time bucket: {event.time_bucket}/{TIME_BUCKETS - 1}\n"
        f"nodes={ig.x.shape[0]}  edges={ig.edge_index.shape[1] // 2}  "
        f"target zone: {target_zone} (w={target_w}, locality={locality})",
        fontsize=11, fontweight='bold', pad=10)
    ax_map.set_aspect('equal')
    ax_map.axis('off')
    ax_map.legend(loc='lower right', fontsize=9)

    # ── prediction panel (or "no trained model" notice) ─────────────────
    _render_prediction_panel(ax_pred, ig, pos, event, k, pred)

    # ── side panel: event meta + label ──────────────────────────────────
    ax_info.axis('off')
    msg = (
        "INPUT TO GNN\n"
        f"-------------------------------\n"
        f"category       : {event.category}\n"
        f"found_station  : {event.found_station}\n"
        f"time_bucket    : {event.time_bucket}\n"
        f"\n"
        f"DRIFT PROFILE (synthetic prior)\n"
        f"-------------------------------\n"
        f"target_zone    : {target_zone}\n"
        f"zone_weight    : {target_w}\n"
        f"distance_decay : {locality}\n"
        f"\n"
        f"LABEL (supervision)\n"
        f"-------------------------------\n"
        f"claim_station  : {event.claim_station}\n"
        f"y_idx (local)  : {ig.y_idx}\n"
        f"found_idx      : {ig.found_idx}\n"
        f"\n"
        f"GRAPH STATS\n"
        f"-------------------------------\n"
        f"sub-nodes      : {ig.x.shape[0]}\n"
        f"sub-edges      : {ig.edge_index.shape[1]}\n"
        f"feature dim    : {NODE_FEAT_DIM}"
    )
    ax_info.text(0, 1, msg, fontsize=9, family='monospace', va='top', ha='left',
                 transform=ax_info.transAxes)

    # ── feature heatmap ─────────────────────────────────────────────────
    feat = ig.x.numpy()
    ax_feat.imshow(feat, aspect='auto', interpolation='nearest', cmap='viridis')
    ax_feat.set_title(f"Node feature matrix [{feat.shape[0]} × {feat.shape[1]}]", fontsize=9)
    ax_feat.set_xlabel("feature dim", fontsize=8)
    ax_feat.set_ylabel("node (local idx)", fontsize=8)
    ax_feat.tick_params(labelsize=7)
    ax_feat.axhline(ig.found_idx, color='#FFD700', lw=1.5, alpha=0.9)
    if ig.y_idx >= 0:
        ax_feat.axhline(ig.y_idx, color='#1B5E20', lw=1.5, alpha=0.9)

    fig.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"saved → {out_path}")


def _render_prediction_panel(ax, ig, pos, event, k, pred):
    """Right-side panel: either the model's claim-station heatmap, or a
    placeholder explaining how to train one."""
    sub_set = set(ig.nodes)

    # faded base lines
    for line_name, stations in ALL_LINES.items():
        color = LINE_COLORS.get(line_name, {}).get("hex", "#888")
        for i in range(len(stations) - 1):
            s1, s2 = stations[i], stations[i + 1]
            if s1 in sub_set and s2 in sub_set and s1 in pos and s2 in pos:
                ax.plot([pos[s1][0], pos[s2][0]], [pos[s1][1], pos[s2][1]],
                        color=color, lw=2.0, alpha=0.45, zorder=2,
                        solid_capstyle='round')

    if pred is not None:
        cmap = matplotlib.colormaps.get_cmap('plasma')
        pmax = max(pred.max(), 1e-6)
        for i, s in enumerate(ig.nodes):
            if s not in pos:
                continue
            p = pred[i]
            ax.scatter(*pos[s], s=120 + 700 * p / pmax,
                       c=[cmap(0.05 + 0.95 * p / pmax)],
                       edgecolors='#222', linewidths=0.6, zorder=4)
            ax.annotate(f"{p:.2f}", pos[s], fontsize=6.5, ha='center', va='center',
                        color='white' if p / pmax > 0.4 else 'black', zorder=5)

        # mark true claim location
        if event.claim_station in pos:
            ax.scatter(*pos[event.claim_station], marker='o', s=380,
                       facecolors='none', edgecolors='#1B5E20',
                       linewidths=3.0, zorder=6, label='true claim')
        if event.found_station in pos:
            ax.scatter(*pos[event.found_station], marker='*', s=420,
                       c='#FFD700', edgecolors='black', linewidths=1.2, zorder=6,
                       label='found')

        top1 = ig.nodes[int(np.argmax(pred))]
        title = (f"OUTPUT: GNN claim-station probability\n"
                 f"argmax = {top1}    P(claim=true) = {pred[ig.y_idx]:.3f}    "
                 f"P_max = {pmax:.3f}")
        ax.legend(loc='lower right', fontsize=9)
    else:
        for s in ig.nodes:
            if s not in pos:
                continue
            ax.scatter(*pos[s], s=80, c='#BDC3C7', edgecolors='#222',
                       linewidths=0.5, zorder=4)
        title = "OUTPUT: (run train.py first to see predictions)"

    sub_xy = np.array([pos[s] for s in ig.nodes if s in pos])
    ax.set_xlim(sub_xy[:, 0].min() - 0.012, sub_xy[:, 0].max() + 0.012)
    ax.set_ylim(sub_xy[:, 1].min() - 0.008, sub_xy[:, 1].max() + 0.008)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)


def pick_event(seed: int, idx: int, category_filter: str | None, k: int, in_subgraph_only: bool):
    """Pick an event by (seed, idx, optional category). If in_subgraph_only,
    filter to events whose claim lies inside the k-hop ego graph (so y_idx >= 0
    and the label marker lands on a visible subgraph node)."""
    events = generate_dataset(500, seed=seed)
    if category_filter:
        events = [e for e in events if e.category == category_filter]
        if not events:
            raise SystemExit(f"no events for category {category_filter!r}")
    if in_subgraph_only:
        events = [e for e in events if build_item_graph(e, k=k).y_idx >= 0]
        if not events:
            raise SystemExit("no events with claim inside subgraph at this k")
    return events[idx % len(events)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx",      type=int, default=0)
    ap.add_argument("--seed",     type=int, default=7)
    ap.add_argument("--k",        type=int, default=5)
    ap.add_argument("--category", type=str, default=None)
    ap.add_argument("--any-claim", action="store_true",
                    help="allow events whose claim falls outside the subgraph")
    ap.add_argument("--model",    type=str, default="artifacts/model.pt",
                    help="path to trained checkpoint (skip overlay if missing)")
    ap.add_argument("--out",      type=str, default="artifacts/single_input.png")
    args = ap.parse_args()

    e = pick_event(args.seed, args.idx, args.category, args.k,
                   in_subgraph_only=not args.any_claim)
    print(f"event → category={e.category}  found={e.found_station}  "
          f"time={e.time_bucket}  claim={e.claim_station}")
    out_path = Path(__file__).parent / args.out
    out_path.parent.mkdir(exist_ok=True)
    model_path = Path(__file__).parent / args.model
    render(e, k=args.k, out_path=str(out_path), model_path=model_path)


if __name__ == "__main__":
    main()
