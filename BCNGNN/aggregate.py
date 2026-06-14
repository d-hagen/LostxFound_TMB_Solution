"""
Run the trained GNN over every synthetic event and aggregate predictions
per station — the demand heatmap that would feed a facility-location step.

Two aggregations:
  • argmax_count : how often the station is the model's top-1 prediction
  • prob_sum     : total predicted probability mass over all events
                   (a softer demand estimate)

Outputs:
  artifacts/aggregation.json   per-station tallies + per-category breakdown
  artifacts/aggregation_map.png  Barcelona map sized/coloured by demand
  stdout                       ranked top-N stations
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from metro import ALL_LINES, LINE_COLORS
from network import G, STATION_LIST, STATION_ZONE, TRANSFERS
from network_map import _GEO, _ZONE_COLORS
from synthetic import generate_dataset, CATEGORIES, CATEGORY_PROFILES
from subgraph import build_item_graph, NODE_FEAT_DIM
from model import StorageGNN, per_item_softmax


def load_model(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = StorageGNN(ckpt["node_feat_dim"], hidden=a["hidden"], n_layers=a["layers"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, a


def aggregate(model, events, k: int):
    """Per-station tallies across the whole event set."""
    argmax_count = Counter()
    prob_sum     = defaultdict(float)
    appearances  = Counter()
    # by category
    argmax_by_cat = defaultdict(Counter)
    prob_by_cat   = defaultdict(lambda: defaultdict(float))

    with torch.no_grad():
        for ev in events:
            ig = build_item_graph(ev, k=k)
            logits = model(ig.x, ig.edge_index)
            batch_idx = torch.zeros(logits.shape[0], dtype=torch.long)
            probs = per_item_softmax(logits, batch_idx, n_items=1).numpy()

            top = ig.nodes[int(np.argmax(probs))]
            argmax_count[top] += 1
            argmax_by_cat[ev.category][top] += 1

            for i, s in enumerate(ig.nodes):
                prob_sum[s] += float(probs[i])
                prob_by_cat[ev.category][s] += float(probs[i])
                appearances[s] += 1

    return {
        "argmax_count": dict(argmax_count),
        "prob_sum":     dict(prob_sum),
        "appearances":  dict(appearances),
        "argmax_by_cat": {c: dict(v) for c, v in argmax_by_cat.items()},
        "prob_by_cat":   {c: dict(v) for c, v in prob_by_cat.items()},
    }


_BAR = "█"
_BAR_W = 40


def _bar(value: float, vmax: float, width: int = _BAR_W) -> str:
    if vmax <= 0:
        return ""
    n = int(round(width * value / vmax))
    return _BAR * n + " " * (width - n)


_ZONE_TERM_COLOR = {
    "tourist":     "\033[91m",   # red
    "business":    "\033[94m",   # blue
    "university":  "\033[95m",   # magenta
    "hospital":    "\033[92m",   # green
    "airport":     "\033[93m",   # yellow
    "leisure":     "\033[96m",   # cyan
    "industrial":  "\033[90m",   # grey
    "residential": "\033[37m",   # light grey
}
_RESET = "\033[0m"


def _color(zone: str) -> tuple[str, str]:
    return _ZONE_TERM_COLOR.get(zone, ""), _RESET


def print_top(agg, k: int = 20):
    # ── argmax bars ──────────────────────────────────────────────────────
    print(f"\n── Top {k} stations by argmax pick count ──")
    items = sorted(agg["argmax_count"].items(), key=lambda kv: -kv[1])[:k]
    vmax = items[0][1] if items else 1
    for stn, n in items:
        zone = STATION_ZONE.get(stn, "?")
        rate = n / max(agg["appearances"].get(stn, 1), 1)
        c, r = _color(zone)
        bar = _bar(n, vmax)
        print(f"  {stn:32s} {c}{bar}{r} {n:4d}  zone={c}{zone:11s}{r}  rate={rate:.2f}")

    # ── prob-sum bars ────────────────────────────────────────────────────
    print(f"\n── Top {k} stations by total predicted prob mass ──")
    items = sorted(agg["prob_sum"].items(), key=lambda kv: -kv[1])[:k]
    vmax = items[0][1] if items else 1.0
    for stn, p in items:
        zone = STATION_ZONE.get(stn, "?")
        c, r = _color(zone)
        bar = _bar(p, vmax)
        print(f"  {stn:32s} {c}{bar}{r} {p:7.2f}  zone={c}{zone}{r}")

    # ── per-category top-5 bars ─────────────────────────────────────────
    print("\n── Per-category argmax distribution (top 5 stations) ──")
    for cat, ctr in agg["argmax_by_cat"].items():
        if not ctr:
            continue
        target_zone = CATEGORY_PROFILES[cat][0]
        tc, tr = _color(target_zone)
        rows = sorted(ctr.items(), key=lambda kv: -kv[1])[:5]
        vmax = rows[0][1] if rows else 1
        total = sum(ctr.values())
        match_top = "✓" if STATION_ZONE.get(rows[0][0]) == target_zone else "·"
        print(f"\n  {cat:13s}  target_zone={tc}{target_zone}{tr}  total_picks={total}  top-match={match_top}")
        for stn, n in rows:
            zone = STATION_ZONE.get(stn, "?")
            c, r = _color(zone)
            bar = _bar(n, vmax, width=30)
            print(f"    {stn:30s} {c}{bar}{r} {n:4d}  ({zone})")

    # ── zone share of all argmax picks ──────────────────────────────────
    zone_counts = Counter()
    for stn, n in agg["argmax_count"].items():
        zone_counts[STATION_ZONE.get(stn, "?")] += n
    total = sum(zone_counts.values()) or 1
    print(f"\n── Zone share of all argmax picks ──")
    vmax = max(zone_counts.values()) if zone_counts else 1
    for zone, n in sorted(zone_counts.items(), key=lambda kv: -kv[1]):
        c, r = _color(zone)
        bar = _bar(n, vmax)
        pct = 100 * n / total
        print(f"  {zone:12s} {c}{bar}{r} {n:5d}  ({pct:5.1f}%)")


def render_map(agg, out_path: Path, mode: str = "argmax"):
    pos = {s: np.array([lon, lat]) for s, (lat, lon) in _GEO.items() if s in STATION_LIST}

    if mode == "argmax":
        values = agg["argmax_count"]
        title = "GNN demand heatmap — argmax pick count"
        label = "picks"
    else:
        values = agg["prob_sum"]
        title = "GNN demand heatmap — total predicted probability mass"
        label = "prob sum"

    vmax = max(values.values()) if values else 1.0

    fig, ax = plt.subplots(figsize=(17, 16))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F5F5F0")

    # base lines
    for line_name, stations in ALL_LINES.items():
        color = LINE_COLORS.get(line_name, {}).get("hex", "#888")
        for i in range(len(stations) - 1):
            s1, s2 = stations[i], stations[i + 1]
            if s1 in pos and s2 in pos:
                ax.plot([pos[s1][0], pos[s2][0]], [pos[s1][1], pos[s2][1]],
                        color=color, lw=2.0, alpha=0.30, zorder=1,
                        solid_capstyle='round')

    # stations sized by value
    cmap = matplotlib.colormaps.get_cmap('plasma')
    for stn in STATION_LIST:
        if stn not in pos:
            continue
        v = values.get(stn, 0)
        if v <= 0:
            ax.scatter(*pos[stn], s=14, c='#DDDDDD', edgecolors='#666',
                       linewidths=0.3, zorder=2)
            continue
        frac = v / vmax
        ax.scatter(*pos[stn], s=80 + 1400 * frac,
                   c=[cmap(0.10 + 0.85 * frac)],
                   edgecolors='#111', linewidths=0.7, zorder=4)

    # label top-15 by value
    top = sorted(values.items(), key=lambda kv: -kv[1])[:15]
    for stn, v in top:
        if stn not in pos:
            continue
        ax.annotate(f"{stn}\n({v:.0f})" if mode == "argmax" else f"{stn}\n({v:.1f})",
                    pos[stn], fontsize=8, ha='center', va='bottom',
                    xytext=(0, 14), textcoords='offset points',
                    fontweight='bold', color='#111',
                    bbox=dict(boxstyle='round,pad=0.22', fc='white',
                              ec='#888', alpha=0.92, lw=0.5),
                    zorder=6)

    ax.set_title(
        f"{title}\n(n_events = {sum(agg['argmax_count'].values())}, "
        f"max {label} = {vmax:.0f})",
        fontsize=13, fontweight='bold', pad=12)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.savefig(out_path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"saved → {out_path}")


def render_per_category(agg, out_path: Path):
    """4 × 2 grid of mini Barcelona maps, one per category, by argmax count."""
    pos = {s: np.array([lon, lat]) for s, (lat, lon) in _GEO.items() if s in STATION_LIST}
    cats = list(agg["argmax_by_cat"].keys())
    n = len(cats)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    fig.patch.set_facecolor("white")
    if rows == 1:
        axes = [axes]

    cmap = matplotlib.colormaps.get_cmap('plasma')

    for idx, cat in enumerate(cats):
        ax = axes[idx // cols][idx % cols] if rows > 1 else axes[0][idx]
        ax.set_facecolor("#F5F5F0")
        values = agg["argmax_by_cat"][cat]
        vmax = max(values.values()) if values else 1.0
        target_zone = CATEGORY_PROFILES[cat][0]

        for line_name, stations in ALL_LINES.items():
            color = LINE_COLORS.get(line_name, {}).get("hex", "#888")
            for i in range(len(stations) - 1):
                s1, s2 = stations[i], stations[i + 1]
                if s1 in pos and s2 in pos:
                    ax.plot([pos[s1][0], pos[s2][0]], [pos[s1][1], pos[s2][1]],
                            color=color, lw=1.4, alpha=0.22, zorder=1,
                            solid_capstyle='round')

        for stn in STATION_LIST:
            if stn not in pos:
                continue
            v = values.get(stn, 0)
            if v <= 0:
                ax.scatter(*pos[stn], s=6, c='#DDDDDD', alpha=0.5, zorder=2)
                continue
            frac = v / vmax
            ax.scatter(*pos[stn], s=40 + 700 * frac,
                       c=[cmap(0.10 + 0.85 * frac)],
                       edgecolors='#111', linewidths=0.4, zorder=4)

        # label top-3
        for stn, v in sorted(values.items(), key=lambda kv: -kv[1])[:3]:
            if stn in pos:
                ax.annotate(stn, pos[stn], fontsize=6, ha='center', va='bottom',
                            xytext=(0, 6), textcoords='offset points',
                            fontweight='bold', color='#111',
                            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                      ec='#999', alpha=0.85, lw=0.4),
                            zorder=5)

        ax.set_title(f"{cat}   (target: {target_zone})", fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')

    # hide unused subplots
    for j in range(n, rows * cols):
        ax = axes[j // cols][j % cols] if rows > 1 else axes[0][j]
        ax.axis('off')

    fig.suptitle("GNN argmax pick count per category", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="artifacts/model.pt")
    ap.add_argument("--events", type=str, default="artifacts/events.json",
                    help="JSON file produced by synthetic.py; if missing, generates")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="artifacts")
    args = ap.parse_args()

    out_dir = Path(__file__).parent / args.out_dir
    out_dir.mkdir(exist_ok=True)
    model_path = Path(__file__).parent / args.model
    events_path = Path(__file__).parent / args.events

    model, model_args = load_model(model_path)
    print(f"loaded model: {model_args}")

    if events_path.exists():
        from synthetic import load_dataset
        events = load_dataset(str(events_path))
        print(f"loaded {len(events)} events from {events_path}")
    else:
        events = generate_dataset(args.n, seed=args.seed)
        print(f"generated {len(events)} events")

    agg = aggregate(model, events, k=args.k)
    print_top(agg, k=15)

    # save
    with open(out_dir / "aggregation.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nsaved aggregation → {out_dir/'aggregation.json'}")

    render_map(agg, out_dir / "aggregation_map_argmax.png", mode="argmax")
    render_map(agg, out_dir / "aggregation_map_probsum.png", mode="prob")
    render_per_category(agg, out_dir / "aggregation_by_category.png")


if __name__ == "__main__":
    main()
