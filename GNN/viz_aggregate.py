"""
Visualise the aggregation — per-station predicted weekly storage load.

Run the trained v2 model over every event in a corpus, tally which station
the model would route each item to, and plot it on the Barcelona map.

Output: artifacts/viz/03_aggregate_map.png
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn.functional as F

from events import load_events
from graph_build import events_to_tensors
from model import HetGNN
from decision import build_dist_matrix, recommend_storage
from network import STATION_LIST, N_STN, STATION_ZONE
from network_map import _GEO
from metro import ALL_LINES
from aggregate import find_default_model, load_model_and_graph


OUT = Path(__file__).parent / "artifacts" / "viz"
OUT.mkdir(parents=True, exist_ok=True)


def geo_xy(station):
    lat, lon = _GEO.get(station, (41.39, 2.17))
    return lon, lat


def tally_storage(events, model, graph, dist_matrix, batch_size=1024):
    tally = torch.zeros(N_STN, dtype=torch.long)
    et = events_to_tensors(events)
    with torch.no_grad():
        h_st, h_ctx = model.encode(graph)
        n = et["pickup"].shape[0]
        for i in range(0, n, batch_size):
            sl = slice(i, i + batch_size)
            logits = model.score(
                h_ctx[et["ctx"][sl]], h_st,
                et["found_st"][sl],
                lost_st_idx=et.get("lost_st", et["found_st"])[sl],
            )
            probs = F.softmax(logits, dim=-1)
            _, idx = recommend_storage(probs, dist_matrix, top_k=1)
            chosen = idx[:, 0]
            tally.index_add_(0, chosen, torch.ones_like(chosen, dtype=torch.long))
    return tally


def weeks_spanned(events):
    ts = [datetime.fromisoformat(e.found_dt) for e in events]
    return max(((max(ts) - min(ts)).days + 1) / 7.0, 1.0)


def draw_map(tally, weeks, top_label=40, out_path=None):
    """Two-panel figure:
       LEFT  — map. Nodes sized by items/week, each non-zero node has its
               items/week number written inside it. No station labels.
       RIGHT — a ranked text table of stations (like the CLI output).
    """
    total = int(tally.sum().item())
    per_week = tally.float().numpy() / weeks
    max_pw = per_week.max() if per_week.max() > 0 else 1.0

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.05)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])

    fig.suptitle(
        f"v2 GNN — predicted storage load   "
        f"({total} items over {weeks:.0f} weeks)",
        fontsize=15, fontweight="bold", y=0.97,
    )

    # ── LEFT: metro map ─────────────────────────────────────────────────
    drawn = set()
    for stations in ALL_LINES.values():
        for a, b in zip(stations[:-1], stations[1:]):
            if a in _GEO and b in _GEO and (a, b) not in drawn:
                xa, ya = geo_xy(a); xb, yb = geo_xy(b)
                ax_map.plot([xa, xb], [ya, yb], color="#d8d8d8",
                            lw=1.0, zorder=1)
                drawn.add((a, b)); drawn.add((b, a))

    cmap = LinearSegmentedColormap.from_list(
        "load", ["#f3f4f6", "#fff7b3", "#f59e0b", "#dc2626", "#7f1d1d"]
    )

    sizes, colors, pts = [], [], []
    for i, s in enumerate(STATION_LIST):
        if s not in _GEO: continue
        pw = per_week[i]
        if pw <= 0:
            x, y = geo_xy(s)
            ax_map.scatter(x, y, s=10, marker="o", facecolor="white",
                           edgecolor="#bbbbbb", linewidths=0.5, zorder=2)
        else:
            x, y = geo_xy(s)
            # Sub-linear (square-root-ish) growth so the top stations don't
            # overwhelm the map visually.
            size = 100 + 600 * (pw / max_pw) ** 0.55
            sizes.append(size); colors.append(pw / max_pw)
            pts.append((x, y, s, pw, size))

    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    sc = ax_map.scatter(xs, ys, s=sizes, c=colors, cmap=cmap, vmin=0, vmax=1,
                        edgecolor="black", linewidths=0.7, zorder=3, alpha=0.95)

    # Number inside each non-zero node — only if the node is big enough
    # that the text is readable.
    for x, y, _s, pw, size in pts:
        if size < 220:
            continue
        color = "white" if (pw / max_pw) > 0.55 else "#111"
        fontsize = 7 + min(7, int(size / 240))
        label = f"{pw:.0f}" if pw >= 1 else f"{pw:.1f}"
        ax_map.text(x, y, label, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold",
                    color=color, zorder=4)

    n_used = int((tally > 0).sum().item())
    pts_sorted = sorted(pts, key=lambda p: p[3], reverse=True)
    top5_share = sum(p[3] for p in pts_sorted[:5]) / max(per_week.sum(), 1e-6)
    summary = (f"Stations used: {n_used} / {N_STN}  ({n_used/N_STN:.0%})\n"
               f"Top-5 absorb {top5_share:.0%} of all items")
    ax_map.text(0.01, 0.01, summary, transform=ax_map.transAxes,
                fontsize=10, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#888", alpha=0.9))

    ax_map.set_aspect("equal")
    ax_map.set_xticks([]); ax_map.set_yticks([])
    for spine in ax_map.spines.values(): spine.set_visible(False)

    # ── RIGHT: ranked CLI-style table ───────────────────────────────────
    ax_tbl.axis("off")
    ax_tbl.set_xlim(0, 1); ax_tbl.set_ylim(0, 1)

    rows = []
    for i in range(N_STN):
        n = int(tally[i].item())
        if n == 0: continue
        rows.append((STATION_LIST[i], STATION_ZONE[STATION_LIST[i]],
                     n, n / weeks, n / total))
    rows.sort(key=lambda r: r[3], reverse=True)
    rows = rows[:top_label]

    # Per-column x-anchors (axes coords). Right-aligned for numbers,
    # left-aligned for the station name.
    COL_X = {
        "rank":    (0.06, "right"),
        "station": (0.10, "left"),
        "wk":      (0.74, "right"),
        "tot":     (0.86, "right"),
        "pct":     (0.97, "right"),
    }
    header_cells = {
        "rank": "#", "station": "Station",
        "wk": "/wk", "tot": "tot", "pct": "%",
    }
    for col, txt in header_cells.items():
        x, ha = COL_X[col]
        ax_tbl.text(x, 0.97, txt, family="monospace",
                    fontsize=9.5, fontweight="bold",
                    va="top", ha=ha, transform=ax_tbl.transAxes)
    ax_tbl.plot([0.02, 0.99], [0.952, 0.952], color="#aaaaaa",
                lw=0.6, transform=ax_tbl.transAxes)

    row_h = 0.92 / max(len(rows), 1)
    for k, (s, z, n, pw, sh) in enumerate(rows):
        y_row = 0.94 - (k + 0.5) * row_h
        sname = s if len(s) <= 30 else s[:28] + "…"
        bg = "#f7f7f7" if k % 2 == 0 else "white"
        ax_tbl.add_patch(mpatches.Rectangle(
            (0.02, y_row - row_h / 2 * 0.95), 0.97, row_h * 0.95,
            transform=ax_tbl.transAxes, facecolor=bg, edgecolor="none", zorder=1,
        ))
        cells = {
            "rank":    str(k + 1),
            "station": sname,
            "wk":      f"{pw:.1f}",
            "tot":     str(n),
            "pct":     f"{sh * 100:.1f}%",
        }
        for col, txt in cells.items():
            x, ha = COL_X[col]
            ax_tbl.text(x, y_row, txt, family="monospace",
                        fontsize=9, va="center", ha=ha,
                        transform=ax_tbl.transAxes, zorder=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = out_path or (OUT / "03_aggregate_map.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/synth.jsonl")
    parser.add_argument("--model", default=None)
    parser.add_argument("--top_label", type=int, default=40,
                        help="How many top stations to list in the right-hand table.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model_path = args.model or find_default_model()
    if model_path is None or not Path(model_path).exists():
        print("No trained model. Run train.py first."); return

    print(f"Loading events: {args.data}")
    events = load_events(args.data)
    print(f"Loading model:  {model_path}")
    model, graph, _ = load_model_and_graph(model_path)
    dist_matrix = build_dist_matrix()

    weeks = weeks_spanned(events)
    print(f"Tallying over {len(events)} events ({weeks:.1f} weeks)...")
    tally = tally_storage(events, model, graph, dist_matrix)
    out = draw_map(tally, weeks, top_label=args.top_label,
                    out_path=Path(args.out) if args.out else None)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
