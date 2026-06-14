"""
Visualise one event flowing through the v2 heterogeneous-graph model.

Single figure styled like viz_aggregate.py:
  - LEFT  : Barcelona map. Stations sized + colour-coded by predicted P(pickup).
            Found-at, lost-at, and top-5 storage stations highlighted.
  - RIGHT : event card + ranked tables of the top-5 preferred pickup stations
            and the top-5 storage recommendations.

Output: artifacts/viz/01_event_prediction.png
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn.functional as F

from contexts import context_id, context_label
from events import FoundEvent
from network import STATION_LIST, STATION_INDEX, STATION_ZONE
from network_map import _GEO
from metro import ALL_LINES
from aggregate import find_default_model, load_model_and_graph
from decision import build_dist_matrix, recommend_storage


OUT = Path(__file__).parent / "artifacts" / "viz"
OUT.mkdir(parents=True, exist_ok=True)


# Example event — passenger lost their phone at Sagrada Família, item carried
# downstream to Tetuan on L2 before being handed in.
EXAMPLE = FoundEvent(
    found_at="Tetuan",
    lost_at="Sagrada Família",
    found_dt="2024-03-04T18:30:00",   # Monday evening
    item_type="phone",
    pickup=None,
    source="demo",
)


def geo_xy(station):
    lat, lon = _GEO.get(station, (41.39, 2.17))
    return lon, lat


def run_model_on_event(event):
    model_path = find_default_model()
    assert model_path and model_path.exists(), "train a model first"
    model, graph, _ = load_model_and_graph(model_path)
    dist_matrix = build_dist_matrix()

    cid = context_id(event.item_type, event.hour, event.dow)
    fi = torch.tensor([STATION_INDEX[event.found_at]], dtype=torch.long)
    li = torch.tensor(
        [STATION_INDEX[event.lost_at or event.found_at]], dtype=torch.long,
    )
    ci = torch.tensor([cid], dtype=torch.long)
    with torch.no_grad():
        h_st, h_ctx = model.encode(graph)
        logits = model.score(h_ctx[ci], h_st, fi, lost_st_idx=li)
        probs = F.softmax(logits, dim=-1)[0]
    cost, storage_idx = recommend_storage(probs, dist_matrix, top_k=10)
    return cid, probs, cost, storage_idx


# ─────────────────────────────────────────────────────────────────────────────

def draw_event_figure(event, cid, probs, cost, storage_idx, out_path):
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.55, 1.0],
        height_ratios=[0.95, 1.0],
        wspace=0.06, hspace=0.18,
    )
    ax_map = fig.add_subplot(gs[:, 0])
    ax_top_event = fig.add_subplot(gs[0, 1])
    ax_top_pref  = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "v2 GNN — single-event prediction",
        fontsize=16, fontweight="bold", y=0.97,
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

    # Subtler colormap: light grey → soft amber, no deep red.
    cmap = LinearSegmentedColormap.from_list(
        "load", ["#eef0f3", "#fde7a8", "#f4b740"]
    )

    probs_np = probs.cpu().numpy()
    max_p = probs_np.max() if probs_np.max() > 0 else 1.0

    sizes, colors, xs, ys = [], [], [], []
    for i, s in enumerate(STATION_LIST):
        if s not in _GEO: continue
        p = probs_np[i]
        x, y = geo_xy(s)
        if p < max_p * 0.02:
            ax_map.scatter(x, y, s=8, marker="o", facecolor="white",
                           edgecolor="#bbbbbb", linewidths=0.4, zorder=2)
            continue
        # Gentler growth: range ~25-180 instead of 100-800
        sizes.append(25 + 180 * (p / max_p) ** 0.6)
        colors.append(p / max_p)
        xs.append(x); ys.append(y)
    ax_map.scatter(xs, ys, s=sizes, c=colors, cmap=cmap, vmin=0, vmax=1,
                   edgecolor="#444", linewidths=0.5, zorder=3, alpha=0.95)

    # found_at and lost_at rings (red + orange) — kept
    fx, fy = geo_xy(event.found_at)
    ax_map.scatter(fx, fy, s=300, marker="o", facecolor="none",
                   edgecolor="#d62728", linewidths=2.2, zorder=4)
    ax_map.text(fx, fy - 0.0025, "found", ha="center", va="top",
                fontsize=8, fontweight="bold", color="#d62728", zorder=5)

    if event.lost_at and event.lost_at != event.found_at and event.lost_at in _GEO:
        lx, ly = geo_xy(event.lost_at)
        ax_map.scatter(lx, ly, s=300, marker="o", facecolor="none",
                       edgecolor="#ff7f0e", linewidths=2.2, zorder=4)
        ax_map.text(lx, ly - 0.0025, "lost", ha="center", va="top",
                    fontsize=8, fontweight="bold", color="#ff7f0e", zorder=5)

    # Top-1 storage station: a labelled arrow pointing at it.
    s1_idx = int(storage_idx[0].item())
    s1_name = STATION_LIST[s1_idx]
    if s1_name in _GEO:
        sx, sy = geo_xy(s1_name)
        x_range = max(xs + [fx, lx if event.lost_at and event.lost_at in _GEO else fx]) - \
                  min(xs + [fx, lx if event.lost_at and event.lost_at in _GEO else fx])
        offset = max(0.012, x_range * 0.10)
        ax_map.annotate(
            f"STORE HERE\n{s1_name}",
            xy=(sx, sy), xytext=(sx + offset, sy + offset * 0.9),
            fontsize=10, fontweight="bold", color="#1a3d6b",
            ha="left", va="bottom", zorder=6,
            arrowprops=dict(arrowstyle="->", color="#1a3d6b", lw=2.0,
                            shrinkA=0, shrinkB=8),
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff",
                      edgecolor="#1a3d6b", linewidth=1.4, alpha=0.95),
        )

    # legend
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='#d62728', markeredgewidth=2.0,
                   markersize=10, label='found_at'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='#ff7f0e', markeredgewidth=2.0,
                   markersize=10, label='lost_at (reported)'),
        plt.Line2D([0], [0], marker=r'$\rightarrow$', color='#1a3d6b',
                   markersize=12, linestyle='None',
                   label='top-1 storage recommendation'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#f4b740', markeredgecolor='#444',
                   markeredgewidth=0.5, markersize=10,
                   label='station, sized by predicted P(pickup)'),
    ]
    ax_map.legend(handles=legend_handles, loc='lower right',
                  fontsize=9, framealpha=0.9)

    # corner summary
    n_active = int((probs_np >= max_p * 0.02).sum())
    summary = (f"Predicted pickup mass concentrated on {n_active} stations\n"
               f"Top pick: {STATION_LIST[int(probs.argmax())]} "
               f"({probs.max().item():.1%})")
    ax_map.text(0.01, 0.01, summary, transform=ax_map.transAxes,
                fontsize=9.5, va='bottom', ha='left',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#888", alpha=0.9))

    ax_map.set_aspect("equal")
    ax_map.set_xticks([]); ax_map.set_yticks([])
    for spine in ax_map.spines.values(): spine.set_visible(False)

    # ── RIGHT TOP: event card + top-5 preferred stations table ──────────
    ax_top_event.axis("off")
    ax_top_event.set_xlim(0, 1); ax_top_event.set_ylim(0, 1)

    ax_top_event.text(0.0, 0.97, "INPUT EVENT", fontsize=11,
                      fontweight="bold", color="#1a3d6b",
                      va="top", ha="left")
    ax_top_event.plot([0.0, 1.0], [0.92, 0.92], color="#cfd9e6", lw=1.0)

    rows_event = [
        ("found_at",  event.found_at),
        ("lost_at",   event.lost_at or "—"),
        ("item",      event.item_type),
        ("hour",      f"{event.hour:02d}:00"),
        ("day",       ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][event.dow]),
        ("context",   f"#{cid}  {context_label(cid)}"),
    ]
    y = 0.85
    for k, v in rows_event:
        ax_top_event.text(0.02, y, k, family="monospace",
                          fontsize=10, color="#666")
        ax_top_event.text(0.25, y, str(v), family="monospace",
                          fontsize=10.5, fontweight="bold", color="#111")
        y -= 0.08

    # Top-5 preferred section (in same axes, below the event card)
    ax_top_event.text(0.0, 0.36, "TOP-5 PREFERRED PICKUP",
                      fontsize=11, fontweight="bold", color="#1a3d6b",
                      va="top", ha="left")
    ax_top_event.plot([0.0, 1.0], [0.31, 0.31], color="#cfd9e6", lw=1.0)

    top5 = torch.topk(probs, k=5)
    pref_rows = [(int(i.item()), float(p.item()))
                 for i, p in zip(top5.indices, top5.values)]
    y = 0.24
    ax_top_event.text(0.02, y, "#", family="monospace", fontsize=9,
                      fontweight="bold", color="#444")
    ax_top_event.text(0.08, y, "station", family="monospace", fontsize=9,
                      fontweight="bold", color="#444")
    ax_top_event.text(0.85, y, "P", family="monospace", fontsize=9,
                      fontweight="bold", color="#444", ha="right")
    y -= 0.045
    for k, (idx, p) in enumerate(pref_rows):
        name = STATION_LIST[idx]
        # alternating row background
        if k % 2 == 0:
            ax_top_event.add_patch(mpatches.Rectangle(
                (0.0, y - 0.020), 1.0, 0.040,
                facecolor="#f7f7f7", edgecolor="none", zorder=0,
            ))
        ax_top_event.text(0.02, y, str(k + 1), family="monospace",
                          fontsize=10, color="#111", va="center")
        ax_top_event.text(0.08, y, name[:28], family="monospace",
                          fontsize=10, color="#111", va="center")
        ax_top_event.text(0.85, y, f"{p:.1%}", family="monospace",
                          fontsize=10, fontweight="bold",
                          color="#dc2626", va="center", ha="right")
        y -= 0.045

    # ── RIGHT BOTTOM: top-5 storage stations table ──────────────────────
    ax_top_pref.axis("off")
    ax_top_pref.set_xlim(0, 1); ax_top_pref.set_ylim(0, 1)

    ax_top_pref.text(0.0, 0.97, "TOP-5 STORAGE RECOMMENDATIONS",
                     fontsize=11, fontweight="bold", color="#1a3d6b",
                     va="top", ha="left")
    ax_top_pref.text(0.0, 0.91, "decision rule: argmin E[claimant travel]",
                     fontsize=9, color="#666", va="top", ha="left",
                     style="italic")
    ax_top_pref.plot([0.0, 1.0], [0.86, 0.86], color="#cfd9e6", lw=1.0)

    cost5 = cost[:5].tolist()
    idx5 = [int(i.item()) for i in storage_idx[:5]]

    y = 0.79
    ax_top_pref.text(0.02, y, "#", family="monospace", fontsize=9,
                     fontweight="bold", color="#444")
    ax_top_pref.text(0.08, y, "station", family="monospace", fontsize=9,
                     fontweight="bold", color="#444")
    ax_top_pref.text(0.62, y, "zone", family="monospace", fontsize=9,
                     fontweight="bold", color="#444")
    ax_top_pref.text(0.97, y, "E[hops]", family="monospace", fontsize=9,
                     fontweight="bold", color="#444", ha="right")
    y -= 0.06
    for k, (idx, c) in enumerate(zip(idx5, cost5)):
        name = STATION_LIST[idx]
        zone = STATION_ZONE.get(name, "—")
        if k % 2 == 0:
            ax_top_pref.add_patch(mpatches.Rectangle(
                (0.0, y - 0.028), 1.0, 0.055,
                facecolor="#fff8dc", edgecolor="none", zorder=0,
            ))
        ax_top_pref.text(0.02, y, str(k + 1), family="monospace",
                         fontsize=10, color="#111", va="center")
        ax_top_pref.text(0.08, y, name[:24], family="monospace",
                         fontsize=10, color="#111", va="center")
        ax_top_pref.text(0.62, y, zone, family="monospace",
                         fontsize=9, color="#555", va="center")
        ax_top_pref.text(0.97, y, f"{c:.2f}", family="monospace",
                         fontsize=10, fontweight="bold",
                         color="#a37800", va="center", ha="right")
        y -= 0.060

    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return out_path


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Event: found_at={EXAMPLE.found_at}  lost_at={EXAMPLE.lost_at}  "
          f"item={EXAMPLE.item_type}  hour={EXAMPLE.hour:02d}  dow={EXAMPLE.dow}")
    cid, probs, cost, idx = run_model_on_event(EXAMPLE)
    print(f"Context id: {cid}  ({context_label(cid)})")
    print(f"Top-5 preferred: {[STATION_LIST[i.item()] for i in torch.topk(probs, 5).indices]}")
    print(f"Top-5 storage:   {[STATION_LIST[i.item()] for i in idx[:5]]}")

    out = OUT / "01_event_prediction.png"
    draw_event_figure(EXAMPLE, cid, probs, cost, idx, out)
    print(f"\nSaved → {out}")
