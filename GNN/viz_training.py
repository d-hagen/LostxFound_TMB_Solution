"""
Visualize how the GNN's station embeddings evolve over training epochs.

At each checkpoint we run encode() on the graph, get the [N_STN, hidden]
matrix of post-GNN station vectors, PCA-project to 2D, and plot. Stations
get coloured by primary line membership so we can see whether the GNN's
embedding space starts to recover line structure as message passing learns.

Output: artifacts/figures/training_evolution.png — a 2x3 grid of scatter
plots, one per checkpoint epoch, with metro edges drawn faintly underneath.
"""
import os, copy, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import TSNE

from events import load_events, split_events
from graph_build import build_hetero_graph, events_to_tensors, graph_to_device
from model import HetGNN
from network import STATION_LIST, STATION_INDEX, N_STN, G
from contexts import N_CONTEXTS, context_label
from metro import ALL_LINES, LINE_COLORS
from network_map import get_positions
from train import _device, evaluate
from decision import build_dist_matrix


# Per-station: primary line (first one alphabetically, or the smaller line if
# multiple). Used only for colouring.
_STATION_PRIMARY_LINE = {}
for line, stns in ALL_LINES.items():
    for s in stns:
        if s not in _STATION_PRIMARY_LINE:
            _STATION_PRIMARY_LINE[s] = line


def _line_color(line_name):
    """Look up line colour from metro.LINE_COLORS, fallback to grey."""
    entry = LINE_COLORS.get(line_name)
    if entry is None:
        return "#888888"
    return entry.get("hex", "#888888") if isinstance(entry, dict) else entry


def get_station_embeddings(model, graph):
    """Run encode() and return the post-GNN station-vector matrix."""
    model.eval()
    with torch.no_grad():
        h_st, _ = model.encode(graph)
    return h_st.cpu().numpy()


def plot_panel(ax, emb2d, title, draw_edges=True):
    """Plot one PCA panel: stations as dots, metro edges as faint lines."""
    # Edges (under)
    if draw_edges:
        for u, v in G.edges():
            ui, vi = STATION_INDEX[u], STATION_INDEX[v]
            ax.plot([emb2d[ui, 0], emb2d[vi, 0]],
                    [emb2d[ui, 1], emb2d[vi, 1]],
                    color="#cccccc", linewidth=0.4, alpha=0.6, zorder=1)
    # Stations (over) coloured by primary line — batched single scatter call
    colors = [_line_color(_STATION_PRIMARY_LINE.get(s, "L1")) for s in STATION_LIST]
    ax.scatter(emb2d[:, 0], emb2d[:, 1], c=colors, s=22,
               edgecolors="black", linewidths=0.3, zorder=2)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    data_path = "artifacts/data/synth_v2.jsonl"
    out_path = Path("artifacts/figures/training_evolution.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = _device()
    print(f"Device: {device}")

    seed = 0
    torch.manual_seed(seed); np.random.seed(seed)

    events = load_events(data_path)
    train_ev, val_ev, _test_ev = split_events(events, val_frac=0.1, test_frac=0.1, seed=seed)
    graph = build_hetero_graph(train_ev, include_picked_up=True)
    graph = graph_to_device(graph, device)

    model = HetGNN(
        station_feat_dim=graph["station_feats"].shape[1],
        n_contexts=N_CONTEXTS,
        hidden=64, n_layers=2, dropout=0.1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    dist_matrix = build_dist_matrix().to(device)

    train_t = events_to_tensors(train_ev)
    val_t   = events_to_tensors(val_ev)
    for d in (train_t, val_t):
        for k in d: d[k] = d[k].to(device)

    # Checkpoints we want embeddings at
    checkpoints = [0, 1, 3, 10, 25, 40]
    saved_embs = {}
    saved_metrics = {}

    # Epoch-0 snapshot (random-init) before any gradient step
    saved_embs[0] = get_station_embeddings(model, graph)
    saved_metrics[0] = evaluate(model, graph, val_t, dist_matrix, device)
    print(f"epoch  0  E[hops]={saved_metrics[0]['expected_hops']:.2f}  (random init)")

    n_train = train_t["pickup"].shape[0]
    batch_size = 512
    max_epoch = max(checkpoints)
    for epoch in range(1, max_epoch + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            h_st, h_ctx = model.encode(graph)
            logits = model.score(h_ctx[train_t["ctx"][idx]], h_st,
                                 found_st_idx=train_t["found_st"][idx],
                                 lost_st_idx=train_t["lost_st"][idx])
            loss = (F.cross_entropy(logits, train_t["pickup"][idx], reduction="none")
                    * train_t["weight"][idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        if epoch in checkpoints:
            saved_embs[epoch] = get_station_embeddings(model, graph)
            saved_metrics[epoch] = evaluate(model, graph, val_t, dist_matrix, device)
            print(f"epoch {epoch:2d}  E[hops]={saved_metrics[epoch]['expected_hops']:.2f}  "
                  f"top1={saved_metrics[epoch]['top1']:.3f}")

    # t-SNE each snapshot to 2D — fit independently so we see how the
    # geometry of the embedding space changes. t-SNE preserves local
    # neighbourhoods better than PCA, so same-line clusters should pop.
    print("\nt-SNE-projecting all snapshots…")
    panels = []
    for ep in checkpoints:
        emb = saved_embs[ep]
        emb_c = emb - emb.mean(axis=0, keepdims=True)
        emb2d = TSNE(n_components=2, perplexity=15, random_state=0,
                     init="pca", learning_rate="auto").fit_transform(emb_c)
        panels.append((ep, emb2d, saved_metrics[ep]))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (ep, emb2d, m) in zip(axes.flatten(), panels):
        title = f"Epoch {ep}   E[hops]={m['expected_hops']:.2f}   top-1={m['top1']:.2f}"
        plot_panel(ax, emb2d, title)

    # Legend across the bottom — one entry per line
    seen_lines = sorted({_STATION_PRIMARY_LINE.get(s, "L1") for s in STATION_LIST})
    handles = [mlines.Line2D([], [], marker='o', linestyle='',
                             color=_line_color(ln), label=ln, markersize=8,
                             markeredgecolor="black", markeredgewidth=0.3)
               for ln in seen_lines]
    fig.legend(handles=handles, loc="lower center", ncol=len(seen_lines),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("GNN station embeddings across training\n"
                 "t-SNE-projected to 2D; faint lines are metro edges; "
                 "colour = primary line",
                 fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"\nSaved → {out_path}")

    # ── Second figure: one example event's predicted pickup distribution
    #    overlaid on the actual Barcelona metro map.
    test_t = events_to_tensors(_test_ev)
    for k in test_t: test_t[k] = test_t[k].to(device)
    plot_pickup_overlay(model, graph, _test_ev, test_t, device,
                        Path("artifacts/figures/predicted_pickup_example.png"))


def plot_pickup_overlay(model, graph, test_ev, test_t, device, out_path):
    """Render one example event's predicted pickup distribution on the
    Barcelona metro map. Stations are sized + coloured by P(pickup); the
    found_at, lost_at, and true pickup stations are ringed for reference."""
    import matplotlib.colors as mcolors

    # Pick a non-trivial event: lost_at != found_at, true pickup != either,
    # so the panel actually shows three distinct stations.
    chosen_i = None
    for i, e in enumerate(test_ev):
        if e.lost_at and e.lost_at != e.found_at and e.pickup not in (e.found_at, e.lost_at):
            chosen_i = i; break
    if chosen_i is None:
        chosen_i = 0
    e = test_ev[chosen_i]

    # Forward pass for this single event
    model.eval()
    with torch.no_grad():
        h_st, h_ctx = model.encode(graph)
        logits = model.score(h_ctx[test_t["ctx"][chosen_i:chosen_i+1]],
                              h_st,
                              found_st_idx=test_t["found_st"][chosen_i:chosen_i+1],
                              lost_st_idx=test_t["lost_st"][chosen_i:chosen_i+1])
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]   # [157]

    # Station positions from network_map (WGS84 lon/lat)
    pos = get_positions()
    pos_xy = np.array([pos[s] for s in STATION_LIST])

    fig, ax = plt.subplots(figsize=(12, 10))

    # Edges (faint)
    for u, v in G.edges():
        ui, vi = STATION_INDEX[u], STATION_INDEX[v]
        ax.plot([pos_xy[ui, 0], pos_xy[vi, 0]],
                [pos_xy[ui, 1], pos_xy[vi, 1]],
                color="#bbbbbb", linewidth=0.6, alpha=0.7, zorder=1)

    # Stations sized by P(pickup); colour by same probability
    sizes = 30 + 1200 * probs                               # min ~30, max ~1200
    cmap = plt.get_cmap("plasma")
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=max(probs.max(), 0.05))
    colors = [cmap(norm(p)) for p in probs]
    sc = ax.scatter(pos_xy[:, 0], pos_xy[:, 1], s=sizes, c=colors,
                    edgecolors="black", linewidths=0.3, zorder=2)

    # Highlight rings: found_at (red), lost_at (orange), true pickup (green)
    def _ring(stn, color, label, size_scale=2.0):
        if stn is None or stn not in STATION_INDEX: return
        i = STATION_INDEX[stn]
        ax.scatter(pos_xy[i, 0], pos_xy[i, 1],
                    s=300 * size_scale, facecolors='none',
                    edgecolors=color, linewidths=2.5, zorder=3, label=label)

    _ring(e.found_at, "#d62728", f"found_at: {e.found_at}")
    _ring(e.lost_at,  "#ff7f0e", f"lost_at:  {e.lost_at}")
    _ring(e.pickup,   "#2ca02c", f"true pickup: {e.pickup}", size_scale=2.5)

    # Top-1 prediction marker
    top1_idx = int(probs.argmax())
    top1_stn = STATION_LIST[top1_idx]
    if top1_stn not in (e.found_at, e.lost_at, e.pickup):
        ax.scatter(pos_xy[top1_idx, 0], pos_xy[top1_idx, 1],
                    s=900, facecolors='none', edgecolors="#1f77b4",
                    linewidths=2.0, linestyle="--", zorder=3,
                    label=f"top-1 predicted: {top1_stn} ({probs[top1_idx]:.0%})")

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("predicted P(pickup)")

    # Annotation block: event metadata
    ctx_label = context_label(e.context_id)
    info = (f"Event: {e.item_type}\n"
            f"context: {ctx_label}\n"
            f"hour={e.hour:02d}  dow={e.dow}\n"
            f"top-3: " +
            ", ".join(f"{STATION_LIST[i]}({probs[i]:.0%})"
                      for i in probs.argsort()[-3:][::-1]))
    ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black",
                      alpha=0.85, linewidth=0.5))

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.9)
    ax.set_title("Predicted pickup distribution for one test event\n"
                 "dot size and colour = P(pickup)", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
