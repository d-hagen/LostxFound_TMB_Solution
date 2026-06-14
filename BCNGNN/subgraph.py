"""
Per-item subgraph construction.

Given a found_station, extract the k-hop ego-graph in the metro network.
Each item is its own training graph (inductive GNN setup).

Returned tensor bundle:
  x           [N_sub, F]   node features
  edge_index  [2, E_sub]   sub-graph edges (re-labelled 0..N_sub-1)
  y_idx       int          index of the claim station inside the subgraph
                           (or -1 if claim outside the subgraph)
  found_idx   int          index of the found station inside the subgraph
  nodes       list[str]    original station names per local index
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import networkx as nx
import torch

sys.path.insert(0, os.path.dirname(__file__))
from network import (
    G,
    STATION_LIST,
    STATION_INDEX,
    STATION_ZONE,
    ZONE_INDEX,
    NUM_ZONES,
    TRANSFERS,
    DIST_MATRIX,
)
from metro import ALL_LINES
from synthetic import Event, CATEGORIES, CAT_INDEX, N_CAT, N_TIME

# ── Line membership precompute ────────────────────────────────────────────

LINE_LIST = sorted(ALL_LINES.keys())
LINE_INDEX = {ln: i for i, ln in enumerate(LINE_LIST)}
N_LINES = len(LINE_LIST)

STATION_LINES = {s: set() for s in STATION_LIST}
for ln, stations in ALL_LINES.items():
    for s in stations:
        if s in STATION_LINES:
            STATION_LINES[s].add(ln)


# Feature dimensions ────────────────────────────────────────────────────────
# per-node features:
#   zone one-hot (NUM_ZONES)
#   line membership multi-hot (N_LINES)
#   is_found_station (1)
#   is_transfer (1)
#   degree / 10 (1)
#   normalised dist-to-found / 10 (1)
#   category one-hot (N_CAT)             ← broadcast item conditioning
#   time-bucket one-hot (N_TIME)         ← broadcast item conditioning
NODE_FEAT_DIM = NUM_ZONES + N_LINES + 4 + N_CAT + N_TIME


@dataclass
class ItemGraph:
    x: torch.Tensor          # [N_sub, F]
    edge_index: torch.Tensor # [2, E_sub]
    y_idx: int               # local index of claim station
    found_idx: int           # local index of found station
    nodes: list[str]         # local index -> station name


def build_item_graph(event: Event, k: int = 4) -> ItemGraph:
    """Construct the per-item ego-graph at radius k around the found station."""
    found = event.found_station
    sub_nodes = sorted(nx.ego_graph(G, found, radius=k).nodes())
    local = {s: i for i, s in enumerate(sub_nodes)}
    N = len(sub_nodes)

    # ── edges ──────────────────────────────────────────────────────────
    src, dst = [], []
    sub_set = set(sub_nodes)
    for u, v in G.edges():
        if u in sub_set and v in sub_set:
            src += [local[u], local[v]]
            dst += [local[v], local[u]]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # ── features ────────────────────────────────────────────────────────
    cat_oh = np.zeros(N_CAT, dtype=np.float32)
    cat_oh[CAT_INDEX[event.category]] = 1.0
    time_oh = np.zeros(N_TIME, dtype=np.float32)
    time_oh[event.time_bucket] = 1.0

    d_from_found = DIST_MATRIX[found]
    feats = np.zeros((N, NODE_FEAT_DIM), dtype=np.float32)

    for i, s in enumerate(sub_nodes):
        ofs = 0
        # zone one-hot
        feats[i, ofs + ZONE_INDEX[STATION_ZONE[s]]] = 1.0
        ofs += NUM_ZONES
        # line membership multi-hot
        for ln in STATION_LINES[s]:
            feats[i, ofs + LINE_INDEX[ln]] = 1.0
        ofs += N_LINES
        # scalars
        feats[i, ofs + 0] = 1.0 if s == found else 0.0
        feats[i, ofs + 1] = 1.0 if s in TRANSFERS else 0.0
        feats[i, ofs + 2] = G.degree(s) / 10.0
        feats[i, ofs + 3] = d_from_found.get(s, 30) / 10.0
        ofs += 4
        # item conditioning (broadcast)
        feats[i, ofs:ofs + N_CAT] = cat_oh
        ofs += N_CAT
        feats[i, ofs:ofs + N_TIME] = time_oh

    x = torch.from_numpy(feats)

    # ── labels ──────────────────────────────────────────────────────────
    y_idx = local.get(event.claim_station, -1)
    found_idx = local[found]

    return ItemGraph(x=x, edge_index=edge_index, y_idx=y_idx,
                     found_idx=found_idx, nodes=sub_nodes)


def collate_batch(items: list[ItemGraph]):
    """Stack per-item subgraphs into one disconnected mega-batch.

    Returns:
      x                [sum N, F]
      edge_index       [2, sum E]
      batch_idx        [sum N]   which item each node belongs to
      y_local          [B]       local-index of claim station per item
      found_global     [B]       global index (within batched x) of each
                                 item's found station
      n_per_item       [B]       node count per item
      offsets          [B]       starting offset in the batched node array
    """
    xs, eis, batch_idx, y_local, found_global = [], [], [], [], []
    n_per_item, offsets = [], []
    cursor = 0
    for b, it in enumerate(items):
        N = it.x.shape[0]
        xs.append(it.x)
        eis.append(it.edge_index + cursor)
        batch_idx.append(torch.full((N,), b, dtype=torch.long))
        y_local.append(it.y_idx)
        found_global.append(cursor + it.found_idx)
        n_per_item.append(N)
        offsets.append(cursor)
        cursor += N

    return {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(eis, dim=1),
        "batch_idx": torch.cat(batch_idx, dim=0),
        "y_local": torch.tensor(y_local, dtype=torch.long),
        "found_global": torch.tensor(found_global, dtype=torch.long),
        "n_per_item": torch.tensor(n_per_item, dtype=torch.long),
        "offsets": torch.tensor(offsets, dtype=torch.long),
    }


if __name__ == "__main__":
    from synthetic import generate_dataset

    events = generate_dataset(5, seed=0)
    for e in events:
        ig = build_item_graph(e, k=4)
        in_sub = ig.y_idx >= 0
        print(f"{e.category:12s} found={e.found_station:25s} "
              f"claim={e.claim_station:25s} "
              f"N={ig.x.shape[0]:3d} E={ig.edge_index.shape[1]:4d} "
              f"in_subgraph={in_sub}")
    print(f"\nNODE_FEAT_DIM = {NODE_FEAT_DIM}")
