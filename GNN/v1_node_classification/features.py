"""
Static graph features + per-event tensor batching for the GNN.

build_static_node_features() — per-station feature vector (zone, lines,
transfer flag, log-degree, distances to K anchor stations chosen by
farthest-point sampling). Computed once, cached.

build_normalized_adj() — symmetric-normalized adjacency with self-loops.

batch_events()         — turn a list of FoundEvent into query tensors.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from events import ITEM_INDEX
from network import (
    G, STATION_LIST, STATION_INDEX, N_STN, STN_EDGE_INDEX,
    STATION_ZONE, ZONE_INDEX, NUM_ZONES, TRANSFERS, transfer_aware_distances,
)
from metro import ALL_LINES


LINE_LIST = sorted(ALL_LINES.keys())
LINE_INDEX = {l: i for i, l in enumerate(LINE_LIST)}
N_LINES = len(LINE_LIST)

STATION_LINES = {}
for _line, _stations in ALL_LINES.items():
    for _s in _stations:
        STATION_LINES.setdefault(_s, []).append(_line)


_ANCHORS_CACHE = None
def pick_anchors(k=8):
    """Farthest-point sampling over transfer-aware distance. Deterministic."""
    global _ANCHORS_CACHE
    if _ANCHORS_CACHE is not None and len(_ANCHORS_CACHE) == k:
        return _ANCHORS_CACHE

    start = "Catalunya" if "Catalunya" in STATION_INDEX else STATION_LIST[0]
    anchors = [start]
    min_dist = {s: float("inf") for s in STATION_LIST}
    while len(anchors) < k:
        d = transfer_aware_distances(anchors[-1])
        for s in STATION_LIST:
            min_dist[s] = min(min_dist[s], d.get(s, 1e6))
        # Next anchor: max of current min distance (break ties lexicographically)
        nxt = max(STATION_LIST, key=lambda s: (min_dist[s], s))
        anchors.append(nxt)
    _ANCHORS_CACHE = anchors
    return anchors


_NODE_FEATS_CACHE = None
def build_static_node_features(k_anchors=8):
    global _NODE_FEATS_CACHE
    if _NODE_FEATS_CACHE is not None:
        return _NODE_FEATS_CACHE

    anchors = pick_anchors(k=k_anchors)
    anchor_dists = {a: transfer_aware_distances(a) for a in anchors}

    feats = []
    for s in STATION_LIST:
        f = []
        # Zone one-hot
        zoh = np.zeros(NUM_ZONES, dtype=np.float32)
        zoh[ZONE_INDEX[STATION_ZONE[s]]] = 1.0
        f.extend(zoh)
        # Line multi-hot
        lmh = np.zeros(N_LINES, dtype=np.float32)
        for ln in STATION_LINES.get(s, []):
            lmh[LINE_INDEX[ln]] = 1.0
        f.extend(lmh)
        # Transfer-hub flag, number-of-lines, log degree
        f.append(1.0 if s in TRANSFERS else 0.0)
        f.append(float(len(STATION_LINES.get(s, []))))
        f.append(float(np.log1p(G.degree(s))))
        # Distance to each anchor (raw; the model can scale)
        for a in anchors:
            f.append(float(anchor_dists[a].get(s, 50.0)))
        feats.append(f)

    T = torch.tensor(feats, dtype=torch.float32)
    # Per-column z-score so the linear projection has a sane input scale
    mean = T.mean(dim=0, keepdim=True)
    std = T.std(dim=0, keepdim=True).clamp_min(1e-6)
    T = (T - mean) / std
    _NODE_FEATS_CACHE = T
    return T


_ADJ_CACHE = None
def build_normalized_adj():
    """D^-1/2 (A + I) D^-1/2 — standard GCN normalization. [N, N] dense."""
    global _ADJ_CACHE
    if _ADJ_CACHE is not None:
        return _ADJ_CACHE
    A = torch.zeros(N_STN, N_STN, dtype=torch.float32)
    ei = STN_EDGE_INDEX
    A[ei[0], ei[1]] = 1.0
    A = A + torch.eye(N_STN)
    deg = A.sum(dim=1)
    d_inv_sqrt = deg.clamp_min(1.0).pow(-0.5)
    A = d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)
    _ADJ_CACHE = A
    return A


def batch_events(events):
    """Build tensors from a list of FoundEvent. Pickup may be None for inference."""
    found_idx, item_idx, hours, dows, pickups, weights = [], [], [], [], [], []
    for e in events:
        found_idx.append(STATION_INDEX[e.found_at])
        item_idx.append(ITEM_INDEX.get(e.item_type, ITEM_INDEX["other"]))
        hours.append(e.hour)
        dows.append(e.dow)
        pickups.append(STATION_INDEX[e.pickup] if e.pickup else -1)
        weights.append(e.weight)
    return {
        "found_idx": torch.tensor(found_idx, dtype=torch.long),
        "item_idx": torch.tensor(item_idx, dtype=torch.long),
        "hour": torch.tensor(hours, dtype=torch.long),
        "dow": torch.tensor(dows, dtype=torch.long),
        "pickup": torch.tensor(pickups, dtype=torch.long),
        "weight": torch.tensor(weights, dtype=torch.float32),
    }
