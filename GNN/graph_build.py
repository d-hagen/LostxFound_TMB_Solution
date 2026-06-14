"""
Build the heterogeneous graph the v2 GNN operates on.

Node types
  - station  (180 nodes, fixed)
  - context  (N_CONTEXTS = 104 nodes, fixed vocabulary)

Edge types  (all stored as (src_index, dst_index, weight) triples)
  - metro          (station ↔ station)   physical metro adjacency, weight=1
  - found_at       (station →  context)  count of events with that (station, context)
  - picked_up_at   (context →  station)  count of events with that pickup    [SUPERVISED]

Per-event tensors  (held out from the graph but used for loss / inference)
  - ctx[i], pickup_st[i], weight[i], found_st[i]

Static station features
  - zone one-hot, line multi-hot, transfer flag, log-degree
"""
from collections import Counter

import numpy as np
import torch

from contexts import N_CONTEXTS
from events import FoundEvent
from network import (
    G, STATION_LIST, STATION_INDEX, N_STN, STN_EDGE_INDEX, TRANSFERS,
)
from metro import ALL_LINES


LINE_LIST = sorted(ALL_LINES.keys())
LINE_INDEX = {l: i for i, l in enumerate(LINE_LIST)}
N_LINES = len(LINE_LIST)

STATION_LINES = {}
for _line, _stns in ALL_LINES.items():
    for _s in _stns:
        STATION_LINES.setdefault(_s, []).append(_line)


def build_station_features():
    """[N_STN, F] of static features.

    Zones are deliberately excluded — the ablation showed the GNN recovers
    zone-like structure from message passing on its own. Keeping zones out of
    the model lets us claim "no hand-coded semantic labels" while the synth
    generator continues to use zones for generation realism.

    Features kept: line multi-hot, transfer flag, # lines served, log-degree.
    """
    feats = []
    for s in STATION_LIST:
        f = []
        lmh = np.zeros(N_LINES, dtype=np.float32)
        for l in STATION_LINES.get(s, []):
            lmh[LINE_INDEX[l]] = 1.0
        f.extend(lmh)
        f.append(1.0 if s in TRANSFERS else 0.0)
        f.append(float(len(STATION_LINES.get(s, []))))
        f.append(float(np.log1p(G.degree(s))))
        feats.append(f)
    T = torch.tensor(feats, dtype=torch.float32)
    mu = T.mean(0, keepdim=True)
    sd = T.std(0, keepdim=True).clamp_min(1e-6)
    return (T - mu) / sd


def _norm_log(counts):
    """Log1p + L1 normalise per source — keeps edge weights bounded and
    treats a context with many observations the same as one with few."""
    w = torch.log1p(counts.float())
    return w


def build_hetero_graph(events, include_picked_up=True):
    """
    Assemble edge tensors from a list of events.

    Returns dict with:
      "station_feats": [N_STN, F]
      "metro_edge_index": [2, E_metro]  (already bidirectional from network.py)
      "found_at_edge_index":   [2, E_found]  row 0 = station idx, row 1 = context id
      "lost_at_edge_index":    [2, E_lost]   row 0 = station idx, row 1 = context id
      "picked_up_edge_index":  [2, E_pickup] row 0 = context id, row 1 = station idx
    """
    found_counts = Counter()
    lost_counts = Counter()
    pickup_counts = Counter()
    for e in events:
        ctx = e.context_id
        found_counts[(STATION_INDEX[e.found_at], ctx)] += e.weight
        if e.lost_at:
            lost_counts[(STATION_INDEX[e.lost_at], ctx)] += e.weight
        if include_picked_up and e.pickup:
            pickup_counts[(ctx, STATION_INDEX[e.pickup])] += e.weight

    def _tensors(counts):
        if not counts:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, dtype=torch.float32)
        src = torch.tensor([k[0] for k in counts], dtype=torch.long)
        dst = torch.tensor([k[1] for k in counts], dtype=torch.long)
        w = torch.tensor([counts[k] for k in counts], dtype=torch.float32)
        return torch.stack([src, dst], dim=0), w

    found_idx, found_w = _tensors(found_counts)
    lost_idx, lost_w = _tensors(lost_counts)
    pickup_idx, pickup_w = _tensors(pickup_counts)
    found_w = _norm_log(found_w)
    lost_w = _norm_log(lost_w)
    pickup_w = _norm_log(pickup_w)

    return {
        "station_feats":         build_station_features(),
        "metro_edge_index":      STN_EDGE_INDEX,            # [2, 2*E_undirected]
        "metro_weight":          torch.ones(STN_EDGE_INDEX.shape[1], dtype=torch.float32),
        "found_at_edge_index":   found_idx,                 # row0=st, row1=ctx
        "found_at_weight":       found_w,
        "lost_at_edge_index":    lost_idx,                  # row0=st, row1=ctx
        "lost_at_weight":        lost_w,
        "picked_up_edge_index":  pickup_idx,                # row0=ctx, row1=st
        "picked_up_weight":      pickup_w,
        "n_stations":            N_STN,
        "n_contexts":            N_CONTEXTS,
    }


def events_to_tensors(events):
    """For training: per-event (context_id, pickup_station_idx, weight,
    found_st, lost_st). lost_st falls back to found_st for events with no
    lost_at field, so the model query is well-defined for legacy data."""
    ctx = torch.tensor([e.context_id for e in events], dtype=torch.long)
    pickup = torch.tensor(
        [STATION_INDEX[e.pickup] if e.pickup else -1 for e in events],
        dtype=torch.long,
    )
    w = torch.tensor([e.weight for e in events], dtype=torch.float32)
    found_st = torch.tensor([STATION_INDEX[e.found_at] for e in events], dtype=torch.long)
    lost_st = torch.tensor(
        [STATION_INDEX[e.lost_at] if e.lost_at else STATION_INDEX[e.found_at]
         for e in events],
        dtype=torch.long,
    )
    return {"ctx": ctx, "pickup": pickup, "weight": w,
            "found_st": found_st, "lost_st": lost_st}


def graph_to_device(graph, device):
    out = {}
    for k, v in graph.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out
