"""Adapter between the TMB demo server and the trained heterogeneous GNN.

Lazy-loads the model + graph + distance matrix on first call. Exposes one
public function: `route_item(found_at, item_type, when_iso, lost_at=None)`.
"""
import os, sys
from datetime import datetime
from pathlib import Path
from typing import Optional


_GNN_DIR = Path(__file__).resolve().parent.parent / "GNN"
sys.path.insert(0, str(_GNN_DIR))

_STATE = {"model": None, "graph": None, "dist": None, "cfg": None}


def _ensure_loaded():
    if _STATE["model"] is not None:
        return
    import torch  # heavy imports deferred until first call
    from aggregate import find_default_model, load_model_and_graph
    from decision import build_dist_matrix

    model_path = find_default_model()
    if model_path is None or not Path(model_path).exists():
        raise RuntimeError(
            "No trained GNN found. Train one first: "
            "cd ../GNN && python3 train.py"
        )
    model, graph, cfg = load_model_and_graph(model_path)
    _STATE["model"] = model
    _STATE["graph"] = graph
    _STATE["dist"] = build_dist_matrix()
    _STATE["cfg"] = cfg
    _STATE["torch"] = torch


def is_available() -> bool:
    """Cheap check: does a trained model exist? Doesn't actually load it."""
    from aggregate import find_default_model
    p = find_default_model()
    return p is not None and Path(p).exists()


def _hops_to_days(hops: float) -> int:
    if hops <= 3:   return 1
    if hops <= 8:   return 2
    return 3


def route_item(found_at: str, item_type: str, when_iso: str,
               lost_at: Optional[str] = None) -> dict:
    """Run the GNN on a single registration event.

    Returns:
        storage_station, expected_hops (E[hops] under top-1 storage),
        hops_found_to_storage (exact transfer-aware distance),
        arrival_days (1-3), top5_destinations, top5_storage,
        error (only if something failed).
    """
    try:
        _ensure_loaded()
    except Exception as e:
        return {"error": f"GNN unavailable: {e}"}

    torch = _STATE["torch"]
    import torch.nn.functional as F
    from contexts import context_id, ITEM_INDEX
    from network import STATION_LIST, STATION_INDEX
    from decision import recommend_storage

    if found_at not in STATION_INDEX:
        return {"error": f"Unknown station: {found_at}"}
    if lost_at and lost_at not in STATION_INDEX:
        lost_at = None
    item = item_type if item_type in ITEM_INDEX else "other"

    try:
        dt = datetime.fromisoformat(when_iso.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.now()
    hour, dow = dt.hour, dt.weekday()

    cid = context_id(item, hour, dow)
    fi = torch.tensor([STATION_INDEX[found_at]], dtype=torch.long)
    li = torch.tensor([STATION_INDEX[lost_at or found_at]], dtype=torch.long)
    ci = torch.tensor([cid], dtype=torch.long)

    model, graph, dist = _STATE["model"], _STATE["graph"], _STATE["dist"]
    with torch.no_grad():
        h_st, h_ctx = model.encode(graph)
        logits = model.score(h_ctx[ci], h_st, found_st_idx=fi, lost_st_idx=li)
        probs = F.softmax(logits, dim=-1)[0]

    cost, idx = recommend_storage(probs, dist, top_k=5)
    storage_idx = int(idx[0].item())
    storage_station = STATION_LIST[storage_idx]
    expected_hops = float(cost[0].item())

    hops_fs = float(dist[STATION_INDEX[found_at], storage_idx].item())
    arrival_days = _hops_to_days(hops_fs)

    top5_dest_v, top5_dest_i = torch.topk(probs, k=5)
    top5_destinations = [
        (STATION_LIST[i.item()], float(p.item()))
        for p, i in zip(top5_dest_v, top5_dest_i)
    ]
    top5_storage = [
        (STATION_LIST[int(j.item())], float(c.item()))
        for c, j in zip(cost, idx)
    ]

    return {
        "storage_station": storage_station,
        "expected_hops": round(expected_hops, 2),
        "hops_found_to_storage": int(round(hops_fs)),
        "arrival_days": arrival_days,
        "top5_destinations": top5_destinations,
        "top5_storage": top5_storage,
    }
