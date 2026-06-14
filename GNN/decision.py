"""
Decision rule. Identical to v1 — once the model produces a destination
distribution, picking a storage station is a deterministic argmin over
transfer-aware distances.
"""
import torch

from network import STATION_LIST, STATION_INDEX, N_STN, transfer_aware_distances


_DIST_MATRIX = None
def build_dist_matrix():
    """Passenger-perceived hop matrix: 1 per stop, +1 per transfer."""
    global _DIST_MATRIX
    if _DIST_MATRIX is not None:
        return _DIST_MATRIX
    M = torch.full((N_STN, N_STN), 100.0, dtype=torch.float32)
    for i, s in enumerate(STATION_LIST):
        d = transfer_aware_distances(s, step_cost=1.0, transfer_penalty=1.0)
        for t, v in d.items():
            M[i, STATION_INDEX[t]] = float(v)
    _DIST_MATRIX = M
    return M


_MOVE_COST_MATRIX = None
def build_move_cost_matrix(step_cost=0.05, transfer_penalty=1.0):
    """Operator movement-cost matrix: cheap to move along a line, costly to
    transfer between lines. Cached for the default parameters; rebuild via
    direct call if you want to sweep."""
    global _MOVE_COST_MATRIX
    if _MOVE_COST_MATRIX is not None and (step_cost, transfer_penalty) == (0.05, 1.0):
        return _MOVE_COST_MATRIX
    M = torch.full((N_STN, N_STN), 100.0, dtype=torch.float32)
    for i, s in enumerate(STATION_LIST):
        d = transfer_aware_distances(s, step_cost=step_cost,
                                      transfer_penalty=transfer_penalty)
        for t, v in d.items():
            M[i, STATION_INDEX[t]] = float(v)
    if (step_cost, transfer_penalty) == (0.05, 1.0):
        _MOVE_COST_MATRIX = M
    return M


def recommend_storage(prob_destination, dist_matrix, top_k=10, feasible_mask=None):
    """argmin_s  sum_d  p(d) * dist(d, s).   prob_destination: [B, N] or [N]."""
    single = prob_destination.dim() == 1
    if single:
        prob_destination = prob_destination.unsqueeze(0)
    expected_cost = prob_destination @ dist_matrix          # [B, N]
    if feasible_mask is not None:
        penalty = torch.where(feasible_mask, 0.0, float("inf"))
        expected_cost = expected_cost + penalty
    k = min(top_k, expected_cost.shape[-1])
    cost, idx = torch.topk(expected_cost, k=k, dim=-1, largest=False)
    if single:
        return cost[0], idx[0]
    return cost, idx


def recommend_storage_combined(prob_destination, dist_matrix, move_cost_matrix,
                                found_st_idx, lam=1.0, top_k=10):
    """argmin_s  E[hops](s) + lam * move_cost(found_at, s).

    E[hops] is the passenger-perceived cost: expected metro-hops from claimant.
    move_cost is the operator cost: cheap along a line, expensive at transfers.
    lam sets the operator/passenger tradeoff. lam=0 → original behaviour.
    """
    expected_hops = prob_destination @ dist_matrix          # [B, N]
    move_cost = move_cost_matrix[found_st_idx]              # [B, N]
    total = expected_hops + lam * move_cost
    k = min(top_k, total.shape[-1])
    cost, idx = torch.topk(total, k=k, dim=-1, largest=False)
    return cost, idx, expected_hops, move_cost
