"""
Storage-station decision rule. The model predicts P(destination | event);
this module picks  s* = argmin_s  E_d[ transfer_aware_distance(d, s) ].

This is deterministic given the distribution — not learned. Keeping it
separate means we can change the operational constraint set (e.g. only
stations with a lost-and-found desk) without retraining.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from network import STATION_LIST, STATION_INDEX, N_STN, transfer_aware_distances


_DIST_MATRIX = None
def build_dist_matrix():
    """[N, N] transfer-aware distance. Symmetric for undirected paths."""
    global _DIST_MATRIX
    if _DIST_MATRIX is not None:
        return _DIST_MATRIX
    M = torch.full((N_STN, N_STN), 100.0, dtype=torch.float32)
    for i, s in enumerate(STATION_LIST):
        d = transfer_aware_distances(s)
        for t, v in d.items():
            M[i, STATION_INDEX[t]] = float(v)
    _DIST_MATRIX = M
    return M


def recommend_storage(prob_destination, dist_matrix, top_k=10, feasible_mask=None):
    """
    prob_destination: [B, N] or [N]   predicted destination distribution
    dist_matrix:      [N, N]          transfer-aware cost from dest -> storage
    feasible_mask:    [N] bool or None — restrict candidate storage stations

    Returns (cost[B, k], idx[B, k]) sorted ascending by expected cost.
    """
    single = prob_destination.dim() == 1
    if single:
        prob_destination = prob_destination.unsqueeze(0)

    # expected_cost[b, s] = sum_v p[b, v] * dist[v, s]
    expected_cost = prob_destination @ dist_matrix          # [B, N]

    if feasible_mask is not None:
        penalty = torch.where(feasible_mask, 0.0, float("inf"))
        expected_cost = expected_cost + penalty

    k = min(top_k, expected_cost.shape[-1])
    cost, idx = torch.topk(expected_cost, k=k, dim=-1, largest=False)

    if single:
        return cost[0], idx[0]
    return cost, idx


def expected_pickup_hops(prob_destination, dist_matrix, true_pickup_idx, feasible_mask=None):
    """Realised pickup cost: distance from the (true) pickup to the chosen storage."""
    cost, idx = recommend_storage(prob_destination, dist_matrix, top_k=1, feasible_mask=feasible_mask)
    if idx.dim() == 1:
        idx = idx.unsqueeze(0)
        true_pickup_idx = torch.tensor([true_pickup_idx])
    chosen = idx[:, 0]
    hops = dist_matrix[true_pickup_idx, chosen]
    return hops
