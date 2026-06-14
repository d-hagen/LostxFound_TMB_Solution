"""
Non-learned baselines. The GNN has to beat these on real-data evaluation
or it isn't earning its keep.

  - SameStationBaseline:   predict pickup = found_at (mass on found station)
  - ModeLookupBaseline:    per (found_at, hour-bucket, item) empirical pickup
                           distribution, with backoff to (found_at) marginal,
                           then global marginal.
  - CentroidBaseline:      always recommend the network's transfer-aware
                           centroid (one station, no conditioning).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
import torch

from network import STATION_INDEX, N_STN


class SameStationBaseline:
    name = "same_station"

    def fit(self, events):  # no-op
        return self

    def predict_proba(self, events):
        p = torch.zeros(len(events), N_STN)
        for i, e in enumerate(events):
            p[i, STATION_INDEX[e.found_at]] = 1.0
        return p


class CentroidBaseline:
    """Constant prediction: one station with mass 1.0, chosen as the
    transfer-aware centroid (minimum total distance to all stations)."""
    name = "centroid"

    def __init__(self, dist_matrix):
        total = dist_matrix.sum(dim=1)
        self.centroid_idx = int(torch.argmin(total).item())

    def fit(self, events):
        return self

    def predict_proba(self, events):
        p = torch.zeros(len(events), N_STN)
        p[:, self.centroid_idx] = 1.0
        return p


class ModeLookupBaseline:
    """Empirical pickup distribution per (found_at, hour-bucket, item_type),
    with Laplace smoothing and graceful backoff."""
    name = "mode_lookup"

    def __init__(self, hour_buckets=4, smoothing=0.1):
        self.hour_buckets = hour_buckets
        self.smoothing = smoothing
        self.table = None
        self.found_marginal = None
        self.global_marginal = None

    def _bucket(self, hour):
        return int(hour) * self.hour_buckets // 24

    def fit(self, events):
        self.table = defaultdict(
            lambda: torch.full((N_STN,), self.smoothing, dtype=torch.float32)
        )
        self.found_marginal = defaultdict(
            lambda: torch.full((N_STN,), self.smoothing, dtype=torch.float32)
        )
        self.global_marginal = torch.full((N_STN,), self.smoothing, dtype=torch.float32)

        for e in events:
            if not e.pickup:
                continue
            pi = STATION_INDEX[e.pickup]
            key = (e.found_at, self._bucket(e.hour), e.item_type)
            self.table[key][pi] += e.weight
            self.found_marginal[e.found_at][pi] += e.weight
            self.global_marginal[pi] += e.weight
        return self

    def predict_proba(self, events):
        p = torch.zeros(len(events), N_STN, dtype=torch.float32)
        floor = self.smoothing * N_STN + 1.0
        for i, e in enumerate(events):
            key = (e.found_at, self._bucket(e.hour), e.item_type)
            if key in self.table and self.table[key].sum() > floor:
                counts = self.table[key]
            elif e.found_at in self.found_marginal and self.found_marginal[e.found_at].sum() > floor:
                counts = self.found_marginal[e.found_at]
            else:
                counts = self.global_marginal
            p[i] = counts / counts.sum()
        return p
