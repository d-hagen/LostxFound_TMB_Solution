"""
Light synthetic event generator for v2.

Three coarse passenger profiles + a uniform noise floor. Designed to be
explainable in one screen; the v1 generator is richer but heavier.

Each event is a passenger trip: sample origin and destination zones, sample
specific stations within those zones, compute the metro path, pick the loss
station along the path. Pickup is the destination with small noise.
"""
import math
from datetime import datetime, timedelta

import numpy as np
import networkx as nx

from events import FoundEvent
from contexts import ITEM_TYPES
from network import G, STATION_LIST, TRANSFERS, STATION_ZONE, ZONE_LIST, transfer_aware_distances


# Each profile defines:
#   weight: mixture share
#   origin/dest zones: dict {zone: prob} or None for uniform over all zones
#   hour_mean/std: Gaussian on hour (None = uniform 5..23)
#   weekday_only: if True reject weekend draws
#   items: dict {item: prob}; None = uniform
PROFILES = {
    "commuter": dict(
        weight=0.40,
        origin_zones={"residential": 0.75, "leisure": 0.15, "tourist": 0.10},
        dest_zones={"business": 0.50, "university": 0.25, "hospital": 0.15, "industrial": 0.10},
        hour_mean=8.5, hour_std=2.0, weekday_only=True,
        items={"keys": 0.13, "wallet": 0.13, "phone": 0.16, "umbrella": 0.13,
               "laptop": 0.16, "bag": 0.10, "document": 0.08, "book": 0.06, "jacket": 0.05},
    ),
    "tourist": dict(
        weight=0.25,
        origin_zones={"tourist": 0.55, "residential": 0.30, "business": 0.15},
        dest_zones={"tourist": 0.50, "leisure": 0.25, "business": 0.10, "airport": 0.15},
        hour_mean=14.0, hour_std=3.5, weekday_only=False,
        items={"passport": 0.14, "camera": 0.12, "bag": 0.15, "phone": 0.20,
               "wallet": 0.12, "jacket": 0.08, "umbrella": 0.05, "other": 0.14},
    ),
    "leisure": dict(
        weight=0.20,
        origin_zones={"residential": 0.65, "tourist": 0.15, "leisure": 0.20},
        dest_zones={"leisure": 0.45, "tourist": 0.30, "business": 0.10, "residential": 0.15},
        hour_mean=20.0, hour_std=3.0, weekday_only=False,
        items={"phone": 0.20, "wallet": 0.15, "jacket": 0.15, "bag": 0.12,
               "keys": 0.10, "umbrella": 0.10, "camera": 0.08, "other": 0.10},
    ),
    "noise": dict(
        weight=0.15,
        origin_zones=None, dest_zones=None,
        hour_mean=None, hour_std=None, weekday_only=False,
        items=None,
    ),
}


_ZONE_STATIONS = {z: [] for z in ZONE_LIST}
for _s, _z in STATION_ZONE.items():
    _ZONE_STATIONS[_z].append(_s)


def _sample_zone(rng, dist):
    if dist is None:
        return ZONE_LIST[rng.choice(len(ZONE_LIST))]
    zones = list(dist.keys())
    p = np.array([dist[z] for z in zones], dtype=float)
    p /= p.sum()
    return zones[rng.choice(len(zones), p=p)]


def _draw_dirichlet_weights(rng, profile_name, alpha=4.0):
    """One Dirichlet draw per (profile, zone) per run → spread within zone."""
    out = {}
    for zone, stations in _ZONE_STATIONS.items():
        if stations:
            w = rng.dirichlet(np.full(len(stations), alpha))
            out[zone] = (stations, w)
    return out


def _sample_station(rng, zone, zone_weights):
    if zone not in zone_weights:
        return STATION_LIST[rng.choice(len(STATION_LIST))]
    stations, w = zone_weights[zone]
    return stations[rng.choice(len(stations), p=w)]


def _sample_hour(rng, mean, std):
    if mean is None:
        return int(rng.integers(5, 24))
    return int(np.clip(round(rng.normal(mean, std)), 5, 23))


def _sample_day(rng, weekday_only, start, days):
    for _ in range(20):
        d = start + timedelta(days=int(rng.integers(0, days)))
        if weekday_only and d.weekday() >= 5:
            continue
        return d
    return d


def _sample_loss_on_path(rng, path):
    if len(path) == 1:
        return path[0]
    w = np.ones(len(path))
    for i, s in enumerate(path):
        if s in TRANSFERS:
            w[i] *= 1.2
        w[i] *= 1.0 + 0.15 * (i / (len(path) - 1))   # mild tail bias
    w /= w.sum()
    return path[rng.choice(len(path), p=w)]


_TAD = {}
def _tad(s):
    if s not in _TAD:
        _TAD[s] = transfer_aware_distances(s)
    return _TAD[s]


def _sample_pickup(rng, dest, noise=0.25, max_hops=4.0):
    if rng.random() > noise:
        return dest
    dists = _tad(dest)
    cands = [(s, d) for s, d in dists.items() if s != dest and d <= max_hops]
    if not cands:
        return dest
    stations, ds = zip(*cands)
    inv = np.array([1.0 / (d + 0.5) for d in ds])
    inv /= inv.sum()
    return stations[rng.choice(len(stations), p=inv)]


# Per-item carry-distance mean (Geometric mean: more bulky / easy-to-forget items
# travel further before staff pick them up). Bounded by remaining path length.
_CARRY_MEAN = {
    "wallet":      0.6,
    "keys":        0.6,
    "phone":       1.1,
    "passport":    0.7,
    "document":    1.0,
    "camera":      1.2,
    "book":        1.6,
    "umbrella":    2.6,
    "jacket":      2.2,
    "bag":         1.8,
    "laptop":      1.5,
    "child_item":  1.0,
    "other":       1.4,
}


def _sample_found_from_lost(rng, path, lost_idx, item):
    """Walk k stops downstream of lost_at along the passenger's path.

    Path is origin → ... → dest. Item is "carried" k stops by the train past
    lost_at before being noticed/handed in. Bounded by remaining path length.
    Returns the found_at station and the realised carry distance k.
    """
    remaining = (len(path) - 1) - lost_idx
    if remaining <= 0:
        return path[lost_idx], 0
    mean = _CARRY_MEAN.get(item, 1.5)
    p_stop = 1.0 / (mean + 1.0)
    k = int(rng.geometric(p_stop)) - 1
    k = max(0, min(k, remaining))
    return path[lost_idx + k], k


def _sample_item(rng, items):
    if items is None:
        return ITEM_TYPES[rng.choice(len(ITEM_TYPES))]
    keys = list(items.keys())
    p = np.array([items[k] for k in keys], dtype=float)
    p /= p.sum()
    return keys[rng.choice(len(keys), p=p)]


def generate_events(n, seed=0, start_date="2024-01-01", duration_days=365, alpha=4.0):
    """Synth event corpus. n events sampled over `duration_days` calendar days."""
    rng = np.random.default_rng(seed)
    start = datetime.fromisoformat(start_date)
    pnames = list(PROFILES.keys())
    probs = np.array([PROFILES[p]["weight"] for p in pnames], dtype=float)
    probs /= probs.sum()
    zone_weights = {p: _draw_dirichlet_weights(rng, p, alpha=alpha) for p in pnames}

    out = []
    attempts, max_attempts = 0, n * 5
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        pname = pnames[rng.choice(len(pnames), p=probs)]
        prof = PROFILES[pname]

        day = _sample_day(rng, prof["weekday_only"], start, duration_days)
        hour = _sample_hour(rng, prof["hour_mean"], prof["hour_std"])
        minute = int(rng.integers(0, 60))
        dt = day.replace(hour=hour, minute=minute, second=0, microsecond=0)

        ozone = _sample_zone(rng, prof["origin_zones"])
        dzone = _sample_zone(rng, prof["dest_zones"])
        origin = _sample_station(rng, ozone, zone_weights[pname])
        dest = _sample_station(rng, dzone, zone_weights[pname])
        if origin == dest:
            continue
        try:
            path = nx.shortest_path(G, origin, dest)
        except nx.NetworkXNoPath:
            continue
        if len(path) < 2:
            continue

        # Sample the *true loss* station on the journey path
        lost_st = _sample_loss_on_path(rng, path)
        lost_idx = path.index(lost_st)
        item = _sample_item(rng, prof["items"])
        # The train carries the item k stops downstream before staff retrieve it
        found_st, _carry_k = _sample_found_from_lost(rng, path, lost_idx, item)
        pickup = _sample_pickup(rng, dest)

        out.append(FoundEvent(
            found_at=found_st, found_dt=dt.isoformat(),
            item_type=item, pickup=pickup,
            lost_at=lost_st,
            source="synth", weight=1.0,
        ))
    return out


def corpus_stats(events):
    from collections import Counter
    found = Counter(e.found_at for e in events)
    pickup = Counter(e.pickup for e in events if e.pickup)
    lost = Counter(e.lost_at for e in events if e.lost_at)
    total = len(events)
    same_lost_found = sum(1 for e in events if e.lost_at and e.lost_at == e.found_at)
    return {
        "n": total,
        "n_labeled": sum(1 for e in events if e.pickup),
        "n_with_lost": sum(1 for e in events if e.lost_at),
        "unique_found": len(found),
        "unique_pickup": len(pickup),
        "unique_lost": len(lost),
        "lost==found pct": f"{same_lost_found / max(total,1):.1%}",
        "top_found": f"{found.most_common(1)[0][0]} ({found.most_common(1)[0][1] / max(total, 1):.1%})" if found else "—",
    }


if __name__ == "__main__":
    import argparse
    from events import save_events
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="artifacts/data/synth.jsonl")
    args = parser.parse_args()

    print(f"Generating {args.n} events (seed={args.seed})")
    events = generate_events(args.n, seed=args.seed)
    print(corpus_stats(events))
    save_events(events, args.out)
    print(f"Saved -> {args.out}")
