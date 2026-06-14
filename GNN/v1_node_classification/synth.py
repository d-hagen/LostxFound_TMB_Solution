"""
Synthetic event generator.

Each event is a simulated lost item: pick a passenger profile, sample an OD,
shortest-path between them, drop the item somewhere on the path (biased to
transfers and trip end), and record the destination (with noise) as pickup.

Variety is enforced by:
  - mixture of 8 profiles (incl. an explicit uniform "noise" profile),
  - Dirichlet(alpha>=2) station weights per (profile, zone), drawn per run,
  - pickup noise (passenger occasionally picks a nearby station instead),
  - validate_corpus(): hard caps on per-station and per-zone share + entropy.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from collections import Counter

import numpy as np
import networkx as nx

from events import FoundEvent, ITEM_TYPES
from network import (
    G, STATION_LIST, N_STN, TRANSFERS,
    STATION_ZONE, ZONE_LIST, transfer_aware_distances,
)


PROFILES = {
    "morning_commuter": dict(
        weight=0.16,
        origin_zones={"residential": 0.85, "leisure": 0.10, "tourist": 0.05},
        dest_zones={"business": 0.50, "university": 0.20, "industrial": 0.20, "hospital": 0.10},
        hour_mean=8.5, hour_std=1.2,
        weekday_only=True, weekend_boost=0.0,
        item_weights={"keys": 0.13, "wallet": 0.13, "umbrella": 0.15, "laptop": 0.18,
                      "phone": 0.13, "bag": 0.10, "document": 0.08, "jacket": 0.08, "book": 0.02},
    ),
    "evening_commuter": dict(
        weight=0.16,
        origin_zones={"business": 0.50, "university": 0.20, "industrial": 0.20, "hospital": 0.10},
        dest_zones={"residential": 0.85, "leisure": 0.10, "tourist": 0.05},
        hour_mean=18.5, hour_std=1.2,
        weekday_only=True, weekend_boost=0.0,
        item_weights={"keys": 0.13, "wallet": 0.13, "umbrella": 0.15, "laptop": 0.18,
                      "phone": 0.13, "bag": 0.10, "document": 0.05, "jacket": 0.10, "book": 0.03},
    ),
    "tourist": dict(
        weight=0.13,
        origin_zones={"tourist": 0.55, "residential": 0.30, "business": 0.15},
        dest_zones={"tourist": 0.50, "leisure": 0.25, "business": 0.15, "airport": 0.10},
        hour_mean=14.0, hour_std=3.0,
        weekday_only=False, weekend_boost=2.0,
        item_weights={"passport": 0.15, "camera": 0.15, "bag": 0.15, "phone": 0.15,
                      "wallet": 0.12, "jacket": 0.08, "umbrella": 0.05, "book": 0.05, "other": 0.10},
    ),
    "student": dict(
        weight=0.11,
        origin_zones={"residential": 0.78, "tourist": 0.10, "leisure": 0.12},
        dest_zones={"university": 0.85, "business": 0.15},
        hour_mean=11.0, hour_std=3.5,
        weekday_only=True, weekend_boost=0.0,
        item_weights={"book": 0.20, "laptop": 0.20, "bag": 0.15, "phone": 0.13,
                      "keys": 0.10, "wallet": 0.10, "document": 0.05, "umbrella": 0.07},
    ),
    "business_traveler": dict(
        weight=0.05,
        origin_zones={"airport": 0.60, "business": 0.30, "tourist": 0.10},
        dest_zones={"business": 0.55, "tourist": 0.30, "airport": 0.15},
        hour_mean=12.0, hour_std=4.0,
        weekday_only=False, weekend_boost=0.5,
        item_weights={"laptop": 0.20, "document": 0.20, "bag": 0.18, "wallet": 0.12,
                      "passport": 0.10, "phone": 0.10, "jacket": 0.05, "umbrella": 0.05},
    ),
    "leisure": dict(
        weight=0.11,
        origin_zones={"residential": 0.70, "tourist": 0.15, "leisure": 0.15},
        dest_zones={"leisure": 0.45, "tourist": 0.30, "business": 0.10, "residential": 0.15},
        hour_mean=20.0, hour_std=3.0,
        weekday_only=False, weekend_boost=2.0,
        item_weights={"phone": 0.18, "wallet": 0.15, "jacket": 0.15, "bag": 0.12,
                      "keys": 0.10, "umbrella": 0.10, "camera": 0.08, "other": 0.12},
    ),
    "hospital_visitor": dict(
        weight=0.06,
        origin_zones={"residential": 0.85, "leisure": 0.15},
        dest_zones={"hospital": 1.0},
        hour_mean=13.0, hour_std=3.5,
        weekday_only=False, weekend_boost=1.0,
        item_weights={"bag": 0.20, "document": 0.15, "wallet": 0.15, "phone": 0.13,
                      "keys": 0.10, "umbrella": 0.10, "jacket": 0.08, "book": 0.05, "other": 0.04},
    ),
    "noise": dict(
        weight=0.22,
        origin_zones=None, dest_zones=None,
        hour_mean=None, hour_std=None,
        weekday_only=False, weekend_boost=1.0,
        item_weights=None,
    ),
}


def _zone_to_stations():
    out = {z: [] for z in ZONE_LIST}
    for s, z in STATION_ZONE.items():
        out[z].append(s)
    return out


_ZONE_STATIONS = _zone_to_stations()


def _dirichlet_weights(rng, alpha):
    """Per (profile, zone), draw Dirichlet weights across stations in that zone."""
    out = {}
    for pname in PROFILES:
        for zone, stations in _ZONE_STATIONS.items():
            if not stations:
                continue
            w = rng.dirichlet(np.full(len(stations), alpha))
            out[(pname, zone)] = (stations, w)
    return out


def _sample_zone(rng, zone_dist):
    if zone_dist is None:
        return rng.choice(ZONE_LIST)
    zones = list(zone_dist.keys())
    p = np.array([zone_dist[z] for z in zones], dtype=float)
    p /= p.sum()
    return zones[rng.choice(len(zones), p=p)]


def _sample_station(rng, zone, pname, zone_weights):
    key = (pname, zone)
    if key not in zone_weights:
        stations = _ZONE_STATIONS.get(zone) or STATION_LIST
        return stations[rng.choice(len(stations))]
    stations, w = zone_weights[key]
    return stations[rng.choice(len(stations), p=w)]


def _sample_hour(rng, mean, std):
    if mean is None:
        return int(rng.integers(5, 24))
    return int(np.clip(round(rng.normal(mean, std)), 5, 23))


def _sample_day(rng, profile, start, duration_days):
    wb = profile["weekend_boost"]
    for _ in range(50):
        d = start + timedelta(days=int(rng.integers(0, duration_days)))
        is_weekend = d.weekday() >= 5
        if profile["weekday_only"] and is_weekend:
            continue
        # Accept-reject for weekend boost. wb=1 -> neutral; wb>1 -> prefer weekend.
        if wb > 1.0 and not is_weekend:
            if rng.random() < 1.0 / wb:
                return d
            continue
        if 0.0 < wb < 1.0 and is_weekend:
            if rng.random() < wb:
                return d
            continue
        return d
    return d


def _sample_loss_on_path(rng, path):
    if len(path) == 1:
        return path[0]
    w = np.ones(len(path))
    for i, s in enumerate(path):
        if s in TRANSFERS:
            w[i] *= 1.15
        # tail bias: items more often noticed/lost near end of trip
        w[i] *= 1.0 + 0.15 * (i / (len(path) - 1))
    w /= w.sum()
    return path[rng.choice(len(path), p=w)]


_TAD_CACHE = {}
def _tad(s):
    if s not in _TAD_CACHE:
        _TAD_CACHE[s] = transfer_aware_distances(s)
    return _TAD_CACHE[s]


def _sample_pickup(rng, dest, noise_prob=0.30, max_hops=5.0):
    if rng.random() > noise_prob:
        return dest
    dists = _tad(dest)
    cands = [(s, d) for s, d in dists.items() if s != dest and d <= max_hops]
    if not cands:
        return dest
    stations, ds = zip(*cands)
    inv = np.array([1.0 / (d + 0.5) for d in ds])
    inv /= inv.sum()
    return stations[rng.choice(len(stations), p=inv)]


def _sample_item(rng, profile):
    iw = profile["item_weights"]
    if iw is None:
        return ITEM_TYPES[rng.choice(len(ITEM_TYPES))]
    items = list(iw.keys())
    p = np.array([iw[i] for i in items], dtype=float)
    p /= p.sum()
    return items[rng.choice(len(items), p=p)]


def _generate_from_profiles(profiles, n_events, seed, start_date,
                            duration_days, alpha,
                            pickup_noise_prob, pickup_max_hops, source_label):
    """Shared sampling loop. profiles: dict of profile_name -> profile spec."""
    rng = np.random.default_rng(seed)
    start = datetime.fromisoformat(start_date)

    pnames = list(profiles.keys())
    # Re-derive zone_weights for this profile set (each generator gets its own draw)
    zone_weights = {}
    for pname in pnames:
        for zone, stations in _ZONE_STATIONS.items():
            if not stations:
                continue
            zone_weights[(pname, zone)] = (stations, rng.dirichlet(np.full(len(stations), alpha)))

    probs = np.array([profiles[p]["weight"] for p in pnames], dtype=float)
    probs /= probs.sum()

    events = []
    attempts = 0
    max_attempts = n_events * 6
    while len(events) < n_events and attempts < max_attempts:
        attempts += 1
        pname = pnames[rng.choice(len(pnames), p=probs)]
        profile = profiles[pname]

        day = _sample_day(rng, profile, start, duration_days)
        hour = _sample_hour(rng, profile["hour_mean"], profile["hour_std"])
        minute = int(rng.integers(0, 60))
        dt = day.replace(hour=hour, minute=minute, second=0, microsecond=0)

        ozone = _sample_zone(rng, profile["origin_zones"])
        dzone = _sample_zone(rng, profile["dest_zones"])
        origin = _sample_station(rng, ozone, pname, zone_weights)
        dest = _sample_station(rng, dzone, pname, zone_weights)
        if origin == dest:
            continue

        try:
            path = nx.shortest_path(G, origin, dest)
        except nx.NetworkXNoPath:
            continue
        if len(path) < 2:
            continue

        loss_station = _sample_loss_on_path(rng, path)
        item = _sample_item(rng, profile)
        pickup = _sample_pickup(rng, dest, noise_prob=pickup_noise_prob, max_hops=pickup_max_hops)

        events.append(FoundEvent(
            found_at=loss_station,
            found_dt=dt.isoformat(),
            item_type=item,
            pickup=pickup,
            source=source_label,
            weight=1.0,
        ))
    return events


def generate_synthetic_events(n_events, seed=0, start_date="2024-01-01",
                              duration_days=365, alpha=6.0):
    """Prior-corpus generator. Uses the hand-coded PROFILES dict — our domain
    knowledge about who travels where and what they carry."""
    return _generate_from_profiles(
        PROFILES, n_events, seed, start_date, duration_days, alpha,
        pickup_noise_prob=0.30, pickup_max_hops=5.0, source_label="synth",
    )


# ── Pseudo-real generator ─────────────────────────────────────────────────
#
# Same machinery, perturbed parameters. Models the inevitable divergence of
# real-world questionnaire data from our hand-coded prior:
#   - commute peaks land later than assumed (9 / 19 instead of 8.5 / 18.5)
#   - tourists wander beyond the tourist zone
#   - phones gained share at the expense of cameras
#   - student hours have higher variance (online classes shift schedules)
#   - an "elderly_visitor" segment exists that the prior didn't model
#   - passengers pick more noisily (less optimal than assumed)
#   - within-zone Dirichlet concentration is different (alpha=4 vs 6)

def _make_real_profiles():
    real = {name: {**spec, "item_weights": dict(spec["item_weights"])
                   if spec["item_weights"] else None,
                   "origin_zones": dict(spec["origin_zones"])
                   if spec["origin_zones"] else None,
                   "dest_zones": dict(spec["dest_zones"])
                   if spec["dest_zones"] else None}
            for name, spec in PROFILES.items()}

    # Commute peaks shifted ~30 min later
    real["morning_commuter"]["hour_mean"] = 9.0
    real["evening_commuter"]["hour_mean"] = 19.0

    # Tourists wander
    real["tourist"]["dest_zones"] = {
        "tourist": 0.40, "leisure": 0.30, "business": 0.10,
        "residential": 0.10, "airport": 0.10,
    }

    # Student hours more variable
    real["student"]["hour_std"] = 4.5

    # Phones gain, cameras lose
    def _shift(iw, gainer, loser, delta):
        iw = dict(iw)
        iw[gainer] = iw.get(gainer, 0.05) + delta
        iw[loser] = max(0.01, iw.get(loser, 0.05) - delta)
        s = sum(iw.values())
        return {k: v / s for k, v in iw.items()}

    for pname in ("morning_commuter", "evening_commuter"):
        real[pname]["item_weights"] = _shift(real[pname]["item_weights"], "phone", "book", 0.05)
    real["tourist"]["item_weights"] = _shift(real["tourist"]["item_weights"], "phone", "camera", 0.08)

    # New profile not in the prior: elderly visitors
    real["elderly_visitor"] = dict(
        weight=0.08,
        origin_zones={"residential": 0.85, "leisure": 0.15},
        dest_zones={"hospital": 0.35, "business": 0.15, "leisure": 0.20,
                    "residential": 0.20, "tourist": 0.10},
        hour_mean=11.5, hour_std=2.5,
        weekday_only=False, weekend_boost=0.7,
        item_weights={"document": 0.18, "keys": 0.15, "wallet": 0.15,
                      "bag": 0.13, "umbrella": 0.12, "phone": 0.10,
                      "jacket": 0.07, "other": 0.10},
    )

    # Renormalise weights
    total = sum(p["weight"] for p in real.values())
    for p in real.values():
        p["weight"] /= total
    return real


REAL_PROFILES = _make_real_profiles()


def generate_real_events(n_events, seed=0, start_date="2024-01-01",
                         duration_days=365, alpha=4.0):
    """Pseudo-real questionnaire generator. For demo only — in production
    this stream is the actual questionnaire CSV/JSONL ingest."""
    return _generate_from_profiles(
        REAL_PROFILES, n_events, seed, start_date, duration_days, alpha,
        pickup_noise_prob=0.40, pickup_max_hops=6.0, source_label="real",
    )


def validate_corpus(events, max_station_frac=0.05, max_zone_frac=0.45,
                    min_entropy_gap=1.5):
    """Raise on overconcentration. Returns summary stats on pass."""
    if not events:
        raise ValueError("Empty corpus")

    found_counts = Counter(e.found_at for e in events)
    pickup_counts = Counter(e.pickup for e in events if e.pickup)

    for label, counts in (("found_at", found_counts), ("pickup", pickup_counts)):
        if not counts:
            continue
        total = sum(counts.values())
        top_station, top_count = counts.most_common(1)[0]
        frac = top_count / total
        if frac > max_station_frac:
            raise ValueError(
                f"[{label}] '{top_station}' = {frac:.1%} of corpus (cap {max_station_frac:.1%}); "
                "increase alpha or lower profile zone concentration."
            )

    for label, counts in (("found_at", found_counts), ("pickup", pickup_counts)):
        if not counts:
            continue
        zone_counts = Counter()
        for s, c in counts.items():
            zone_counts[STATION_ZONE[s]] += c
        total = sum(zone_counts.values())
        top_zone, top_count = zone_counts.most_common(1)[0]
        frac = top_count / total
        if frac > max_zone_frac:
            raise ValueError(
                f"[{label}] zone '{top_zone}' = {frac:.1%} (cap {max_zone_frac:.1%}); "
                "rebalance profile weights."
            )

    total_found = sum(found_counts.values())
    probs = [c / total_found for c in found_counts.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(N_STN)
    if max_entropy - entropy > min_entropy_gap:
        raise ValueError(
            f"found_at entropy {entropy:.2f} too far from log(N)={max_entropy:.2f} "
            f"(gap {max_entropy - entropy:.2f} > {min_entropy_gap})."
        )

    return {
        "n_events": len(events),
        "n_labeled": sum(1 for e in events if e.pickup),
        "unique_found_at": len(found_counts),
        "unique_pickup": len(pickup_counts),
        "top_found_frac": found_counts.most_common(1)[0][1] / total_found,
        "found_entropy": entropy,
        "max_entropy": max_entropy,
    }


if __name__ == "__main__":
    import argparse, json
    from pathlib import Path
    from events import save_events

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=6.0)
    parser.add_argument("--mode", choices=["prior", "real"], default="prior",
                        help="prior = hand-coded behavioural assumptions; "
                             "real  = pseudo-real questionnaire (perturbed for demo)")
    parser.add_argument("--out", default="artifacts/data/dev/events.jsonl")
    parser.add_argument("--no_validate", action="store_true")
    args = parser.parse_args()

    print(f"Generating {args.n} {args.mode} events (seed={args.seed}, alpha={args.alpha})")
    if args.mode == "prior":
        events = generate_synthetic_events(args.n, seed=args.seed, alpha=args.alpha)
    else:
        events = generate_real_events(args.n, seed=args.seed, alpha=args.alpha)

    if not args.no_validate:
        try:
            stats = validate_corpus(events)
            print("Validation OK:")
            print(json.dumps(stats, indent=2))
        except ValueError as e:
            print(f"Validation note ({args.mode}): {e}")

    out = Path(args.out)
    save_events(events, out)
    print(f"Saved {len(events)} events -> {out}")
