"""
Synthetic lost-and-found event generator for Barcelona.

Each event = (item_category, found_station, time_bucket, claim_station).

We don't have real claim-side data, so we simulate plausible drift:
each category has a target zone that biases where the claimant picks up.
Distance from found_station also matters (heavy local bias).

The model has to recover these patterns from the graph + features.
"""
from __future__ import annotations

import os
import sys
import json
import random
from dataclasses import dataclass, asdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from network import (
    STATION_LIST,
    STATION_INDEX,
    STATION_ZONE,
    DIST_MATRIX,
    CANDIDATE_STATIONS,
)

# ── Item categories and their drift profiles ──────────────────────────────
# Each category has a "target zone" the claim is biased toward, and a
# "locality" — how strongly distance from the found station matters.

CATEGORIES = [
    "wallet",
    "phone",
    "umbrella",
    "keys",
    "backpack",
    "id_passport",
    "headphones",
]

CATEGORY_PROFILES = {
    # category : (target_zone, target_weight, locality)
    "wallet":      ("tourist",     4.0, 0.35),
    "phone":       ("tourist",     3.0, 0.40),
    "umbrella":    ("residential", 2.5, 0.55),
    "keys":        ("residential", 3.0, 0.60),
    "backpack":    ("university",  4.0, 0.45),
    "id_passport": ("airport",     6.0, 0.30),
    "headphones":  ("business",    2.5, 0.50),
}

CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
N_CAT = len(CATEGORIES)

TIME_BUCKETS = 6  # 4-hour windows over the day
N_TIME = TIME_BUCKETS


@dataclass
class Event:
    category: str
    found_station: str
    time_bucket: int
    claim_station: str

    def to_dict(self):
        return asdict(self)


def _draw_claim(found_station: str, category: str, rng: random.Random) -> str:
    """Sample a claim station biased toward the category's target zone and
    distance from the found station."""
    profile = CATEGORY_PROFILES[category]
    target_zone, target_w, locality = profile

    d_from_found = DIST_MATRIX[found_station]

    candidates = []
    weights = []
    for s in STATION_LIST:
        if s == found_station:
            # rule out same-station pickup (user said pickup is a day or two
            # later, almost never the same station)
            continue
        d = d_from_found.get(s, 30)
        if d == 0 or d > 25:
            continue
        # distance term: exp(-locality * d)
        w = np.exp(-locality * d)
        # zone bias
        if STATION_ZONE.get(s) == target_zone:
            w *= target_w
        candidates.append(s)
        weights.append(w)

    if not candidates:
        # fallback: random non-same neighbour
        return rng.choice([s for s in STATION_LIST if s != found_station])

    weights = np.array(weights)
    weights /= weights.sum()
    return rng.choices(candidates, weights=weights, k=1)[0]


def generate_event(rng: random.Random) -> Event:
    cat = rng.choice(CATEGORIES)
    found = rng.choice(CANDIDATE_STATIONS)
    tb = rng.randint(0, TIME_BUCKETS - 1)
    claim = _draw_claim(found, cat, rng)
    return Event(category=cat, found_station=found, time_bucket=tb, claim_station=claim)


def generate_dataset(n: int, seed: int = 0) -> list[Event]:
    rng = random.Random(seed)
    return [generate_event(rng) for _ in range(n)]


def save_dataset(events: list[Event], path: str):
    with open(path, "w") as f:
        json.dump([e.to_dict() for e in events], f, indent=2)


def load_dataset(path: str) -> list[Event]:
    with open(path) as f:
        rows = json.load(f)
    return [Event(**r) for r in rows]


if __name__ == "__main__":
    events = generate_dataset(2000, seed=42)
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    save_dataset(events, os.path.join(out_dir, "events.json"))

    # quick summary
    from collections import Counter

    cat_counts = Counter(e.category for e in events)
    print(f"Generated {len(events)} events")
    print("By category:")
    for c, n in cat_counts.most_common():
        print(f"  {c:14s} {n:5d}")
    print(f"Example event: {events[0]}")
