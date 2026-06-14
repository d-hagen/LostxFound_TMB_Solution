"""
FoundEvent schema, vocabulary, I/O, and split utilities.

A FoundEvent represents one lost-item event:
  - found_at:  station where the item was discovered
  - found_dt:  ISO timestamp
  - item_type: category from ITEM_TYPES
  - pickup:    station the passenger eventually used to pick it up
               (the supervised target; None for unlabeled events)
  - source:    "synth" or "real"
  - weight:    loss weight (used to upweight real over synth at training)
"""
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Literal, Optional, Iterable, List
from pathlib import Path
import json, random


ITEM_TYPES = [
    "wallet", "phone", "keys", "bag", "umbrella", "laptop",
    "passport", "camera", "book", "jacket", "child_item",
    "document", "other",
]
ITEM_INDEX = {t: i for i, t in enumerate(ITEM_TYPES)}
N_ITEMS = len(ITEM_TYPES)


@dataclass(frozen=True)
class FoundEvent:
    found_at: str
    found_dt: str
    item_type: str
    pickup: Optional[str]
    source: Literal["synth", "real"] = "synth"
    weight: float = 1.0

    @property
    def hour(self) -> int:
        return datetime.fromisoformat(self.found_dt).hour

    @property
    def dow(self) -> int:
        return datetime.fromisoformat(self.found_dt).weekday()


def save_events(events: Iterable[FoundEvent], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(asdict(e)) + "\n")


def load_events(path) -> List[FoundEvent]:
    events: List[FoundEvent] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            events.append(FoundEvent(**d))
    return events


def split_events(events, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42):
    """Random split. For time-aware retraining, use real events' timestamp ordering instead."""
    events = list(events)
    rng = random.Random(seed)
    rng.shuffle(events)
    n = len(events)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = events[:n_test]
    val = events[n_test:n_test + n_val]
    train = events[n_test + n_val:]
    return train, val, test


def reweight(events, weight: float):
    """Return a new list with weights overridden."""
    return [FoundEvent(e.found_at, e.found_dt, e.item_type, e.pickup, e.source, weight)
            for e in events]
