"""
Event schema for the v2 (heterogeneous-graph link-prediction) pipeline.

A FoundEvent here is the same as in v1 but with a derived context_id used
to map the event to a context node in the graph.
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Iterable, List
from pathlib import Path
import json, random

from contexts import context_id as _context_id


@dataclass(frozen=True)
class FoundEvent:
    found_at: str
    found_dt: str
    item_type: str
    pickup: Optional[str]
    lost_at: Optional[str] = None
    source: str = "synth"
    weight: float = 1.0

    @property
    def hour(self) -> int:
        return datetime.fromisoformat(self.found_dt).hour

    @property
    def dow(self) -> int:
        return datetime.fromisoformat(self.found_dt).weekday()

    @property
    def context_id(self) -> int:
        return _context_id(self.item_type, self.hour, self.dow)


def save_events(events: Iterable[FoundEvent], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(asdict(e)) + "\n")


def load_events(path) -> List[FoundEvent]:
    out: List[FoundEvent] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Permit extra fields (e.g. v1's trajectory) — drop them silently.
            d = {k: v for k, v in d.items()
                 if k in {"found_at", "found_dt", "item_type", "pickup",
                          "lost_at", "source", "weight"}}
            out.append(FoundEvent(**d))
    return out


def split_events(events, val_frac=0.1, test_frac=0.1, seed=42):
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
