"""
Context bucket vocabulary.

A context is a coarsened (item_type, hour-bucket, day-bucket) tuple. Each
context is one node in the heterogeneous graph; questionnaire responses
become edges from contexts to stations.

Total contexts = N_ITEMS × HOUR_BUCKETS × DAY_BUCKETS.
"""

ITEM_TYPES = [
    "wallet", "phone", "keys", "bag", "umbrella", "laptop",
    "passport", "camera", "book", "jacket", "child_item",
    "document", "other",
]
ITEM_INDEX = {t: i for i, t in enumerate(ITEM_TYPES)}
N_ITEMS = len(ITEM_TYPES)

# 4 six-hour buckets: 0=00-06, 1=06-12, 2=12-18, 3=18-24
HOUR_BUCKETS = 4
def hour_bucket(hour: int) -> int:
    return int(hour) // 6

# 2 day buckets: 0=weekday, 1=weekend
DAY_BUCKETS = 2
def day_bucket(dow: int) -> int:
    return 1 if int(dow) >= 5 else 0

N_CONTEXTS = N_ITEMS * HOUR_BUCKETS * DAY_BUCKETS  # 13 * 4 * 2 = 104


def context_id(item_type: str, hour: int, dow: int) -> int:
    """Linearise (item, hour-bucket, day-bucket) → single int in [0, N_CONTEXTS)."""
    i = ITEM_INDEX.get(item_type, ITEM_INDEX["other"])
    h = hour_bucket(hour)
    d = day_bucket(dow)
    return (i * HOUR_BUCKETS + h) * DAY_BUCKETS + d


def context_label(cid: int) -> str:
    """Human-readable label for context id (for debugging / aggregate output)."""
    d = cid % DAY_BUCKETS
    cid //= DAY_BUCKETS
    h = cid % HOUR_BUCKETS
    cid //= HOUR_BUCKETS
    item = ITEM_TYPES[cid]
    hr = ["00-06", "06-12", "12-18", "18-24"][h]
    day = ["weekday", "weekend"][d]
    return f"{item}|{hr}|{day}"
