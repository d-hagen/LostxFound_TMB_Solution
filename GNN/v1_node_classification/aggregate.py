"""
Aggregate demo: run the trained model over all past events and report,
for every station with non-zero load, the average items-per-week the GNN
would have routed there.

No training. Loads the events corpus + the current model checkpoint, predicts
the best storage station per event (decision rule), tallies, then divides by
the number of calendar weeks the corpus spans.

Usage:
  python aggregate.py
  python aggregate.py --data artifacts/data/dev/events.jsonl
  python aggregate.py --model artifacts/models/dev/model.pt --csv loads.csv
  python aggregate.py --top 30          # only show top-30 stations
  python aggregate.py --item phone      # filter to one item type
"""
import sys, os, argparse, csv
from pathlib import Path
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from events import load_events, ITEM_INDEX
from features import build_static_node_features, build_normalized_adj, batch_events
from model import GNNDestinationPredictor
from decision import recommend_storage, build_dist_matrix
from network import STATION_LIST, N_STN, STATION_ZONE
from demo import find_default_model, load_model


def _weeks_spanned(events):
    if not events:
        return 1.0
    ts = [datetime.fromisoformat(e.found_dt) for e in events]
    span_days = (max(ts) - min(ts)).days + 1
    return max(span_days / 7.0, 1.0)


def aggregate(events, model, node_feats, adj, dist_matrix, batch_size=512, device="cpu"):
    """Return [N] tensor with the predicted-storage tally per station."""
    tally = torch.zeros(N_STN, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            b = batch_events(batch)
            for k in b:
                b[k] = b[k].to(device)
            logits = model(node_feats, adj, b["found_idx"], b["item_idx"], b["hour"], b["dow"])
            probs = F.softmax(logits, dim=-1)
            _, idx = recommend_storage(probs, dist_matrix, top_k=1)
            chosen = idx[:, 0].cpu()
            tally.index_add_(0, chosen, torch.ones_like(chosen, dtype=torch.long))
    return tally


def render(tally, weeks, top=None):
    total = int(tally.sum().item())
    rows = []
    for i in range(N_STN):
        n = int(tally[i].item())
        if n == 0:
            continue
        rows.append((STATION_LIST[i], STATION_ZONE[STATION_LIST[i]], n, n / weeks, n / total))
    rows.sort(key=lambda r: r[3], reverse=True)
    if top is not None:
        rows = rows[:top]

    n_nonzero = sum(1 for i in range(N_STN) if tally[i].item() > 0)
    print(f"\nCorpus: {total} events over {weeks:.1f} weeks "
          f"({weeks/52*12:.1f} months equiv).")
    print(f"Stations actually used by the model: {n_nonzero} / {N_STN}  "
          f"(coverage {n_nonzero/N_STN:.0%})\n")

    max_per_week = rows[0][3] if rows else 1.0
    header = f"{'Rank':>4}  {'Station':<35s} {'Zone':<12s}  {'items/week':>10s}  {'total':>7s}  {'share':>6s}   load"
    print(header)
    print("-" * len(header))
    for rank, (s, z, n, per_week, share) in enumerate(rows, 1):
        bar = "█" * int(round(28 * per_week / max(max_per_week, 1e-6)))
        print(f"{rank:>4}  {s:<35s} {z:<12s}  {per_week:>10.2f}  {n:>7d}  {share:>5.1%}   {bar}")
    print()
    return rows


def to_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "station", "zone", "total_items", "items_per_week", "share"])
        for rank, (s, z, n, per_week, share) in enumerate(rows, 1):
            w.writerow([rank, s, z, n, f"{per_week:.4f}", f"{share:.6f}"])
    print(f"CSV -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/dev/events.jsonl")
    parser.add_argument("--model", default=None)
    parser.add_argument("--item", default=None,
                        help="If set, only aggregate over events of this item type")
    parser.add_argument("--dow", type=int, default=None,
                        help="If set, only aggregate events on this day-of-week (0=Mon)")
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    model_path = args.model or find_default_model()
    if model_path is None or not Path(model_path).exists():
        print("No trained model found. Train one first:")
        print("  python train.py --data artifacts/data/dev/events.jsonl --out artifacts/models/dev")
        sys.exit(1)

    print(f"Loading events: {args.data}")
    events = load_events(args.data)
    if args.item:
        if args.item not in ITEM_INDEX:
            print(f"Unknown item '{args.item}'. Valid: {list(ITEM_INDEX)}"); sys.exit(1)
        events = [e for e in events if e.item_type == args.item]
        print(f"  filtered to item='{args.item}' -> {len(events)} events")
    if args.dow is not None:
        events = [e for e in events if e.dow == args.dow]
        print(f"  filtered to dow={args.dow} -> {len(events)} events")
    if not events:
        print("No events after filtering."); sys.exit(1)

    print(f"Loading model: {model_path}")
    model = load_model(model_path)
    node_feats = build_static_node_features()
    adj = build_normalized_adj()
    dist_matrix = build_dist_matrix()

    weeks = _weeks_spanned(events)
    tally = aggregate(events, model, node_feats, adj, dist_matrix)
    rows = render(tally, weeks, top=args.top)

    if args.csv:
        to_csv(rows, args.csv)


if __name__ == "__main__":
    main()
