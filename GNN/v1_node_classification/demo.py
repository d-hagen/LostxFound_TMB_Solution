"""
Interactive CLI demo.

Choose:
  - found-at station (fuzzy match by substring)
  - item type
  - hour (0–23)
  - day-of-week (0=Mon … 6=Sun)

Outputs:
  - top-5 destination predictions (what the GNN thinks)
  - top-10 storage stations ranked by expected pickup hops (the decision)

Usage:
  python demo.py                                 # interactive
  python demo.py --found "sagrada" --item phone --hour 18 --dow 1
"""
import sys, os, argparse
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from events import ITEM_TYPES, ITEM_INDEX
from features import build_static_node_features, build_normalized_adj
from model import GNNDestinationPredictor
from decision import recommend_storage, build_dist_matrix
from network import STATION_LIST, STATION_INDEX, N_STN, STATION_ZONE


DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def find_default_model():
    """Resolve artifacts/models/current_version.txt -> path/to/model.pt."""
    root = Path(__file__).parent / "artifacts" / "models"
    ptr = root / "current_version.txt"
    if ptr.exists():
        v = ptr.read_text().strip()
        candidate = root / v / "model.pt"
        if candidate.exists():
            return candidate
    # fallback: any *dev* model
    devp = root / "dev" / "model.pt"
    if devp.exists():
        return devp
    return None


def load_model(model_path, device="cpu"):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = GNNDestinationPredictor(
        n_stations=cfg["n_stations"], node_feat_dim=cfg["node_feat_dim"],
        n_items=cfg["n_items"], hidden=cfg["hidden"],
        n_layers=cfg["layers"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def fuzzy_station(query):
    if not query: return None
    q = query.strip().lower()
    if not q: return None
    # exact case-insensitive first
    for s in STATION_LIST:
        if s.lower() == q:
            return s
    matches = [s for s in STATION_LIST if q in s.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None
    return matches  # ambiguous → list


def predict(model, node_feats, adj, dist_matrix, found_at, item, hour, dow, top_k=10):
    fi = torch.tensor([STATION_INDEX[found_at]], dtype=torch.long)
    ii = torch.tensor([ITEM_INDEX.get(item, ITEM_INDEX["other"])], dtype=torch.long)
    h = torch.tensor([int(hour)], dtype=torch.long)
    d = torch.tensor([int(dow)], dtype=torch.long)
    with torch.no_grad():
        logits = model(node_feats, adj, fi, ii, h, d)
        probs = F.softmax(logits, dim=-1)[0]
    cost, idx = recommend_storage(probs, dist_matrix, top_k=top_k)
    return probs, cost, idx


def render(found_at, item, hour, dow, probs, cost, idx, top_dest_k=5):
    print(f"\n  Found:  {found_at}  ({STATION_ZONE[found_at]})")
    print(f"  Item:   {item}    Hour: {int(hour):02d}:00    Day: {DAY_NAMES[int(dow)]}")

    print(f"\n  Top {top_dest_k} predicted DESTINATIONS (passenger likely going to):")
    td = torch.topk(probs, k=top_dest_k)
    for p, i in zip(td.values, td.indices):
        s = STATION_LIST[i.item()]
        bar = "█" * int(round(p.item() * 30))
        print(f"    {p.item():.3f}  {s:<35s} ({STATION_ZONE[s]:<12s}) {bar}")

    print(f"\n  Top {len(idx)} STORAGE RECOMMENDATIONS (lowest expected pickup hops):")
    max_cost = float(cost.max().item()) if len(cost) else 1.0
    for rank, (c, i) in enumerate(zip(cost.tolist(), idx.tolist()), 1):
        s = STATION_LIST[i]
        bar_len = int(round(20 * (1 - c / max(max_cost, 1e-6))))
        bar = "█" * bar_len
        print(f"    {rank:2d}. {s:<35s} ({STATION_ZONE[s]:<12s})  E[hops]={c:5.2f}  {bar}")
    print()


def interactive(model_path):
    print(f"Loading model: {model_path}")
    model = load_model(model_path)
    node_feats = build_static_node_features()
    adj = build_normalized_adj()
    dist_matrix = build_dist_matrix()
    print(f"Ready.  {N_STN} stations, {len(ITEM_TYPES)} item types.")
    print(f"Items: {', '.join(ITEM_TYPES)}")
    print(f"Day:   0=Mon … 6=Sun.    Hour: 0–23.")
    print(f"Type 'q' / 'quit' to exit.\n")

    last_item, last_hour, last_dow = "phone", 18, 1

    while True:
        try:
            raw = input("Found at station: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if raw.lower() in ("q", "quit", "exit"): break
        match = fuzzy_station(raw)
        if match is None:
            print(f"  No match for '{raw}'.  Try a partial name like 'sagrada'.")
            continue
        if isinstance(match, list):
            print(f"  Ambiguous ({len(match)} matches): {', '.join(match[:10])}"
                  + (" …" if len(match) > 10 else ""))
            continue

        item = input(f"Item type [{last_item}]: ").strip().lower() or last_item
        if item not in ITEM_INDEX:
            print(f"  Unknown item '{item}', using 'other'.")
            item = "other"

        hour_in = input(f"Hour 0–23 [{last_hour}]: ").strip() or str(last_hour)
        dow_in = input(f"Day 0=Mon..6=Sun [{last_dow}]: ").strip() or str(last_dow)
        try:
            hour = int(hour_in) % 24
            dow = int(dow_in) % 7
        except ValueError:
            print("  Bad hour/day, skipping."); continue

        last_item, last_hour, last_dow = item, hour, dow
        probs, cost, idx = predict(model, node_feats, adj, dist_matrix,
                                    match, item, hour, dow)
        render(match, item, hour, dow, probs, cost, idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--found", default=None, help="Station (partial match OK)")
    parser.add_argument("--item", default="phone")
    parser.add_argument("--hour", type=int, default=18)
    parser.add_argument("--dow", type=int, default=1)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    model_path = args.model or find_default_model()
    if model_path is None or not Path(model_path).exists():
        print("No trained model found. Run:")
        print("  python synth.py --n 50000 --out artifacts/data/dev/events.jsonl")
        print("  python train.py --data artifacts/data/dev/events.jsonl --out artifacts/models/dev")
        sys.exit(1)

    if args.found is None:
        interactive(model_path)
        return

    model = load_model(model_path)
    node_feats = build_static_node_features()
    adj = build_normalized_adj()
    dist_matrix = build_dist_matrix()

    match = fuzzy_station(args.found)
    if match is None:
        print(f"No station matches '{args.found}'"); sys.exit(1)
    if isinstance(match, list):
        print(f"Ambiguous: {', '.join(match[:10])}"); sys.exit(1)

    probs, cost, idx = predict(model, node_feats, adj, dist_matrix,
                                match, args.item, args.hour, args.dow, args.top)
    render(match, args.item, args.hour, args.dow, probs, cost, idx)


if __name__ == "__main__":
    main()
