"""
Interactive demo for the v2 (heterogeneous-graph) model.

Pick found-at station, item type, hour, day → top-5 destination predictions
and top-10 storage recommendations.
"""
import sys, os, argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from contexts import context_id, ITEM_TYPES, ITEM_INDEX, context_label
from graph_build import graph_to_device
from model import HetGNN
from decision import build_dist_matrix, recommend_storage
from network import STATION_LIST, STATION_INDEX, N_STN, STATION_ZONE
from aggregate import find_default_model, load_model_and_graph


DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def fuzzy_station(query):
    if not query: return None
    q = query.strip().lower()
    if not q: return None
    for s in STATION_LIST:
        if s.lower() == q: return s
    matches = [s for s in STATION_LIST if q in s.lower()]
    if len(matches) == 1: return matches[0]
    if not matches: return None
    return matches


def predict(model, graph, dist_matrix, found_at, item, hour, dow,
            lost_at=None, top_k=10):
    """`lost_at` is the passenger-reported loss station. Defaults to found_at
    when the questionnaire is unavailable (legacy / staff-only events)."""
    cid = context_id(item, hour, dow)
    fi = torch.tensor([STATION_INDEX[found_at]], dtype=torch.long)
    li = torch.tensor([STATION_INDEX[lost_at or found_at]], dtype=torch.long)
    ci = torch.tensor([cid], dtype=torch.long)
    with torch.no_grad():
        h_st, h_ctx = model.encode(graph)
        logits = model.score(h_ctx[ci], h_st, found_st_idx=fi, lost_st_idx=li)
        probs = F.softmax(logits, dim=-1)[0]
    cost, idx = recommend_storage(probs, dist_matrix, top_k=top_k)
    return probs, cost, idx, cid


def render(found_at, item, hour, dow, cid, probs, cost, idx, top_dest=5,
           lost_at=None):
    print(f"\n  Found:  {found_at}  ({STATION_ZONE[found_at]})")
    if lost_at and lost_at != found_at:
        print(f"  Lost:   {lost_at}  ({STATION_ZONE[lost_at]})   "
              f"(passenger-reported loss station)")
    print(f"  Item:   {item}    Hour: {int(hour):02d}:00    Day: {DAY_NAMES[int(dow)]}")
    print(f"  Context: id={cid}  →  {context_label(cid)}")

    print(f"\n  Top {top_dest} DESTINATION predictions (where passenger is going):")
    td = torch.topk(probs, k=top_dest)
    for p, i in zip(td.values, td.indices):
        s = STATION_LIST[i.item()]
        bar = "█" * int(round(p.item() * 30))
        print(f"    {p.item():.3f}  {s:<35s} ({STATION_ZONE[s]:<12s}) {bar}")

    print(f"\n  Top {len(idx)} STORAGE recommendations (lowest expected pickup hops):")
    max_cost = float(cost.max().item()) if len(cost) else 1.0
    for rank, (c, i) in enumerate(zip(cost.tolist(), idx.tolist()), 1):
        s = STATION_LIST[i]
        bar = "█" * int(round(20 * (1 - c / max(max_cost, 1e-6))))
        print(f"    {rank:2d}. {s:<35s} ({STATION_ZONE[s]:<12s})  E[hops]={c:5.2f}  {bar}")
    print()


def interactive(model_path):
    print(f"Loading {model_path}")
    model, graph, cfg = load_model_and_graph(model_path)
    dist_matrix = build_dist_matrix()
    print(f"Ready. {N_STN} stations, {len(ITEM_TYPES)} items, {cfg['n_contexts']} contexts.")
    print(f"Items: {', '.join(ITEM_TYPES)}")
    print(f"'q' to quit.\n")
    last = {"item": "phone", "hour": 18, "dow": 1}
    while True:
        try:
            raw = input("Found at: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if raw.lower() in ("q", "quit", "exit"): break
        m = fuzzy_station(raw)
        if m is None:
            print(f"  No match for '{raw}'"); continue
        if isinstance(m, list):
            print(f"  Ambiguous: {', '.join(m[:10])}"); continue

        # Optional: passenger-reported loss station (different from where staff
        # found it). Press Enter to default to the found_at station.
        lost_raw = input("Lost at (Enter = same as found): ").strip()
        lost_at = None
        if lost_raw:
            lm = fuzzy_station(lost_raw)
            if lm is None:
                print(f"  No match for '{lost_raw}', defaulting to found_at"); lost_at = None
            elif isinstance(lm, list):
                print(f"  Ambiguous: {', '.join(lm[:10])}; defaulting to found_at")
                lost_at = None
            else:
                lost_at = lm

        item = input(f"Item [{last['item']}]: ").strip().lower() or last["item"]
        if item not in ITEM_INDEX:
            print(f"  unknown item, using 'other'"); item = "other"
        try:
            hour = int(input(f"Hour [{last['hour']}]: ").strip() or last["hour"]) % 24
            dow = int(input(f"Day 0=Mon..6=Sun [{last['dow']}]: ").strip() or last["dow"]) % 7
        except ValueError:
            print("  bad hour/day"); continue
        last = {"item": item, "hour": hour, "dow": dow}
        probs, cost, idx, cid = predict(model, graph, dist_matrix, m, item, hour, dow,
                                         lost_at=lost_at)
        render(m, item, hour, dow, cid, probs, cost, idx, lost_at=lost_at)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--found", default=None)
    parser.add_argument("--lost",  default=None,
                        help="Passenger-reported loss station. Defaults to --found.")
    parser.add_argument("--item", default="phone")
    parser.add_argument("--hour", type=int, default=18)
    parser.add_argument("--dow", type=int, default=1)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    model_path = args.model or find_default_model()
    if model_path is None or not Path(model_path).exists():
        print("No trained model. Run train.py first."); sys.exit(1)
    if args.found is None:
        interactive(model_path); return
    model, graph, _ = load_model_and_graph(model_path)
    dist_matrix = build_dist_matrix()
    m = fuzzy_station(args.found)
    if not isinstance(m, str):
        print(f"Bad station: {args.found}"); sys.exit(1)
    lost = None
    if args.lost:
        lm = fuzzy_station(args.lost)
        if not isinstance(lm, str):
            print(f"Bad lost station: {args.lost}"); sys.exit(1)
        lost = lm
    probs, cost, idx, cid = predict(model, graph, dist_matrix, m,
                                     args.item, args.hour, args.dow,
                                     lost_at=lost, top_k=args.top)
    render(m, args.item, args.hour, args.dow, cid, probs, cost, idx, lost_at=lost)


if __name__ == "__main__":
    main()
