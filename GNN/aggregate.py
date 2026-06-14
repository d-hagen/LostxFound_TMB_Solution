"""
Aggregate: rerun the trained heterogeneous GNN over every event in a corpus
and tally, per station, how often it is the recommended storage.
"""
import sys, os, argparse, csv
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F

from events import load_events
from graph_build import build_hetero_graph, events_to_tensors, graph_to_device
from model import HetGNN
from decision import build_dist_matrix, recommend_storage
from network import STATION_LIST, N_STN, STATION_ZONE
from contexts import N_CONTEXTS


def find_default_model():
    root = Path(__file__).parent / "artifacts" / "models"
    ptr = root / "current_version.txt"
    if ptr.exists():
        v = ptr.read_text().strip()
        p = root / v / "model.pt"
        if p.exists(): return p
    p = root / "dev" / "model.pt"
    return p if p.exists() else None


def load_model_and_graph(model_path, device="cpu"):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = HetGNN(
        station_feat_dim=cfg["station_feat_dim"],
        n_contexts=cfg["n_contexts"],
        hidden=cfg["hidden"], n_layers=cfg["layers"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    graph = torch.load(Path(model_path).parent / "graph.pt", map_location=device, weights_only=False)
    graph = graph_to_device(graph, device)
    return model, graph, cfg


def _weeks_spanned(events):
    if not events: return 1.0
    ts = [datetime.fromisoformat(e.found_dt) for e in events]
    return max(((max(ts) - min(ts)).days + 1) / 7.0, 1.0)


def aggregate(events, model, graph, dist_matrix, batch_size=512, device="cpu"):
    tally = torch.zeros(N_STN, dtype=torch.long)
    et = events_to_tensors(events)
    for k in et: et[k] = et[k].to(device)
    with torch.no_grad():
        h_st, h_ctx = model.encode(graph)
        n = et["pickup"].shape[0]
        for i in range(0, n, batch_size):
            sl = slice(i, i + batch_size)
            logits = model.score(
                h_ctx[et["ctx"][sl]], h_st,
                found_st_idx=et["found_st"][sl],
                lost_st_idx=et.get("lost_st", et["found_st"])[sl],
            )
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
        if n == 0: continue
        rows.append((STATION_LIST[i], STATION_ZONE[STATION_LIST[i]], n, n / weeks, n / total))
    rows.sort(key=lambda r: r[3], reverse=True)
    if top: rows = rows[:top]
    print(f"\nCorpus: {total} events over {weeks:.1f} weeks "
          f"({weeks/52*12:.1f} months equiv).")
    print(f"Stations used: {sum(1 for i in range(N_STN) if tally[i].item() > 0)} / {N_STN}\n")
    max_pw = rows[0][3] if rows else 1.0
    header = f"{'Rank':>4}  {'Station':<35s} {'Zone':<12s}  {'items/week':>10s}  {'total':>7s}  {'share':>6s}   load"
    print(header); print("-" * len(header))
    for rank, (s, z, n, pw, sh) in enumerate(rows, 1):
        bar = "█" * int(round(28 * pw / max(max_pw, 1e-6)))
        print(f"{rank:>4}  {s:<35s} {z:<12s}  {pw:>10.2f}  {n:>7d}  {sh:>5.1%}   {bar}")
    print()
    return rows


def to_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "station", "zone", "total_items", "items_per_week", "share"])
        for rank, (s, z, n, pw, sh) in enumerate(rows, 1):
            w.writerow([rank, s, z, n, f"{pw:.4f}", f"{sh:.6f}"])
    print(f"CSV -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/synth.jsonl")
    parser.add_argument("--model", default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    model_path = args.model or find_default_model()
    if model_path is None or not Path(model_path).exists():
        print("No trained model. Run: python3 train.py --data ... --out artifacts/models/dev")
        sys.exit(1)
    print(f"Loading events: {args.data}")
    events = load_events(args.data)
    print(f"Loading model:  {model_path}")
    model, graph, _ = load_model_and_graph(model_path)
    dist_matrix = build_dist_matrix()
    weeks = _weeks_spanned(events)
    tally = aggregate(events, model, graph, dist_matrix)
    rows = render(tally, weeks, top=args.top)
    if args.csv: to_csv(rows, args.csv)


if __name__ == "__main__":
    main()
