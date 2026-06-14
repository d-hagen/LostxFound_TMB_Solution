"""
Yearly retrain pipeline with a promotion gate.

Inputs each year:
  - fresh synthetic corpus (new Dirichlet draws → different station hotspots)
  - new questionnaire CSV/JSONL of real events (optional; upweighted at train)

Pipeline:
  1) generate synth, ingest real (if present), merge → versioned data dir
  2) train candidate model → versioned model dir
  3) evaluate candidate AND current production on the held-out test split
  4) promotion gate:
       a. E[hops] must improve by >= min_improvement
       b. coverage must not collapse
       c. no per-zone E[hops] regression > max_per_zone_regression
  5) if gate passes, point  artifacts/models/current_version.txt -> new version

Run with no args after the first time and it just regenerates synth + retrains.
"""
import sys, os, json, argparse
from datetime import datetime
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events import FoundEvent, load_events, save_events
from synth import generate_synthetic_events, validate_corpus
from train import train


ARTIFACTS = Path(__file__).parent / "artifacts"
CURRENT_PTR = ARTIFACTS / "models" / "current_version.txt"


def gate_decision(candidate, current, min_improvement=0.03, max_zone_regression=0.10):
    """Return (promote: bool, reasons: list[str]). Uses test_metrics dict from metadata.json."""
    reasons = []
    if current is None:
        reasons.append("No current production model — promote candidate.")
        return True, reasons

    cand_h = candidate["expected_hops"]
    curr_h = current["expected_hops"]
    if cand_h > curr_h * (1.0 - min_improvement):
        reasons.append(
            f"E[hops] regression: {curr_h:.3f} -> {cand_h:.3f} "
            f"(needs >={min_improvement*100:.0f}% drop)"
        )
        return False, reasons

    if candidate["coverage"] < 0.5 * current["coverage"] and candidate["coverage"] < 0.1:
        reasons.append(
            f"Coverage collapse: {current['coverage']:.2f} -> {candidate['coverage']:.2f}"
        )
        return False, reasons

    for zone, cand_z in candidate["zone_hops"].items():
        curr_z = current["zone_hops"].get(zone, cand_z)
        if curr_z > 0 and (cand_z - curr_z) / curr_z > max_zone_regression:
            reasons.append(
                f"Zone '{zone}' regressed by {(cand_z - curr_z) / curr_z:.1%} "
                f"({curr_z:.2f} -> {cand_z:.2f})"
            )
            return False, reasons

    reasons.append(
        f"E[hops] {curr_h:.3f} -> {cand_h:.3f}  |  "
        f"cov {current['coverage']:.2f} -> {candidate['coverage']:.2f}"
    )
    return True, reasons


def get_current_version():
    if CURRENT_PTR.exists():
        return CURRENT_PTR.read_text().strip()
    return None


def set_current_version(version):
    CURRENT_PTR.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_PTR.write_text(version)


def run(real_data_path=None, n_synth=50000, alpha=3.0, seed=None,
        real_weight=5.0, epochs=30, hidden=64, layers=3):
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    data_dir = ARTIFACTS / "data" / version
    model_dir = ARTIFACTS / "models" / version
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    if seed is None:
        seed = int(datetime.now().timestamp()) % (2**31)

    print(f"=== Retrain {version} (seed={seed}) ===")

    # 1) generate synth
    synth = generate_synthetic_events(n_synth, seed=seed, alpha=alpha)
    stats = validate_corpus(synth)
    print(f"Synth: {stats['n_events']} events, top-station {stats['top_found_frac']:.1%}, "
          f"entropy gap {stats['max_entropy'] - stats['found_entropy']:.2f}")

    # 2) ingest real, upweight
    real = []
    if real_data_path and Path(real_data_path).exists():
        raw = load_events(real_data_path)
        real = [FoundEvent(e.found_at, e.found_dt, e.item_type, e.pickup, "real", real_weight)
                for e in raw]
        print(f"Real: {len(real)} questionnaire events  (upweighted to {real_weight}x)")

    all_events = synth + real
    data_path = data_dir / "events.jsonl"
    save_events(all_events, data_path)

    # 3) train candidate
    train(
        data_path=str(data_path), output_dir=str(model_dir),
        epochs=epochs, hidden=hidden, layers=layers, seed=seed,
    )

    # 4) gate
    cand_meta = json.loads((model_dir / "metadata.json").read_text())
    cand_metrics = cand_meta["test_metrics"]

    current_version = get_current_version()
    if current_version and (ARTIFACTS / "models" / current_version / "metadata.json").exists():
        curr_meta = json.loads((ARTIFACTS / "models" / current_version / "metadata.json").read_text())
        current_metrics = curr_meta["test_metrics"]
        print(f"\nCurrent production: {current_version}  "
              f"E[hops]={current_metrics['expected_hops']:.3f}  "
              f"cov={current_metrics['coverage']:.2f}")
    else:
        current_metrics = None

    promote, reasons = gate_decision(cand_metrics, current_metrics)
    print(f"\n=== GATE: {'PROMOTE' if promote else 'REJECT'} ===")
    for r in reasons:
        print(f"  - {r}")

    if promote:
        set_current_version(version)
        print(f"\nProduction is now: {version}")
    else:
        print(f"\nKeeping production = {current_version}.  Candidate retained at {model_dir}")

    return version, promote


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default=None, help="Path to real questionnaire events (jsonl)")
    parser.add_argument("--n_synth", type=int, default=50000)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--real_weight", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    args = parser.parse_args()
    run(
        real_data_path=args.real, n_synth=args.n_synth, alpha=args.alpha,
        seed=args.seed, real_weight=args.real_weight,
        epochs=args.epochs, hidden=args.hidden, layers=args.layers,
    )
