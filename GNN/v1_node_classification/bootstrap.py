"""
First-iteration GNN bootstrap — simulates the end-of-Year-0 state.

Two corpora:
  1) prior   — large, generated from hand-coded behavioural profiles. This is
               our domain knowledge encoded as data.
  2) real    — small, generated from a PERTURBED profile set. Stands in for
               the actual Year-0 questionnaire dump. In production this file
               is the real questionnaire JSONL.

Pipeline:
  1) generate prior corpus + real corpus
  2) PRETRAIN the GNN on the prior corpus
  3) FINE-TUNE that checkpoint on the real corpus (heavily upweighted, lower lr)
  4) (optional) train a real-only baseline for comparison
  5) promote the fine-tuned model as production
  6) aggregate over real events to size lockers

Run:
  python3 bootstrap.py
  python3 bootstrap.py --n_prior 50000 --n_real 2000 \
                       --epochs_pretrain 20 --epochs_finetune 25
"""
import sys, os, json, argparse
from pathlib import Path
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events import save_events, load_events, FoundEvent
from synth import generate_synthetic_events, generate_real_events, validate_corpus
from train import train


ARTIFACTS = Path(__file__).parent / "artifacts"


def set_current_version(version):
    p = ARTIFACTS / "models" / "current_version.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(version)


def _stats(events, label):
    from collections import Counter
    c = Counter(e.found_at for e in events)
    p = Counter(e.pickup for e in events if e.pickup)
    top_found = c.most_common(1)[0] if c else (None, 0)
    print(f"{label:8s}  n={len(events):>6d}  unique_found={len(c):>3d}  "
          f"unique_pickup={len(p):>3d}  top_found={top_found[0]}({top_found[1]/max(len(events),1):.1%})")


def run(n_prior=50000, n_real=2000, seed=0,
        epochs_pretrain=20, epochs_finetune=25,
        lr_pretrain=1e-3, lr_finetune=3e-4,
        real_weight=10.0, train_real_only=True,
        hidden=64, layers=3):
    version = datetime.now().strftime("v%Y%m%d_%H%M%S_bootstrap")
    data_dir = ARTIFACTS / "data" / version
    models_dir = ARTIFACTS / "models" / version
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── 1) generate corpora ──────────────────────────────────────────────
    print("=" * 70)
    print(f"BOOTSTRAP {version}  (seed={seed})")
    print("=" * 70)

    print("\n[1/5] Generating prior corpus from hand-coded behavioural assumptions...")
    prior = generate_synthetic_events(n_prior, seed=seed)
    try:
        validate_corpus(prior)
    except ValueError as e:
        print(f"  prior validation warning: {e}")
    prior_path = data_dir / "prior.jsonl"
    save_events(prior, prior_path)
    _stats(prior, "prior")

    print(f"\n[2/5] Generating pseudo-real questionnaire (Year-0 corpus, {n_real} events)...")
    real = generate_real_events(n_real, seed=seed + 1)
    # Upweight at training; the synth corpus dominates by count, so without
    # weight the real signal would be drowned.
    real = [FoundEvent(e.found_at, e.found_dt, e.item_type, e.pickup, "real", real_weight)
            for e in real]
    real_path = data_dir / "real.jsonl"
    save_events(real, real_path)
    _stats(real, "real")
    print(f"  (real events upweighted to {real_weight}x at training)")

    # ── 3) pretrain on prior ─────────────────────────────────────────────
    pretrain_dir = models_dir / "pretrain"
    print(f"\n[3/5] Pretraining on prior corpus  →  {pretrain_dir}")
    print("-" * 70)
    train(
        data_path=str(prior_path), output_dir=str(pretrain_dir),
        epochs=epochs_pretrain, lr=lr_pretrain, seed=seed,
        hidden=hidden, layers=layers,
    )

    # ── 4) fine-tune on real ─────────────────────────────────────────────
    finetune_dir = models_dir / "production"
    print(f"\n[4/5] Fine-tuning on real corpus  →  {finetune_dir}")
    print("-" * 70)
    train(
        data_path=str(real_path), output_dir=str(finetune_dir),
        epochs=epochs_finetune, lr=lr_finetune, seed=seed,
        init_from=str(pretrain_dir / "model.pt"),
        patience=8,
    )
    set_current_version(f"{version}/production")
    print(f"\nProduction model: {finetune_dir}/model.pt")
    print(f"current_version.txt -> {version}/production")

    # ── 4b) real-only baseline (optional but informative) ────────────────
    if train_real_only:
        real_only_dir = models_dir / "real_only"
        print(f"\n[+] Real-only baseline (no synth prior)  →  {real_only_dir}")
        print("-" * 70)
        train(
            data_path=str(real_path), output_dir=str(real_only_dir),
            epochs=epochs_finetune, lr=lr_pretrain, seed=seed,
            hidden=hidden, layers=layers, patience=8,
        )

    # ── 5) headline comparison ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[5/5] Comparison on real-corpus test split")
    print("=" * 70)

    def _show(label, model_dir):
        meta = json.loads((model_dir / "metadata.json").read_text())
        m = meta["test_metrics"]
        print(f"  {label:24s}  top1={m['top1']:.3f}  top5={m['top5']:.3f}  "
              f"E[hops]={m['expected_hops']:.2f}  cov={m['coverage']:.2f}")

    _show("pretrain (prior only)", pretrain_dir)
    if train_real_only:
        _show("real-only (no prior)", real_only_dir)
    _show("production (pretrain+ft)", finetune_dir)

    print()
    print("Note: the 'pretrain' row tests on the PRIOR data, so its numbers")
    print("      are not directly comparable to the real-test rows.")
    print()
    print("Next step: run aggregate over the REAL corpus to size lockers:")
    print(f"  python3 aggregate.py --data {real_path} --top 40")

    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_prior", type=int, default=50000)
    parser.add_argument("--n_real", type=int, default=2000,
                        help="Year-0 questionnaire response count")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs_pretrain", type=int, default=20)
    parser.add_argument("--epochs_finetune", type=int, default=25)
    parser.add_argument("--lr_pretrain", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=3e-4)
    parser.add_argument("--real_weight", type=float, default=10.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--no_real_only", action="store_true",
                        help="Skip the real-only baseline run.")
    args = parser.parse_args()

    run(
        n_prior=args.n_prior, n_real=args.n_real, seed=args.seed,
        epochs_pretrain=args.epochs_pretrain, epochs_finetune=args.epochs_finetune,
        lr_pretrain=args.lr_pretrain, lr_finetune=args.lr_finetune,
        real_weight=args.real_weight,
        train_real_only=not args.no_real_only,
        hidden=args.hidden, layers=args.layers,
    )
