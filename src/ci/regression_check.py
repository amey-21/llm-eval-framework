"""
Compares a new eval run against the committed baseline.
Exits with code 1 if any metric regresses beyond the threshold.

Exit code matters for CI:
  0 = success (CI passes)
  1 = failure (CI blocks the PR)
"""

import json
import sys
from pathlib import Path
from loguru import logger



def load_baseline(path: str = "results/baseline.json") -> dict:
    baseline_path = Path(path)
    if not baseline_path.exists():
        logger.error(f"Baseline not found at {path}")
        logger.error("Run: python scripts/save_baseline.py")
        sys.exit(1)
    return json.loads(baseline_path.read_text())


def load_current_results(results_path: str) -> dict:
    """Load results written by the CI eval run."""
    path = Path(results_path)
    if not path.exists():
        logger.error(f"Results not found at {results_path}")
        sys.exit(1)
    return json.loads(path.read_text())


SKIP_METRICS = {"latency", "cost_efficiency"}


def check_regressions(
    baseline: dict,
    current: dict,
    threshold: float = 0.05,
    skip_metrics: set = SKIP_METRICS,
) -> list[dict]:
    regressions = []

    for model, metrics in baseline.items():
        if model not in current:
            logger.warning(f"Model {model!r} in baseline but not in current run — skipping")
            continue

        for metric, baseline_score in metrics.items():

            # infrastructure metrics vary by external factors
            # monitor them in the dashboard, don't gate CI on them
            if metric in skip_metrics:
                logger.info(f"Skipping {metric} for {model} (infrastructure metric)")
                continue

            if metric not in current[model]:
                logger.warning(f"Metric {metric!r} missing for {model} — skipping")
                continue

            current_score = current[model][metric]
            drop = baseline_score - current_score

            if round(drop, 4) > threshold:
                regressions.append({
                    "model": model,
                    "metric": metric,
                    "baseline": baseline_score,
                    "current": current_score,
                    "drop": drop,
                })

    return regressions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/ci_run.json")
    parser.add_argument("--baseline", default="results/baseline.json")
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    baseline = load_baseline(args.baseline)
    current = load_current_results(args.results)
    regressions = check_regressions(baseline, current, args.threshold)

    # always print the full comparison table
    print("\n── Regression Check Results ──────────────────────")
    print(f"{'Model':<30} {'Metric':<25} {'Baseline':>10} {'Current':>10} {'Delta':>8}")
    print("─" * 85)

    all_models = set(baseline) | set(current)
    for model in sorted(all_models):
        all_metrics = set(baseline.get(model, {})) | set(current.get(model, {}))
        for metric in sorted(all_metrics):
            b_score = baseline.get(model, {}).get(metric, None)
            c_score = current.get(model, {}).get(metric, None)
            if b_score is None or c_score is None:
                continue
            delta = c_score - b_score
            flag = " ← REGRESSION" if (b_score - c_score) > args.threshold else ""
            print(
                f"{model:<30} {metric:<25} "
                f"{b_score:>10.4f} {c_score:>10.4f} "
                f"{delta:>+8.4f}{flag}"
            )

    print("─" * 85)

    if regressions:
        print(f"\n❌ {len(regressions)} regression(s) detected (threshold={args.threshold}):\n")
        for r in regressions:
            print(
                f"  {r['model']} / {r['metric']}: "
                f"{r['baseline']:.4f} → {r['current']:.4f} "
                f"(dropped {r['drop']:.4f})"
            )
        print("\nFailing CI build.")
        sys.exit(1)
    else:
        print(f"\n✅ No regressions detected. All metrics within {args.threshold} of baseline.")
        sys.exit(0)


if __name__ == "__main__":
    main()