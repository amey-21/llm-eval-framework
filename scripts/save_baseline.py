# scripts/save_baseline.py
"""
Run this locally after a good eval run to save the baseline.
Commit the output file. CI will compare future runs against it.
"""

import json
from pathlib import Path
from src.db.repository import EvalRepository


def main():
    repo = EvalRepository()

    # get the most recent run
    run_id = repo.get_latest_run_id()
    if not run_id:
        print("No runs found. Run scripts/run_eval.py first.")
        return

    leaderboard = repo.get_leaderboard(run_id)

    # structure: {model_name: {metric_name: avg_score}}
    baseline = {}
    for entry in leaderboard:
        model = entry["model_name"]
        metric = entry["metric_name"]
        score = entry["avg_score"]

        if model not in baseline:
            baseline[model] = {}
        baseline[model][metric] = score

    output_path = Path("results/baseline.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(baseline, indent=2))

    print(f"Baseline saved from run_id={run_id}")
    print(f"Models: {list(baseline.keys())}")
    print(f"Metrics: {list(next(iter(baseline.values())).keys())}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()