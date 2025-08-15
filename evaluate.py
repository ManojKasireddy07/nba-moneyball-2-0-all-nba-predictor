"""
Evaluation utilities for the All‑NBA models.

This script reads the metrics generated during training and aggregates them
across years to produce summary statistics.  It also serializes the
aggregate metrics back into the ``reports/`` directory for easy consumption
by the Streamlit app or the README.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def aggregate_metrics(per_year: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute the mean of precision, recall and F1 across seasons."""
    agg: Dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    n = len(per_year)
    if n == 0:
        return agg
    for metrics in per_year.values():
        for k in agg:
            agg[k] += metrics.get(k, 0.0)
    for k in agg:
        agg[k] /= n
    return agg


def main(args: argparse.Namespace) -> None:
    reports_dir = Path(args.reports_dir)
    metrics_path = reports_dir / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"metrics.json not found at {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    summary: Dict[str, Dict[str, float]] = {}
    for model_name, per_year in metrics.items():
        summary[model_name] = aggregate_metrics(per_year)
    summary_path = reports_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved aggregated metrics to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate per‑season metrics")
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory containing metrics.json and where to write metrics_summary.json",
    )
    args = parser.parse_args()
    main(args)