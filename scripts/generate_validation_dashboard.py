#!/usr/bin/env python
"""Generate the local model-performance dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc26_predictor.reporting.validation_dashboard import (
    ValidationDashboardConfig,
    generate_validation_dashboard,
)
from wc26_predictor.reporting.world_cup_dashboard import generate_world_cup_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root directory.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Optional output HTML path. Defaults to reports/model_performance_dashboard.html.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2_000,
        help="Number of bootstrap resamples for validation intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for bootstrap intervals.",
    )
    args = parser.parse_args()

    reports_dir = args.project_root / "reports"
    validation_destination = (
        args.destination
        if args.destination is not None
        else reports_dir / "dashboard" / "model_performance.html"
    )
    validation_destination.parent.mkdir(parents=True, exist_ok=True)
    path = generate_validation_dashboard(
        project_root=args.project_root,
        destination=validation_destination,
        config=ValidationDashboardConfig(
            bootstrap_samples=args.bootstrap_samples,
            random_seed=args.seed,
        ),
    )
    dashboard_paths = generate_world_cup_dashboard(
        project_root=args.project_root,
        index_path=reports_dir / "model_performance_dashboard.html",
        dashboard_dir=reports_dir / "dashboard",
    )
    print(f"Validation dashboard -> {path}")
    print(f"World Cup dashboard -> {dashboard_paths.index}")
    print(f"Future matches page -> {dashboard_paths.future_matches}")
    print(f"Group-stage page -> {dashboard_paths.group_stage}")
    print(f"Next-phases page -> {dashboard_paths.next_phases}")


if __name__ == "__main__":
    main()
