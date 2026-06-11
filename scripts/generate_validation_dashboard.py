#!/usr/bin/env python
"""Generate the local model-performance dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc26_predictor.reporting.validation_dashboard import (
    ValidationDashboardConfig,
    generate_validation_dashboard,
)


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

    path = generate_validation_dashboard(
        project_root=args.project_root,
        destination=args.destination,
        config=ValidationDashboardConfig(
            bootstrap_samples=args.bootstrap_samples,
            random_seed=args.seed,
        ),
    )
    print(f"Validation dashboard -> {path}")


if __name__ == "__main__":
    main()
