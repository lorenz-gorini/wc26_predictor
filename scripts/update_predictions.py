#!/usr/bin/env python
"""Refresh post-match data and regenerate World Cup 2026 predictions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from wc26_predictor.reporting.prediction_history import archive_latest_prediction_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root directory.",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=2000,
        help="Monte Carlo simulations for tournament forecasts.",
    )
    parser.add_argument(
        "--impact-simulations",
        type=int,
        default=400,
        help="Monte Carlo simulations for per-match impact diagnostics.",
    )
    parser.add_argument(
        "--top-scorer-simulations",
        type=int,
        default=100_000,
        help="Monte Carlo simulations for top-scorer probabilities.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for final report top-scorer probabilities.",
    )
    parser.add_argument(
        "--refresh-static-data",
        action="store_true",
        help="Also refresh fixtures, squads, and club-form tables.",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    python = sys.executable
    archived = archive_latest_prediction_snapshot(root / "data" / "processed")
    if archived is not None:
        print(f"Archived pre-update prediction snapshot -> {archived}")

    download_command = [
        python,
        str(root / "scripts" / "download_data.py"),
        "--project-root",
        str(root),
    ]
    if not args.refresh_static_data:
        download_command.append("--skip-fixtures")

    _run(download_command, root)
    _run(
        [
            python,
            str(root / "scripts" / "run_baselines.py"),
            "--project-root",
            str(root),
            "--n-simulations",
            str(args.n_simulations),
            "--impact-simulations",
            str(args.impact_simulations),
        ],
        root,
    )
    _run(
        [
            python,
            str(root / "scripts" / "generate_final_reports.py"),
            "--project-root",
            str(root),
            "--top-scorer-simulations",
            str(args.top_scorer_simulations),
            "--seed",
            str(args.seed),
        ],
        root,
    )
    _run(
        [
            python,
            str(root / "scripts" / "generate_validation_dashboard.py"),
            "--project-root",
            str(root),
        ],
        root,
    )


def _run(command: list[str], cwd: Path) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    main()
