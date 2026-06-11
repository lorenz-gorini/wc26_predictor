#!/usr/bin/env python
"""Generate final World Cup 2026 forecast reports from processed pipeline outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc26_predictor.reporting.final_reports import FinalReportConfig, generate_final_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root directory.",
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
        help="Random seed for top-scorer probability simulation.",
    )
    args = parser.parse_args()

    paths = generate_final_reports(
        processed_dir=args.project_root / "data" / "processed",
        reports_dir=args.project_root / "reports",
        config=FinalReportConfig(
            top_scorer_simulations=args.top_scorer_simulations,
            random_seed=args.seed,
        ),
    )

    print(f"Group advancement probabilities -> {paths.group_advancement}")
    print(f"Round-by-round probabilities -> {paths.round_by_round}")
    print(f"Winner probabilities -> {paths.winners}")
    print(f"Top-scorer probabilities -> {paths.top_scorers}")
    print(f"Group match forecasts -> {paths.group_matches}")
    print(f"Knockout match forecasts -> {paths.knockout_matches}")
    print(f"Full match forecasts -> {paths.full_matches}")
    print(f"Markdown report -> {paths.markdown_report}")


if __name__ == "__main__":
    main()
