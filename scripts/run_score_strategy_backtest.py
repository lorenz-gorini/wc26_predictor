#!/usr/bin/env python
"""Backtest exact-score strategies on historical World Cup windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wc26_predictor.data.ingest_results import load_results_csv
from wc26_predictor.evaluation.score_strategy import (
    rolling_score_strategy_backtest,
    write_score_strategy_report,
)
from wc26_predictor.models.elo import EloConfig
from wc26_predictor.models.form_adjusted_poisson import FormAdjustedPoissonConfig
from wc26_predictor.models.poisson import PoissonConfig
from wc26_predictor.pipelines.baselines import BaselineModelConfigs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root directory.",
    )
    args = parser.parse_args()

    processed_dir = args.project_root / "data" / "processed"
    reports_dir = args.project_root / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = load_results_csv(processed_dir / "international_results.csv")
    configs = _load_model_configs(processed_dir / "world_cup_2026_selected_hyperparameters.csv")

    print("Running rolling score-strategy backtest...", flush=True)
    result = rolling_score_strategy_backtest(
        results,
        configs=configs,
        progress=lambda message: print(message, flush=True),
    )

    matches_path = processed_dir / "world_cup_score_strategy_backtest_matches.csv"
    summary_path = reports_dir / "score_strategy_backtest_summary.csv"
    report_path = reports_dir / "score_strategy_backtest_report.md"
    result.predictions.to_csv(matches_path, index=False)
    result.summary.to_csv(summary_path, index=False)
    write_score_strategy_report(result.summary, result.predictions, str(report_path))

    print(f"Score strategy match predictions -> {matches_path}", flush=True)
    print(f"Score strategy summary -> {summary_path}", flush=True)
    print(f"Score strategy report -> {report_path}", flush=True)


def _load_model_configs(path: Path) -> BaselineModelConfigs:
    frame = pd.read_csv(path)
    required = {"model", "parameter", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing selected hyperparameter columns: {sorted(missing)}")
    values = {
        (str(row.model), str(row.parameter)): row.value
        for row in frame.itertuples(index=False)
    }
    poisson = PoissonConfig(
        prior_strength=float(values[("poisson", "prior_strength")]),
        recency_half_life_days=int(values[("poisson", "recency_half_life_days")]),
        use_tournament_importance=_parse_bool(
            values[("poisson", "use_tournament_importance")]
        ),
    )
    return BaselineModelConfigs(
        elo=EloConfig(
            k_factor=float(values[("elo", "k_factor")]),
            draw_probability=float(values[("elo", "draw_probability")]),
            home_advantage=float(values[("elo", "home_advantage")]),
            neutral_home_advantage=float(values[("elo", "neutral_home_advantage")]),
        ),
        poisson=poisson,
        form_adjusted_poisson=FormAdjustedPoissonConfig(
            poisson=poisson,
            form_window_matches=int(values[("form_adjusted_poisson", "form_window_matches")]),
            form_prior_matches=float(values[("form_adjusted_poisson", "form_prior_matches")]),
            form_strength=float(values[("form_adjusted_poisson", "form_strength")]),
        ),
    )


def _parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


if __name__ == "__main__":
    main()
