#!/usr/bin/env python
"""Evaluate recent completed World Cup matches with chronological holdouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wc26_predictor.data.ingest_results import load_results_csv
from wc26_predictor.data.odds import load_football_data_world_cup_odds
from wc26_predictor.evaluation.contest_points import APP_PROBABILITY_COLUMNS
from wc26_predictor.evaluation.recent_world_cup import (
    evaluate_recent_world_cup_holdouts,
    load_market_probabilities_csv,
    validate_market_probability_frame,
    write_recent_holdout_report,
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
    parser.add_argument(
        "--holdout-sizes",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Recent completed-match block sizes to evaluate.",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    processed_dir = root / "data" / "processed"
    raw_dir = root / "data" / "raw"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = load_results_csv(processed_dir / "international_results.csv")
    fixtures = pd.read_csv(raw_dir / "world_cup_2026_fixtures.csv")
    configs = _load_model_configs(processed_dir / "world_cup_2026_selected_hyperparameters.csv")
    historical_odds = _load_optional_historical_odds(raw_dir / "odds" / "WorldCup2026.xlsx")
    current_market = _load_optional_current_market(raw_dir)

    result = evaluate_recent_world_cup_holdouts(
        results=results,
        fixtures=fixtures,
        market_probabilities=current_market,
        historical_odds=historical_odds,
        holdout_sizes=args.holdout_sizes,
        configs=configs,
    )

    predictions_path = processed_dir / "world_cup_2026_recent_holdout_predictions.csv"
    summary_path = reports_dir / "world_cup_2026_recent_holdout_summary.csv"
    report_path = reports_dir / "world_cup_2026_recent_holdout_report.md"
    result.predictions.to_csv(predictions_path, index=False)
    result.summary.to_csv(summary_path, index=False)
    write_recent_holdout_report(result, report_path)

    print(f"Recent holdout predictions -> {predictions_path}")
    print(f"Recent holdout summary -> {summary_path}")
    print(f"Recent holdout report -> {report_path}")
    if current_market is None:
        print("No current 2026 market probabilities were available.")
    if result.market_weight is not None:
        print(
            "Historical model-market weight -> "
            f"model={result.market_weight:.2f}, market={1.0 - result.market_weight:.2f}"
        )


def _load_optional_historical_odds(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return load_football_data_world_cup_odds(path)


def _load_optional_current_market(raw_dir: Path) -> pd.DataFrame | None:
    market_path = raw_dir / "world_cup_2026_market_probabilities.csv"
    if market_path.exists():
        return load_market_probabilities_csv(market_path)

    app_points_path = raw_dir / "world_cup_2026_app_points.csv"
    if not app_points_path.exists():
        return None
    app_points = pd.read_csv(app_points_path)
    required_app_columns = {"match_number", *APP_PROBABILITY_COLUMNS.values()}
    if not required_app_columns.issubset(app_points.columns):
        return None
    market = app_points.loc[:, list(required_app_columns)].rename(
        columns={
            "app_home_win_probability": "market_home_win",
            "app_draw_probability": "market_draw",
            "app_away_win_probability": "market_away_win",
        }
    )
    market["market_source"] = "app_points"
    return validate_market_probability_frame(market)


def _load_model_configs(path: Path) -> BaselineModelConfigs:
    frame = pd.read_csv(path)
    required = {"model", "parameter", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing selected hyperparameter columns: {sorted(missing)}")
    values = {
        (str(row.model), str(row.parameter)): row.value for row in frame.itertuples(index=False)
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
