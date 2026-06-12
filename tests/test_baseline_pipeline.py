from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from wc26_predictor.data.ingest_results import load_results_csv
from wc26_predictor.models.ensemble import EnsembleWeights
from wc26_predictor.pipelines import baselines
from wc26_predictor.pipelines.baselines import (
    EvaluationWindow,
    attach_completed_fixture_results,
    build_full_match_forecast_table,
    default_model_configs,
    forecast_2026_group_fixtures,
    model_configs_to_frame,
    tune_baseline_model_configs,
)
from wc26_predictor.reporting.upcoming_forecasts import (
    UpcomingForecastConfig,
    build_final_stage_match_impacts,
    build_match_driver_table,
    build_upcoming_match_details,
)


def _sample_results() -> pd.DataFrame:
    path = Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_results.csv"
    return load_results_csv(path)


def test_forecast_2026_group_fixtures_returns_probability_columns() -> None:
    results = _sample_results()
    fixtures = pd.DataFrame(
        {
            "date": ["2026-06-11"],
            "time_local": ["1:00 p.m."],
            "utc_offset": ["UTC-6"],
            "group": ["A"],
            "match_number": [1],
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
            "stadium": ["Example Stadium"],
            "city": ["Example City"],
        }
    )

    forecasts = forecast_2026_group_fixtures(results, fixtures)

    assert len(forecasts) == 1
    assert forecasts.loc[0, "poisson_home_expected_goals"] > 0
    assert forecasts.loc[0, "form_poisson_home_expected_goals"] > 0
    assert forecasts.loc[0, "predicted_score_probability"] > 0
    assert forecasts.loc[0, "predicted_score"] == (
        f"{forecasts.loc[0, 'predicted_home_score']}-"
        f"{forecasts.loc[0, 'predicted_away_score']}"
    )
    assert forecasts.loc[0, "predicted_score_outcome"] in {"home_win", "draw", "away_win"}
    assert ";" in forecasts.loc[0, "top_scorelines"]
    assert forecasts.loc[0, "ensemble_home_win"] == pytest.approx(
        (
            forecasts.loc[0, "elo_home_win"]
            + forecasts.loc[0, "poisson_home_win"]
            + forecasts.loc[0, "form_poisson_home_win"]
        )
        / 3
    )


def test_forecast_2026_group_fixtures_accepts_calibrated_weights() -> None:
    results = _sample_results()
    fixtures = pd.DataFrame(
        {
            "date": ["2026-06-11"],
            "time_local": ["1:00 p.m."],
            "utc_offset": ["UTC-6"],
            "group": ["A"],
            "match_number": [1],
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
            "stadium": ["Example Stadium"],
            "city": ["Example City"],
        }
    )

    forecasts = forecast_2026_group_fixtures(
        results,
        fixtures,
        ensemble_weights=EnsembleWeights(elo=1.0, poisson=0.0, form_adjusted_poisson=0.0),
    )

    assert forecasts.loc[0, "ensemble_home_win"] == pytest.approx(
        forecasts.loc[0, "elo_home_win"]
    )
    assert forecasts.loc[0, "ensemble_equal_weight_home_win"] != pytest.approx(
        forecasts.loc[0, "ensemble_home_win"]
    )


def test_build_full_match_forecast_table_combines_group_and_knockout_rows() -> None:
    group_forecasts = pd.DataFrame(
        {
            "match_number": [1],
            "group": ["A"],
            "home_team": ["A1"],
            "away_team": ["A2"],
            "predicted_score": ["1-0"],
            "predicted_score_outcome": ["home_win"],
            "predicted_score_probability": [0.14],
            "top_scorelines": ["1-0 (0.140); 1-1 (0.120)"],
            "ensemble_home_win": [0.5],
            "ensemble_draw": [0.25],
            "ensemble_away_win": [0.25],
        }
    )
    knockout_forecasts = pd.DataFrame(
        {
            "match_number": [73],
            "round": ["round_of_32"],
            "first_team": ["A1"],
            "second_team": ["B2"],
            "pairing_probability": [0.4],
            "first_advancement_probability": [0.7],
            "second_advancement_probability": [0.3],
            "simulation_count": [40],
        }
    )

    full = build_full_match_forecast_table(group_forecasts, knockout_forecasts)

    assert len(full) == 2
    assert full.loc[0, "round"] == "group"
    assert full.loc[0, "pairing_probability"] == 1.0
    assert full.loc[0, "predicted_score"] == "1-0"
    assert full.loc[0, "predicted_score_outcome"] == "home_win"
    assert full.loc[1, "round"] == "round_of_32"
    assert full.loc[1, "first_advancement_probability"] == 0.7


def test_attach_completed_fixture_results_marks_played_fixture() -> None:
    forecasts = pd.DataFrame(
        {
            "date": ["2026-06-11", "2026-06-12"],
            "home_team": ["Argentina", "Brazil"],
            "away_team": ["Germany", "Japan"],
        }
    )
    results = pd.DataFrame(
        {
            "date": ["2026-06-11"],
            "home_team": ["Argentina"],
            "away_team": ["Germany"],
            "home_score": [2],
            "away_score": [1],
            "tournament": ["FIFA World Cup"],
            "city": ["Example City"],
            "country": ["Example Country"],
            "neutral": [True],
        }
    )

    attached = attach_completed_fixture_results(forecasts, results)

    assert attached.loc[0, "is_completed"]
    assert attached.loc[0, "completed_home_score"] == 2
    assert attached.loc[0, "completed_away_score"] == 1
    assert not attached.loc[1, "is_completed"]


def test_upcoming_match_details_and_drivers_are_coherent() -> None:
    results = _sample_results()
    fixtures = pd.DataFrame(
        {
            "date": ["2026-06-11"],
            "time_local": ["1:00 p.m."],
            "utc_offset": ["UTC-6"],
            "group": ["A"],
            "match_number": [1],
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
            "stadium": ["Example Stadium"],
            "city": ["Example City"],
        }
    )
    forecasts = forecast_2026_group_fixtures(results, fixtures)

    details = build_upcoming_match_details(forecasts)
    drivers = build_match_driver_table(results, details)

    assert details.loc[0, "predicted_score"] == (
        f"{details.loc[0, 'predicted_home_score']}-"
        f"{details.loc[0, 'predicted_away_score']}"
    )
    total_outcome_probability = (
        details.loc[0, "home_win_probability"]
        + details.loc[0, "draw_probability"]
        + details.loc[0, "away_win_probability"]
    )
    assert total_outcome_probability == pytest.approx(1.0)
    assert set(drivers["driver"]).issuperset(
        {
            "team_attack_strength",
            "opponent_defensive_vulnerability",
            "recent_attack_form",
            "availability_adjustment",
        }
    )
    assert drivers["match_number"].eq(1).all()


def test_final_stage_match_impacts_returns_one_row_per_group_match() -> None:
    results = _sample_results()
    forecasts = _synthetic_tournament_forecasts()
    teams = sorted(set(forecasts["home_team"]).union(forecasts["away_team"]))
    advancement_probabilities = {
        (first, second): 0.5
        for first in teams
        for second in teams
        if first != second
    }

    impacts = build_final_stage_match_impacts(
        results,
        forecasts,
        advancement_probabilities=advancement_probabilities,
        config=UpcomingForecastConfig(impact_simulations=10, random_seed=3),
    )

    assert len(impacts) == len(forecasts)
    assert impacts["total_final_stage_impact"].ge(0.0).all()
    assert {"largest_winner_delta_team", "largest_winner_probability_delta"}.issubset(
        impacts.columns
    )


def test_model_configs_to_frame_contains_expected_parameters() -> None:
    frame = model_configs_to_frame(default_model_configs())

    assert {"model", "parameter", "value"}.issubset(frame.columns)
    assert set(frame["model"]) == {"elo", "poisson", "form_adjusted_poisson"}
    assert "k_factor" in set(frame["parameter"])
    assert "form_strength" in set(frame["parameter"])


def test_tune_baseline_model_configs_returns_ranked_results(monkeypatch) -> None:
    results = _sample_results()
    window = EvaluationWindow(
        name="sample",
        train_until=date(2022, 11, 24),
        test_start=date(2022, 11, 25),
        test_end=date(2022, 12, 31),
    )
    monkeypatch.setattr(baselines, "WORLD_CUP_WINDOWS", [window])

    configs, tuning = tune_baseline_model_configs(results)

    assert configs.elo.k_factor > 0
    assert configs.poisson.prior_strength > 0
    assert configs.form_adjusted_poisson.form_window_matches > 0
    assert set(tuning["model"]) == {"elo", "poisson", "form_adjusted_poisson"}
    assert tuning.groupby("model")["rank"].min().eq(1).all()


def test_evaluate_window_rejects_empty_holdout() -> None:
    results = _sample_results()
    window = EvaluationWindow(
        name="empty",
        train_until=date(2022, 12, 31),
        test_start=date(2023, 1, 1),
        test_end=date(2023, 12, 31),
    )

    with pytest.raises(ValueError, match="Test split is empty"):
        baselines._evaluate_window(results, window)


def _synthetic_tournament_forecasts() -> pd.DataFrame:
    rows = []
    match_number = 1
    for group in "ABCDEFGHIJKL":
        teams = [f"{group}{index}" for index in range(1, 5)]
        for home_index in range(4):
            for away_index in range(home_index + 1, 4):
                rows.append(
                    {
                        "group": group,
                        "match_number": match_number,
                        "home_team": teams[home_index],
                        "away_team": teams[away_index],
                        "form_poisson_home_expected_goals": 1.25,
                        "form_poisson_away_expected_goals": 1.05,
                    }
                )
                match_number += 1
    return pd.DataFrame(rows)
