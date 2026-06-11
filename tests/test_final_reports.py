from __future__ import annotations

import pandas as pd
import pytest

from wc26_predictor.reporting.final_reports import (
    FinalReportConfig,
    estimate_top_scorer_probabilities,
    generate_final_reports,
)


def test_estimate_top_scorer_probabilities_splits_symmetric_players() -> None:
    top_scorers = pd.DataFrame(
        {
            "team": ["A", "B"],
            "scorer": ["Player One", "Player Two"],
            "club": ["Club A", "Club B"],
            "expected_tournament_goals_transfermarkt_adjusted": [2.0, 2.0],
            "expected_tournament_goals": [1.8, 1.9],
            "transfermarkt_match_quality": ["exact", "exact"],
        }
    )

    probabilities = estimate_top_scorer_probabilities(
        top_scorers,
        n_simulations=20_000,
        seed=2026,
    )

    assert probabilities["top_scorer_probability"].sum() == pytest.approx(1.0)
    assert probabilities.loc[0, "top_scorer_probability"] == pytest.approx(0.5, abs=0.03)
    assert probabilities.loc[1, "top_scorer_probability"] == pytest.approx(0.5, abs=0.03)


def test_generate_final_reports_writes_expected_outputs(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()

    _write_processed_fixture(processed_dir)

    paths = generate_final_reports(
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        config=FinalReportConfig(top_scorer_simulations=5_000, random_seed=7),
    )

    assert paths.markdown_report.exists()
    winners = pd.read_csv(paths.winners)
    top_scorers = pd.read_csv(paths.top_scorers)
    group_advancement = pd.read_csv(paths.group_advancement)

    assert list(winners["team"]) == ["A", "B"]
    assert top_scorers.loc[0, "rank"] == 1
    assert set(group_advancement["advance_probability"]) == {0.9, 0.8}
    assert "Winner probabilities" in paths.markdown_report.read_text(encoding="utf-8")


def test_generate_final_reports_rejects_missing_group_membership(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    _write_processed_fixture(processed_dir)
    forecasts = pd.read_csv(processed_dir / "world_cup_2026_baseline_forecasts.csv")
    forecasts = forecasts[forecasts["home_team"] != "A"]
    forecasts.to_csv(processed_dir / "world_cup_2026_baseline_forecasts.csv", index=False)

    with pytest.raises(ValueError, match="Missing group labels"):
        generate_final_reports(processed_dir=processed_dir, reports_dir=reports_dir)


def _write_processed_fixture(processed_dir) -> None:
    pd.DataFrame(
        {
            "team": ["A", "B"],
            "round_of_32_probability": [0.9, 0.8],
            "round_of_16_probability": [0.7, 0.6],
            "quarter_final_probability": [0.5, 0.4],
            "semi_final_probability": [0.3, 0.25],
            "final_probability": [0.2, 0.15],
            "third_place_match_probability": [0.1, 0.08],
            "winner_probability": [0.12, 0.10],
            "expected_knockout_matches": [2.1, 1.8],
            "expected_team_matches": [5.1, 4.8],
        }
    ).to_csv(processed_dir / "world_cup_2026_official_tournament_probabilities.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2026-06-11"],
            "time_local": ["12:00"],
            "utc_offset": ["UTC-5"],
            "group": ["A"],
            "match_number": [1],
            "home_team": ["A"],
            "away_team": ["B"],
            "stadium": ["Example"],
            "city": ["City"],
            "ensemble_home_win": [0.5],
            "ensemble_draw": [0.25],
            "ensemble_away_win": [0.25],
            "form_poisson_home_expected_goals": [1.6],
            "form_poisson_away_expected_goals": [1.1],
        }
    ).to_csv(processed_dir / "world_cup_2026_baseline_forecasts.csv", index=False)
    pd.DataFrame(
        {
            "match_number": [73],
            "round": ["round_of_32"],
            "first_team": ["A"],
            "second_team": ["B"],
            "pairing_probability": [0.4],
            "first_advancement_probability": [0.65],
            "second_advancement_probability": [0.35],
            "simulation_count": [400],
        }
    ).to_csv(processed_dir / "world_cup_2026_knockout_match_forecasts.csv", index=False)
    pd.DataFrame(
        {
            "match_number": [1, 73],
            "round": ["group", "round_of_32"],
            "group": ["A", None],
            "first_team": ["A", "A"],
            "second_team": ["B", "B"],
            "pairing_probability": [1.0, 0.4],
            "first_win_probability": [0.5, None],
            "draw_probability": [0.25, None],
            "second_win_probability": [0.25, None],
            "first_advancement_probability": [None, 0.65],
            "second_advancement_probability": [None, 0.35],
            "simulation_count": [None, 400],
        }
    ).to_csv(processed_dir / "world_cup_2026_full_match_forecasts.csv", index=False)
    pd.DataFrame(
        {
            "team": ["A", "B"],
            "scorer": ["Player One", "Player Two"],
            "club": ["Club A", "Club B"],
            "expected_tournament_goals_transfermarkt_adjusted": [2.2, 1.7],
            "expected_tournament_goals": [2.0, 1.5],
            "transfermarkt_match_quality": ["exact", "exact"],
        }
    ).to_csv(
        processed_dir / "world_cup_2026_top_scorer_transfermarkt_adjusted_top100.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "use_model_market": [False],
            "reason": ["model_market did not improve validation log loss"],
            "model_log_loss": [0.9],
            "market_log_loss": [0.85],
            "model_market_log_loss": [0.87],
        }
    ).to_csv(processed_dir / "world_cup_market_gate_decision.csv", index=False)
    pd.DataFrame(
        {
            "team": ["A", "B"],
            "team_availability_burden": [0.0, 0.1],
            "unavailable_players": [0, 1],
        }
    ).to_csv(processed_dir / "world_cup_2026_team_availability_burden.csv", index=False)
