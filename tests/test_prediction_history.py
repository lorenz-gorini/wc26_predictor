from __future__ import annotations

import math
from datetime import UTC, datetime

import pandas as pd
import pytest

from wc26_predictor.reporting.prediction_history import (
    archive_latest_prediction_snapshot,
    build_played_match_prediction_checks,
    write_latest_prediction_snapshot,
)


def test_snapshot_archive_and_played_match_checks(tmp_path) -> None:
    latest_path = tmp_path / "world_cup_2026_latest_prediction_snapshot.csv"
    history_path = tmp_path / "world_cup_2026_prediction_snapshots.csv"
    details = _match_details(is_completed=False)

    write_latest_prediction_snapshot(
        details,
        latest_path,
        generated_at=datetime(2026, 6, 10, 12, tzinfo=UTC),
    )
    archived = archive_latest_prediction_snapshot(tmp_path, history_path=history_path)

    assert archived == history_path
    history = pd.read_csv(history_path)
    assert len(history) == 1
    assert history.loc[0, "predicted_score"] == "1-0"

    completed = _match_details(is_completed=True)
    checks = build_played_match_prediction_checks(completed, history)

    assert len(checks) == 1
    assert checks.loc[0, "observed_score"] == "2-1"
    assert checks.loc[0, "predicted_score"] == "1-0"
    assert not bool(checks.loc[0, "exact_score_hit"])
    assert bool(checks.loc[0, "outcome_hit"])
    assert checks.loc[0, "observed_score_probability"] > 0
    assert checks.loc[0, "outcome_log_loss"] == pytest.approx(-math.log(0.55))
    assert checks.loc[0, "most_likely_outcome"] == "home_win"
    assert checks.loc[0, "observed_outcome_rank"] == 1
    assert checks.loc[0, "predicted_outcome_probability"] == pytest.approx(0.55)
    assert checks.loc[0, "outcome_probability_gap"] == pytest.approx(0.0)
    assert checks.loc[0, "result_diagnostic"] == "right_outcome_wrong_score"


def test_played_match_checks_empty_without_archived_prediction() -> None:
    checks = build_played_match_prediction_checks(
        _match_details(is_completed=True),
        prediction_history=None,
    )

    assert checks.empty


def test_played_match_checks_ignore_snapshots_after_fixture_date(tmp_path) -> None:
    details = _match_details(is_completed=False)
    snapshot = write_latest_prediction_snapshot(
        details,
        tmp_path / "latest_prediction_snapshot_after_fixture.csv",
        generated_at=datetime(2026, 6, 12, 12, tzinfo=UTC),
    )
    history = pd.read_csv(snapshot)

    checks = build_played_match_prediction_checks(
        _match_details(is_completed=True),
        history,
    )

    assert checks.empty


def _match_details(is_completed: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-06-11",
                "time_local": "1:00 p.m.",
                "utc_offset": "UTC-6",
                "group": "A",
                "match_number": 1,
                "home_team": "A1",
                "away_team": "A2",
                "stadium": "Example Stadium",
                "city": "Example City",
                "is_completed": is_completed,
                "completed_home_score": 2 if is_completed else pd.NA,
                "completed_away_score": 1 if is_completed else pd.NA,
                "home_expected_goals": 1.4,
                "away_expected_goals": 0.8,
                "predicted_home_score": 1,
                "predicted_away_score": 0,
                "predicted_score": "1-0",
                "predicted_score_outcome": "home_win",
                "predicted_score_probability": 0.16,
                "top_scorelines": "1-0 (0.160); 1-1 (0.120)",
                "home_win_probability": 0.55,
                "draw_probability": 0.25,
                "away_win_probability": 0.20,
                "ensemble_home_win": 0.56,
                "ensemble_draw": 0.24,
                "ensemble_away_win": 0.20,
                "home_availability_burden": 0.0,
                "away_availability_burden": 0.0,
                "home_availability_goal_multiplier": 1.0,
                "away_availability_goal_multiplier": 1.0,
            }
        ]
    )
