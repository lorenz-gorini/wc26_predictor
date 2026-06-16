from __future__ import annotations

import pandas as pd
import pytest

from wc26_predictor.evaluation.contest_points import (
    build_contest_pick_recommendations,
    validate_contest_points_frame,
)


def test_contest_picks_maximize_expected_points_not_modal_probability() -> None:
    forecasts = pd.DataFrame(
        [
            {
                "match_number": 1,
                "date": "2026-06-11",
                "home_team": "France",
                "away_team": "Senegal",
                "home_win_probability": 0.62,
                "draw_probability": 0.33,
                "away_win_probability": 0.05,
                "is_completed": False,
            }
        ]
    )
    points = pd.DataFrame(
        [
            {
                "match_number": 1,
                "home_win_points": 46,
                "draw_points": 128,
                "away_win_points": 153,
                "app_home_win_probability": 0.88,
                "app_draw_probability": 0.09,
                "app_away_win_probability": 0.03,
            }
        ]
    )

    picks = build_contest_pick_recommendations(forecasts, points)

    assert picks.loc[0, "modal_model_outcome"] == "home_win"
    assert picks.loc[0, "recommended_outcome"] == "draw"
    assert bool(picks.loc[0, "recommendation_differs_from_modal"])
    assert picks.loc[0, "recommended_expected_points"] == pytest.approx(42.24)
    assert picks.loc[0, "expected_points_gain_vs_modal"] == pytest.approx(13.72)


def test_validate_contest_points_rejects_duplicate_match_numbers() -> None:
    points = pd.DataFrame(
        [
            {
                "match_number": 1,
                "home_win_points": 46,
                "draw_points": 128,
                "away_win_points": 153,
            },
            {
                "match_number": 1,
                "home_win_points": 60,
                "draw_points": 100,
                "away_win_points": 140,
            },
        ]
    )

    with pytest.raises(ValueError, match="Duplicate contest point match numbers"):
        validate_contest_points_frame(points)
