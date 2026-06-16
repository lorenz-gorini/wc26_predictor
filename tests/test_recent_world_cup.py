from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wc26_predictor.evaluation.recent_world_cup import (
    evaluate_recent_world_cup_holdouts,
    validate_market_probability_frame,
)


def test_recent_holdout_uses_latest_completed_matches_and_market_probabilities() -> None:
    result = evaluate_recent_world_cup_holdouts(
        results=_results(),
        fixtures=_fixtures(),
        market_probabilities=pd.DataFrame(
            [
                {
                    "match_number": 3,
                    "market_home_win": 0.30,
                    "market_draw": 0.25,
                    "market_away_win": 0.45,
                },
                {
                    "match_number": 4,
                    "market_home_win": 0.35,
                    "market_draw": 0.40,
                    "market_away_win": 0.25,
                },
            ]
        ),
        holdout_sizes=[2],
    )

    assert result.predictions["match_number"].tolist() == [3, 4]
    assert set(result.summary["model"]) == {
        "elo",
        "poisson",
        "form_adjusted_poisson",
        "ensemble",
        "market",
    }
    market = result.summary[result.summary["model"] == "market"].iloc[0]
    assert market["n_matches"] == 2
    assert np.isfinite(market["log_loss"])


def test_recent_holdout_omits_market_metrics_without_current_probabilities() -> None:
    result = evaluate_recent_world_cup_holdouts(
        results=_results(),
        fixtures=_fixtures(),
        holdout_sizes=[2],
    )

    assert "market" not in set(result.summary["model"])


def test_validate_market_probability_frame_requires_probabilities_to_sum_to_one() -> None:
    probabilities = pd.DataFrame(
        [
            {
                "match_number": 1,
                "market_home_win": 0.60,
                "market_draw": 0.30,
                "market_away_win": 0.30,
            }
        ]
    )

    with pytest.raises(ValueError, match="Market probabilities must sum to one"):
        validate_market_probability_frame(probabilities)


def _results() -> pd.DataFrame:
    rows = [
        ("2025-01-01", "A", "B", 2, 0, "Friendly"),
        ("2025-01-02", "C", "D", 1, 0, "Friendly"),
        ("2025-02-01", "A", "C", 1, 1, "Friendly"),
        ("2025-02-02", "B", "D", 0, 1, "Friendly"),
        ("2025-03-01", "A", "D", 2, 1, "Friendly"),
        ("2025-03-02", "B", "C", 1, 2, "Friendly"),
        ("2026-06-11", "A", "B", 1, 0, "FIFA World Cup"),
        ("2026-06-12", "C", "D", 0, 0, "FIFA World Cup"),
        ("2026-06-13", "A", "C", 0, 1, "FIFA World Cup"),
        ("2026-06-14", "B", "D", 1, 1, "FIFA World Cup"),
    ]
    return pd.DataFrame(
        [
            {
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": tournament,
                "city": "Example City",
                "country": "Example Country",
                "neutral": True,
            }
            for date, home_team, away_team, home_score, away_score, tournament in rows
        ]
    )


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-06-11",
                "match_number": 1,
                "home_team": "A",
                "away_team": "B",
            },
            {
                "date": "2026-06-12",
                "match_number": 2,
                "home_team": "C",
                "away_team": "D",
            },
            {
                "date": "2026-06-13",
                "match_number": 3,
                "home_team": "A",
                "away_team": "C",
            },
            {
                "date": "2026-06-14",
                "match_number": 4,
                "home_team": "B",
                "away_team": "D",
            },
        ]
    )
