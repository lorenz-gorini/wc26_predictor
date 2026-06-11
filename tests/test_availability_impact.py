from __future__ import annotations

import pandas as pd

from wc26_predictor.models.availability_impact import estimate_goal_penalty_per_burden


def test_estimate_goal_penalty_per_burden_returns_positive_penalty() -> None:
    results_rows = []
    availability_rows = []
    for index in range(40):
        date = f"2024-01-{index % 28 + 1:02d}"
        burden_a = 1.0 if index % 2 else 0.0
        burden_b = 0.0
        results_rows.append(
            {
                "date": date,
                "home_team": "A",
                "away_team": "B",
                "home_score": 0 if burden_a else 2,
                "away_score": 1,
                "tournament": "Friendly",
                "city": "X",
                "country": "Y",
                "neutral": True,
            }
        )
        availability_rows.extend(
            [
                {"date": date, "team": "A", "team_availability_burden": burden_a},
                {"date": date, "team": "B", "team_availability_burden": burden_b},
            ]
        )

    penalty = estimate_goal_penalty_per_burden(
        pd.DataFrame(results_rows),
        pd.DataFrame(availability_rows),
    )

    assert penalty > 0
