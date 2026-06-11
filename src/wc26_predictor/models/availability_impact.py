"""Estimate match impact from historical team availability burden."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wc26_predictor.data.schema import validate_results_frame


HISTORICAL_AVAILABILITY_COLUMNS = {"date", "team", "team_availability_burden"}


def validate_historical_team_availability(history: pd.DataFrame) -> pd.DataFrame:
    """Validate historical team-level availability burden rows."""

    missing = HISTORICAL_AVAILABILITY_COLUMNS.difference(history.columns)
    if missing:
        raise ValueError(f"Missing historical availability columns: {sorted(missing)}")
    normalized = history.loc[:, sorted(HISTORICAL_AVAILABILITY_COLUMNS)].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.date
    normalized["team"] = normalized["team"].astype("string").str.strip()
    if normalized["team"].isna().any() or (normalized["team"] == "").any():
        raise ValueError("Column 'team' contains missing or empty values.")
    normalized["team_availability_burden"] = pd.to_numeric(
        normalized["team_availability_burden"],
        errors="raise",
    )
    if (normalized["team_availability_burden"] < 0).any():
        raise ValueError("team_availability_burden cannot be negative.")
    return normalized.sort_values(["date", "team"]).reset_index(drop=True)


def estimate_goal_penalty_per_burden(
    results: pd.DataFrame,
    historical_availability: pd.DataFrame,
) -> float:
    """Estimate expected-goal penalty per unit of availability burden.

    This is intentionally simple and conservative. It estimates the slope from a
    team-match panel regression of goals scored on team availability burden. The
    returned value is clipped to [0, 0.20] and represents a multiplicative
    expected-goal penalty per burden unit.
    """

    validated_results = validate_results_frame(results)
    availability = validate_historical_team_availability(historical_availability)
    team_rows = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": validated_results["date"],
                    "team": validated_results["home_team"],
                    "goals_for": validated_results["home_score"],
                }
            ),
            pd.DataFrame(
                {
                    "date": validated_results["date"],
                    "team": validated_results["away_team"],
                    "goals_for": validated_results["away_score"],
                }
            ),
        ],
        ignore_index=True,
    )
    matched = team_rows.merge(availability, on=["date", "team"], how="inner")
    if len(matched) < 30:
        raise ValueError("At least 30 matched team-match availability rows are required.")

    x = matched["team_availability_burden"].to_numpy(dtype=float)
    y = matched["goals_for"].to_numpy(dtype=float)
    if np.isclose(x.std(), 0.0):
        raise ValueError("Historical availability burden has no variation.")
    slope = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    baseline_goals = max(float(y.mean()), 0.1)
    penalty = max(0.0, -slope / baseline_goals)
    return float(np.clip(penalty, 0.0, 0.20))
