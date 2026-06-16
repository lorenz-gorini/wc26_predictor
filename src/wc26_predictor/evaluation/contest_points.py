"""Contest-point utilities for 1X2 prediction games."""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ("home_win", "draw", "away_win")
POINT_COLUMNS = {
    "home_win": "home_win_points",
    "draw": "draw_points",
    "away_win": "away_win_points",
}
PROBABILITY_COLUMNS = {
    "home_win": "home_win_probability",
    "draw": "draw_probability",
    "away_win": "away_win_probability",
}
APP_PROBABILITY_COLUMNS = {
    "home_win": "app_home_win_probability",
    "draw": "app_draw_probability",
    "away_win": "app_away_win_probability",
}


def validate_contest_points_frame(points: pd.DataFrame) -> pd.DataFrame:
    """Validate app point offers for World Cup 1X2 contest picks.

    Required columns are ``match_number``, ``home_win_points``, ``draw_points``,
    and ``away_win_points``. Optional app-implied probability columns are kept
    when present, but they are diagnostic only; expected points are computed from
    the model probabilities and the app point offers.
    """

    required = {"match_number", *POINT_COLUMNS.values()}
    missing = required.difference(points.columns)
    if missing:
        raise ValueError(f"Missing contest point columns: {sorted(missing)}")

    optional_probabilities = [
        column for column in APP_PROBABILITY_COLUMNS.values() if column in points.columns
    ]
    columns = ["match_number", *POINT_COLUMNS.values(), *optional_probabilities]
    normalized = points.loc[:, columns].copy()
    normalized["match_number"] = pd.to_numeric(
        normalized["match_number"],
        errors="raise",
    )
    if (normalized["match_number"] % 1 != 0).any():
        raise ValueError("Contest point match numbers must be integers.")
    normalized["match_number"] = normalized["match_number"].astype(int)
    if normalized["match_number"].duplicated().any():
        duplicated = sorted(
            normalized.loc[normalized["match_number"].duplicated(), "match_number"].unique()
        )
        raise ValueError(f"Duplicate contest point match numbers: {duplicated}")

    for column in POINT_COLUMNS.values():
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        if (normalized[column] <= 0).any():
            raise ValueError(f"Column {column!r} must contain positive point values.")

    for column in optional_probabilities:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        if ((normalized[column] < 0.0) | (normalized[column] > 1.0)).any():
            raise ValueError(f"Column {column!r} must contain probabilities in [0, 1].")

    if len(optional_probabilities) == len(APP_PROBABILITY_COLUMNS):
        probability_sum = normalized[optional_probabilities].sum(axis=1)
        if not np.allclose(probability_sum, 1.0, atol=1e-3):
            raise ValueError("App-implied outcome probabilities must sum to one.")

    return normalized.sort_values("match_number", ignore_index=True)


def build_contest_pick_recommendations(
    forecasts: pd.DataFrame,
    contest_points: pd.DataFrame,
) -> pd.DataFrame:
    """Choose 1X2 picks by maximizing model-implied expected contest points."""

    required_forecasts = {
        "match_number",
        "date",
        "home_team",
        "away_team",
        *PROBABILITY_COLUMNS.values(),
    }
    missing_forecasts = required_forecasts.difference(forecasts.columns)
    if missing_forecasts:
        raise ValueError(f"Missing forecast columns: {sorted(missing_forecasts)}")

    points = validate_contest_points_frame(contest_points)
    frame = forecasts.loc[
        :,
        [
            "match_number",
            "date",
            "home_team",
            "away_team",
            *PROBABILITY_COLUMNS.values(),
            *[column for column in ["is_completed"] if column in forecasts.columns],
        ],
    ].copy()
    frame["match_number"] = pd.to_numeric(frame["match_number"], errors="raise").astype(int)
    joined = frame.merge(points, on="match_number", how="inner", validate="one_to_one")
    if joined.empty:
        return _empty_recommendations(points)

    for outcome in OUTCOMES:
        joined[f"{outcome}_expected_points"] = (
            joined[PROBABILITY_COLUMNS[outcome]] * joined[POINT_COLUMNS[outcome]]
        )

    expected_columns = [f"{outcome}_expected_points" for outcome in OUTCOMES]
    expected_values = joined[expected_columns].to_numpy(dtype=float)
    best_indices = expected_values.argmax(axis=1)
    modal_probability_values = joined[
        [PROBABILITY_COLUMNS[outcome] for outcome in OUTCOMES]
    ].to_numpy(dtype=float)
    modal_indices = modal_probability_values.argmax(axis=1)

    joined["recommended_outcome"] = [OUTCOMES[index] for index in best_indices]
    joined["modal_model_outcome"] = [OUTCOMES[index] for index in modal_indices]
    joined["recommendation_differs_from_modal"] = (
        joined["recommended_outcome"] != joined["modal_model_outcome"]
    )
    joined["recommended_expected_points"] = expected_values[
        np.arange(len(joined)),
        best_indices,
    ]
    joined["modal_expected_points"] = expected_values[np.arange(len(joined)), modal_indices]
    joined["expected_points_gain_vs_modal"] = (
        joined["recommended_expected_points"] - joined["modal_expected_points"]
    )
    joined["recommended_model_probability"] = _select_outcome_values(
        joined,
        PROBABILITY_COLUMNS,
        joined["recommended_outcome"],
    )
    joined["recommended_app_points"] = _select_outcome_values(
        joined,
        POINT_COLUMNS,
        joined["recommended_outcome"],
    )

    ordered_columns = [
        "match_number",
        "date",
        "home_team",
        "away_team",
        "recommended_outcome",
        "recommended_model_probability",
        "recommended_app_points",
        "recommended_expected_points",
        "modal_model_outcome",
        "modal_expected_points",
        "expected_points_gain_vs_modal",
        "recommendation_differs_from_modal",
        *PROBABILITY_COLUMNS.values(),
        *POINT_COLUMNS.values(),
        *[column for column in APP_PROBABILITY_COLUMNS.values() if column in joined.columns],
        *expected_columns,
        *[column for column in ["is_completed"] if column in joined.columns],
    ]
    return joined.loc[:, ordered_columns].sort_values("match_number", ignore_index=True)


def load_contest_points_csv(path: str) -> pd.DataFrame:
    """Load and validate app point offers from CSV."""

    return validate_contest_points_frame(pd.read_csv(path))


def _select_outcome_values(
    frame: pd.DataFrame,
    column_map: dict[str, str],
    outcomes: pd.Series,
) -> pd.Series:
    values = np.empty(len(frame), dtype=float)
    for index, outcome in enumerate(outcomes):
        values[index] = float(frame.iloc[index][column_map[str(outcome)]])
    return pd.Series(values, index=frame.index)


def _empty_recommendations(points: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "match_number",
        "date",
        "home_team",
        "away_team",
        "recommended_outcome",
        "recommended_model_probability",
        "recommended_app_points",
        "recommended_expected_points",
        "modal_model_outcome",
        "modal_expected_points",
        "expected_points_gain_vs_modal",
        "recommendation_differs_from_modal",
        *PROBABILITY_COLUMNS.values(),
        *POINT_COLUMNS.values(),
        *[column for column in APP_PROBABILITY_COLUMNS.values() if column in points.columns],
        *[f"{outcome}_expected_points" for outcome in OUTCOMES],
        "is_completed",
    ]
    return pd.DataFrame(columns=columns)
