"""Prediction snapshot history for post-match forecast evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from wc26_predictor.models.poisson import independent_poisson_score_matrix

LATEST_SNAPSHOT_FILE = "world_cup_2026_latest_prediction_snapshot.csv"
SNAPSHOT_HISTORY_FILE = "world_cup_2026_prediction_snapshots.csv"
PLAYED_CHECKS_FILE = "world_cup_2026_played_match_prediction_checks.csv"

SNAPSHOT_COLUMNS = [
    "snapshot_generated_at",
    "date",
    "time_local",
    "utc_offset",
    "group",
    "match_number",
    "home_team",
    "away_team",
    "predicted_home_score",
    "predicted_away_score",
    "predicted_score",
    "predicted_score_outcome",
    "predicted_score_probability",
    "top_scorelines",
    "home_expected_goals",
    "away_expected_goals",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
]


def write_latest_prediction_snapshot(
    match_details: pd.DataFrame,
    destination: str | Path,
    generated_at: datetime | None = None,
) -> Path:
    """Write the current unplayed-fixture predictions as the latest snapshot."""

    _require_columns(
        match_details, set(SNAPSHOT_COLUMNS).difference({"snapshot_generated_at"}), "match details"
    )
    output = match_details.copy()
    if "is_completed" in output.columns:
        output = output[~output["is_completed"].map(_parse_boolish)].copy()
    output.insert(0, "snapshot_generated_at", _timestamp(generated_at))
    output = output.loc[:, SNAPSHOT_COLUMNS].sort_values("match_number", ignore_index=True)

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return path


def archive_latest_prediction_snapshot(
    processed_dir: str | Path,
    history_path: str | Path | None = None,
) -> Path | None:
    """Append the last saved pre-update prediction snapshot to the history file."""

    processed_path = Path(processed_dir)
    latest_path = processed_path / LATEST_SNAPSHOT_FILE
    if not latest_path.exists():
        return None

    latest = pd.read_csv(latest_path)
    if latest.empty:
        return None
    _require_columns(latest, set(SNAPSHOT_COLUMNS), "latest prediction snapshot")

    destination = (
        Path(history_path) if history_path is not None else processed_path / SNAPSHOT_HISTORY_FILE
    )
    if destination.exists():
        history = pd.read_csv(destination)
        _require_columns(history, set(SNAPSHOT_COLUMNS), "prediction snapshot history")
        combined = pd.concat([history, latest], ignore_index=True)
    else:
        combined = latest.copy()
    combined = combined.drop_duplicates(
        ["snapshot_generated_at", "match_number"],
        keep="last",
    ).sort_values(["snapshot_generated_at", "match_number"], ignore_index=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    return destination


def build_played_match_prediction_checks(
    match_details: pd.DataFrame,
    prediction_history: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compare completed group matches with the last archived pre-match prediction."""

    required_details = {
        "date",
        "group",
        "match_number",
        "home_team",
        "away_team",
        "is_completed",
        "completed_home_score",
        "completed_away_score",
    }
    _require_columns(match_details, required_details, "match details")
    empty = _empty_played_checks()
    completed = match_details[match_details["is_completed"].map(_parse_boolish)].copy()
    if completed.empty:
        return empty
    if prediction_history is None or prediction_history.empty:
        return empty
    _require_columns(prediction_history, set(SNAPSHOT_COLUMNS), "prediction history")

    joined = completed.merge(
        prediction_history,
        on="match_number",
        how="inner",
        suffixes=("", "_prediction"),
        validate="one_to_many",
    )
    joined["match_date"] = pd.to_datetime(joined["date"], errors="raise").dt.date
    joined["snapshot_date"] = pd.to_datetime(
        joined["snapshot_generated_at"],
        errors="raise",
    ).dt.date
    joined = joined[joined["snapshot_date"] <= joined["match_date"]].copy()
    if joined.empty:
        return empty
    joined = (
        joined.sort_values(["match_number", "snapshot_generated_at"])
        .groupby("match_number", as_index=False)
        .tail(1)
    )

    rows = []
    for row in joined.itertuples(index=False):
        observed_home = int(row.completed_home_score)
        observed_away = int(row.completed_away_score)
        observed_outcome = _scoreline_outcome(observed_home, observed_away)
        predicted_outcome = str(row.predicted_score_outcome)
        outcome_probabilities = {
            "home_win": float(row.home_win_probability),
            "draw": float(row.draw_probability),
            "away_win": float(row.away_win_probability),
        }
        observed_outcome_probability = outcome_probabilities[observed_outcome]
        score_probability = _observed_score_probability(
            home_expected_goals=float(row.home_expected_goals),
            away_expected_goals=float(row.away_expected_goals),
            observed_home=observed_home,
            observed_away=observed_away,
        )
        rows.append(
            {
                "match_number": int(row.match_number),
                "group": row.group,
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "observed_score": f"{observed_home}-{observed_away}",
                "observed_outcome": observed_outcome,
                "snapshot_generated_at": row.snapshot_generated_at,
                "predicted_score": row.predicted_score,
                "predicted_score_probability": float(row.predicted_score_probability),
                "predicted_outcome": predicted_outcome,
                "observed_score_probability": score_probability,
                "observed_outcome_probability": observed_outcome_probability,
                "exact_score_hit": row.predicted_home_score == observed_home
                and row.predicted_away_score == observed_away,
                "outcome_hit": predicted_outcome == observed_outcome,
                "home_goal_error": int(row.predicted_home_score) - observed_home,
                "away_goal_error": int(row.predicted_away_score) - observed_away,
                "goal_abs_error": (
                    abs(int(row.predicted_home_score) - observed_home)
                    + abs(int(row.predicted_away_score) - observed_away)
                )
                / 2.0,
                "total_goal_abs_error": abs(
                    int(row.predicted_home_score)
                    + int(row.predicted_away_score)
                    - observed_home
                    - observed_away
                ),
                "exact_score_log_loss": float(-np.log(np.clip(score_probability, 1e-15, 1.0))),
                "outcome_log_loss": float(
                    -np.log(np.clip(observed_outcome_probability, 1e-15, 1.0))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "match_number"], ignore_index=True)


def load_prediction_history(path: str | Path) -> pd.DataFrame | None:
    """Load archived prediction history if it exists."""

    history_path = Path(path)
    if not history_path.exists():
        return None
    history = pd.read_csv(history_path)
    _require_columns(history, set(SNAPSHOT_COLUMNS), "prediction history")
    return history


def _observed_score_probability(
    home_expected_goals: float,
    away_expected_goals: float,
    observed_home: int,
    observed_away: int,
) -> float:
    matrix = independent_poisson_score_matrix(home_expected_goals, away_expected_goals)
    if observed_home >= matrix.shape[0] or observed_away >= matrix.shape[1]:
        return 0.0
    return float(matrix[observed_home, observed_away])


def _empty_played_checks() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "match_number",
            "group",
            "date",
            "home_team",
            "away_team",
            "observed_score",
            "observed_outcome",
            "snapshot_generated_at",
            "predicted_score",
            "predicted_score_probability",
            "predicted_outcome",
            "observed_score_probability",
            "observed_outcome_probability",
            "exact_score_hit",
            "outcome_hit",
            "home_goal_error",
            "away_goal_error",
            "goal_abs_error",
            "total_goal_abs_error",
            "exact_score_log_loss",
            "outcome_log_loss",
        ]
    )


def _scoreline_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"


def _timestamp(value: datetime | None) -> str:
    timestamp = value or datetime.now(tz=UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.isoformat(timespec="seconds")


def _parse_boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {label}: {sorted(missing)}")
