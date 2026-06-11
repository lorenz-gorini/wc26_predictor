"""Load and validate international goalscorer data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

GOALSCORER_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "team",
    "scorer",
    "minute",
    "own_goal",
    "penalty",
}


def load_goalscorers_csv(path_or_url: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path_or_url)
    return validate_goalscorers_frame(raw)


def load_known_goalscorers_csv(path_or_url: str | Path) -> pd.DataFrame:
    """Load goalscorer rows with known individual scorer names."""

    raw = pd.read_csv(path_or_url)
    known = raw.dropna(subset=["scorer"]).copy()
    return validate_goalscorers_frame(known)


def validate_goalscorers_frame(goalscorers: pd.DataFrame) -> pd.DataFrame:
    missing = GOALSCORER_COLUMNS.difference(goalscorers.columns)
    if missing:
        raise ValueError(f"Missing goalscorer columns: {sorted(missing)}")

    normalized = goalscorers.loc[:, sorted(GOALSCORER_COLUMNS)].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.date

    for column in ["home_team", "away_team", "team", "scorer"]:
        normalized[column] = normalized[column].astype("string").str.strip()
        if normalized[column].isna().any() or (normalized[column] == "").any():
            raise ValueError(f"Column {column!r} contains missing or empty values.")

    normalized["own_goal"] = _normalize_boolean_series(normalized["own_goal"])
    normalized["penalty"] = _normalize_boolean_series(normalized["penalty"])
    return normalized.sort_values(["date", "team", "scorer"]).reset_index(drop=True)


def _normalize_boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    normalized = values.astype("string").str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        bad_values = sorted(values[normalized.isna()].astype(str).unique())
        raise ValueError(f"Boolean column contains invalid values: {bad_values}")
    return normalized.astype(bool)
