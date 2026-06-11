"""Typed records and dataframe validation for international football data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import ClassVar

import pandas as pd


@dataclass(frozen=True, slots=True)
class MatchResult:
    """A completed international match."""

    date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: str
    city: str
    country: str
    neutral: bool

    required_columns: ClassVar[set[str]] = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    }


@dataclass(frozen=True, slots=True)
class Fixture:
    """A future match to be forecast."""

    home_team: str
    away_team: str
    neutral: bool = True
    tournament: str = "FIFA World Cup"
    city: str | None = None
    country: str | None = None


@dataclass(frozen=True, slots=True)
class OutcomeProbabilities:
    """Home/draw/away probabilities for a match."""

    home_win: float
    draw: float
    away_win: float

    def as_dict(self) -> dict[str, float]:
        return {
            "home_win": self.home_win,
            "draw": self.draw,
            "away_win": self.away_win,
        }


def validate_results_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a results dataframe.

    The function intentionally raises on invalid input. Silent coercion is limited to
    standardizing dates, booleans, and string trimming.
    """

    missing = MatchResult.required_columns.difference(results.columns)
    if missing:
        raise ValueError(f"Missing required result columns: {sorted(missing)}")

    normalized = results.loc[:, sorted(MatchResult.required_columns)].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.date

    string_columns = ["home_team", "away_team", "tournament", "city", "country"]
    for column in string_columns:
        normalized[column] = normalized[column].astype("string").str.strip()
        if normalized[column].isna().any() or (normalized[column] == "").any():
            raise ValueError(f"Column {column!r} contains missing or empty values.")

    for column in ["home_score", "away_score"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        if (normalized[column] < 0).any():
            raise ValueError(f"Column {column!r} contains negative scores.")
        if (normalized[column] % 1 != 0).any():
            raise ValueError(f"Column {column!r} contains non-integer scores.")
        normalized[column] = normalized[column].astype(int)

    normalized["neutral"] = _normalize_boolean_series(normalized["neutral"])

    same_team = normalized["home_team"] == normalized["away_team"]
    if same_team.any():
        raise ValueError("At least one match has identical home and away teams.")

    return normalized.sort_values("date").reset_index(drop=True)


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
        raise ValueError(f"Column 'neutral' contains invalid booleans: {bad_values}")
    return normalized.astype(bool)

