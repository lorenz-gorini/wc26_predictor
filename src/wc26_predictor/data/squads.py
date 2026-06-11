"""Load and validate World Cup squad data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SQUAD_COLUMNS = {"team", "number", "position", "player", "caps", "goals", "club"}


def load_squads_csv(path_or_url: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path_or_url)
    return validate_squads_frame(raw)


def validate_squads_frame(squads: pd.DataFrame) -> pd.DataFrame:
    missing = SQUAD_COLUMNS.difference(squads.columns)
    if missing:
        raise ValueError(f"Missing squad columns: {sorted(missing)}")

    normalized = squads.loc[:, sorted(SQUAD_COLUMNS)].copy()
    for column in ["team", "position", "player", "club"]:
        normalized[column] = normalized[column].astype("string").str.strip()
        if normalized[column].isna().any() or (normalized[column] == "").any():
            raise ValueError(f"Column {column!r} contains missing or empty values.")

    for column in ["number", "caps", "goals"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(int)
        if (normalized[column] < 0).any():
            raise ValueError(f"Column {column!r} contains negative values.")

    return normalized.sort_values(["team", "number"]).reset_index(drop=True)

