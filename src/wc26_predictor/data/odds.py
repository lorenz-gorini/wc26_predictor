"""Bookmaker odds ingestion and margin removal."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc26_predictor.data.club_form import normalize_name


FOOTBALL_DATA_ODDS_COLUMNS = {
    "H-Avg": "home_odds",
    "D-Avg": "draw_odds",
    "A-Avg": "away_odds",
}

TEAM_ALIASES = {
    "bosnia herz egovina": "bosnia and herzegovina",
    "bosnia herzegovina": "bosnia and herzegovina",
    "czech republic": "czech republic",
    "czechia": "czech republic",
    "d r congo": "dr congo",
    "iran": "iran",
    "ir iran": "iran",
    "ivory coast": "ivory coast",
    "cote d ivoire": "ivory coast",
    "south korea": "south korea",
    "korea republic": "south korea",
    "usa": "united states",
    "united states": "united states",
}


def load_football_data_world_cup_odds(path: str | Path) -> pd.DataFrame:
    """Load World Cup 1X2 odds from a Football-Data workbook."""

    source_path = Path(path)
    workbook = pd.ExcelFile(source_path)
    rows = []
    for sheet_name in workbook.sheet_names:
        if not sheet_name.startswith("WorldCup20") or "Qualifiers" in sheet_name:
            continue
        sheet = pd.read_excel(source_path, sheet_name=sheet_name)
        if sheet.empty:
            continue
        rows.append(_parse_football_data_sheet(sheet, sheet_name, str(source_path)))

    if not rows:
        raise ValueError(f"No World Cup odds sheets found in {source_path}.")

    odds = pd.concat(rows, ignore_index=True)
    return validate_odds_frame(odds)


def validate_odds_frame(odds: pd.DataFrame) -> pd.DataFrame:
    """Validate normalized odds and derive margin-adjusted probabilities."""

    required_columns = {
        "source",
        "competition",
        "date",
        "home_team",
        "away_team",
        "home_odds",
        "draw_odds",
        "away_odds",
    }
    missing = required_columns.difference(odds.columns)
    if missing:
        raise ValueError(f"Missing odds columns: {sorted(missing)}")

    normalized = odds.loc[:, sorted(required_columns)].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.date
    for column in ["source", "competition", "home_team", "away_team"]:
        normalized[column] = normalized[column].astype("string").str.strip()
        if normalized[column].isna().any() or (normalized[column] == "").any():
            raise ValueError(f"Column {column!r} contains missing or empty values.")

    for column in ["home_odds", "draw_odds", "away_odds"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        if (normalized[column] <= 1.0).any():
            raise ValueError(f"Column {column!r} contains invalid decimal odds.")

    implied = pd.DataFrame(
        {
            "market_home_win": 1.0 / normalized["home_odds"],
            "market_draw": 1.0 / normalized["draw_odds"],
            "market_away_win": 1.0 / normalized["away_odds"],
        }
    )
    overround = implied.sum(axis=1)
    normalized["market_overround"] = overround
    normalized["market_home_win"] = implied["market_home_win"] / overround
    normalized["market_draw"] = implied["market_draw"] / overround
    normalized["market_away_win"] = implied["market_away_win"] / overround
    normalized["home_team_key"] = normalized["home_team"].map(team_key)
    normalized["away_team_key"] = normalized["away_team"].map(team_key)
    return normalized.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def team_key(value: object) -> str:
    """Normalize team names for odds/result matching."""

    key = normalize_name(value)
    return TEAM_ALIASES.get(key, key)


def _parse_football_data_sheet(sheet: pd.DataFrame, sheet_name: str, source: str) -> pd.DataFrame:
    missing = {"Date", "Home", "Away"}.difference(sheet.columns)
    if missing:
        raise ValueError(f"Sheet {sheet_name!r} missing columns: {sorted(missing)}")

    odds_columns = _available_odds_columns(sheet)
    if odds_columns is None:
        raise ValueError(f"Sheet {sheet_name!r} does not contain average 1X2 odds.")

    parsed = sheet.loc[:, ["Date", "Home", "Away", *odds_columns]].copy()
    parsed = parsed.rename(
        columns={
            "Date": "date",
            "Home": "home_team",
            "Away": "away_team",
            odds_columns[0]: "home_odds",
            odds_columns[1]: "draw_odds",
            odds_columns[2]: "away_odds",
        }
    )
    parsed["competition"] = sheet_name
    parsed["source"] = source
    return parsed


def _available_odds_columns(sheet: pd.DataFrame) -> tuple[str, str, str] | None:
    preferred = ("H-Avg", "D-Avg", "A-Avg")
    if set(preferred).issubset(sheet.columns):
        return preferred
    alternate = ("H_Avg", "D_Avg", "A_Avg")
    if set(alternate).issubset(sheet.columns):
        return alternate
    return None
