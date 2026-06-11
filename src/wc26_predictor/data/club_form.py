"""Club-form data parsing and player matching."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

CLUB_FORM_COLUMNS = {
    "competition",
    "source_url",
    "player",
    "club",
    "goals",
    "minutes",
}


def load_club_top_scorers_csv(path_or_url: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path_or_url)
    return validate_club_top_scorers_frame(raw)


def validate_club_top_scorers_frame(club_form: pd.DataFrame) -> pd.DataFrame:
    missing = CLUB_FORM_COLUMNS.difference(club_form.columns)
    if missing:
        raise ValueError(f"Missing club-form columns: {sorted(missing)}")

    normalized = club_form.loc[:, sorted(CLUB_FORM_COLUMNS)].copy()
    for column in ["competition", "source_url", "player", "club"]:
        normalized[column] = normalized[column].astype("string").str.strip()
        if normalized[column].isna().any() or (normalized[column] == "").any():
            raise ValueError(f"Column {column!r} contains missing or empty values.")

    normalized["goals"] = pd.to_numeric(normalized["goals"], errors="raise").astype(int)
    normalized["minutes"] = pd.to_numeric(normalized["minutes"], errors="coerce")
    if (normalized["goals"] < 0).any():
        raise ValueError("Club goals cannot be negative.")
    return normalized.sort_values(["competition", "goals"], ascending=[True, False])


def parse_wikipedia_top_scorers(html: str, competition: str, source_url: str) -> pd.DataFrame:
    """Parse top-scorer tables from a Wikipedia competition page."""

    rows = []
    for table in _html_tables(html):
        table = _flatten_columns(table)
        player_col = _find_column(table, ["player", "scorer"])
        club_col = _find_column(table, ["club", "club(s)", "team"])
        goals_col = _find_column(table, ["goals"])
        if player_col is None or club_col is None or goals_col is None:
            continue
        minutes_col = _find_column(table, ["minutes played", "minutes", "mins"])

        for _, item in table.iterrows():
            player = _clean_text(item[player_col])
            club = _clean_text(item[club_col])
            goals = _parse_int(item[goals_col])
            minutes = _parse_int(item[minutes_col]) if minutes_col is not None else None
            if (
                not _is_missing_text(player)
                and not _is_missing_text(club)
                and goals is not None
                and not _is_aggregate_player_row(player)
            ):
                rows.append(
                    {
                        "competition": competition,
                        "source_url": source_url,
                        "player": player,
                        "club": club,
                        "goals": goals,
                        "minutes": minutes,
                    }
                )

    if not rows:
        raise ValueError(f"No top-scorer rows parsed for {competition}.")

    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["competition", "player", "club", "goals", "minutes"]
    )
    return validate_club_top_scorers_frame(frame)


def enrich_top_scorers_with_club_form(
    top_scorers: pd.DataFrame,
    squads: pd.DataFrame,
    club_form: pd.DataFrame,
    top_n: int = 100,
) -> pd.DataFrame:
    """Add club-form features to the current top-N scorer candidates."""

    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    candidates = top_scorers.head(top_n).copy()
    squad_lookup = squads.loc[:, ["team", "player", "position", "caps", "goals", "club"]].copy()
    squad_lookup["player_key"] = squad_lookup["player"].map(normalize_name)
    squad_lookup["team_key"] = squad_lookup["team"].map(normalize_name)

    candidates["player_key"] = candidates["scorer"].map(normalize_name)
    candidates["team_key"] = candidates["team"].map(normalize_name)
    candidates = candidates.merge(
        squad_lookup.drop_duplicates(["team_key", "player_key"]),
        on=["team_key", "player_key"],
        how="left",
        suffixes=("", "_squad"),
    )

    form = club_form.copy()
    form["player_key"] = form["player"].map(normalize_name)
    form["club_key"] = form["club"].map(normalize_name)
    form = form.sort_values(["player_key", "goals"], ascending=[True, False])

    enriched_rows = []
    for row in candidates.itertuples(index=False):
        match = _best_club_form_match(row, form)
        enriched_rows.append({**row._asdict(), **match})

    enriched = pd.DataFrame(enriched_rows)
    enriched["club_form_goals"] = pd.to_numeric(enriched["club_form_goals"], errors="coerce")
    enriched["club_form_minutes"] = pd.to_numeric(enriched["club_form_minutes"], errors="coerce")
    enriched["club_form_goals_per90"] = (
        90.0 * enriched["club_form_goals"] / enriched["club_form_minutes"]
    )
    enriched.loc[enriched["club_form_minutes"].isna(), "club_form_goals_per90"] = pd.NA
    enriched["club_form_multiplier"] = _club_form_multiplier(enriched["club_form_goals"])
    enriched["expected_tournament_goals_club_adjusted"] = (
        enriched["expected_tournament_goals"] * enriched["club_form_multiplier"]
    )
    return enriched.sort_values(
        "expected_tournament_goals_club_adjusted",
        ascending=False,
        ignore_index=True,
    )


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\([^)]*\)", "", text.lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _best_club_form_match(row: object, form: pd.DataFrame) -> dict[str, object]:
    player_matches = form[form["player_key"] == row.player_key]
    if player_matches.empty:
        return _empty_club_form_match("unmatched")

    squad_club_key = normalize_name(getattr(row, "club", ""))
    club_matches = player_matches[player_matches["club_key"].map(lambda club: _clubs_overlap(squad_club_key, club))]
    if not club_matches.empty:
        best = club_matches.iloc[0]
        return _club_form_match(best, "player_and_club")

    best = player_matches.iloc[0]
    return _club_form_match(best, "player_only")


def _clubs_overlap(first: str, second: str) -> bool:
    if not first or not second:
        return False
    return first in second or second in first


def _club_form_match(row: pd.Series, quality: str) -> dict[str, object]:
    return {
        "club_form_match_quality": quality,
        "club_form_competition": row["competition"],
        "club_form_source_url": row["source_url"],
        "club_form_player": row["player"],
        "club_form_club": row["club"],
        "club_form_goals": row["goals"],
        "club_form_minutes": row["minutes"],
    }


def _empty_club_form_match(quality: str) -> dict[str, object]:
    return {
        "club_form_match_quality": quality,
        "club_form_competition": pd.NA,
        "club_form_source_url": pd.NA,
        "club_form_player": pd.NA,
        "club_form_club": pd.NA,
        "club_form_goals": pd.NA,
        "club_form_minutes": pd.NA,
    }


def _club_form_multiplier(goals: pd.Series) -> pd.Series:
    signal = goals.fillna(goals.median()).fillna(0.0).map(lambda value: float(value))
    if signal.std() == 0:
        return pd.Series(1.0, index=goals.index)
    standardized = (signal - signal.median()) / signal.std()
    return (1.0 + 0.12 * standardized).clip(lower=0.80, upper=1.25)


def _flatten_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = [
            " ".join(str(part) for part in column if not str(part).startswith("Unnamed")).strip()
            for column in table.columns
        ]
    table.columns = [str(column).strip() for column in table.columns]
    return table


def _html_tables(html: str) -> list[pd.DataFrame]:
    soup = BeautifulSoup(html, "html.parser")
    tables = []
    for table in soup.find_all("table"):
        header_cells = table.find("tr")
        if header_cells is None:
            continue
        headers = [
            _clean_text(cell.get_text(" ", strip=True))
            for cell in header_cells.find_all(["th", "td"])
        ]
        if not headers:
            continue
        rows = []
        for tr in table.find_all("tr")[1:]:
            cells = [_clean_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
            if not cells:
                continue
            if len(cells) < len(headers):
                cells = [pd.NA] * (len(headers) - len(cells)) + cells
            if len(cells) > len(headers):
                cells = cells[: len(headers)]
            rows.append(cells)
        if rows:
            tables.append(pd.DataFrame(rows, columns=headers))
    return tables


def _find_column(table: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {normalize_name(column): column for column in table.columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _clean_text(value: object) -> str:
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_int(value: object) -> int | None:
    match = re.search(r"\d+", str(value).replace(",", ""))
    return int(match.group()) if match else None


def _is_missing_text(value: str) -> bool:
    return normalize_name(value) in {"", "nan", "none", "na"}


def _is_aggregate_player_row(player: str) -> bool:
    normalized = normalize_name(player)
    return bool(re.fullmatch(r"\d+", normalized) or re.search(r"\d+ players?", normalized))
