"""Sportsgambler current injury feed parsing and squad matching."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

from wc26_predictor.data.club_form import normalize_name

SPORTSGAMBLER_FOOTBALL_INJURIES_URL = "https://www.sportsgambler.com/injuries/football/"


@dataclass(frozen=True, slots=True)
class SportsgamblerConfig:
    """Configuration for current Sportsgambler injury extraction."""

    source_url: str = SPORTSGAMBLER_FOOTBALL_INJURIES_URL
    include_club_suspensions: bool = False
    timeout_seconds: int = 30


def fetch_sportsgambler_football_injuries(
    config: SportsgamblerConfig | None = None,
) -> pd.DataFrame:
    """Fetch and parse the all-football current injury table."""

    cfg = config or SportsgamblerConfig()
    response = requests.get(
        cfg.source_url,
        headers={"User-Agent": "wc26-predictor/0.1"},
        timeout=cfg.timeout_seconds,
    )
    response.raise_for_status()
    return parse_sportsgambler_football_injuries(response.text, source_url=cfg.source_url)


def parse_sportsgambler_football_injuries(html: str, source_url: str) -> pd.DataFrame:
    """Parse current football injury rows from Sportsgambler HTML."""

    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, object]] = []
    for heading in soup.select("h3"):
        club_team = heading.get_text(" ", strip=True)
        sibling = heading.find_next_sibling()
        while isinstance(sibling, Tag) and sibling.name != "h3":
            classes = sibling.get("class", [])
            if "inj-row" in classes:
                parsed = _parse_injury_row(sibling, club_team, source_url)
                if parsed is not None:
                    rows.append(parsed)
            sibling = sibling.find_next_sibling()

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "club_team",
                "player",
                "position",
                "matches",
                "goals",
                "assists",
                "info",
                "expected_return",
                "type_class",
                "source_url",
            ]
        )
    return frame.sort_values(["club_team", "player"]).reset_index(drop=True)


def match_sportsgambler_injuries_to_squads(
    injuries: pd.DataFrame,
    squads: pd.DataFrame,
    config: SportsgamblerConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match current club injury rows to World Cup squad players.

    Returns an audit table with all name matches and a model-ready availability
    override table restricted to name+club matches and physical injury rows.
    """

    cfg = config or SportsgamblerConfig()
    required_injuries = {
        "club_team",
        "player",
        "info",
        "expected_return",
        "type_class",
        "source_url",
    }
    missing_injuries = required_injuries.difference(injuries.columns)
    if missing_injuries:
        raise ValueError(f"Missing Sportsgambler injury columns: {sorted(missing_injuries)}")

    required_squads = {"team", "player", "club", "position"}
    missing_squads = required_squads.difference(squads.columns)
    if missing_squads:
        raise ValueError(f"Missing squad columns: {sorted(missing_squads)}")

    injury_rows = injuries.copy()
    squad_rows = squads.loc[:, ["team", "player", "club", "position"]].copy()
    injury_rows["player_key"] = injury_rows["player"].map(normalize_name)
    squad_rows["player_key"] = squad_rows["player"].map(normalize_name)
    matched = injury_rows.merge(
        squad_rows,
        on="player_key",
        how="inner",
        suffixes=("_source", "_squad"),
    )
    if matched.empty:
        return matched, _empty_availability_overrides()

    matched["club_match"] = matched.apply(
        lambda row: _clubs_overlap(row["club_team"], row["club"]),
        axis=1,
    )
    matched["status"] = matched["type_class"].map(_status_from_type_class)
    matched["is_club_suspension"] = matched["type_class"].str.contains(
        "redcard",
        case=False,
        na=False,
    )
    matched = matched.sort_values(
        ["club_match", "team", "player_squad"],
        ascending=[False, True, True],
        ignore_index=True,
    )

    usable = matched[matched["club_match"]].copy()
    if not cfg.include_club_suspensions:
        usable = usable[~usable["is_club_suspension"]].copy()
    if usable.empty:
        return matched, _empty_availability_overrides()

    usable["reason"] = (
        "Sportsgambler club feed: "
        + usable["info"].astype(str)
        + "; club="
        + usable["club_team"].astype(str)
        + "; expected_return="
        + usable["expected_return"].astype(str)
    )
    overrides = (
        usable.loc[:, ["team", "player_squad", "status", "reason", "source_url"]]
        .rename(columns={"player_squad": "player"})
        .sort_values(["team", "player"], ignore_index=True)
    )
    return matched, overrides


def _parse_injury_row(row: Tag, club_team: str, source_url: str) -> dict[str, object] | None:
    container = row.select_one(".inj-container")
    if container is None:
        return None
    fields = {
        "club_team": club_team,
        "player": _text(container, ".inj-player"),
        "position": _text(container, ".inj-position"),
        "matches": _text(container, ".inj-game"),
        "goals": _text(container, ".inj-goals"),
        "assists": _text(container, ".inj-assist"),
        "info": _text(container, ".inj-info"),
        "expected_return": _text(container, ".inj-return"),
        "type_class": " ".join(container.select_one(".inj-type").get("class", []))
        if container.select_one(".inj-type")
        else "",
        "source_url": source_url,
    }
    if not fields["player"]:
        return None
    return fields


def _text(container: Tag, selector: str) -> str:
    node = container.select_one(selector)
    return node.get_text(" ", strip=True) if node is not None else ""


def _status_from_type_class(type_class: object) -> str:
    normalized = str(type_class).lower()
    if "questionmark" in normalized:
        return "doubtful"
    if "redcard" in normalized:
        return "suspended"
    return "injured"


def _clubs_overlap(left: object, right: object) -> bool:
    left_tokens = {token for token in normalize_name(left).split() if len(token) >= 4}
    right_tokens = {token for token in normalize_name(right).split() if len(token) >= 4}
    return bool(left_tokens.intersection(right_tokens))


def _empty_availability_overrides() -> pd.DataFrame:
    return pd.DataFrame(columns=["team", "player", "status", "reason", "source_url"])
