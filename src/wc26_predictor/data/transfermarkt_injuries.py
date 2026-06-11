"""Transfermarkt injury-history features for World Cup squad players."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from wc26_predictor.data.availability import validate_player_availability_frame
from wc26_predictor.data.club_form import normalize_name
from wc26_predictor.data.transfermarkt import TEAM_COUNTRY_ALIASES

TRANSFERMARKT_INJURY_COLUMNS = {
    "player_id",
    "season_name",
    "injury_reason",
    "from_date",
    "end_date",
    "days_missed",
    "games_missed",
}


@dataclass(frozen=True, slots=True)
class TransfermarktInjuryConfig:
    """Configuration for injury feature construction."""

    reference_date: date = date(2026, 6, 11)
    lookback_days: int = 730
    current_data_max_age_days: int = 45
    source_snapshot_date: date | None = None


def load_transfermarkt_injuries_csv(path_or_url: str | Path) -> pd.DataFrame:
    """Load and validate Transfermarkt injury-history rows."""

    raw = pd.read_csv(path_or_url)
    return validate_transfermarkt_injuries_frame(raw)


def validate_transfermarkt_injuries_frame(injuries: pd.DataFrame) -> pd.DataFrame:
    """Validate the Transfermarkt injury-history schema."""

    missing = TRANSFERMARKT_INJURY_COLUMNS.difference(injuries.columns)
    if missing:
        raise ValueError(f"Missing Transfermarkt injury columns: {sorted(missing)}")

    normalized = injuries.loc[:, sorted(TRANSFERMARKT_INJURY_COLUMNS)].copy()
    normalized["player_id"] = pd.to_numeric(normalized["player_id"], errors="raise").astype(int)
    if (normalized["player_id"] <= 0).any():
        raise ValueError("player_id must be positive.")

    for column in ["season_name", "injury_reason"]:
        normalized[column] = normalized[column].astype("string").str.strip()
        if normalized[column].isna().any() or (normalized[column] == "").any():
            raise ValueError(f"Column {column!r} contains missing or empty values.")

    normalized["from_date"] = pd.to_datetime(
        normalized["from_date"],
        errors="raise",
    )
    normalized["end_date"] = pd.to_datetime(
        normalized["end_date"],
        errors="coerce",
    )
    closed = normalized["end_date"].notna()
    if (normalized.loc[closed, "end_date"] < normalized.loc[closed, "from_date"]).any():
        raise ValueError("end_date cannot be before from_date.")

    for column in ["days_missed", "games_missed"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0.0)
        if (normalized[column] < 0).any():
            raise ValueError(f"{column} cannot be negative.")

    return normalized.sort_values(["player_id", "from_date"]).reset_index(drop=True)


def build_transfermarkt_squad_injury_features(
    squads: pd.DataFrame,
    availability: pd.DataFrame,
    transfermarkt_dir: str | Path,
    injuries_path: str | Path,
    config: TransfermarktInjuryConfig | None = None,
) -> pd.DataFrame:
    """Join World Cup squad players to Transfermarkt injury-history features."""

    cfg = config or TransfermarktInjuryConfig()
    if cfg.lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")

    required_squad = {"team", "player", "club", "position", "caps", "goals"}
    missing_squad = required_squad.difference(squads.columns)
    if missing_squad:
        raise ValueError(f"Missing squad columns for injury features: {sorted(missing_squad)}")

    base_dir = Path(transfermarkt_dir)
    players_path = base_dir / "players.csv"
    if not players_path.exists():
        raise ValueError(f"Missing Transfermarkt players file: {players_path}")
    injuries_file = Path(injuries_path)
    if not injuries_file.exists():
        raise ValueError(f"Missing Transfermarkt injuries file: {injuries_file}")

    squad = squads.loc[:, sorted(required_squad)].copy()
    squad["team_key"] = squad["team"].map(normalize_name)
    squad["player_key"] = squad["player"].map(normalize_name)
    squad["club_key"] = squad["club"].map(normalize_name)

    avail = validate_player_availability_frame(availability)
    expected_minutes = avail.loc[
        :,
        ["team_key", "player_key", "expected_minutes_share"],
    ].copy()
    squad = squad.merge(expected_minutes, on=["team_key", "player_key"], how="left")
    if squad["expected_minutes_share"].isna().any():
        missing_players = squad.loc[
            squad["expected_minutes_share"].isna(),
            ["team", "player"],
        ].to_dict("records")
        raise ValueError(f"Availability features are missing squad players: {missing_players}")

    players = _load_transfermarkt_players(players_path)
    matched = _match_squad_players_to_transfermarkt(squad, players)
    injuries = load_transfermarkt_injuries_csv(injuries_file)
    aggregates = _aggregate_injury_history(injuries, cfg)

    features = matched.merge(aggregates, on="transfermarkt_player_id", how="left")
    numeric_zero_columns = [
        "career_injury_spells",
        "career_days_missed",
        "career_games_missed",
        "recent_injury_spells",
        "recent_injury_days",
        "recent_games_missed",
        "active_injury_flag",
        "open_injury_at_source_flag",
    ]
    for column in numeric_zero_columns:
        features[column] = features[column].fillna(0)
    features["active_injury_reason"] = features["active_injury_reason"].fillna("")
    features["active_injury_since"] = features["active_injury_since"].fillna("")
    features["source_snapshot_date"] = features["source_snapshot_date"].fillna("")
    features["recent_injury_days_share"] = (
        features["recent_injury_days"] / float(cfg.lookback_days)
    ).clip(0.0, 1.0)
    features["injury_recurrence_score"] = (
        features["recent_injury_days_share"] + 0.03 * features["recent_injury_spells"]
    ).clip(0.0, 1.0)
    features["historical_injury_burden"] = (
        features["expected_minutes_share"] * features["injury_recurrence_score"]
    )
    features["current_open_injury_burden"] = (
        features["expected_minutes_share"] * features["active_injury_flag"]
    )

    return features.sort_values(["team", "player"]).reset_index(drop=True)


def aggregate_transfermarkt_team_injury_burden(player_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Transfermarkt player injury-history features to team burden."""

    required = {
        "team",
        "transfermarkt_player_id",
        "recent_injury_spells",
        "active_injury_flag",
        "historical_injury_burden",
        "current_open_injury_burden",
    }
    missing = required.difference(player_features.columns)
    if missing:
        raise ValueError(f"Missing Transfermarkt injury feature columns: {sorted(missing)}")

    return (
        player_features.groupby("team", as_index=False)
        .agg(
            transfermarkt_players_matched=(
                "transfermarkt_player_id",
                lambda values: int(values.notna().sum()),
            ),
            players_with_recent_injury=(
                "recent_injury_spells",
                lambda values: int((values > 0).sum()),
            ),
            players_with_open_injury=(
                "active_injury_flag",
                lambda values: int((values > 0).sum()),
            ),
            team_historical_injury_burden=("historical_injury_burden", "sum"),
            team_current_open_injury_burden=("current_open_injury_burden", "sum"),
        )
        .assign(
            team_historical_injury_burden=lambda frame: frame[
                "team_historical_injury_burden"
            ].clip(0.0, 3.0),
            team_current_open_injury_burden=lambda frame: frame[
                "team_current_open_injury_burden"
            ].clip(0.0, 3.0),
        )
        .sort_values(
            ["team_current_open_injury_burden", "team_historical_injury_burden", "team"],
            ascending=[False, False, True],
            ignore_index=True,
        )
    )


def _load_transfermarkt_players(path: Path) -> pd.DataFrame:
    columns = [
        "player_id",
        "name",
        "country_of_citizenship",
        "current_club_name",
        "market_value_in_eur",
        "international_caps",
    ]
    players = pd.read_csv(path, usecols=columns)
    players["transfermarkt_player_id"] = pd.to_numeric(
        players["player_id"],
        errors="raise",
    ).astype("Int64")
    players["player_key"] = players["name"].map(normalize_name)
    players["country_key"] = players["country_of_citizenship"].map(normalize_name)
    players["club_key"] = players["current_club_name"].map(normalize_name)
    players["market_value_in_eur"] = pd.to_numeric(
        players["market_value_in_eur"],
        errors="coerce",
    )
    players["international_caps"] = pd.to_numeric(
        players["international_caps"],
        errors="coerce",
    )
    return players


def _match_squad_players_to_transfermarkt(
    squad: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for player in squad.itertuples(index=False):
        candidates = players[players["player_key"] == player.player_key].copy()
        if candidates.empty:
            rows.append(_unmatched_player(player))
            continue

        aliases = _team_aliases(player.team_key)
        player_club_key = player.club_key
        candidates["match_score"] = 0
        candidates["match_score"] += candidates["country_key"].isin(aliases).astype(int) * 4
        candidates["match_score"] += candidates["club_key"].map(
            lambda club_key, player_club_key=player_club_key: _clubs_overlap(
                player_club_key,
                club_key,
            )
        ).astype(int) * 3
        candidates["match_score"] += candidates["international_caps"].notna().astype(int)
        candidates = candidates.sort_values(
            ["match_score", "market_value_in_eur", "international_caps"],
            ascending=[False, False, False],
        )
        best = candidates.iloc[0]
        rows.append(_matched_player(player, best))
    return pd.DataFrame(rows)


def _aggregate_injury_history(
    injuries: pd.DataFrame,
    config: TransfermarktInjuryConfig,
) -> pd.DataFrame:
    reference = pd.Timestamp(config.reference_date)
    window_start = reference - pd.Timedelta(days=config.lookback_days)
    snapshot = (
        pd.Timestamp(config.source_snapshot_date)
        if config.source_snapshot_date
        else _infer_snapshot_date(injuries)
    )
    source_is_current = 0 <= (reference - snapshot).days <= config.current_data_max_age_days
    frame = injuries.copy()
    effective_end = _effective_end_dates(frame, snapshot)
    frame["overlap_days"] = _overlap_days(
        frame["from_date"],
        effective_end,
        window_start,
        reference,
    )
    frame["is_recent"] = frame["overlap_days"] > 0
    frame["is_open_at_source"] = frame["end_date"].isna()
    frame["is_active"] = frame["is_open_at_source"] & source_is_current

    active = (
        frame[frame["is_open_at_source"]]
        .sort_values(["player_id", "from_date"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .loc[:, ["player_id", "injury_reason", "from_date"]]
        .rename(
            columns={
                "injury_reason": "active_injury_reason",
                "from_date": "active_injury_since",
            }
        )
    )
    active["open_injury_at_source_flag"] = 1
    active["active_injury_flag"] = int(source_is_current)
    active["active_injury_since"] = active["active_injury_since"].dt.date.astype("string")
    active["source_snapshot_date"] = snapshot.date().isoformat()

    recent = frame[frame["is_recent"]].copy()
    recent_agg = (
        recent.groupby("player_id", as_index=False)
        .agg(
            recent_injury_spells=("injury_reason", "size"),
            recent_injury_days=("overlap_days", "sum"),
            recent_games_missed=("games_missed", "sum"),
        )
        if not recent.empty
        else pd.DataFrame(
            columns=[
                "player_id",
                "recent_injury_spells",
                "recent_injury_days",
                "recent_games_missed",
            ]
        )
    )
    career = (
        frame.groupby("player_id", as_index=False)
        .agg(
            career_injury_spells=("injury_reason", "size"),
            career_days_missed=("days_missed", "sum"),
            career_games_missed=("games_missed", "sum"),
        )
        .rename(columns={"player_id": "transfermarkt_player_id"})
    )
    return (
        career.merge(
            recent_agg.rename(columns={"player_id": "transfermarkt_player_id"}),
            on="transfermarkt_player_id",
            how="left",
        )
        .merge(
            active.rename(columns={"player_id": "transfermarkt_player_id"}),
            on="transfermarkt_player_id",
            how="left",
        )
    )


def _overlap_days(
    starts: pd.Series,
    ends: pd.Series,
    window_start: pd.Timestamp,
    reference: pd.Timestamp,
) -> pd.Series:
    overlap_start = starts.clip(lower=window_start)
    overlap_end = ends.clip(upper=reference)
    days = (overlap_end - overlap_start).dt.days + 1
    return days.clip(lower=0).astype(float)


def _infer_snapshot_date(injuries: pd.DataFrame) -> pd.Timestamp:
    closed_dates = injuries["end_date"].dropna()
    open_rows = injuries[injuries["end_date"].isna()].copy()
    inferred_open_dates = pd.Series(dtype="datetime64[ns]")
    if not open_rows.empty:
        inferred_open_dates = open_rows["from_date"] + pd.to_timedelta(
            open_rows["days_missed"].clip(lower=1) - 1,
            unit="D",
        )
    candidates = pd.concat([closed_dates, inferred_open_dates], ignore_index=True)
    if candidates.empty:
        raise ValueError("Cannot infer Transfermarkt injury source snapshot date.")
    return candidates.max()


def _effective_end_dates(injuries: pd.DataFrame, snapshot: pd.Timestamp) -> pd.Series:
    open_estimate = injuries["from_date"] + pd.to_timedelta(
        injuries["days_missed"].clip(lower=1) - 1,
        unit="D",
    )
    open_estimate = open_estimate.clip(upper=snapshot)
    return injuries["end_date"].fillna(open_estimate)


def _matched_player(squad_player: object, transfermarkt_player: pd.Series) -> dict[str, object]:
    score = int(transfermarkt_player["match_score"])
    return {
        "team": squad_player.team,
        "player": squad_player.player,
        "club": squad_player.club,
        "position": squad_player.position,
        "caps": squad_player.caps,
        "goals": squad_player.goals,
        "expected_minutes_share": squad_player.expected_minutes_share,
        "transfermarkt_player_id": int(transfermarkt_player["transfermarkt_player_id"]),
        "transfermarkt_player_name": transfermarkt_player["name"],
        "transfermarkt_current_club": transfermarkt_player["current_club_name"],
        "transfermarkt_country": transfermarkt_player["country_of_citizenship"],
        "transfermarkt_match_quality": _match_quality(score),
        "latest_market_value_in_eur": transfermarkt_player["market_value_in_eur"],
    }


def _unmatched_player(squad_player: object) -> dict[str, object]:
    return {
        "team": squad_player.team,
        "player": squad_player.player,
        "club": squad_player.club,
        "position": squad_player.position,
        "caps": squad_player.caps,
        "goals": squad_player.goals,
        "expected_minutes_share": squad_player.expected_minutes_share,
        "transfermarkt_player_id": pd.NA,
        "transfermarkt_player_name": "",
        "transfermarkt_current_club": "",
        "transfermarkt_country": "",
        "transfermarkt_match_quality": "unmatched",
        "latest_market_value_in_eur": np.nan,
    }


def _match_quality(score: int) -> str:
    if score >= 7:
        return "name_country_club"
    if score >= 4:
        return "name_country"
    return "name_only"


def _team_aliases(team_key: str) -> set[str]:
    return TEAM_COUNTRY_ALIASES.get(team_key, {team_key})


def _clubs_overlap(left: str, right: str) -> bool:
    if not left or not right:
        return False
    left_tokens = {token for token in left.split() if len(token) >= 4}
    right_tokens = {token for token in right.split() if len(token) >= 4}
    return bool(left_tokens.intersection(right_tokens))
