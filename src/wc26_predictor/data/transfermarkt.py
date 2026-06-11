"""Transfermarkt Kaggle feature engineering for top scorer candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wc26_predictor.data.club_form import normalize_name


@dataclass(frozen=True, slots=True)
class TransfermarktConfig:
    """Configuration for club-form aggregation from the Kaggle dump."""

    season_start: str = "2025-07-01"
    season_end: str = "2026-06-10"
    club_form_half_life_days: float = 120.0
    top_n: int = 100
    chunk_size: int = 250_000


TEAM_COUNTRY_ALIASES = {
    "czech republic": {"czech republic", "czechia"},
    "curacao": {"curacao", "curaçao"},
    "dr congo": {"dr congo", "congo dr", "congo"},
    "england": {"england"},
    "ivory coast": {"ivory coast", "cote d ivoire"},
    "south korea": {"south korea", "korea south", "korea republic"},
    "turkey": {"turkey", "turkiye", "türkiye"},
    "united states": {"united states", "usa", "united states of america"},
}


def build_transfermarkt_top_scorer_features(
    top_scorers: pd.DataFrame,
    squads: pd.DataFrame,
    transfermarkt_dir: str | Path,
    config: TransfermarktConfig | None = None,
) -> pd.DataFrame:
    """Enrich current top scorer candidates with Transfermarkt player features."""

    cfg = config or TransfermarktConfig()
    if cfg.top_n <= 0:
        raise ValueError("top_n must be positive.")
    if cfg.club_form_half_life_days <= 0:
        raise ValueError("club_form_half_life_days must be positive.")

    base_dir = Path(transfermarkt_dir)
    required_files = [
        "appearances.csv",
        "competitions.csv",
        "game_events.csv",
        "game_lineups.csv",
        "games.csv",
        "player_valuations.csv",
        "players.csv",
    ]
    missing_files = [name for name in required_files if not (base_dir / name).exists()]
    if missing_files:
        raise ValueError(f"Missing Transfermarkt files: {missing_files}")

    candidates = _candidate_frame(top_scorers.head(cfg.top_n), squads)
    players = _load_players(base_dir / "players.csv")
    matched = _match_candidates_to_players(candidates, players)
    matched_ids = set(matched["transfermarkt_player_id"].dropna().astype(int))

    competition_context = _competition_context(base_dir, cfg)
    appearances = _aggregate_appearances(base_dir, matched_ids, cfg, competition_context)
    starts = _aggregate_starts(base_dir, matched_ids, cfg)
    event_goals = _aggregate_event_goals(base_dir, matched_ids, cfg, competition_context)
    valuations = _latest_valuations(base_dir / "player_valuations.csv", matched_ids, cfg)

    enriched = (
        matched.merge(appearances, on="transfermarkt_player_id", how="left")
        .merge(starts, on="transfermarkt_player_id", how="left")
        .merge(event_goals, on="transfermarkt_player_id", how="left")
        .merge(valuations, on="transfermarkt_player_id", how="left")
    )
    numeric_fill_zero = [
        "club_appearances",
        "club_goals",
        "club_weighted_goals",
        "club_assists",
        "club_minutes",
        "club_starts",
        "event_goals",
        "event_weighted_goals",
    ]
    for column in numeric_fill_zero:
        enriched[column] = enriched[column].fillna(0)

    has_appearance_coverage = enriched["club_appearances"] > 0
    has_event_goal_coverage = enriched["event_goals"] > 0
    enriched["club_goals_source"] = np.select(
        [has_appearance_coverage, has_event_goal_coverage],
        ["appearances", "game_events"],
        default="missing",
    )
    enriched["club_goals_model"] = np.where(
        has_appearance_coverage,
        enriched["club_goals"],
        enriched["event_goals"],
    )
    enriched["club_weighted_goals_model"] = np.where(
        has_appearance_coverage,
        enriched["club_weighted_goals"],
        enriched["event_weighted_goals"],
    )
    enriched["club_minutes_estimated"] = (~has_appearance_coverage) & (enriched["club_starts"] > 0)
    enriched["club_minutes_model"] = np.where(
        has_appearance_coverage,
        enriched["club_minutes"],
        enriched["club_starts"] * 75.0,
    )

    enriched["club_goals_per90"] = np.where(
        enriched["club_minutes_model"] > 0,
        90.0 * enriched["club_goals_model"] / enriched["club_minutes_model"],
        np.nan,
    )
    enriched["club_weighted_goals_per90"] = np.where(
        enriched["club_minutes_model"] > 0,
        90.0 * enriched["club_weighted_goals_model"] / enriched["club_minutes_model"],
        np.nan,
    )
    enriched["club_assists_per90"] = np.where(
        enriched["club_minutes"] > 0,
        90.0 * enriched["club_assists"] / enriched["club_minutes"],
        np.nan,
    )
    enriched["club_starts_per90"] = np.where(
        enriched["club_minutes"] > 0,
        90.0 * enriched["club_starts"] / enriched["club_minutes"],
        np.nan,
    )
    enriched["transfermarkt_recent_appearance_coverage"] = np.where(
        enriched["club_minutes_model"] > 0,
        np.clip(enriched["club_minutes_model"] / 900.0, 0.0, 1.0),
        0.0,
    )
    enriched["transfermarkt_multiplier"] = _transfermarkt_multiplier(enriched)
    enriched["expected_tournament_goals_transfermarkt_adjusted"] = (
        enriched["expected_tournament_goals"] * enriched["transfermarkt_multiplier"]
    )
    return enriched.sort_values(
        "expected_tournament_goals_transfermarkt_adjusted",
        ascending=False,
        ignore_index=True,
    )


def _candidate_frame(top_scorers: pd.DataFrame, squads: pd.DataFrame) -> pd.DataFrame:
    required_top = {"team", "scorer", "expected_tournament_goals"}
    missing_top = required_top.difference(top_scorers.columns)
    if missing_top:
        raise ValueError(f"Missing top-scorer columns: {sorted(missing_top)}")

    required_squad = {"team", "player", "club", "position", "caps", "goals"}
    missing_squad = required_squad.difference(squads.columns)
    if missing_squad:
        raise ValueError(f"Missing squad columns: {sorted(missing_squad)}")

    candidates = top_scorers.copy()
    candidates["team_key"] = candidates["team"].map(normalize_name)
    candidates["player_key"] = candidates["scorer"].map(normalize_name)
    squad_lookup = squads.loc[:, ["team", "player", "club", "position", "caps", "goals"]].copy()
    squad_lookup["team_key"] = squad_lookup["team"].map(normalize_name)
    squad_lookup["player_key"] = squad_lookup["player"].map(normalize_name)
    return candidates.merge(
        squad_lookup.drop_duplicates(["team_key", "player_key"]),
        on=["team_key", "player_key"],
        how="left",
        suffixes=("", "_squad"),
    )


def _load_players(path: Path) -> pd.DataFrame:
    columns = [
        "player_id",
        "name",
        "country_of_citizenship",
        "current_club_name",
        "current_club_id",
        "position",
        "sub_position",
        "international_caps",
        "international_goals",
        "market_value_in_eur",
        "highest_market_value_in_eur",
    ]
    players = pd.read_csv(path, usecols=columns)
    players["player_key"] = players["name"].map(normalize_name)
    players["country_key"] = players["country_of_citizenship"].map(normalize_name)
    players["club_key"] = players["current_club_name"].map(normalize_name)
    return players


def _match_candidates_to_players(candidates: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in candidates.itertuples(index=False):
        player_matches = players[players["player_key"] == candidate.player_key].copy()
        if player_matches.empty:
            rows.append(_unmatched_candidate(candidate))
            continue

        squad_club_key = normalize_name(getattr(candidate, "club", ""))
        team_aliases = _team_aliases(candidate.team_key)
        player_matches["match_score"] = 0
        player_matches["match_score"] += player_matches["country_key"].isin(team_aliases).astype(int) * 4
        player_matches["match_score"] += player_matches["club_key"].map(
            lambda value: _clubs_overlap(squad_club_key, value)
        ).astype(int) * 3
        player_matches["match_score"] += player_matches["international_caps"].notna().astype(int)
        player_matches = player_matches.sort_values(
            ["match_score", "market_value_in_eur", "international_caps"],
            ascending=[False, False, False],
        )
        best = player_matches.iloc[0]
        rows.append(_matched_candidate(candidate, best))
    return pd.DataFrame(rows)


def _aggregate_appearances(
    base_dir: Path,
    player_ids: set[int],
    config: TransfermarktConfig,
    competition_context: pd.DataFrame,
) -> pd.DataFrame:
    if not player_ids:
        return _empty_player_aggregate("transfermarkt_player_id")

    club_competitions = set(competition_context["competition_id"])
    chunks = []
    columns = [
        "player_id",
        "date",
        "competition_id",
        "goals",
        "assists",
        "minutes_played",
    ]
    for chunk in pd.read_csv(base_dir / "appearances.csv", usecols=columns, chunksize=config.chunk_size):
        chunk = chunk[chunk["player_id"].isin(player_ids)]
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk[
            (chunk["date"] >= pd.Timestamp(config.season_start))
            & (chunk["date"] <= pd.Timestamp(config.season_end))
            & (chunk["competition_id"].isin(club_competitions))
        ]
        if not chunk.empty:
            chunk = chunk.merge(
                competition_context[
                    ["competition_id", "competition_goal_weight", "competition_goals_per90_factor"]
                ],
                on="competition_id",
                how="left",
            )
            chunk["club_form_recency_weight"] = _recency_weight(chunk["date"], config)
            chunk["weighted_goals"] = (
                chunk["goals"]
                * chunk["competition_goal_weight"]
                * chunk["competition_goals_per90_factor"]
                * chunk["club_form_recency_weight"]
            )
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(
            {
                "transfermarkt_player_id": list(player_ids),
                "club_appearances": 0,
                "club_goals": 0,
                "club_assists": 0,
                "club_minutes": 0,
                "club_weighted_goals": 0.0,
            }
        )

    appearances = pd.concat(chunks, ignore_index=True)
    return (
        appearances.groupby("player_id", as_index=False)
        .agg(
            club_appearances=("player_id", "size"),
            club_goals=("goals", "sum"),
            club_weighted_goals=("weighted_goals", "sum"),
            club_assists=("assists", "sum"),
            club_minutes=("minutes_played", "sum"),
        )
        .rename(columns={"player_id": "transfermarkt_player_id"})
    )


def _aggregate_starts(
    base_dir: Path,
    player_ids: set[int],
    config: TransfermarktConfig,
) -> pd.DataFrame:
    if not player_ids:
        return _empty_player_aggregate("transfermarkt_player_id")

    game_context = _club_game_context(base_dir, config)
    chunks = []
    columns = ["player_id", "date", "game_id", "type"]
    for chunk in pd.read_csv(base_dir / "game_lineups.csv", usecols=columns, chunksize=config.chunk_size):
        chunk = chunk[chunk["player_id"].isin(player_ids)]
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk[
            (chunk["date"] >= pd.Timestamp(config.season_start))
            & (chunk["date"] <= pd.Timestamp(config.season_end))
            & (chunk["type"] == "starting_lineup")
        ]
        chunk = chunk[chunk["game_id"].isin(game_context)]
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame({"transfermarkt_player_id": list(player_ids), "club_starts": 0})

    starts = pd.concat(chunks, ignore_index=True)
    return (
        starts.groupby("player_id", as_index=False)
        .agg(club_starts=("player_id", "size"))
        .rename(columns={"player_id": "transfermarkt_player_id"})
    )


def _aggregate_event_goals(
    base_dir: Path,
    player_ids: set[int],
    config: TransfermarktConfig,
    competition_context: pd.DataFrame,
) -> pd.DataFrame:
    if not player_ids:
        return _empty_player_aggregate("transfermarkt_player_id")

    games = pd.read_csv(
        base_dir / "games.csv",
        usecols=["game_id", "competition_id", "competition_type"],
    )
    games = games[games["competition_type"] != "national_team_competition"]
    games = games.merge(
        competition_context[
            ["competition_id", "competition_goal_weight", "competition_goals_per90_factor"]
        ],
        on="competition_id",
        how="inner",
    )

    chunks = []
    columns = ["player_id", "date", "game_id", "type"]
    for chunk in pd.read_csv(base_dir / "game_events.csv", usecols=columns, chunksize=config.chunk_size):
        chunk = chunk[(chunk["player_id"].isin(player_ids)) & (chunk["type"] == "Goals")]
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk[
            (chunk["date"] >= pd.Timestamp(config.season_start))
            & (chunk["date"] <= pd.Timestamp(config.season_end))
        ]
        if chunk.empty:
            continue
        chunk = chunk.merge(games, on="game_id", how="inner")
        if not chunk.empty:
            chunk["club_form_recency_weight"] = _recency_weight(chunk["date"], config)
            chunk["weighted_goals"] = (
                chunk["competition_goal_weight"] * chunk["competition_goals_per90_factor"]
                * chunk["club_form_recency_weight"]
            )
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(
            {
                "transfermarkt_player_id": list(player_ids),
                "event_goals": 0,
                "event_weighted_goals": 0.0,
            }
        )

    goals = pd.concat(chunks, ignore_index=True)
    return (
        goals.groupby("player_id", as_index=False)
        .agg(event_goals=("player_id", "size"), event_weighted_goals=("weighted_goals", "sum"))
        .rename(columns={"player_id": "transfermarkt_player_id"})
    )


def _latest_valuations(
    path: Path,
    player_ids: set[int],
    config: TransfermarktConfig,
) -> pd.DataFrame:
    if not player_ids:
        return _empty_player_aggregate("transfermarkt_player_id")

    chunks = []
    columns = ["player_id", "date", "market_value_in_eur"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=config.chunk_size):
        chunk = chunk[chunk["player_id"].isin(player_ids)]
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk[chunk["date"] <= pd.Timestamp(config.season_end)]
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(
            {"transfermarkt_player_id": list(player_ids), "latest_market_value_in_eur": np.nan}
        )

    valuations = pd.concat(chunks, ignore_index=True).sort_values("date")
    return (
        valuations.groupby("player_id", as_index=False)
        .tail(1)
        .rename(
            columns={
                "player_id": "transfermarkt_player_id",
                "market_value_in_eur": "latest_market_value_in_eur",
                "date": "latest_market_value_date",
            }
        )
    )


def _transfermarkt_multiplier(enriched: pd.DataFrame) -> pd.Series:
    matched = enriched["transfermarkt_match_quality"] != "unmatched"
    has_recent_appearances = enriched["club_minutes_model"] > 0
    coverage = enriched["transfermarkt_recent_appearance_coverage"].fillna(0.0)

    rate = pd.Series(enriched["club_weighted_goals_per90"], dtype="float64")
    median_rate = rate[has_recent_appearances].median()
    if pd.isna(median_rate):
        median_rate = 0.0
    shrunk_rate = coverage * rate.fillna(median_rate) + (1.0 - coverage) * median_rate

    goals_per90 = _standardized_signal(shrunk_rate)
    minutes = _standardized_signal(np.log1p(enriched["club_minutes"]))
    market_value = _standardized_signal(np.log1p(enriched["latest_market_value_in_eur"]))

    multiplier = 1.0 + coverage * (0.18 * goals_per90 + 0.08 * minutes) + 0.06 * market_value
    multiplier = multiplier.clip(lower=0.70, upper=1.45)
    multiplier.loc[(~matched) | (~has_recent_appearances)] = 1.0
    return multiplier


def _competition_context(base_dir: Path, config: TransfermarktConfig) -> pd.DataFrame:
    competitions = pd.read_csv(
        base_dir / "competitions.csv",
        usecols=["competition_id", "type", "sub_type", "country_name", "name"],
    )
    competitions = competitions[competitions["type"] != "national_team_competition"].copy()
    appearances = []
    columns = ["competition_id", "date", "goals", "minutes_played"]
    for chunk in pd.read_csv(base_dir / "appearances.csv", usecols=columns, chunksize=config.chunk_size):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk[
            (chunk["date"] >= pd.Timestamp(config.season_start))
            & (chunk["date"] <= pd.Timestamp(config.season_end))
        ]
        if not chunk.empty:
            appearances.append(chunk)

    if appearances:
        app = pd.concat(appearances, ignore_index=True)
        scoring = (
            app.groupby("competition_id", as_index=False)
            .agg(total_goals=("goals", "sum"), total_minutes=("minutes_played", "sum"))
        )
        scoring["raw_goals_per90"] = np.where(
            scoring["total_minutes"] > 0,
            90.0 * scoring["total_goals"] / scoring["total_minutes"],
            np.nan,
        )
        global_rate = scoring["raw_goals_per90"].median()
        scoring["competition_goals_per90_factor"] = (
            global_rate / scoring["raw_goals_per90"]
        ).clip(lower=0.80, upper=1.20)
    else:
        scoring = pd.DataFrame(columns=["competition_id", "competition_goals_per90_factor"])

    context = competitions.merge(
        scoring[["competition_id", "competition_goals_per90_factor"]],
        on="competition_id",
        how="left",
    )
    context["competition_goals_per90_factor"] = context["competition_goals_per90_factor"].fillna(1.0)
    context["competition_goal_weight"] = context.apply(_competition_goal_weight, axis=1)
    return context


def _competition_goal_weight(row: pd.Series) -> float:
    competition_id = str(row["competition_id"])
    competition_type = str(row["type"])
    sub_type = str(row["sub_type"])
    country = normalize_name(row["country_name"])

    if competition_id == "CL":
        return 1.15
    if competition_id in {"EL", "UCOL"}:
        return 1.05
    if competition_type == "international_cup":
        return 1.10
    if sub_type == "first_tier" and country in {"england", "spain", "germany", "italy", "france"}:
        return 1.00
    if sub_type == "first_tier" and country in {"portugal", "netherlands", "belgium"}:
        return 0.90
    if sub_type == "first_tier" and country in {"saudi arabia", "united states", "mexico", "japan", "turkey"}:
        return 0.75
    if competition_type == "domestic_cup":
        return 0.85
    return 0.80


def _club_game_context(base_dir: Path, config: TransfermarktConfig) -> set[int]:
    games = pd.read_csv(base_dir / "games.csv", usecols=["game_id", "date", "competition_type"])
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games[
        (games["date"] >= pd.Timestamp(config.season_start))
        & (games["date"] <= pd.Timestamp(config.season_end))
        & (games["competition_type"] != "national_team_competition")
    ]
    return set(games["game_id"])


def _recency_weight(dates: pd.Series, config: TransfermarktConfig) -> pd.Series:
    season_end = pd.Timestamp(config.season_end)
    days_old = (season_end - pd.to_datetime(dates, errors="raise")).dt.days.clip(lower=0)
    return 0.5 ** (days_old / config.club_form_half_life_days)


def _standardized_signal(values: pd.Series) -> pd.Series:
    signal = pd.Series(values, dtype="float64")
    signal = signal.fillna(signal.median()).fillna(0.0)
    std = signal.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=signal.index)
    return (signal - signal.median()) / std


def _matched_candidate(candidate: object, player: pd.Series) -> dict[str, object]:
    return {
        **candidate._asdict(),
        "transfermarkt_player_id": int(player["player_id"]),
        "transfermarkt_player_name": player["name"],
        "transfermarkt_country": player["country_of_citizenship"],
        "transfermarkt_current_club": player["current_club_name"],
        "transfermarkt_position": player["position"],
        "transfermarkt_sub_position": player["sub_position"],
        "transfermarkt_international_caps": player["international_caps"],
        "transfermarkt_international_goals": player["international_goals"],
        "transfermarkt_profile_market_value_in_eur": player["market_value_in_eur"],
        "transfermarkt_highest_market_value_in_eur": player["highest_market_value_in_eur"],
        "transfermarkt_match_score": player["match_score"],
        "transfermarkt_match_quality": _match_quality(player["match_score"]),
    }


def _unmatched_candidate(candidate: object) -> dict[str, object]:
    return {
        **candidate._asdict(),
        "transfermarkt_player_id": np.nan,
        "transfermarkt_player_name": pd.NA,
        "transfermarkt_country": pd.NA,
        "transfermarkt_current_club": pd.NA,
        "transfermarkt_position": pd.NA,
        "transfermarkt_sub_position": pd.NA,
        "transfermarkt_international_caps": np.nan,
        "transfermarkt_international_goals": np.nan,
        "transfermarkt_profile_market_value_in_eur": np.nan,
        "transfermarkt_highest_market_value_in_eur": np.nan,
        "transfermarkt_match_score": 0,
        "transfermarkt_match_quality": "unmatched",
    }


def _match_quality(score: int) -> str:
    if score >= 7:
        return "player_country_club"
    if score >= 4:
        return "player_country"
    if score >= 3:
        return "player_club"
    return "player_only"


def _team_aliases(team_key: str) -> set[str]:
    return TEAM_COUNTRY_ALIASES.get(team_key, {team_key})


def _clubs_overlap(first: str, second: str) -> bool:
    if not first or not second:
        return False
    return first in second or second in first


def _empty_player_aggregate(id_column: str) -> pd.DataFrame:
    return pd.DataFrame({id_column: []})
