"""Player availability, expected-minutes, and penalty-role features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wc26_predictor.data.club_form import normalize_name


AVAILABILITY_COLUMNS = {
    "team",
    "player",
    "status",
    "expected_minutes_share",
    "penalty_taker_rank",
    "reason",
    "return_date",
    "source_url",
}

STATUS_AVAILABILITY_MULTIPLIER = {
    "available": 1.0,
    "doubtful": 0.5,
    "injured": 0.0,
    "suspended": 0.0,
    "out": 0.0,
}


def load_player_availability_csv(path_or_url: str | Path) -> pd.DataFrame:
    """Load current player availability rows from CSV."""

    raw = pd.read_csv(path_or_url)
    return validate_player_availability_frame(raw)


def validate_player_availability_frame(availability: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a player availability dataframe."""

    required = {"team", "player", "status"}
    missing_required = required.difference(availability.columns)
    if missing_required:
        raise ValueError(f"Missing availability columns: {sorted(missing_required)}")

    normalized = availability.copy()
    for column in AVAILABILITY_COLUMNS.difference(normalized.columns):
        normalized[column] = pd.NA
    normalized = normalized.loc[:, sorted(AVAILABILITY_COLUMNS)].copy()

    for column in ["team", "player", "status"]:
        normalized[column] = normalized[column].astype("string").str.strip()
        if normalized[column].isna().any() or (normalized[column] == "").any():
            raise ValueError(f"Column {column!r} contains missing or empty values.")
    normalized["status"] = normalized["status"].str.lower()
    unknown_status = sorted(set(normalized["status"]).difference(STATUS_AVAILABILITY_MULTIPLIER))
    if unknown_status:
        raise ValueError(f"Unknown availability status values: {unknown_status}")

    normalized["expected_minutes_share"] = pd.to_numeric(
        normalized["expected_minutes_share"],
        errors="coerce",
    )
    invalid_minutes = normalized["expected_minutes_share"].notna() & ~normalized[
        "expected_minutes_share"
    ].between(0.0, 1.0)
    if invalid_minutes.any():
        raise ValueError("expected_minutes_share must be in [0, 1] when provided.")

    normalized["penalty_taker_rank"] = pd.to_numeric(
        normalized["penalty_taker_rank"],
        errors="coerce",
    )
    invalid_rank = normalized["penalty_taker_rank"].notna() & (
        normalized["penalty_taker_rank"] < 1
    )
    if invalid_rank.any():
        raise ValueError("penalty_taker_rank must be positive when provided.")

    normalized["return_date"] = pd.to_datetime(normalized["return_date"], errors="coerce").dt.date
    for column in ["reason", "source_url"]:
        normalized[column] = normalized[column].astype("string").fillna("").str.strip()

    normalized["team_key"] = normalized["team"].map(normalize_name)
    normalized["player_key"] = normalized["player"].map(normalize_name)
    normalized["availability_multiplier"] = normalized["status"].map(
        STATUS_AVAILABILITY_MULTIPLIER
    )
    return normalized.sort_values(["team", "player"]).reset_index(drop=True)


def build_default_player_availability(
    squads: pd.DataFrame,
    goalscorers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a neutral availability table with expected-minutes and penalty-role priors."""

    required = {"team", "player", "position", "caps", "goals"}
    missing = required.difference(squads.columns)
    if missing:
        raise ValueError(f"Missing squad columns for availability features: {sorted(missing)}")

    availability = squads.loc[:, ["team", "player", "position", "caps", "goals"]].copy()
    availability["status"] = "available"
    availability["reason"] = ""
    availability["return_date"] = pd.NA
    availability["source_url"] = ""
    availability["expected_minutes_share"] = _estimate_expected_minutes_share(availability)
    availability["penalty_taker_rank"] = _penalty_taker_ranks(availability, goalscorers)
    return validate_player_availability_frame(availability)


def merge_player_availability(
    base_availability: pd.DataFrame,
    overrides: pd.DataFrame | None,
) -> pd.DataFrame:
    """Apply manual availability overrides to a default availability table."""

    base = validate_player_availability_frame(base_availability)
    if overrides is None:
        return base
    override = validate_player_availability_frame(overrides)
    merged = base.set_index(["team_key", "player_key"])
    updates = override.set_index(["team_key", "player_key"])
    unknown = updates.index.difference(merged.index)
    if len(unknown) > 0:
        unknown_rows = updates.loc[unknown, ["team", "player"]].reset_index(drop=True)
        raise ValueError(f"Availability overrides include unknown squad players: {unknown_rows.to_dict('records')}")

    update_columns = [
        "status",
        "expected_minutes_share",
        "penalty_taker_rank",
        "reason",
        "return_date",
        "source_url",
        "availability_multiplier",
    ]
    merged.update(updates[update_columns])
    return validate_player_availability_frame(merged.reset_index(drop=True))


def apply_availability_to_top_scorers(
    top_scorers: pd.DataFrame,
    availability: pd.DataFrame,
) -> pd.DataFrame:
    """Add availability and expected-minutes adjustments to scorer predictions."""

    required_top = {"team", "scorer", "expected_tournament_goals", "expected_group_goals"}
    missing_top = required_top.difference(top_scorers.columns)
    if missing_top:
        raise ValueError(f"Missing top-scorer columns: {sorted(missing_top)}")

    avail = validate_player_availability_frame(availability)
    candidates = top_scorers.copy()
    candidates["team_key"] = candidates["team"].map(normalize_name)
    candidates["player_key"] = candidates["scorer"].map(normalize_name)
    candidates = candidates.merge(
        avail[
            [
                "team_key",
                "player_key",
                "status",
                "expected_minutes_share",
                "penalty_taker_rank",
                "reason",
                "return_date",
                "availability_multiplier",
            ]
        ],
        on=["team_key", "player_key"],
        how="left",
    )
    candidates["status"] = candidates["status"].fillna("available")
    candidates["availability_multiplier"] = candidates["availability_multiplier"].fillna(1.0)
    candidates["expected_minutes_share"] = candidates["expected_minutes_share"].fillna(1.0)
    candidates["expected_minutes_multiplier"] = (
        0.5 + 0.5 * candidates["expected_minutes_share"]
    )
    candidates["penalty_role_multiplier"] = candidates["penalty_taker_rank"].map(
        lambda rank: _penalty_multiplier(rank)
    )
    candidates["penalty_role_multiplier"] = candidates["penalty_role_multiplier"].fillna(1.0)
    candidates["availability_expected_goals_multiplier"] = (
        candidates["availability_multiplier"]
        * candidates["expected_minutes_multiplier"]
        * candidates["penalty_role_multiplier"]
    )
    candidates["expected_group_goals_availability_adjusted"] = (
        candidates["expected_group_goals"]
        * candidates["availability_expected_goals_multiplier"]
    )
    candidates["expected_tournament_goals_availability_adjusted"] = (
        candidates["expected_tournament_goals"]
        * candidates["availability_expected_goals_multiplier"]
    )
    return candidates.sort_values(
        "expected_tournament_goals_availability_adjusted",
        ascending=False,
        ignore_index=True,
    )


def aggregate_team_availability_burden(availability: pd.DataFrame) -> pd.DataFrame:
    """Aggregate unavailable expected-minutes share to team-level injury burden."""

    avail = validate_player_availability_frame(availability)
    avail["unavailable_minutes_share"] = (
        1.0 - avail["availability_multiplier"]
    ) * avail["expected_minutes_share"].fillna(0.0)
    return (
        avail.groupby("team", as_index=False)
        .agg(
            team_availability_burden=("unavailable_minutes_share", "sum"),
            unavailable_players=("availability_multiplier", lambda values: int((values < 1).sum())),
        )
        .assign(team_availability_burden=lambda frame: frame["team_availability_burden"].clip(0.0, 3.0))
        .sort_values("team")
        .reset_index(drop=True)
    )


def adjust_fixture_forecasts_for_team_availability(
    forecasts: pd.DataFrame,
    team_availability: pd.DataFrame,
    goal_penalty_per_burden: float = 0.08,
) -> pd.DataFrame:
    """Apply conservative expected-goal penalties from team availability burden."""

    if goal_penalty_per_burden < 0 or goal_penalty_per_burden > 0.5:
        raise ValueError("goal_penalty_per_burden must be in [0, 0.5].")
    required = {
        "home_team",
        "away_team",
        "form_poisson_home_expected_goals",
        "form_poisson_away_expected_goals",
    }
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")

    burden = team_availability.loc[:, ["team", "team_availability_burden"]].copy()
    adjusted = forecasts.merge(
        burden.rename(
            columns={
                "team": "home_team",
                "team_availability_burden": "home_availability_burden",
            }
        ),
        on="home_team",
        how="left",
    ).merge(
        burden.rename(
            columns={
                "team": "away_team",
                "team_availability_burden": "away_availability_burden",
            }
        ),
        on="away_team",
        how="left",
    )
    adjusted[["home_availability_burden", "away_availability_burden"]] = adjusted[
        ["home_availability_burden", "away_availability_burden"]
    ].fillna(0.0)
    adjusted["home_availability_goal_multiplier"] = (
        1.0 - goal_penalty_per_burden * adjusted["home_availability_burden"]
    ).clip(lower=0.75)
    adjusted["away_availability_goal_multiplier"] = (
        1.0 - goal_penalty_per_burden * adjusted["away_availability_burden"]
    ).clip(lower=0.75)
    adjusted["availability_adjusted_home_expected_goals"] = (
        adjusted["form_poisson_home_expected_goals"]
        * adjusted["home_availability_goal_multiplier"]
    )
    adjusted["availability_adjusted_away_expected_goals"] = (
        adjusted["form_poisson_away_expected_goals"]
        * adjusted["away_availability_goal_multiplier"]
    )
    return adjusted


def _estimate_expected_minutes_share(squad: pd.DataFrame) -> pd.Series:
    features = squad.copy()
    features["caps"] = pd.to_numeric(features["caps"], errors="raise")
    features["goals"] = pd.to_numeric(features["goals"], errors="raise")
    features["squad_importance"] = features["caps"] + 3.0 * features["goals"]
    shares = []
    for _, group in features.groupby("team", sort=False):
        ranks = group["squad_importance"].rank(method="first", ascending=False)
        team_size = len(group)
        raw = 1.0 - (ranks - 1.0) / max(team_size - 1.0, 1.0)
        position = group["position"].astype(str).str.upper()
        expected = 0.25 + 0.70 * raw
        expected = expected.mask((position == "GK") & (ranks > 1), 0.08)
        expected = expected.mask((position == "GK") & (ranks == 1), 1.0)
        shares.append(expected.clip(0.05, 1.0))
    return pd.concat(shares).sort_index()


def _penalty_taker_ranks(
    squad: pd.DataFrame,
    goalscorers: pd.DataFrame | None,
) -> pd.Series:
    if goalscorers is None or goalscorers.empty or "penalty" not in goalscorers.columns:
        return pd.Series(pd.NA, index=squad.index, dtype="Int64")

    penalties = goalscorers[goalscorers["penalty"]].copy()
    if penalties.empty:
        return pd.Series(pd.NA, index=squad.index, dtype="Int64")
    penalties["team_key"] = penalties["team"].map(normalize_name)
    penalties["player_key"] = penalties["scorer"].map(normalize_name)
    penalty_counts = (
        penalties.groupby(["team_key", "player_key"], as_index=False)
        .size()
        .rename(columns={"size": "penalty_goals"})
    )

    players = squad.copy()
    players["team_key"] = players["team"].map(normalize_name)
    players["player_key"] = players["player"].map(normalize_name)
    players = players.merge(penalty_counts, on=["team_key", "player_key"], how="left")
    players["penalty_goals"] = players["penalty_goals"].fillna(0)
    ranks = players.groupby("team_key")["penalty_goals"].rank(
        method="first",
        ascending=False,
    )
    ranks = ranks.where(players["penalty_goals"] > 0, pd.NA)
    return ranks.astype("Int64").set_axis(squad.index)


def _penalty_multiplier(rank: object) -> float:
    if pd.isna(rank):
        return 1.0
    rank_value = int(rank)
    if rank_value == 1:
        return 1.06
    if rank_value == 2:
        return 1.03
    return 1.0
