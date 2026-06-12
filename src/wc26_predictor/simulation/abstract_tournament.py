"""Fast abstract tournament simulation for exploratory forecasting.

This module intentionally does not encode the official 2026 knockout bracket.
It simulates group qualification correctly at a high level, then uses random
knockout pairings with Elo-based advancement. It is useful for early expected
matches, winner probabilities, and scorer exposure estimates.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wc26_predictor.data.schema import validate_results_frame
from wc26_predictor.models.elo import EloConfig, EloRatings


@dataclass(frozen=True, slots=True)
class PreparedGroup:
    name: str
    teams: tuple[str, ...]
    home_indices: np.ndarray
    away_indices: np.ndarray
    home_lambdas: np.ndarray
    away_lambdas: np.ndarray
    completed_home_scores: np.ndarray
    completed_away_scores: np.ndarray


def simulate_abstract_tournament(
    results: pd.DataFrame,
    forecasts: pd.DataFrame,
    n_simulations: int = 2000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Estimate advancement and winner probabilities from current forecasts."""

    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    validated_results = validate_results_frame(results)
    prepared_groups = _prepare_groups(forecasts)
    teams = sorted({team for group in prepared_groups for team in group.teams})
    elo = EloRatings(config=EloConfig()).fit(validated_results)
    elo_probabilities = _elo_probability_cache(teams, elo)

    rng = np.random.default_rng(seed)
    round_of_32_counts: Counter[str] = Counter()
    quarter_counts: Counter[str] = Counter()
    semi_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    winner_counts: Counter[str] = Counter()
    knockout_match_counts: Counter[str] = Counter()

    for _ in range(n_simulations):
        remaining = _simulate_round_of_32(prepared_groups, rng)
        round_of_32_counts.update(remaining)

        while len(remaining) > 1:
            knockout_match_counts.update(remaining)
            if len(remaining) == 16:
                quarter_counts.update(remaining)
            elif len(remaining) == 8:
                semi_counts.update(remaining)
            elif len(remaining) == 2:
                final_counts.update(remaining)

            shuffled = list(rng.permutation(remaining))
            remaining = [
                _simulate_knockout_winner(first, second, elo_probabilities, rng)
                for first, second in zip(shuffled[0::2], shuffled[1::2], strict=True)
            ]

        winner_counts.update(remaining)

    return pd.DataFrame(
        {
            "team": teams,
            "round_of_32_probability": [round_of_32_counts[team] / n_simulations for team in teams],
            "quarter_final_probability": [quarter_counts[team] / n_simulations for team in teams],
            "semi_final_probability": [semi_counts[team] / n_simulations for team in teams],
            "final_probability": [final_counts[team] / n_simulations for team in teams],
            "winner_probability": [winner_counts[team] / n_simulations for team in teams],
            "expected_knockout_matches": [
                knockout_match_counts[team] / n_simulations for team in teams
            ],
            "expected_team_matches": [
                3.0 + knockout_match_counts[team] / n_simulations for team in teams
            ],
        }
    ).sort_values("winner_probability", ascending=False, ignore_index=True)


def _prepare_groups(forecasts: pd.DataFrame) -> list[PreparedGroup]:
    required_columns = {
        "group",
        "home_team",
        "away_team",
        "form_poisson_home_expected_goals",
        "form_poisson_away_expected_goals",
    }
    missing = required_columns.difference(forecasts.columns)
    if missing:
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")

    groups = []
    for name, group_frame in forecasts.groupby("group"):
        teams = tuple(sorted(set(group_frame["home_team"]).union(group_frame["away_team"])))
        team_index = {team: index for index, team in enumerate(teams)}
        completed_home_scores, completed_away_scores = _completed_scores(group_frame)
        groups.append(
            PreparedGroup(
                name=str(name),
                teams=teams,
                home_indices=group_frame["home_team"].map(team_index).to_numpy(dtype=int),
                away_indices=group_frame["away_team"].map(team_index).to_numpy(dtype=int),
                home_lambdas=group_frame["form_poisson_home_expected_goals"].to_numpy(dtype=float),
                away_lambdas=group_frame["form_poisson_away_expected_goals"].to_numpy(dtype=float),
                completed_home_scores=completed_home_scores,
                completed_away_scores=completed_away_scores,
            )
        )
    if len(groups) != 12:
        raise ValueError(f"Expected 12 groups, found {len(groups)}.")
    return groups


def _simulate_round_of_32(groups: list[PreparedGroup], rng: np.random.Generator) -> list[str]:
    top_two: list[str] = []
    third_place: list[tuple[int, int, int, str]] = []
    for group in groups:
        table = _simulate_group(group, rng)
        top_two.extend([team for *_, team in table[:2]])
        third_place.append(table[2])

    third_place = sorted(third_place, key=lambda row: (-row[0], -row[1], -row[2], row[3]))
    return top_two + [team for *_, team in third_place[:8]]


def _simulate_group(group: PreparedGroup, rng: np.random.Generator) -> list[tuple[int, int, int, str]]:
    n_teams = len(group.teams)
    points = np.zeros(n_teams, dtype=int)
    goals_for = np.zeros(n_teams, dtype=int)
    goals_against = np.zeros(n_teams, dtype=int)

    home_goals = rng.poisson(group.home_lambdas)
    away_goals = rng.poisson(group.away_lambdas)
    completed = ~np.isnan(group.completed_home_scores)
    home_goals = np.where(completed, group.completed_home_scores, home_goals).astype(int)
    away_goals = np.where(completed, group.completed_away_scores, away_goals).astype(int)

    for home_idx, away_idx, home_score, away_score in zip(
        group.home_indices,
        group.away_indices,
        home_goals,
        away_goals,
        strict=True,
    ):
        goals_for[home_idx] += int(home_score)
        goals_against[home_idx] += int(away_score)
        goals_for[away_idx] += int(away_score)
        goals_against[away_idx] += int(home_score)
        if home_score > away_score:
            points[home_idx] += 3
        elif away_score > home_score:
            points[away_idx] += 3
        else:
            points[home_idx] += 1
            points[away_idx] += 1

    goal_difference = goals_for - goals_against
    rows = [
        (int(points[index]), int(goal_difference[index]), int(goals_for[index]), group.teams[index])
        for index in range(n_teams)
    ]
    return sorted(rows, key=lambda row: (-row[0], -row[1], -row[2], row[3]))


def _completed_scores(group_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    has_home = "completed_home_score" in group_frame.columns
    has_away = "completed_away_score" in group_frame.columns
    if not has_home and not has_away:
        empty = np.full(len(group_frame), np.nan)
        return empty, empty.copy()
    required = {"completed_home_score", "completed_away_score"}
    missing = required.difference(group_frame.columns)
    if missing:
        raise ValueError(f"Missing completed-score columns: {sorted(missing)}")

    home_scores = pd.to_numeric(group_frame["completed_home_score"], errors="coerce")
    away_scores = pd.to_numeric(group_frame["completed_away_score"], errors="coerce")
    mismatch = home_scores.isna() != away_scores.isna()
    if mismatch.any():
        raise ValueError("Completed fixture scores must provide both home and away goals.")
    return home_scores.to_numpy(dtype=float), away_scores.to_numpy(dtype=float)


def _elo_probability_cache(teams: list[str], elo: EloRatings) -> dict[tuple[str, str], float]:
    probabilities = {}
    for first in teams:
        for second in teams:
            if first != second:
                probabilities[(first, second)] = elo.expected_home_score(first, second, neutral=True)
    return probabilities


def _simulate_knockout_winner(
    first: str,
    second: str,
    elo_probabilities: dict[tuple[str, str], float],
    rng: np.random.Generator,
) -> str:
    return first if rng.random() < elo_probabilities[(first, second)] else second
