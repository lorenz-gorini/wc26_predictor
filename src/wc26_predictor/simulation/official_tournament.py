"""Official-bracket World Cup 2026 tournament simulation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from importlib.resources import files

import numpy as np
import pandas as pd

from wc26_predictor.data.schema import validate_results_frame
from wc26_predictor.models.elo import EloConfig, EloRatings
from wc26_predictor.simulation.abstract_tournament import PreparedGroup, _prepare_groups


THIRD_PLACE_SLOTS = ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")

ROUND_OF_32 = {
    73: ("2A", "2B"),
    74: ("1E", "3:1E"),
    75: ("1F", "2C"),
    76: ("1C", "2F"),
    77: ("1I", "3:1I"),
    78: ("2E", "2I"),
    79: ("1A", "3:1A"),
    80: ("1L", "3:1L"),
    81: ("1D", "3:1D"),
    82: ("1G", "3:1G"),
    83: ("2K", "2L"),
    84: ("1H", "2J"),
    85: ("1B", "3:1B"),
    86: ("1J", "2H"),
    87: ("1K", "3:1K"),
    88: ("2D", "2G"),
}

ROUND_OF_16 = {
    89: (73, 75),
    90: (74, 77),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
}

QUARTER_FINALS = {
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
}

SEMI_FINALS = {
    101: (97, 98),
    102: (99, 100),
}

FINAL = {103: (101, 102)}
THIRD_PLACE = {104: (101, 102)}

ROUND_NAMES = {
    **{match_number: "round_of_32" for match_number in ROUND_OF_32},
    **{match_number: "round_of_16" for match_number in ROUND_OF_16},
    **{match_number: "quarter_final" for match_number in QUARTER_FINALS},
    **{match_number: "semi_final" for match_number in SEMI_FINALS},
    103: "final",
    104: "third_place",
}


@dataclass(frozen=True, slots=True)
class KnockoutResult:
    winner: str
    loser: str


def simulate_official_tournament(
    results: pd.DataFrame,
    forecasts: pd.DataFrame,
    n_simulations: int = 2000,
    seed: int = 2026,
    advancement_probabilities: dict[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    """Estimate advancement with the official 2026 knockout bracket."""

    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    validated_results = validate_results_frame(results)
    prepared_groups = _prepare_groups(forecasts)
    teams = sorted({team for group in prepared_groups for team in group.teams})
    knockout_probabilities = advancement_probabilities or _default_elo_probability_cache(
        teams, validated_results
    )
    third_place_assignments = load_third_place_assignments()

    rng = np.random.default_rng(seed)
    round_of_32_counts: Counter[str] = Counter()
    round_of_16_counts: Counter[str] = Counter()
    quarter_counts: Counter[str] = Counter()
    semi_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    third_place_counts: Counter[str] = Counter()
    winner_counts: Counter[str] = Counter()

    for _ in range(n_simulations):
        group_positions, best_third_key = _simulate_group_positions(prepared_groups, rng)
        third_place_assignment = third_place_assignments[best_third_key]

        round_of_32_teams = [
            team
            for fixture in ROUND_OF_32.values()
            for team in _resolve_fixture(fixture, group_positions, third_place_assignment)
        ]
        round_of_32_counts.update(round_of_32_teams)

        match_results: dict[int, KnockoutResult] = {}
        _simulate_match_round(
            ROUND_OF_32,
            group_positions,
            third_place_assignment,
            match_results,
            knockout_probabilities,
            rng,
        )
        round_of_16_counts.update(match_results[match_number].winner for match_number in ROUND_OF_32)

        _simulate_winner_round(ROUND_OF_16, match_results, knockout_probabilities, rng)
        quarter_counts.update(match_results[match_number].winner for match_number in ROUND_OF_16)

        _simulate_winner_round(QUARTER_FINALS, match_results, knockout_probabilities, rng)
        semi_counts.update(match_results[match_number].winner for match_number in QUARTER_FINALS)

        _simulate_winner_round(SEMI_FINALS, match_results, knockout_probabilities, rng)
        finalists = [match_results[match_number].winner for match_number in SEMI_FINALS]
        third_place_teams = [match_results[match_number].loser for match_number in SEMI_FINALS]
        final_counts.update(finalists)
        third_place_counts.update(third_place_teams)

        final_result = _simulate_knockout_match(
            finalists[0],
            finalists[1],
            knockout_probabilities,
            rng,
        )
        winner_counts.update([final_result.winner])

    return pd.DataFrame(
        {
            "team": teams,
            "round_of_32_probability": [round_of_32_counts[team] / n_simulations for team in teams],
            "round_of_16_probability": [round_of_16_counts[team] / n_simulations for team in teams],
            "quarter_final_probability": [quarter_counts[team] / n_simulations for team in teams],
            "semi_final_probability": [semi_counts[team] / n_simulations for team in teams],
            "final_probability": [final_counts[team] / n_simulations for team in teams],
            "third_place_match_probability": [
                third_place_counts[team] / n_simulations for team in teams
            ],
            "winner_probability": [winner_counts[team] / n_simulations for team in teams],
            "expected_knockout_matches": [
                (
                    round_of_32_counts[team]
                    + round_of_16_counts[team]
                    + quarter_counts[team]
                    + semi_counts[team]
                    + final_counts[team]
                    + third_place_counts[team]
                )
                / n_simulations
                for team in teams
            ],
            "expected_team_matches": [
                3.0
                + (
                    round_of_32_counts[team]
                    + round_of_16_counts[team]
                    + quarter_counts[team]
                    + semi_counts[team]
                    + final_counts[team]
                    + third_place_counts[team]
                )
                / n_simulations
                for team in teams
            ],
        }
    ).sort_values("winner_probability", ascending=False, ignore_index=True)


def simulate_official_knockout_match_forecasts(
    results: pd.DataFrame,
    forecasts: pd.DataFrame,
    n_simulations: int = 2000,
    seed: int = 2026,
    advancement_probabilities: dict[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    """Estimate conditional knockout pairings and advancement probabilities."""

    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    validated_results = validate_results_frame(results)
    prepared_groups = _prepare_groups(forecasts)
    teams = sorted({team for group in prepared_groups for team in group.teams})
    knockout_probabilities = advancement_probabilities or _default_elo_probability_cache(
        teams, validated_results
    )
    third_place_assignments = load_third_place_assignments()

    rng = np.random.default_rng(seed)
    pairing_counts: Counter[tuple[int, str, str]] = Counter()
    first_advancement_counts: Counter[tuple[int, str, str]] = Counter()

    for _ in range(n_simulations):
        group_positions, best_third_key = _simulate_group_positions(prepared_groups, rng)
        third_place_assignment = third_place_assignments[best_third_key]
        match_results: dict[int, KnockoutResult] = {}

        _simulate_match_forecast_round(
            ROUND_OF_32,
            group_positions,
            third_place_assignment,
            match_results,
            knockout_probabilities,
            pairing_counts,
            first_advancement_counts,
            rng,
        )
        _simulate_winner_forecast_round(
            ROUND_OF_16,
            match_results,
            knockout_probabilities,
            pairing_counts,
            first_advancement_counts,
            rng,
        )
        _simulate_winner_forecast_round(
            QUARTER_FINALS,
            match_results,
            knockout_probabilities,
            pairing_counts,
            first_advancement_counts,
            rng,
        )
        _simulate_winner_forecast_round(
            SEMI_FINALS,
            match_results,
            knockout_probabilities,
            pairing_counts,
            first_advancement_counts,
            rng,
        )
        _simulate_winner_forecast_round(
            FINAL,
            match_results,
            knockout_probabilities,
            pairing_counts,
            first_advancement_counts,
            rng,
        )
        _simulate_third_place_forecast(
            match_results,
            knockout_probabilities,
            pairing_counts,
            first_advancement_counts,
            rng,
        )

    rows = []
    for (match_number, first_team, second_team), count in pairing_counts.items():
        first_advances = knockout_probabilities[(first_team, second_team)]
        rows.append(
            {
                "match_number": match_number,
                "round": ROUND_NAMES[match_number],
                "first_team": first_team,
                "second_team": second_team,
                "pairing_probability": count / n_simulations,
                "first_advancement_probability": first_advances,
                "second_advancement_probability": 1.0 - first_advances,
                "simulation_count": count,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["match_number", "pairing_probability", "first_team", "second_team"],
        ascending=[True, False, True, True],
        ignore_index=True,
    )


def load_third_place_assignments() -> dict[str, dict[str, str]]:
    """Load FIFA Annex C third-place assignments from the packaged CSV."""

    resource = files("wc26_predictor.simulation").joinpath(
        "data", "third_place_assignments_2026.csv"
    )
    assignments = pd.read_csv(resource)
    required_columns = {"qualified_third_groups", *THIRD_PLACE_SLOTS}
    missing = required_columns.difference(assignments.columns)
    if missing:
        raise ValueError(f"Missing third-place assignment columns: {sorted(missing)}")
    if len(assignments) != 495:
        raise ValueError(f"Expected 495 third-place assignments, found {len(assignments)}.")

    return {
        str(row["qualified_third_groups"]): {slot: str(row[slot]) for slot in THIRD_PLACE_SLOTS}
        for _, row in assignments.iterrows()
    }


def _simulate_group_positions(
    groups: list[PreparedGroup],
    rng: np.random.Generator,
) -> tuple[dict[str, str], str]:
    positions = {}
    third_place_rows: list[tuple[int, int, int, str, str]] = []
    for group in groups:
        table = _simulate_group(group, rng)
        positions[f"1{group.name}"] = table[0][4]
        positions[f"2{group.name}"] = table[1][4]
        positions[f"3{group.name}"] = table[2][4]
        third_place_rows.append((*table[2][:3], group.name, table[2][4]))

    third_place_rows = sorted(
        third_place_rows,
        key=lambda row: (-row[0], -row[1], -row[2], row[3]),
    )
    best_third_key = "".join(sorted(row[3] for row in third_place_rows[:8]))
    return positions, best_third_key


def _simulate_group(group: PreparedGroup, rng: np.random.Generator) -> list[tuple[int, int, int, int, str]]:
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
        (
            int(points[index]),
            int(goal_difference[index]),
            int(goals_for[index]),
            index,
            group.teams[index],
        )
        for index in range(n_teams)
    ]
    return sorted(rows, key=lambda row: (-row[0], -row[1], -row[2], row[3]))


def _simulate_match_round(
    match_specs: dict[int, tuple[str, str]],
    group_positions: dict[str, str],
    third_place_assignment: dict[str, str],
    match_results: dict[int, KnockoutResult],
    elo_probabilities: dict[tuple[str, str], float],
    rng: np.random.Generator,
) -> None:
    for match_number, fixture in match_specs.items():
        first, second = _resolve_fixture(fixture, group_positions, third_place_assignment)
        match_results[match_number] = _simulate_knockout_match(
            first,
            second,
            elo_probabilities,
            rng,
        )


def _simulate_winner_round(
    match_specs: dict[int, tuple[int, int]],
    match_results: dict[int, KnockoutResult],
    elo_probabilities: dict[tuple[str, str], float],
    rng: np.random.Generator,
) -> None:
    for match_number, previous_matches in match_specs.items():
        first = match_results[previous_matches[0]].winner
        second = match_results[previous_matches[1]].winner
        match_results[match_number] = _simulate_knockout_match(
            first,
            second,
            elo_probabilities,
            rng,
        )


def _simulate_match_forecast_round(
    match_specs: dict[int, tuple[str, str]],
    group_positions: dict[str, str],
    third_place_assignment: dict[str, str],
    match_results: dict[int, KnockoutResult],
    advancement_probabilities: dict[tuple[str, str], float],
    pairing_counts: Counter[tuple[int, str, str]],
    first_advancement_counts: Counter[tuple[int, str, str]],
    rng: np.random.Generator,
) -> None:
    for match_number, fixture in match_specs.items():
        first, second = _resolve_fixture(fixture, group_positions, third_place_assignment)
        _record_and_simulate_match(
            match_number,
            first,
            second,
            match_results,
            advancement_probabilities,
            pairing_counts,
            first_advancement_counts,
            rng,
        )


def _simulate_winner_forecast_round(
    match_specs: dict[int, tuple[int, int]],
    match_results: dict[int, KnockoutResult],
    advancement_probabilities: dict[tuple[str, str], float],
    pairing_counts: Counter[tuple[int, str, str]],
    first_advancement_counts: Counter[tuple[int, str, str]],
    rng: np.random.Generator,
) -> None:
    for match_number, previous_matches in match_specs.items():
        first = match_results[previous_matches[0]].winner
        second = match_results[previous_matches[1]].winner
        _record_and_simulate_match(
            match_number,
            first,
            second,
            match_results,
            advancement_probabilities,
            pairing_counts,
            first_advancement_counts,
            rng,
        )


def _simulate_third_place_forecast(
    match_results: dict[int, KnockoutResult],
    advancement_probabilities: dict[tuple[str, str], float],
    pairing_counts: Counter[tuple[int, str, str]],
    first_advancement_counts: Counter[tuple[int, str, str]],
    rng: np.random.Generator,
) -> None:
    first = match_results[101].loser
    second = match_results[102].loser
    _record_and_simulate_match(
        104,
        first,
        second,
        match_results,
        advancement_probabilities,
        pairing_counts,
        first_advancement_counts,
        rng,
    )


def _record_and_simulate_match(
    match_number: int,
    first: str,
    second: str,
    match_results: dict[int, KnockoutResult],
    advancement_probabilities: dict[tuple[str, str], float],
    pairing_counts: Counter[tuple[int, str, str]],
    first_advancement_counts: Counter[tuple[int, str, str]],
    rng: np.random.Generator,
) -> None:
    key = (match_number, first, second)
    pairing_counts.update([key])
    result = _simulate_knockout_match(first, second, advancement_probabilities, rng)
    if result.winner == first:
        first_advancement_counts.update([key])
    match_results[match_number] = result


def _resolve_fixture(
    fixture: tuple[str, str],
    group_positions: dict[str, str],
    third_place_assignment: dict[str, str],
) -> tuple[str, str]:
    return tuple(
        _resolve_entrant(entrant, group_positions, third_place_assignment) for entrant in fixture
    )


def _resolve_entrant(
    entrant: str,
    group_positions: dict[str, str],
    third_place_assignment: dict[str, str],
) -> str:
    if entrant.startswith("3:"):
        slot = entrant.removeprefix("3:")
        return group_positions[f"3{third_place_assignment[slot]}"]
    return group_positions[entrant]


def _default_elo_probability_cache(
    teams: list[str],
    results: pd.DataFrame,
) -> dict[tuple[str, str], float]:
    elo = EloRatings(config=EloConfig()).fit(results)
    probabilities = {}
    for first in teams:
        for second in teams:
            if first != second:
                probabilities[(first, second)] = elo.expected_home_score(first, second, neutral=True)
    return probabilities


def _simulate_knockout_match(
    first: str,
    second: str,
    elo_probabilities: dict[tuple[str, str], float],
    rng: np.random.Generator,
) -> KnockoutResult:
    if rng.random() < elo_probabilities[(first, second)]:
        return KnockoutResult(winner=first, loser=second)
    return KnockoutResult(winner=second, loser=first)
