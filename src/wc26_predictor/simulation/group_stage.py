"""Group-stage simulation with football standings rules."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wc26_predictor.data.schema import Fixture
from wc26_predictor.models.poisson import IndependentPoissonModel
from wc26_predictor.simulation.match import SimulatedScore, simulate_score


@dataclass(slots=True)
class TeamStanding:
    team: str
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    wins: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass(frozen=True, slots=True)
class GroupSimulation:
    scores: list[SimulatedScore]
    table: pd.DataFrame


def simulate_group(
    fixtures: list[Fixture],
    model: IndependentPoissonModel,
    rng: np.random.Generator,
) -> GroupSimulation:
    """Simulate a group and return its standings table."""

    if not fixtures:
        raise ValueError("At least one fixture is required.")

    teams = sorted({fixture.home_team for fixture in fixtures} | {fixture.away_team for fixture in fixtures})
    standings = {team: TeamStanding(team=team) for team in teams}
    scores: list[SimulatedScore] = []

    for fixture in fixtures:
        score = simulate_score(model.predict(fixture), rng)
        scores.append(score)
        _apply_score(standings, score)

    table = _standings_frame(standings.values())
    return GroupSimulation(scores=scores, table=table)


def _apply_score(standings: dict[str, TeamStanding], score: SimulatedScore) -> None:
    home = standings[score.home_team]
    away = standings[score.away_team]
    home.goals_for += score.home_score
    home.goals_against += score.away_score
    away.goals_for += score.away_score
    away.goals_against += score.home_score

    if score.home_score > score.away_score:
        home.points += 3
        home.wins += 1
    elif score.away_score > score.home_score:
        away.points += 3
        away.wins += 1
    else:
        home.points += 1
        away.points += 1


def _standings_frame(standings: list[TeamStanding]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "team": item.team,
                "points": item.points,
                "goal_difference": item.goal_difference,
                "goals_for": item.goals_for,
                "goals_against": item.goals_against,
                "wins": item.wins,
            }
            for item in standings
        ]
    )
    return frame.sort_values(
        ["points", "goal_difference", "goals_for", "wins", "team"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

