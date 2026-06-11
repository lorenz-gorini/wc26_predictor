"""Dynamic Elo ratings for international football."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log

import pandas as pd

from wc26_predictor.data.schema import Fixture, OutcomeProbabilities, validate_results_frame


@dataclass(frozen=True, slots=True)
class EloConfig:
    """Hyperparameters for Elo updating."""

    initial_rating: float = 1500.0
    k_factor: float = 30.0
    home_advantage: float = 60.0
    neutral_home_advantage: float = 0.0
    goal_margin_weight: bool = True
    draw_probability: float = 0.26


@dataclass(slots=True)
class EloRatings:
    """Mutable Elo rating store."""

    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)

    def fit(self, results: pd.DataFrame) -> "EloRatings":
        """Update ratings sequentially from a validated result history."""

        validated = validate_results_frame(results)
        for match in validated.itertuples(index=False):
            self.update(
                home_team=match.home_team,
                away_team=match.away_team,
                home_score=int(match.home_score),
                away_score=int(match.away_score),
                neutral=bool(match.neutral),
            )
        return self

    def rating(self, team: str) -> float:
        """Return a team's current rating, falling back to the initial rating."""

        return self.ratings.get(team, self.config.initial_rating)

    def expected_home_score(self, home_team: str, away_team: str, neutral: bool) -> float:
        """Expected Elo match score for the home team on the [0, 1] scale."""

        advantage = self.config.neutral_home_advantage if neutral else self.config.home_advantage
        rating_diff = self.rating(home_team) + advantage - self.rating(away_team)
        return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))

    def predict_outcome(self, fixture: Fixture) -> OutcomeProbabilities:
        """Convert Elo strength into 1X2 probabilities with a fixed draw prior."""

        expected_home = self.expected_home_score(
            fixture.home_team,
            fixture.away_team,
            fixture.neutral,
        )
        draw = self.config.draw_probability
        decisive_mass = 1.0 - draw
        return OutcomeProbabilities(
            home_win=decisive_mass * expected_home,
            draw=draw,
            away_win=decisive_mass * (1.0 - expected_home),
        )

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        neutral: bool,
    ) -> None:
        """Apply one Elo update."""

        if home_team == away_team:
            raise ValueError("home_team and away_team must be different.")
        if home_score < 0 or away_score < 0:
            raise ValueError("Scores must be non-negative.")

        expected_home = self.expected_home_score(home_team, away_team, neutral)
        actual_home = _actual_score(home_score, away_score)
        margin_multiplier = _goal_margin_multiplier(abs(home_score - away_score))
        if not self.config.goal_margin_weight:
            margin_multiplier = 1.0

        delta = self.config.k_factor * margin_multiplier * (actual_home - expected_home)
        self.ratings[home_team] = self.rating(home_team) + delta
        self.ratings[away_team] = self.rating(away_team) - delta


def _actual_score(home_score: int, away_score: int) -> float:
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5


def _goal_margin_multiplier(goal_margin: int) -> float:
    if goal_margin <= 1:
        return 1.0
    return 1.0 + log(goal_margin)

