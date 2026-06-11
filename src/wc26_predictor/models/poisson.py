"""Regularized independent-Poisson score model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from wc26_predictor.data.schema import Fixture, OutcomeProbabilities, validate_results_frame
from wc26_predictor.features.tournament_importance import tournament_importance_weight


@dataclass(frozen=True, slots=True)
class PoissonConfig:
    """Configuration for the moment-based Poisson model."""

    prior_strength: float = 6.0
    max_goals: int = 10
    recency_half_life_days: int | None = 730
    use_tournament_importance: bool = True
    min_expected_goals: float = 0.05
    max_expected_goals: float = 6.0


@dataclass(frozen=True, slots=True)
class ScorePrediction:
    """Predicted score distribution and derived 1X2 probabilities."""

    home_team: str
    away_team: str
    home_expected_goals: float
    away_expected_goals: float
    score_matrix: np.ndarray
    outcome_probabilities: OutcomeProbabilities


class IndependentPoissonModel:
    """Estimate attack/defense strengths with shrinkage and predict scorelines."""

    def __init__(self, config: PoissonConfig | None = None) -> None:
        self.config = config or PoissonConfig()
        self.global_goal_rate_: float | None = None
        self.home_goal_multiplier_: float | None = None
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}

    def fit(self, results: pd.DataFrame, as_of: date | None = None) -> "IndependentPoissonModel":
        """Fit team attack and defense multipliers from match results."""

        validated = validate_results_frame(results)
        if validated.empty:
            raise ValueError("Cannot fit Poisson model on an empty result history.")

        reference_date = as_of or validated["date"].max()
        weights = _recency_weights(
            dates=validated["date"],
            reference_date=reference_date,
            half_life_days=self.config.recency_half_life_days,
        )
        if self.config.use_tournament_importance:
            weights = weights * validated["tournament"].map(tournament_importance_weight)

        team_rows = self._team_match_rows(validated, weights)
        weighted_matches = team_rows["weight"].sum()
        if weighted_matches <= 0:
            raise ValueError("Total observation weight must be positive.")

        self.global_goal_rate_ = float(
            (team_rows["goals_for"] * team_rows["weight"]).sum() / weighted_matches
        )
        self.home_goal_multiplier_ = self._estimate_home_goal_multiplier(team_rows)

        team_stats = team_rows.groupby("team", as_index=True).agg(
            weighted_goals_for=("weighted_goals_for", "sum"),
            weighted_goals_against=("weighted_goals_against", "sum"),
            weight=("weight", "sum"),
        )

        prior = self.config.prior_strength
        baseline = self.global_goal_rate_
        for team, row in team_stats.iterrows():
            exposure = float(row["weight"])
            attack_rate = (float(row["weighted_goals_for"]) + prior * baseline) / (
                exposure + prior
            )
            defense_rate = (float(row["weighted_goals_against"]) + prior * baseline) / (
                exposure + prior
            )
            self.attack_[str(team)] = attack_rate / baseline
            self.defense_[str(team)] = defense_rate / baseline

        return self

    def predict(self, fixture: Fixture) -> ScorePrediction:
        """Predict scoreline and 1X2 probabilities for a fixture."""

        self._check_is_fitted()
        home_lambda, away_lambda = self.expected_goals(fixture)
        matrix = _independent_poisson_matrix(home_lambda, away_lambda, self.config.max_goals)
        outcomes = _outcomes_from_score_matrix(matrix)
        return ScorePrediction(
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            home_expected_goals=home_lambda,
            away_expected_goals=away_lambda,
            score_matrix=matrix,
            outcome_probabilities=outcomes,
        )

    def predict_outcome(self, fixture: Fixture) -> OutcomeProbabilities:
        """Predict only 1X2 probabilities for evaluation interfaces."""

        return self.predict(fixture).outcome_probabilities

    def expected_goals(self, fixture: Fixture) -> tuple[float, float]:
        """Return expected goals for home and away teams."""

        self._check_is_fitted()
        baseline = self.global_goal_rate_
        home_multiplier = 1.0 if fixture.neutral else self.home_goal_multiplier_
        home_lambda = baseline * self._attack(fixture.home_team) * self._defense(fixture.away_team)
        away_lambda = baseline * self._attack(fixture.away_team) * self._defense(fixture.home_team)

        home_lambda *= home_multiplier
        away_lambda /= home_multiplier
        return (
            float(np.clip(home_lambda, self.config.min_expected_goals, self.config.max_expected_goals)),
            float(np.clip(away_lambda, self.config.min_expected_goals, self.config.max_expected_goals)),
        )

    def _attack(self, team: str) -> float:
        return self.attack_.get(team, 1.0)

    def _defense(self, team: str) -> float:
        return self.defense_.get(team, 1.0)

    def _check_is_fitted(self) -> None:
        if self.global_goal_rate_ is None or self.home_goal_multiplier_ is None:
            raise RuntimeError("Fit the Poisson model before predicting.")

    @staticmethod
    def _team_match_rows(results: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
        home = pd.DataFrame(
            {
                "team": results["home_team"],
                "goals_for": results["home_score"],
                "goals_against": results["away_score"],
                "is_home": ~results["neutral"],
                "weight": weights,
            }
        )
        away = pd.DataFrame(
            {
                "team": results["away_team"],
                "goals_for": results["away_score"],
                "goals_against": results["home_score"],
                "is_home": False,
                "weight": weights,
            }
        )
        team_rows = pd.concat([home, away], ignore_index=True)
        team_rows["weighted_goals_for"] = team_rows["goals_for"] * team_rows["weight"]
        team_rows["weighted_goals_against"] = team_rows["goals_against"] * team_rows["weight"]
        return team_rows

    @staticmethod
    def _estimate_home_goal_multiplier(team_rows: pd.DataFrame) -> float:
        home_rows = team_rows[team_rows["is_home"]]
        away_or_neutral_rows = team_rows[~team_rows["is_home"]]
        if home_rows.empty or away_or_neutral_rows.empty:
            return 1.0

        home_rate = (home_rows["goals_for"] * home_rows["weight"]).sum() / home_rows["weight"].sum()
        other_rate = (
            (away_or_neutral_rows["goals_for"] * away_or_neutral_rows["weight"]).sum()
            / away_or_neutral_rows["weight"].sum()
        )
        if other_rate <= 0:
            return 1.0
        return float(np.clip(np.sqrt(home_rate / other_rate), 0.8, 1.3))


def _recency_weights(
    dates: pd.Series,
    reference_date: date,
    half_life_days: int | None,
) -> pd.Series:
    if half_life_days is None:
        return pd.Series(1.0, index=dates.index)
    if half_life_days <= 0:
        raise ValueError("recency_half_life_days must be positive or None.")

    age_days = dates.map(lambda value: (reference_date - value).days)
    if (age_days < 0).any():
        raise ValueError("Result history contains matches after the requested as_of date.")
    return 0.5 ** (age_days / half_life_days)


def _independent_poisson_matrix(home_lambda: float, away_lambda: float, max_goals: int) -> np.ndarray:
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1.")

    goals = np.arange(max_goals + 1)
    home_probs = _poisson_pmf(goals, home_lambda)
    away_probs = _poisson_pmf(goals, away_lambda)
    matrix = np.outer(home_probs, away_probs)
    return matrix / matrix.sum()


def _poisson_pmf(goals: np.ndarray, expected_goals: float) -> np.ndarray:
    probabilities = np.empty_like(goals, dtype=float)
    probabilities[0] = np.exp(-expected_goals)
    for index in range(1, len(goals)):
        probabilities[index] = probabilities[index - 1] * expected_goals / index
    return probabilities


def _outcomes_from_score_matrix(score_matrix: np.ndarray) -> OutcomeProbabilities:
    home_win = float(np.tril(score_matrix, k=-1).sum())
    draw = float(np.trace(score_matrix))
    away_win = float(np.triu(score_matrix, k=1).sum())
    total = home_win + draw + away_win
    return OutcomeProbabilities(
        home_win=home_win / total,
        draw=draw / total,
        away_win=away_win / total,
    )
