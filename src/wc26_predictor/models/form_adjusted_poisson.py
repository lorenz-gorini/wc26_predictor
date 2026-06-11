"""Poisson score model with recent-form adjustments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from wc26_predictor.data.schema import Fixture, OutcomeProbabilities
from wc26_predictor.features.recent_form import RecentFormTable
from wc26_predictor.models.poisson import (
    IndependentPoissonModel,
    PoissonConfig,
    ScorePrediction,
    _independent_poisson_matrix,
    _outcomes_from_score_matrix,
)


@dataclass(frozen=True, slots=True)
class FormAdjustedPoissonConfig:
    """Configuration for recent-form adjustment."""

    poisson: PoissonConfig = PoissonConfig()
    form_window_matches: int = 10
    form_prior_matches: float = 6.0
    form_strength: float = 0.35


class FormAdjustedPoissonModel:
    """Poisson model adjusted by each team's recent attack and defense form."""

    def __init__(self, config: FormAdjustedPoissonConfig | None = None) -> None:
        self.config = config or FormAdjustedPoissonConfig()
        self.base_model = IndependentPoissonModel(config=self.config.poisson)
        self.form_table: RecentFormTable | None = None

    def fit(self, results: pd.DataFrame, as_of: date | None = None) -> "FormAdjustedPoissonModel":
        self.base_model.fit(results, as_of=as_of)
        self.form_table = RecentFormTable(
            results=results,
            as_of=as_of,
            window_matches=self.config.form_window_matches,
            prior_matches=self.config.form_prior_matches,
        )
        return self

    def predict(self, fixture: Fixture) -> ScorePrediction:
        self._check_is_fitted()
        home_lambda, away_lambda = self.expected_goals(fixture)
        matrix = _independent_poisson_matrix(home_lambda, away_lambda, self.config.poisson.max_goals)
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
        return self.predict(fixture).outcome_probabilities

    def expected_goals(self, fixture: Fixture) -> tuple[float, float]:
        self._check_is_fitted()
        home_lambda, away_lambda = self.base_model.expected_goals(fixture)
        home_form = self.form_table.get(fixture.home_team)
        away_form = self.form_table.get(fixture.away_team)

        home_adjustment = (
            home_form.attack_index * away_form.defensive_vulnerability_index
        ) ** self.config.form_strength
        away_adjustment = (
            away_form.attack_index * home_form.defensive_vulnerability_index
        ) ** self.config.form_strength

        return (
            float(
                np.clip(
                    home_lambda * home_adjustment,
                    self.config.poisson.min_expected_goals,
                    self.config.poisson.max_expected_goals,
                )
            ),
            float(
                np.clip(
                    away_lambda * away_adjustment,
                    self.config.poisson.min_expected_goals,
                    self.config.poisson.max_expected_goals,
                )
            ),
        )

    def _check_is_fitted(self) -> None:
        if self.form_table is None:
            raise RuntimeError("Fit the form-adjusted Poisson model before predicting.")

