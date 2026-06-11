from __future__ import annotations

import pandas as pd
import pytest

from wc26_predictor.data.schema import OutcomeProbabilities
from wc26_predictor.models.ensemble import (
    EnsembleWeights,
    fit_ensemble_weights,
    weighted_average_probabilities,
)


def test_fit_ensemble_weights_prefers_best_validation_model() -> None:
    validation = pd.DataFrame(
        {
            "observed": ["home_win", "away_win", "draw"],
            "elo_home_win": [0.90, 0.05, 0.10],
            "elo_draw": [0.05, 0.05, 0.80],
            "elo_away_win": [0.05, 0.90, 0.10],
            "poisson_home_win": [0.34, 0.33, 0.33],
            "poisson_draw": [0.33, 0.34, 0.33],
            "poisson_away_win": [0.33, 0.33, 0.34],
            "form_adjusted_poisson_home_win": [0.40, 0.30, 0.30],
            "form_adjusted_poisson_draw": [0.30, 0.40, 0.30],
            "form_adjusted_poisson_away_win": [0.30, 0.30, 0.40],
        }
    )

    weights = fit_ensemble_weights(validation, step=0.1)

    assert weights.elo == pytest.approx(1.0)
    assert weights.poisson == pytest.approx(0.0)
    assert weights.form_adjusted_poisson == pytest.approx(0.0)


def test_weighted_average_probabilities_uses_convex_weights() -> None:
    probabilities = weighted_average_probabilities(
        {
            "elo": OutcomeProbabilities(0.6, 0.2, 0.2),
            "poisson": OutcomeProbabilities(0.3, 0.4, 0.3),
            "form_adjusted_poisson": OutcomeProbabilities(0.2, 0.3, 0.5),
        },
        EnsembleWeights(elo=0.5, poisson=0.3, form_adjusted_poisson=0.2),
    )

    assert probabilities.home_win == pytest.approx(0.43)
    assert probabilities.draw == pytest.approx(0.28)
    assert probabilities.away_win == pytest.approx(0.29)
