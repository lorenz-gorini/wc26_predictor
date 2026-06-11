from __future__ import annotations

from pathlib import Path

import numpy as np

from wc26_predictor.data.ingest_results import load_results_csv
from wc26_predictor.data.schema import Fixture
from wc26_predictor.models.elo import EloRatings
from wc26_predictor.models.poisson import IndependentPoissonModel


def _sample_results_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_results.csv"


def test_elo_prediction_is_valid_probability_vector() -> None:
    results = load_results_csv(_sample_results_path())
    model = EloRatings().fit(results)

    probabilities = model.predict_outcome(Fixture("Argentina", "Brazil", neutral=True))

    assert 0.0 <= probabilities.home_win <= 1.0
    assert 0.0 <= probabilities.draw <= 1.0
    assert 0.0 <= probabilities.away_win <= 1.0
    assert probabilities.home_win + probabilities.draw + probabilities.away_win == 1.0


def test_poisson_score_matrix_is_valid_probability_distribution() -> None:
    results = load_results_csv(_sample_results_path())
    model = IndependentPoissonModel().fit(results)

    prediction = model.predict(Fixture("Argentina", "Brazil", neutral=True))

    assert prediction.home_expected_goals > 0
    assert prediction.away_expected_goals > 0
    assert np.isclose(prediction.score_matrix.sum(), 1.0)
    outcomes = prediction.outcome_probabilities
    assert np.isclose(outcomes.home_win + outcomes.draw + outcomes.away_win, 1.0)

