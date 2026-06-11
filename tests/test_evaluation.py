from __future__ import annotations

import numpy as np
import pytest

from wc26_predictor.data.schema import OutcomeProbabilities
from wc26_predictor.evaluation.metrics import brier_score, multiclass_log_loss
from wc26_predictor.evaluation.score_metrics import (
    exact_score_accuracy,
    exact_score_log_loss,
    goal_mae,
    goal_rmse,
    total_goal_mae,
)


def test_metrics_score_valid_predictions() -> None:
    predictions = [
        OutcomeProbabilities(home_win=0.7, draw=0.2, away_win=0.1),
        OutcomeProbabilities(home_win=0.2, draw=0.3, away_win=0.5),
    ]
    observed = ["home_win", "away_win"]

    assert multiclass_log_loss(predictions, observed) > 0
    assert brier_score(predictions, observed) > 0


def test_metrics_raise_on_probabilities_that_do_not_sum_to_one() -> None:
    predictions = [OutcomeProbabilities(home_win=0.8, draw=0.2, away_win=0.2)]

    with pytest.raises(ValueError, match="sum to one"):
        multiclass_log_loss(predictions, ["home_win"])


def test_score_metrics_score_exact_score_predictions() -> None:
    matrix = np.array([[0.10, 0.20], [0.30, 0.40]])
    matrix = matrix / matrix.sum()
    score_matrices = [matrix, matrix]
    predicted_scores = [(1, 1), (1, 0)]
    observed_scores = [(1, 1), (0, 1)]

    assert exact_score_log_loss(score_matrices, observed_scores) > 0
    assert exact_score_accuracy(predicted_scores, observed_scores) == pytest.approx(0.5)
    assert goal_mae(predicted_scores, observed_scores) == pytest.approx(0.5)
    assert total_goal_mae(predicted_scores, observed_scores) == pytest.approx(0.0)
    assert goal_rmse(predicted_scores, observed_scores) > 0
