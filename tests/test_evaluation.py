from __future__ import annotations

import pytest

from wc26_predictor.data.schema import OutcomeProbabilities
from wc26_predictor.evaluation.metrics import brier_score, multiclass_log_loss


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

