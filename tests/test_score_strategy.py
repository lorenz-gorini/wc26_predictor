from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wc26_predictor.evaluation.score_strategy import (
    STRATEGY_1X2_COMPATIBLE,
    STRATEGY_SCORE_ONLY,
    select_compatible_scoreline,
    summarize_score_strategy_predictions,
)


def test_select_compatible_scoreline_respects_requested_outcome() -> None:
    matrix = np.array(
        [
            [0.20, 0.05, 0.01],
            [0.30, 0.10, 0.03],
            [0.25, 0.04, 0.02],
        ]
    )
    matrix = matrix / matrix.sum()

    scoreline, outcome_mass = select_compatible_scoreline(matrix, "home_win")

    assert scoreline.label == "1-0"
    assert outcome_mass == pytest.approx(matrix[1, 0] + matrix[2, 0] + matrix[2, 1])


def test_summarize_score_strategy_predictions_scores_both_strategies() -> None:
    predictions = pd.DataFrame(
        {
            "window": ["w1", "w1"],
            "home_score": [1, 0],
            "away_score": [0, 0],
            "observed": ["home_win", "draw"],
            "score_only_home_score": [1, 1],
            "score_only_away_score": [0, 0],
            "score_only_score_probability": [0.2, 0.3],
            "score_only_observed_score_probability": [0.2, 0.1],
            "compatible_home_score": [1, 0],
            "compatible_away_score": [0, 0],
            "compatible_score_probability": [0.2, 0.4],
            "compatible_observed_score_probability": [0.2, 0.4],
            "compatible_outcome": ["home_win", "draw"],
        }
    )

    summary = summarize_score_strategy_predictions(predictions)

    assert set(summary["strategy"]) == {STRATEGY_SCORE_ONLY, STRATEGY_1X2_COMPATIBLE}
    all_windows = summary[summary["window"] == "all_windows"]
    compatible = all_windows[all_windows["strategy"] == STRATEGY_1X2_COMPATIBLE].iloc[0]
    score_only = all_windows[all_windows["strategy"] == STRATEGY_SCORE_ONLY].iloc[0]
    assert compatible["exact_score_accuracy"] > score_only["exact_score_accuracy"]
