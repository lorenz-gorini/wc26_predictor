"""Proper scoring rules for probabilistic football forecasts."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from wc26_predictor.data.schema import OutcomeProbabilities

OUTCOME_ORDER = ("home_win", "draw", "away_win")


def observed_outcome(home_score: int, away_score: int) -> str:
    """Map a scoreline to `home_win`, `draw`, or `away_win`."""

    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"


def multiclass_log_loss(
    predictions: Iterable[OutcomeProbabilities],
    observed: Iterable[str],
    epsilon: float = 1e-15,
) -> float:
    """Average multiclass log loss."""

    prediction_matrix = _prediction_matrix(predictions)
    observed_indices = _observed_indices(observed)
    chosen_probabilities = prediction_matrix[np.arange(len(observed_indices)), observed_indices]
    return float(-np.log(np.clip(chosen_probabilities, epsilon, 1.0)).mean())


def brier_score(
    predictions: Iterable[OutcomeProbabilities],
    observed: Iterable[str],
) -> float:
    """Average multiclass Brier score."""

    prediction_matrix = _prediction_matrix(predictions)
    observed_indices = _observed_indices(observed)
    observed_matrix = np.zeros_like(prediction_matrix)
    observed_matrix[np.arange(len(observed_indices)), observed_indices] = 1.0
    return float(np.mean(np.sum((prediction_matrix - observed_matrix) ** 2, axis=1)))


def calibration_table(
    predictions: Iterable[OutcomeProbabilities],
    observed: Iterable[str],
    outcome: str = "home_win",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return empirical calibration by probability bin for one outcome."""

    if outcome not in OUTCOME_ORDER:
        raise ValueError(f"Unknown outcome {outcome!r}. Expected one of {OUTCOME_ORDER}.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    prediction_matrix = _prediction_matrix(predictions)
    outcome_index = OUTCOME_ORDER.index(outcome)
    probabilities = prediction_matrix[:, outcome_index]
    observed_values = np.array([item == outcome for item in observed], dtype=float)
    bins = pd.cut(probabilities, bins=np.linspace(0.0, 1.0, n_bins + 1), include_lowest=True)
    return (
        pd.DataFrame({"probability": probabilities, "observed": observed_values, "bin": bins})
        .groupby("bin", observed=True)
        .agg(
            mean_prediction=("probability", "mean"),
            empirical_frequency=("observed", "mean"),
            count=("observed", "size"),
        )
        .reset_index()
    )


def _prediction_matrix(predictions: Iterable[OutcomeProbabilities]) -> np.ndarray:
    rows = [[p.home_win, p.draw, p.away_win] for p in predictions]
    if not rows:
        raise ValueError("At least one prediction is required.")

    matrix = np.array(rows, dtype=float)
    if (matrix < 0).any():
        raise ValueError("Predicted probabilities must be non-negative.")

    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise ValueError("Each prediction must sum to one.")
    return matrix


def _observed_indices(observed: Iterable[str]) -> np.ndarray:
    index = {name: position for position, name in enumerate(OUTCOME_ORDER)}
    values = list(observed)
    if not values:
        raise ValueError("At least one observed outcome is required.")

    unknown = sorted(set(values).difference(index))
    if unknown:
        raise ValueError(f"Unknown observed outcomes: {unknown}")
    return np.array([index[value] for value in values], dtype=int)

