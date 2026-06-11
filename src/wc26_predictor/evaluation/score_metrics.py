"""Evaluation metrics for exact-score football forecasts."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def exact_score_log_loss(
    score_matrices: Iterable[np.ndarray],
    observed_scores: Iterable[tuple[int, int]],
    epsilon: float = 1e-15,
) -> float:
    """Average negative log probability assigned to the realized exact score."""

    losses = []
    for matrix, observed in zip(score_matrices, observed_scores, strict=True):
        home_score, away_score = observed
        _validate_observed_score(home_score, away_score)
        _validate_score_matrix(matrix)
        if home_score >= matrix.shape[0] or away_score >= matrix.shape[1]:
            probability = epsilon
        else:
            probability = float(matrix[home_score, away_score])
        losses.append(-np.log(np.clip(probability, epsilon, 1.0)))
    if not losses:
        raise ValueError("At least one score prediction is required.")
    return float(np.mean(losses))


def exact_score_accuracy(
    predicted_scores: Iterable[tuple[int, int]],
    observed_scores: Iterable[tuple[int, int]],
) -> float:
    """Share of matches where the modal exact score equals the realized score."""

    predicted = list(predicted_scores)
    observed = list(observed_scores)
    if not predicted or not observed:
        raise ValueError("At least one score prediction is required.")
    if len(predicted) != len(observed):
        raise ValueError("predicted_scores and observed_scores must have the same length.")
    return float(np.mean([item == truth for item, truth in zip(predicted, observed, strict=True)]))


def goal_mae(
    predicted_scores: Iterable[tuple[int, int]],
    observed_scores: Iterable[tuple[int, int]],
) -> float:
    """Mean absolute error across home and away goals."""

    predicted, observed = _score_arrays(predicted_scores, observed_scores)
    return float(np.abs(predicted - observed).mean())


def total_goal_mae(
    predicted_scores: Iterable[tuple[int, int]],
    observed_scores: Iterable[tuple[int, int]],
) -> float:
    """Mean absolute error for total goals."""

    predicted, observed = _score_arrays(predicted_scores, observed_scores)
    return float(np.abs(predicted.sum(axis=1) - observed.sum(axis=1)).mean())


def goal_rmse(
    predicted_scores: Iterable[tuple[int, int]],
    observed_scores: Iterable[tuple[int, int]],
) -> float:
    """Root mean squared error across home and away goals."""

    predicted, observed = _score_arrays(predicted_scores, observed_scores)
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def _score_arrays(
    predicted_scores: Iterable[tuple[int, int]],
    observed_scores: Iterable[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.array(list(predicted_scores), dtype=int)
    observed = np.array(list(observed_scores), dtype=int)
    if predicted.size == 0 or observed.size == 0:
        raise ValueError("At least one score prediction is required.")
    if predicted.shape != observed.shape:
        raise ValueError("predicted_scores and observed_scores must have the same length.")
    if predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("Scores must be pairs of home and away goals.")
    if (predicted < 0).any() or (observed < 0).any():
        raise ValueError("Scores must be non-negative.")
    return predicted, observed


def _validate_observed_score(home_score: int, away_score: int) -> None:
    if home_score < 0 or away_score < 0:
        raise ValueError("Observed scores must be non-negative.")


def _validate_score_matrix(score_matrix: np.ndarray) -> None:
    if score_matrix.ndim != 2:
        raise ValueError("score matrices must be two-dimensional.")
    if (score_matrix < 0).any():
        raise ValueError("score matrix probabilities must be non-negative.")
    if not np.isclose(score_matrix.sum(), 1.0, atol=1e-8):
        raise ValueError("score matrix probabilities must sum to one.")
