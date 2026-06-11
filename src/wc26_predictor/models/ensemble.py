"""Utilities for calibrated probability ensembles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wc26_predictor.data.schema import OutcomeProbabilities


MODEL_NAMES = ("elo", "poisson", "form_adjusted_poisson")
OUTCOMES = ("home_win", "draw", "away_win")


@dataclass(frozen=True, slots=True)
class EnsembleWeights:
    """Convex weights for the three baseline outcome models."""

    elo: float
    poisson: float
    form_adjusted_poisson: float

    @classmethod
    def equal(cls) -> "EnsembleWeights":
        """Return equal weights."""

        return cls(elo=1.0 / 3.0, poisson=1.0 / 3.0, form_adjusted_poisson=1.0 / 3.0)

    def as_dict(self) -> dict[str, float]:
        return {
            "elo": self.elo,
            "poisson": self.poisson,
            "form_adjusted_poisson": self.form_adjusted_poisson,
        }

    def validate(self) -> None:
        values = np.array(list(self.as_dict().values()), dtype=float)
        if np.any(values < 0):
            raise ValueError("Ensemble weights must be non-negative.")
        if not np.isclose(values.sum(), 1.0):
            raise ValueError(f"Ensemble weights must sum to 1.0, found {values.sum():.6f}.")


def weighted_average_probabilities(
    predictions: dict[str, OutcomeProbabilities],
    weights: EnsembleWeights,
) -> OutcomeProbabilities:
    """Average model probabilities with convex weights."""

    weights.validate()
    missing_models = set(MODEL_NAMES).difference(predictions)
    if missing_models:
        raise ValueError(f"Missing model predictions: {sorted(missing_models)}")

    weight_map = weights.as_dict()
    return OutcomeProbabilities(
        home_win=sum(weight_map[model] * predictions[model].home_win for model in MODEL_NAMES),
        draw=sum(weight_map[model] * predictions[model].draw for model in MODEL_NAMES),
        away_win=sum(weight_map[model] * predictions[model].away_win for model in MODEL_NAMES),
    )


def fit_ensemble_weights(
    validation_predictions: pd.DataFrame,
    step: float = 0.01,
) -> EnsembleWeights:
    """Fit convex ensemble weights by minimizing multiclass log loss on predictions."""

    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1].")
    required_columns = {"observed"} | {
        f"{model}_{outcome}" for model in MODEL_NAMES for outcome in OUTCOMES
    }
    missing = required_columns.difference(validation_predictions.columns)
    if missing:
        raise ValueError(f"Missing validation prediction columns: {sorted(missing)}")
    if validation_predictions.empty:
        raise ValueError("validation_predictions must not be empty.")

    prediction_tensor = np.stack(
        [
            validation_predictions[[f"{model}_{outcome}" for outcome in OUTCOMES]].to_numpy(
                dtype=float
            )
            for model in MODEL_NAMES
        ],
        axis=0,
    )
    observed = validation_predictions["observed"].map(
        {"home_win": 0, "draw": 1, "away_win": 2}
    )
    if observed.isna().any():
        bad_values = sorted(validation_predictions.loc[observed.isna(), "observed"].unique())
        raise ValueError(f"Invalid observed outcomes: {bad_values}")
    observed_array = observed.to_numpy(dtype=int)

    best_loss = np.inf
    best_weights = EnsembleWeights.equal()
    grid_size = int(round(1.0 / step))
    for elo_idx in range(grid_size + 1):
        elo_weight = elo_idx * step
        for poisson_idx in range(grid_size + 1 - elo_idx):
            poisson_weight = poisson_idx * step
            form_weight = max(0.0, 1.0 - elo_weight - poisson_weight)
            weights_array = np.array([elo_weight, poisson_weight, form_weight])
            ensemble = np.tensordot(weights_array, prediction_tensor, axes=(0, 0))
            loss = _log_loss_array(ensemble, observed_array)
            if loss < best_loss:
                best_loss = loss
                best_weights = EnsembleWeights(
                    elo=float(elo_weight),
                    poisson=float(poisson_weight),
                    form_adjusted_poisson=float(form_weight),
                )
    return best_weights


def prediction_frame_to_probabilities(frame: pd.DataFrame, prefix: str) -> list[OutcomeProbabilities]:
    """Convert probability columns in a frame to `OutcomeProbabilities` records."""

    columns = [f"{prefix}_{outcome}" for outcome in OUTCOMES]
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing probability columns: {sorted(missing)}")
    return [
        OutcomeProbabilities(home_win=row[0], draw=row[1], away_win=row[2])
        for row in frame.loc[:, columns].itertuples(index=False, name=None)
    ]


def add_weighted_ensemble_columns(
    frame: pd.DataFrame,
    weights: EnsembleWeights,
    output_prefix: str = "ensemble",
) -> pd.DataFrame:
    """Return a copy with weighted ensemble probability columns."""

    weights.validate()
    result = frame.copy()
    for outcome in OUTCOMES:
        result[f"{output_prefix}_{outcome}"] = sum(
            weights.as_dict()[model] * result[f"{model}_{outcome}"] for model in MODEL_NAMES
        )
    return result


def _log_loss_array(probabilities: np.ndarray, observed: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.log(clipped[np.arange(len(observed)), observed]).mean())
