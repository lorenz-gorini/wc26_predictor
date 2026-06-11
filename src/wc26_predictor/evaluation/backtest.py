"""Rolling-origin backtesting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd

from wc26_predictor.data.schema import Fixture, OutcomeProbabilities, validate_results_frame
from wc26_predictor.evaluation.metrics import brier_score, multiclass_log_loss, observed_outcome


class OutcomeModel(Protocol):
    def fit(self, results: pd.DataFrame) -> "OutcomeModel":
        ...

    def predict_outcome(self, fixture: Fixture) -> OutcomeProbabilities:
        ...


@dataclass(frozen=True, slots=True)
class BacktestResult:
    n_matches: int
    log_loss: float
    brier_score: float


def evaluate_holdout(
    model: OutcomeModel,
    results: pd.DataFrame,
    train_until: date,
) -> BacktestResult:
    """Fit on matches through `train_until` and score later matches."""

    validated = validate_results_frame(results)
    train = validated[validated["date"] <= train_until]
    holdout = validated[validated["date"] > train_until]
    if train.empty:
        raise ValueError("Training split is empty.")
    if holdout.empty:
        raise ValueError("Holdout split is empty.")

    fitted = model.fit(train.copy())
    predictions = [
        fitted.predict_outcome(
            Fixture(
                home_team=row.home_team,
                away_team=row.away_team,
                neutral=bool(row.neutral),
                tournament=row.tournament,
                city=row.city,
                country=row.country,
            )
        )
        for row in holdout.itertuples(index=False)
    ]
    observed = [observed_outcome(row.home_score, row.away_score) for row in holdout.itertuples()]
    return BacktestResult(
        n_matches=len(holdout),
        log_loss=multiclass_log_loss(predictions, observed),
        brier_score=brier_score(predictions, observed),
    )

