"""Match-level simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wc26_predictor.models.poisson import ScorePrediction


@dataclass(frozen=True, slots=True)
class SimulatedScore:
    home_team: str
    away_team: str
    home_score: int
    away_score: int

    @property
    def winner(self) -> str | None:
        if self.home_score > self.away_score:
            return self.home_team
        if self.away_score > self.home_score:
            return self.away_team
        return None


def simulate_score(prediction: ScorePrediction, rng: np.random.Generator) -> SimulatedScore:
    """Draw one scoreline from a predicted score matrix."""

    flat_index = int(rng.choice(prediction.score_matrix.size, p=prediction.score_matrix.ravel()))
    home_score, away_score = np.unravel_index(flat_index, prediction.score_matrix.shape)
    return SimulatedScore(
        home_team=prediction.home_team,
        away_team=prediction.away_team,
        home_score=int(home_score),
        away_score=int(away_score),
    )

