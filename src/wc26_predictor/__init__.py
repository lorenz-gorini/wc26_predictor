"""World Cup 2026 probabilistic forecasting package."""

from wc26_predictor.data.schema import Fixture, MatchResult
from wc26_predictor.models.elo import EloConfig, EloRatings
from wc26_predictor.models.poisson import IndependentPoissonModel, PoissonConfig

__all__ = [
    "EloConfig",
    "EloRatings",
    "Fixture",
    "IndependentPoissonModel",
    "MatchResult",
    "PoissonConfig",
]

