"""High-level tournament Monte Carlo utilities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import numpy as np
import pandas as pd

from wc26_predictor.data.schema import Fixture
from wc26_predictor.models.poisson import IndependentPoissonModel
from wc26_predictor.simulation.group_stage import simulate_group


def simulate_group_advancement(
    groups: Mapping[str, list[Fixture]],
    model: IndependentPoissonModel,
    n_simulations: int,
    seed: int = 2026,
    automatic_places_per_group: int = 2,
) -> pd.DataFrame:
    """Estimate top-N group advancement probabilities.

    Best-third-place logic is intentionally left for a later full World Cup simulator
    because it depends on all group tables jointly and tie-breaking rules.
    """

    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    rng = np.random.default_rng(seed)
    top_counts: Counter[str] = Counter()
    appearances: Counter[str] = Counter()

    for fixtures in groups.values():
        for fixture in fixtures:
            appearances[fixture.home_team] += 0
            appearances[fixture.away_team] += 0

    for _ in range(n_simulations):
        for fixtures in groups.values():
            simulation = simulate_group(fixtures=fixtures, model=model, rng=rng)
            top_teams = simulation.table.head(automatic_places_per_group)["team"]
            top_counts.update(top_teams)

    teams = sorted(appearances)
    return pd.DataFrame(
        {
            "team": teams,
            "automatic_advancement_probability": [
                top_counts[team] / n_simulations for team in teams
            ],
        }
    ).sort_values("automatic_advancement_probability", ascending=False, ignore_index=True)

