from __future__ import annotations

from pathlib import Path

import numpy as np

from wc26_predictor.data.ingest_results import load_results_csv
from wc26_predictor.data.schema import Fixture
from wc26_predictor.models.poisson import IndependentPoissonModel
from wc26_predictor.simulation.group_stage import simulate_group


def test_group_simulation_returns_complete_table() -> None:
    results = load_results_csv(Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_results.csv")
    model = IndependentPoissonModel().fit(results)
    fixtures = [
        Fixture("Argentina", "Brazil"),
        Fixture("Argentina", "Germany"),
        Fixture("Argentina", "Japan"),
        Fixture("Brazil", "Germany"),
        Fixture("Brazil", "Japan"),
        Fixture("Germany", "Japan"),
    ]

    simulation = simulate_group(fixtures, model, np.random.default_rng(2026))

    assert len(simulation.scores) == 6
    assert list(simulation.table.columns) == [
        "team",
        "points",
        "goal_difference",
        "goals_for",
        "goals_against",
        "wins",
    ]
    assert len(simulation.table) == 4

