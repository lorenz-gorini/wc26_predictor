"""Command-line entry points for smoke tests and demonstrations."""

from __future__ import annotations

from pathlib import Path

from wc26_predictor.data.ingest_results import load_results_csv
from wc26_predictor.data.schema import Fixture
from wc26_predictor.models.elo import EloRatings
from wc26_predictor.models.poisson import IndependentPoissonModel


def run_demo() -> None:
    """Train baselines on the sample data and print one forecast."""

    project_root = Path(__file__).resolve().parents[2]
    results = load_results_csv(project_root / "data" / "raw" / "sample_results.csv")
    fixture = Fixture(home_team="Argentina", away_team="Brazil", neutral=True)

    elo = EloRatings().fit(results)
    elo_prediction = elo.predict_outcome(fixture)

    poisson = IndependentPoissonModel().fit(results)
    score_prediction = poisson.predict(fixture)

    print("Fixture: Argentina vs Brazil, neutral venue")
    print(f"Elo 1X2: {elo_prediction.as_dict()}")
    print(
        "Poisson expected goals: "
        f"{score_prediction.home_expected_goals:.2f} - {score_prediction.away_expected_goals:.2f}"
    )
    print(f"Poisson 1X2: {score_prediction.outcome_probabilities.as_dict()}")


if __name__ == "__main__":
    run_demo()

