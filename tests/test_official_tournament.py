from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc26_predictor.data.ingest_results import load_results_csv
from wc26_predictor.simulation.official_tournament import (
    load_third_place_assignments,
    simulate_official_knockout_match_forecasts,
    simulate_official_tournament,
)


def test_third_place_assignments_load_known_annex_rows() -> None:
    assignments = load_third_place_assignments()

    assert len(assignments) == 495
    assert assignments["EFGHIJKL"] == {
        "1A": "E",
        "1B": "J",
        "1D": "I",
        "1E": "F",
        "1G": "H",
        "1I": "G",
        "1K": "L",
        "1L": "K",
    }
    assert assignments["ABCDEFGH"] == {
        "1A": "H",
        "1B": "G",
        "1D": "B",
        "1E": "C",
        "1G": "A",
        "1I": "F",
        "1K": "D",
        "1L": "E",
    }


def test_official_tournament_probabilities_are_coherent() -> None:
    results = load_results_csv(Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_results.csv")
    forecasts = _synthetic_forecasts()

    probabilities = simulate_official_tournament(
        results,
        forecasts,
        n_simulations=200,
        seed=7,
    )

    assert len(probabilities) == 48
    assert round(probabilities["round_of_32_probability"].sum(), 6) == 32.0
    assert round(probabilities["round_of_16_probability"].sum(), 6) == 16.0
    assert round(probabilities["quarter_final_probability"].sum(), 6) == 8.0
    assert round(probabilities["semi_final_probability"].sum(), 6) == 4.0
    assert round(probabilities["final_probability"].sum(), 6) == 2.0
    assert round(probabilities["third_place_match_probability"].sum(), 6) == 2.0
    assert round(probabilities["winner_probability"].sum(), 6) == 1.0
    assert probabilities["expected_team_matches"].between(3.0, 8.0).all()


def test_official_knockout_match_forecasts_are_conditional_pairings() -> None:
    results = load_results_csv(Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_results.csv")
    forecasts = _synthetic_forecasts()

    knockout = simulate_official_knockout_match_forecasts(
        results,
        forecasts,
        n_simulations=100,
        seed=11,
    )

    assert not knockout.empty
    assert set(knockout["round"]).issuperset({"round_of_32", "final", "third_place"})
    assert knockout["pairing_probability"].between(0.0, 1.0).all()
    assert knockout["first_advancement_probability"].between(0.0, 1.0).all()
    assert knockout.groupby("match_number")["pairing_probability"].sum().round(6).eq(1.0).all()


def _synthetic_forecasts() -> pd.DataFrame:
    rows = []
    match_number = 1
    for group in "ABCDEFGHIJKL":
        teams = [f"{group}{index}" for index in range(1, 5)]
        for home_index in range(4):
            for away_index in range(home_index + 1, 4):
                rows.append(
                    {
                        "group": group,
                        "match_number": match_number,
                        "home_team": teams[home_index],
                        "away_team": teams[away_index],
                        "form_poisson_home_expected_goals": 1.25,
                        "form_poisson_away_expected_goals": 1.05,
                    }
                )
                match_number += 1
    return pd.DataFrame(rows)
