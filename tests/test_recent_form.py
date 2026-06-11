from __future__ import annotations

from pathlib import Path

from wc26_predictor.data.ingest_results import load_results_csv
from wc26_predictor.features.recent_form import RecentFormTable
from wc26_predictor.features.tournament_importance import tournament_importance_weight


def test_recent_form_table_returns_team_indices() -> None:
    results = load_results_csv(Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_results.csv")

    form = RecentFormTable(results, window_matches=5).get("Argentina")

    assert form.matches > 0
    assert form.attack_index > 0
    assert form.defensive_vulnerability_index > 0


def test_tournament_importance_orders_world_cup_above_friendly() -> None:
    assert tournament_importance_weight("FIFA World Cup") > tournament_importance_weight("Friendly")

