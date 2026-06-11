from __future__ import annotations

import pandas as pd

from wc26_predictor.data.odds import validate_odds_frame
from wc26_predictor.models.market import (
    evaluate_market_models,
    market_gate_decision,
    match_validation_predictions_to_odds,
)


def test_match_validation_predictions_to_odds_and_evaluate_market() -> None:
    predictions = pd.DataFrame(
        {
            "window": ["w1", "w2"],
            "date": ["2022-12-18", "2022-12-16"],
            "home_team": ["Argentina", "Croatia"],
            "away_team": ["France", "Morocco"],
            "observed": ["home_win", "home_win"],
            "ensemble_home_win": [0.40, 0.42],
            "ensemble_draw": [0.30, 0.30],
            "ensemble_away_win": [0.30, 0.28],
        }
    )
    odds = validate_odds_frame(
        pd.DataFrame(
            {
                "source": ["test", "test"],
                "competition": ["WorldCup2022", "WorldCup2022"],
                "date": ["2022-12-18", "2022-12-17"],
                "home_team": ["Argentina", "Croatia"],
                "away_team": ["France", "Morocco"],
                "home_odds": [2.0, 2.1],
                "draw_odds": [3.5, 3.2],
                "away_odds": [4.0, 4.2],
            }
        )
    )

    matched = match_validation_predictions_to_odds(predictions, odds)
    metrics, weights = evaluate_market_models(matched, step=0.5)
    decision = market_gate_decision(metrics)

    assert len(matched) == 2
    assert set(matched["odds_date_delta_days"]) == {0, 1}
    assert set(metrics["model"]) == {"model_only", "market_only", "model_market_lowo"}
    assert len(weights) == 2
    assert "use_model_market" in decision
