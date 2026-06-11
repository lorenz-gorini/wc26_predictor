from __future__ import annotations

import pandas as pd
import pytest

from wc26_predictor.data.odds import validate_odds_frame


def test_validate_odds_frame_removes_margin() -> None:
    odds = pd.DataFrame(
        {
            "source": ["test"],
            "competition": ["WorldCup2022"],
            "date": ["2022-12-18"],
            "home_team": ["Argentina"],
            "away_team": ["France"],
            "home_odds": [2.70],
            "draw_odds": [3.10],
            "away_odds": [2.88],
        }
    )

    normalized = validate_odds_frame(odds)

    assert normalized.loc[0, "market_overround"] > 1.0
    assert (
        normalized.loc[0, ["market_home_win", "market_draw", "market_away_win"]].sum()
        == pytest.approx(1.0)
    )
    assert normalized.loc[0, "home_team_key"] == "argentina"
