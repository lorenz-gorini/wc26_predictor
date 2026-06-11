from __future__ import annotations

import pandas as pd
import pytest

from wc26_predictor.data.schema import validate_results_frame


def test_validate_results_frame_normalizes_expected_types() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "home_team": [" A "],
            "away_team": ["B"],
            "home_score": [2],
            "away_score": [1],
            "tournament": ["Friendly"],
            "city": ["Rome"],
            "country": ["Italy"],
            "neutral": ["FALSE"],
        }
    )

    validated = validate_results_frame(raw)

    assert validated.loc[0, "home_team"] == "A"
    assert bool(validated.loc[0, "neutral"]) is False


def test_validate_results_frame_raises_on_same_team() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "home_team": ["A"],
            "away_team": ["A"],
            "home_score": [0],
            "away_score": [0],
            "tournament": ["Friendly"],
            "city": ["Rome"],
            "country": ["Italy"],
            "neutral": ["TRUE"],
        }
    )

    with pytest.raises(ValueError, match="identical"):
        validate_results_frame(raw)
