from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from wc26_predictor.data.availability import validate_player_availability_frame
from wc26_predictor.data.transfermarkt_injuries import (
    TransfermarktInjuryConfig,
    aggregate_transfermarkt_team_injury_burden,
    build_transfermarkt_squad_injury_features,
    validate_transfermarkt_injuries_frame,
)


def test_validate_transfermarkt_injuries_rejects_negative_days() -> None:
    injuries = pd.DataFrame(
        {
            "player_id": [1],
            "season_name": ["25/26"],
            "injury_reason": ["Hamstring"],
            "from_date": ["2026-01-01"],
            "end_date": ["2026-01-05"],
            "days_missed": [-1],
            "games_missed": [1],
        }
    )

    with pytest.raises(ValueError, match="days_missed"):
        validate_transfermarkt_injuries_frame(injuries)


def test_build_transfermarkt_squad_injury_features(tmp_path) -> None:
    transfermarkt_dir = tmp_path / "transfermarkt"
    transfermarkt_dir.mkdir()
    pd.DataFrame(
        {
            "player_id": [10, 20],
            "name": ["Known Forward", "Other Player"],
            "country_of_citizenship": ["Country A", "Country B"],
            "current_club_name": ["Strong FC", "Other FC"],
            "market_value_in_eur": [100_000_000, 10_000_000],
            "international_caps": [50, 5],
        }
    ).to_csv(transfermarkt_dir / "players.csv", index=False)
    injuries_path = tmp_path / "player_injuries.csv"
    pd.DataFrame(
        {
            "player_id": [10, 10],
            "season_name": ["25/26", "24/25"],
            "injury_reason": ["Knee injury", "Ankle injury"],
            "from_date": ["2026-05-20", "2025-01-01"],
            "end_date": ["", "2025-01-20"],
            "days_missed": [23, 20],
            "games_missed": [4, 3],
        }
    ).to_csv(injuries_path, index=False)
    squads = pd.DataFrame(
        {
            "team": ["Country A", "Country A"],
            "player": ["Known Forward", "Unmatched Player"],
            "club": ["Strong FC", "Unknown FC"],
            "position": ["FW", "MF"],
            "caps": [50, 1],
            "goals": [20, 0],
        }
    )
    availability = validate_player_availability_frame(
        pd.DataFrame(
            {
                "team": ["Country A", "Country A"],
                "player": ["Known Forward", "Unmatched Player"],
                "status": ["available", "available"],
                "expected_minutes_share": [0.9, 0.2],
            }
        )
    )

    features = build_transfermarkt_squad_injury_features(
        squads=squads,
        availability=availability,
        transfermarkt_dir=transfermarkt_dir,
        injuries_path=injuries_path,
        config=TransfermarktInjuryConfig(
            reference_date=date(2026, 6, 11),
            source_snapshot_date=date(2026, 6, 11),
            lookback_days=60,
        ),
    )
    burden = aggregate_transfermarkt_team_injury_burden(features)

    known = features[features["player"] == "Known Forward"].iloc[0]
    unmatched = features[features["player"] == "Unmatched Player"].iloc[0]
    assert known["transfermarkt_player_id"] == 10
    assert known["active_injury_flag"] == 1
    assert known["recent_injury_days"] == pytest.approx(23.0)
    assert known["current_open_injury_burden"] == pytest.approx(0.9)
    assert unmatched["transfermarkt_match_quality"] == "unmatched"
    assert burden.loc[0, "players_with_open_injury"] == 1
    assert burden.loc[0, "team_current_open_injury_burden"] == pytest.approx(0.9)
