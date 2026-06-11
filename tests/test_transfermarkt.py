from __future__ import annotations

import pandas as pd

from wc26_predictor.data.transfermarkt import (
    TransfermarktConfig,
    build_transfermarkt_top_scorer_features,
)


def test_build_transfermarkt_top_scorer_features_matches_and_aggregates(tmp_path) -> None:
    transfermarkt_dir = tmp_path / "tm"
    transfermarkt_dir.mkdir()

    pd.DataFrame(
        {
            "competition_id": ["GB1", "FIWC"],
            "type": ["domestic_league", "national_team_competition"],
            "sub_type": ["first_tier", "world_cup"],
            "country_name": ["England", pd.NA],
            "name": ["premier-league", "world-cup"],
        }
    ).to_csv(transfermarkt_dir / "competitions.csv", index=False)
    pd.DataFrame(
        {
            "player_id": [1],
            "name": ["Example Scorer"],
            "country_of_citizenship": ["England"],
            "current_club_name": ["Example FC"],
            "current_club_id": [10],
            "position": ["Attack"],
            "sub_position": ["Centre-Forward"],
            "international_caps": [20],
            "international_goals": [8],
            "market_value_in_eur": [10_000_000],
            "highest_market_value_in_eur": [12_000_000],
        }
    ).to_csv(transfermarkt_dir / "players.csv", index=False)
    pd.DataFrame(
        {
            "game_id": [100, 101, 102],
            "date": ["2025-08-01", "2025-09-01", "2025-10-01"],
            "competition_type": ["domestic_league", "domestic_league", "national_team_competition"],
            "competition_id": ["GB1", "GB1", "FIWC"],
        }
    ).to_csv(transfermarkt_dir / "games.csv", index=False)
    pd.DataFrame(
        {
            "player_id": [1, 1, 1],
            "date": ["2025-08-01", "2025-09-01", "2025-10-01"],
            "competition_id": ["GB1", "GB1", "FIWC"],
            "goals": [1, 2, 9],
            "assists": [0, 1, 0],
            "minutes_played": [90, 80, 90],
        }
    ).to_csv(transfermarkt_dir / "appearances.csv", index=False)
    pd.DataFrame(
        {
            "player_id": [1, 1],
            "date": ["2025-08-01", "2025-09-01"],
            "game_id": [100, 101],
            "type": ["starting_lineup", "substitutes"],
        }
    ).to_csv(transfermarkt_dir / "game_lineups.csv", index=False)
    pd.DataFrame(
        {
            "player_id": [1, 1],
            "date": ["2025-08-01", "2025-10-01"],
            "game_id": [100, 102],
            "type": ["Goals", "Goals"],
        }
    ).to_csv(transfermarkt_dir / "game_events.csv", index=False)
    pd.DataFrame(
        {
            "player_id": [1, 1],
            "date": ["2025-07-01", "2026-02-01"],
            "market_value_in_eur": [8_000_000, 11_000_000],
        }
    ).to_csv(transfermarkt_dir / "player_valuations.csv", index=False)

    top_scorers = pd.DataFrame(
        {
            "team": ["England"],
            "scorer": ["Example Scorer"],
            "expected_tournament_goals": [1.5],
        }
    )
    squads = pd.DataFrame(
        {
            "team": ["England"],
            "player": ["Example Scorer"],
            "club": ["Example FC"],
            "position": ["FW"],
            "caps": [20],
            "goals": [8],
        }
    )

    features = build_transfermarkt_top_scorer_features(
        top_scorers,
        squads,
        transfermarkt_dir,
        config=TransfermarktConfig(chunk_size=2),
    )

    assert features.loc[0, "transfermarkt_match_quality"] == "player_country_club"
    assert features.loc[0, "club_goals"] == 3
    assert features.loc[0, "club_minutes"] == 170
    assert features.loc[0, "club_starts"] == 1
    assert features.loc[0, "latest_market_value_in_eur"] == 11_000_000
    assert features.loc[0, "expected_tournament_goals_transfermarkt_adjusted"] > 0


def test_transfermarkt_features_use_event_goals_when_appearances_are_missing(tmp_path) -> None:
    transfermarkt_dir = tmp_path / "tm"
    transfermarkt_dir.mkdir()

    pd.DataFrame(
        {
            "competition_id": ["SA1"],
            "type": ["domestic_league"],
            "sub_type": ["first_tier"],
            "country_name": ["Saudi Arabia"],
            "name": ["saudi-pro-league"],
        }
    ).to_csv(transfermarkt_dir / "competitions.csv", index=False)
    pd.DataFrame(
        {
            "player_id": [7],
            "name": ["Event Goal Scorer"],
            "country_of_citizenship": ["Portugal"],
            "current_club_name": ["Example Riyadh"],
            "current_club_id": [77],
            "position": ["Attack"],
            "sub_position": ["Centre-Forward"],
            "international_caps": [200],
            "international_goals": [120],
            "market_value_in_eur": [5_000_000],
            "highest_market_value_in_eur": [100_000_000],
        }
    ).to_csv(transfermarkt_dir / "players.csv", index=False)
    pd.DataFrame(
        {
            "game_id": [700, 701, 702],
            "date": ["2026-06-10", "2026-06-10", "2026-06-10"],
            "competition_type": ["domestic_league", "domestic_league", "domestic_league"],
            "competition_id": ["SA1", "SA1", "SA1"],
        }
    ).to_csv(transfermarkt_dir / "games.csv", index=False)
    pd.DataFrame(
        {
            "player_id": [999],
            "date": ["2026-06-10"],
            "competition_id": ["SA1"],
            "goals": [1],
            "assists": [0],
            "minutes_played": [90],
        }
    ).to_csv(transfermarkt_dir / "appearances.csv", index=False)
    pd.DataFrame(
        {
            "player_id": [7, 7, 7],
            "date": ["2026-06-10", "2026-06-10", "2026-06-10"],
            "game_id": [700, 701, 702],
            "type": ["starting_lineup", "starting_lineup", "starting_lineup"],
        }
    ).to_csv(transfermarkt_dir / "game_lineups.csv", index=False)
    pd.DataFrame(
        {
            "player_id": [7, 7, 7, 7],
            "date": ["2026-06-10", "2026-06-10", "2026-06-10", "2026-06-10"],
            "game_id": [700, 701, 702, 702],
            "type": ["Goals", "Goals", "Goals", "Goals"],
        }
    ).to_csv(transfermarkt_dir / "game_events.csv", index=False)
    pd.DataFrame(
        {
            "player_id": [7],
            "date": ["2026-02-01"],
            "market_value_in_eur": [6_000_000],
        }
    ).to_csv(transfermarkt_dir / "player_valuations.csv", index=False)

    top_scorers = pd.DataFrame(
        {
            "team": ["Portugal"],
            "scorer": ["Event Goal Scorer"],
            "expected_tournament_goals": [1.0],
        }
    )
    squads = pd.DataFrame(
        {
            "team": ["Portugal"],
            "player": ["Event Goal Scorer"],
            "club": ["Example Riyadh"],
            "position": ["FW"],
            "caps": [200],
            "goals": [120],
        }
    )

    features = build_transfermarkt_top_scorer_features(
        top_scorers,
        squads,
        transfermarkt_dir,
        config=TransfermarktConfig(chunk_size=2),
    )

    assert features.loc[0, "club_goals_source"] == "game_events"
    assert features.loc[0, "club_goals_model"] == 4
    assert features.loc[0, "club_weighted_goals_model"] == 3.0
    assert features.loc[0, "club_minutes_estimated"]
    assert features.loc[0, "club_minutes_model"] == 225.0
