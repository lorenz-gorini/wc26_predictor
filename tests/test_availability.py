from __future__ import annotations

import pandas as pd
import pytest

from wc26_predictor.data.availability import (
    aggregate_team_availability_burden,
    apply_availability_to_top_scorers,
    build_default_player_availability,
    merge_player_availability,
    validate_player_availability_frame,
)


def test_build_default_availability_adds_minutes_and_penalty_rank() -> None:
    squads = pd.DataFrame(
        {
            "team": ["A", "A", "A"],
            "player": ["Starter", "Backup", "Keeper"],
            "position": ["FW", "FW", "GK"],
            "caps": [50, 5, 40],
            "goals": [20, 1, 0],
        }
    )
    goalscorers = pd.DataFrame(
        {
            "team": ["A", "A"],
            "scorer": ["Starter", "Backup"],
            "penalty": [True, False],
        }
    )

    availability = build_default_player_availability(squads, goalscorers)

    starter = availability[availability["player"] == "Starter"].iloc[0]
    assert starter["status"] == "available"
    assert starter["expected_minutes_share"] > 0.5
    assert starter["penalty_taker_rank"] == 1


def test_merge_player_availability_rejects_unknown_player() -> None:
    base = validate_player_availability_frame(
        pd.DataFrame({"team": ["A"], "player": ["Known"], "status": ["available"]})
    )
    override = pd.DataFrame({"team": ["A"], "player": ["Unknown"], "status": ["injured"]})

    with pytest.raises(ValueError, match="unknown squad players"):
        merge_player_availability(base, override)


def test_availability_adjusts_top_scorer_expectations_and_team_burden() -> None:
    top_scorers = pd.DataFrame(
        {
            "team": ["A", "A"],
            "scorer": ["Fit Player", "Injured Player"],
            "expected_group_goals": [1.0, 1.0],
            "expected_tournament_goals": [2.0, 2.0],
        }
    )
    availability = validate_player_availability_frame(
        pd.DataFrame(
            {
                "team": ["A", "A"],
                "player": ["Fit Player", "Injured Player"],
                "status": ["available", "injured"],
                "expected_minutes_share": [1.0, 0.8],
            }
        )
    )

    adjusted = apply_availability_to_top_scorers(top_scorers, availability)
    burden = aggregate_team_availability_burden(availability)

    injured = adjusted[adjusted["scorer"] == "Injured Player"].iloc[0]
    assert injured["expected_tournament_goals_availability_adjusted"] == pytest.approx(0.0)
    assert burden.loc[0, "team_availability_burden"] == pytest.approx(0.8)
