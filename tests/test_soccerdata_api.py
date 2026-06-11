from __future__ import annotations

import pandas as pd

from wc26_predictor.data.soccerdata_api import (
    extract_sidelined_availability,
    resolve_sidelined_to_squad_availability,
    summarize_upcoming_match_previews,
)


def test_summarize_upcoming_match_previews_flattens_groups() -> None:
    payload = [
        {
            "league_id": 1,
            "league_name": "FIFA - World Cup",
            "country": {"name": "world"},
            "match_previews": [
                {
                    "id": 123,
                    "date": "11/06/2026",
                    "time": "20:00",
                    "teams": {
                        "home": {"id": 10, "name": "Team A"},
                        "away": {"id": 20, "name": "Team B"},
                    },
                    "word_count": 400,
                }
            ],
        }
    ]

    previews = summarize_upcoming_match_previews(payload)

    assert len(previews) == 1
    assert previews.loc[0, "match_id"] == 123
    assert previews.loc[0, "league_name"] == "FIFA - World Cup"
    assert previews.loc[0, "home_team"] == "Team A"


def test_extract_sidelined_availability_maps_statuses() -> None:
    payload = {
        "id": 123,
        "teams": {
            "home": {"name": "Team A"},
            "away": {"name": "Team B"},
        },
        "lineups": {
            "sidelined": {
                "home": [
                    {
                        "player": {"id": 1, "name": "Injured Player"},
                        "status": "out",
                        "desc": "Injury",
                    }
                ],
                "away": [
                    {
                        "player": {"id": 2, "name": "Maybe Player"},
                        "status": "questionable",
                        "desc": "Fitness",
                    }
                ],
            }
        },
    }

    availability = extract_sidelined_availability(payload)

    assert availability["status"].tolist() == ["out", "doubtful"]
    assert set(availability["team"]) == {"Team A", "Team B"}


def test_resolve_sidelined_to_squad_availability_matches_abbreviated_names() -> None:
    sidelined = extract_sidelined_availability(
        {
            "id": 123,
            "teams": {"home": {"name": "Scotland"}, "away": {"name": "Haiti"}},
            "lineups": {
                "sidelined": {
                    "home": [
                        {
                            "player": {"id": 1, "name": "B. Gilmour"},
                            "status": "out",
                            "desc": "Knee Injury",
                        }
                    ]
                }
            },
        }
    )
    squads = pd.DataFrame(
        {
            "team": ["Scotland", "Scotland"],
            "player": ["Billy Gilmour", "John McGinn"],
        }
    )

    matched, unmatched = resolve_sidelined_to_squad_availability(sidelined, squads)

    assert unmatched.empty
    assert matched.loc[0, "player"] == "Billy Gilmour"
    assert matched.loc[0, "source_player"] == "B. Gilmour"


def test_resolve_sidelined_to_squad_availability_keeps_unmatched_rows() -> None:
    sidelined = extract_sidelined_availability(
        {
            "id": 123,
            "teams": {"home": {"name": "Scotland"}, "away": {"name": "Haiti"}},
            "lineups": {
                "sidelined": {
                    "home": [
                        {
                            "player": {"id": 1, "name": "B. Gilmour"},
                            "status": "out",
                            "desc": "Knee Injury",
                        }
                    ]
                }
            },
        }
    )
    squads = pd.DataFrame(
        {
            "team": ["Scotland"],
            "player": ["John McGinn"],
        }
    )

    matched, unmatched = resolve_sidelined_to_squad_availability(sidelined, squads)

    assert matched.empty
    assert unmatched.loc[0, "player"] == "B. Gilmour"
