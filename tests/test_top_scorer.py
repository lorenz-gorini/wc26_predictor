from __future__ import annotations

import pandas as pd

from wc26_predictor.models.top_scorer import NationalTeamTopScorerModel


def test_top_scorer_model_allocates_team_expected_goals() -> None:
    goalscorers = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "home_team": ["A", "A", "B"],
            "away_team": ["B", "C", "A"],
            "team": ["A", "A", "B"],
            "scorer": ["Player One", "Player One", "Player Two"],
            "minute": [10, 20, 30],
            "own_goal": [False, False, False],
            "penalty": [False, True, False],
        }
    )
    forecasts = pd.DataFrame(
        {
            "home_team": ["A"],
            "away_team": ["B"],
            "form_poisson_home_expected_goals": [2.0],
            "form_poisson_away_expected_goals": [1.0],
        }
    )

    predictions = (
        NationalTeamTopScorerModel()
        .fit(goalscorers, eligible_teams={"A", "B"})
        .predict_group_top_scorers(forecasts)
    )

    assert list(predictions["scorer"]) == ["Player One", "Player Two"]
    assert predictions.loc[0, "expected_group_goals"] > predictions.loc[1, "expected_group_goals"]


def test_top_scorer_model_filters_to_squad_players_and_expected_matches() -> None:
    goalscorers = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-02-01"],
            "home_team": ["A", "A"],
            "away_team": ["B", "C"],
            "team": ["A", "A"],
            "scorer": ["Included Player", "Excluded Player"],
            "minute": [10, 20],
            "own_goal": [False, False],
            "penalty": [False, False],
        }
    )
    squads = pd.DataFrame(
        {
            "team": ["A"],
            "player": ["Included Player"],
        }
    )
    forecasts = pd.DataFrame(
        {
            "home_team": ["A"],
            "away_team": ["B"],
            "form_poisson_home_expected_goals": [1.5],
            "form_poisson_away_expected_goals": [0.5],
        }
    )
    team_expected_matches = pd.DataFrame({"team": ["A", "B"], "expected_team_matches": [5.0, 3.0]})

    predictions = (
        NationalTeamTopScorerModel()
        .fit(goalscorers, eligible_teams={"A", "B"}, squads=squads)
        .predict_group_top_scorers(forecasts, team_expected_matches=team_expected_matches)
    )

    assert list(predictions["scorer"]) == ["Included Player"]
    assert predictions.loc[0, "expected_tournament_goals"] > predictions.loc[0, "expected_group_goals"]
