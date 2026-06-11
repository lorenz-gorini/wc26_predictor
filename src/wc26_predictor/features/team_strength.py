"""Team-strength features derived from historical match results."""

from __future__ import annotations

import pandas as pd


def long_team_results(results: pd.DataFrame) -> pd.DataFrame:
    """Convert match-level results into one row per team per match."""

    home = pd.DataFrame(
        {
            "date": results["date"],
            "team": results["home_team"],
            "opponent": results["away_team"],
            "goals_for": results["home_score"],
            "goals_against": results["away_score"],
            "is_home": ~results["neutral"],
        }
    )
    away = pd.DataFrame(
        {
            "date": results["date"],
            "team": results["away_team"],
            "opponent": results["home_team"],
            "goals_for": results["away_score"],
            "goals_against": results["home_score"],
            "is_home": False,
        }
    )
    return pd.concat([home, away], ignore_index=True).sort_values(["date", "team"])

