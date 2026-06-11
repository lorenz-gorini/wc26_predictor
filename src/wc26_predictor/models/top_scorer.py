"""National-team top-scorer baseline model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from wc26_predictor.data.goalscorers import validate_goalscorers_frame


@dataclass(frozen=True, slots=True)
class TopScorerConfig:
    """Configuration for the national-team scorer baseline."""

    recency_half_life_days: int = 730
    prior_team_goals: float = 3.0
    min_player_share: float = 0.005


class NationalTeamTopScorerModel:
    """Allocate forecast team goals to recent national-team scorers."""

    def __init__(self, config: TopScorerConfig | None = None) -> None:
        self.config = config or TopScorerConfig()
        self.player_shares_: pd.DataFrame | None = None

    def fit(
        self,
        goalscorers: pd.DataFrame,
        eligible_teams: set[str],
        squads: pd.DataFrame | None = None,
        as_of: date | None = None,
    ) -> "NationalTeamTopScorerModel":
        validated = validate_goalscorers_frame(goalscorers)
        scorers = validated[
            (~validated["own_goal"]) & (validated["team"].isin(eligible_teams))
        ].copy()
        if scorers.empty:
            raise ValueError("No eligible goalscorer rows after filtering own goals.")

        reference_date = as_of or scorers["date"].max()
        age_days = scorers["date"].map(lambda value: (reference_date - value).days)
        if (age_days < 0).any():
            raise ValueError("Goalscorer history contains rows after the requested as_of date.")

        scorers["weight"] = 0.5 ** (age_days / self.config.recency_half_life_days)
        player_goals = (
            scorers.groupby(["team", "scorer"], as_index=False)
            .agg(weighted_goals=("weight", "sum"), penalties=("penalty", "sum"))
            .sort_values(["team", "weighted_goals"], ascending=[True, False])
        )
        team_totals = player_goals.groupby("team")["weighted_goals"].transform("sum")
        player_goals["goal_share"] = player_goals["weighted_goals"] / (
            team_totals + self.config.prior_team_goals
        )
        if squads is not None:
            eligible_players = squads.loc[:, ["team", "player"]].drop_duplicates()
            player_goals = player_goals.merge(
                eligible_players,
                left_on=["team", "scorer"],
                right_on=["team", "player"],
                how="inner",
            ).drop(columns=["player"])

        player_goals = player_goals[player_goals["goal_share"] >= self.config.min_player_share]
        if player_goals.empty:
            raise ValueError("No eligible squad goalscorer rows after filtering.")
        self.player_shares_ = player_goals.reset_index(drop=True)
        return self

    def predict_group_top_scorers(
        self,
        fixture_forecasts: pd.DataFrame,
        team_expected_matches: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Predict expected tournament goals for known national-team scorers."""

        self._check_is_fitted()
        team_goals = _team_expected_goals_from_fixtures(fixture_forecasts)
        if team_expected_matches is None:
            team_goals["expected_team_matches"] = 3.0
        else:
            team_goals = team_goals.merge(team_expected_matches, on="team", how="left")
            if team_goals["expected_team_matches"].isna().any():
                missing = sorted(team_goals.loc[team_goals["expected_team_matches"].isna(), "team"])
                raise ValueError(f"Missing expected-match estimates for teams: {missing}")

        team_goals["expected_team_goals_per_match"] = (
            team_goals["expected_team_group_goals"] / 3.0
        )
        team_goals["expected_team_tournament_goals"] = (
            team_goals["expected_team_goals_per_match"] * team_goals["expected_team_matches"]
        )

        predictions = self.player_shares_.merge(team_goals, on="team", how="inner")
        predictions["expected_group_goals"] = (
            predictions["goal_share"] * predictions["expected_team_group_goals"]
        )
        predictions["expected_tournament_goals"] = (
            predictions["goal_share"] * predictions["expected_team_tournament_goals"]
        )
        return predictions.sort_values(
            ["expected_tournament_goals", "expected_group_goals", "weighted_goals"],
            ascending=[False, False, False],
            ignore_index=True,
        )

    def _check_is_fitted(self) -> None:
        if self.player_shares_ is None:
            raise RuntimeError("Fit the top-scorer model before predicting.")


def _team_expected_goals_from_fixtures(fixture_forecasts: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "home_team",
        "away_team",
        "form_poisson_home_expected_goals",
        "form_poisson_away_expected_goals",
    }
    missing = required_columns.difference(fixture_forecasts.columns)
    if missing:
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")

    home = pd.DataFrame(
        {
            "team": fixture_forecasts["home_team"],
            "expected_goals": fixture_forecasts["form_poisson_home_expected_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "team": fixture_forecasts["away_team"],
            "expected_goals": fixture_forecasts["form_poisson_away_expected_goals"],
        }
    )
    return (
        pd.concat([home, away], ignore_index=True)
        .groupby("team", as_index=False)
        .agg(expected_team_group_goals=("expected_goals", "sum"))
    )
