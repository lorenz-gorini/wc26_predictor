"""Recent-form features for team-level match prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd

from wc26_predictor.data.schema import validate_results_frame


@dataclass(frozen=True, slots=True)
class TeamRecentForm:
    """Recent team performance summary before a prediction date."""

    team: str
    matches: int
    points_per_match: float
    goals_for_per_match: float
    goals_against_per_match: float
    goal_difference_per_match: float
    attack_index: float
    defensive_vulnerability_index: float


class RecentFormTable:
    """Lookup table for recent team form as of a fixed date."""

    def __init__(
        self,
        results: pd.DataFrame,
        as_of: date | None = None,
        window_matches: int = 10,
        prior_matches: float = 6.0,
    ) -> None:
        if window_matches <= 0:
            raise ValueError("window_matches must be positive.")
        if prior_matches < 0:
            raise ValueError("prior_matches cannot be negative.")

        validated = validate_results_frame(results)
        self.as_of = as_of or validated["date"].max()
        self.window_matches = window_matches
        self.prior_matches = prior_matches
        self.global_goals_per_team_match = float(
            (validated["home_score"].sum() + validated["away_score"].sum()) / (2 * len(validated))
        )
        self._forms = self._build(validated)

    def get(self, team: str) -> TeamRecentForm:
        """Return recent form for a team, falling back to neutral form."""

        return self._forms.get(
            team,
            TeamRecentForm(
                team=team,
                matches=0,
                points_per_match=1.0,
                goals_for_per_match=self.global_goals_per_team_match,
                goals_against_per_match=self.global_goals_per_team_match,
                goal_difference_per_match=0.0,
                attack_index=1.0,
                defensive_vulnerability_index=1.0,
            ),
        )

    def to_frame(self) -> pd.DataFrame:
        """Return all team form records as a dataframe."""

        return pd.DataFrame([asdict(form) for form in self._forms.values()]).sort_values("team")

    def _build(self, results: pd.DataFrame) -> dict[str, TeamRecentForm]:
        history = _long_results(results[results["date"] <= self.as_of])
        forms = {}
        for team, team_history in history.groupby("team"):
            recent = team_history.sort_values("date").tail(self.window_matches)
            forms[str(team)] = self._summarize(str(team), recent)
        return forms

    def _summarize(self, team: str, recent: pd.DataFrame) -> TeamRecentForm:
        matches = len(recent)
        if matches == 0:
            return self.get(team)

        goals_for = float(recent["goals_for"].sum())
        goals_against = float(recent["goals_against"].sum())
        points = float(recent["points"].sum())
        denominator = matches + self.prior_matches
        baseline_goals = self.global_goals_per_team_match

        attack_rate = (goals_for + self.prior_matches * baseline_goals) / denominator
        defense_rate = (goals_against + self.prior_matches * baseline_goals) / denominator

        return TeamRecentForm(
            team=team,
            matches=matches,
            points_per_match=points / matches,
            goals_for_per_match=goals_for / matches,
            goals_against_per_match=goals_against / matches,
            goal_difference_per_match=(goals_for - goals_against) / matches,
            attack_index=float(np.clip(attack_rate / baseline_goals, 0.55, 1.75)),
            defensive_vulnerability_index=float(np.clip(defense_rate / baseline_goals, 0.55, 1.75)),
        )


def _long_results(results: pd.DataFrame) -> pd.DataFrame:
    home_points = np.select(
        [
            results["home_score"] > results["away_score"],
            results["home_score"] == results["away_score"],
        ],
        [3, 1],
        default=0,
    )
    away_points = np.select(
        [
            results["away_score"] > results["home_score"],
            results["home_score"] == results["away_score"],
        ],
        [3, 1],
        default=0,
    )

    home = pd.DataFrame(
        {
            "date": results["date"],
            "team": results["home_team"],
            "goals_for": results["home_score"],
            "goals_against": results["away_score"],
            "points": home_points,
        }
    )
    away = pd.DataFrame(
        {
            "date": results["date"],
            "team": results["away_team"],
            "goals_for": results["away_score"],
            "goals_against": results["home_score"],
            "points": away_points,
        }
    )
    return pd.concat([home, away], ignore_index=True)
