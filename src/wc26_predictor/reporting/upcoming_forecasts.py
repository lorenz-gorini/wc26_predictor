"""Upcoming-match forecast details and model-native explanations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wc26_predictor.data.schema import Fixture, validate_results_frame
from wc26_predictor.models.form_adjusted_poisson import FormAdjustedPoissonModel
from wc26_predictor.models.poisson import (
    independent_poisson_score_matrix,
    most_likely_scoreline,
    top_scorelines,
)
from wc26_predictor.pipelines.baselines import BaselineModelConfigs, default_model_configs
from wc26_predictor.simulation.official_tournament import simulate_official_tournament


@dataclass(frozen=True, slots=True)
class UpcomingForecastConfig:
    """Configuration for dashboard-oriented future match artifacts."""

    impact_simulations: int = 400
    random_seed: int = 2026


def build_upcoming_match_details(
    forecasts: pd.DataFrame,
    availability_forecasts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one row per group match with the score shown in the dashboard."""

    required = {
        "date",
        "time_local",
        "utc_offset",
        "group",
        "match_number",
        "home_team",
        "away_team",
        "stadium",
        "city",
        "form_poisson_home_expected_goals",
        "form_poisson_away_expected_goals",
        "ensemble_home_win",
        "ensemble_draw",
        "ensemble_away_win",
    }
    _require_columns(forecasts, required, "forecasts")
    output = forecasts.copy()
    if availability_forecasts is not None:
        availability_columns = {
            "match_number",
            "availability_adjusted_home_expected_goals",
            "availability_adjusted_away_expected_goals",
            "home_availability_burden",
            "away_availability_burden",
            "home_availability_goal_multiplier",
            "away_availability_goal_multiplier",
        }
        _require_columns(availability_forecasts, availability_columns, "availability forecasts")
        output = output.merge(
            availability_forecasts.loc[:, sorted(availability_columns)],
            on="match_number",
            how="left",
            validate="one_to_one",
        )

    home_column = (
        "availability_adjusted_home_expected_goals"
        if "availability_adjusted_home_expected_goals" in output.columns
        else "form_poisson_home_expected_goals"
    )
    away_column = (
        "availability_adjusted_away_expected_goals"
        if "availability_adjusted_away_expected_goals" in output.columns
        else "form_poisson_away_expected_goals"
    )
    rows = []
    for row in output.itertuples(index=False):
        home_lambda = float(getattr(row, home_column))
        away_lambda = float(getattr(row, away_column))
        matrix = independent_poisson_score_matrix(home_lambda, away_lambda)
        modal = most_likely_scoreline(matrix)
        home_win = float(np.tril(matrix, k=-1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, k=1).sum())
        rows.append(
            {
                "date": row.date,
                "time_local": row.time_local,
                "utc_offset": row.utc_offset,
                "group": row.group,
                "match_number": int(row.match_number),
                "home_team": row.home_team,
                "away_team": row.away_team,
                "stadium": row.stadium,
                "city": row.city,
                "is_completed": bool(getattr(row, "is_completed", False)),
                "completed_home_score": getattr(row, "completed_home_score", np.nan),
                "completed_away_score": getattr(row, "completed_away_score", np.nan),
                "home_expected_goals": home_lambda,
                "away_expected_goals": away_lambda,
                "predicted_home_score": modal.home_score,
                "predicted_away_score": modal.away_score,
                "predicted_score": modal.label,
                "predicted_score_outcome": _scoreline_outcome(
                    modal.home_score,
                    modal.away_score,
                ),
                "predicted_score_probability": modal.probability,
                "top_scorelines": _format_top_scorelines(matrix),
                "home_win_probability": home_win,
                "draw_probability": draw,
                "away_win_probability": away_win,
                "ensemble_home_win": float(row.ensemble_home_win),
                "ensemble_draw": float(row.ensemble_draw),
                "ensemble_away_win": float(row.ensemble_away_win),
                "home_availability_burden": float(
                    getattr(row, "home_availability_burden", 0.0)
                ),
                "away_availability_burden": float(
                    getattr(row, "away_availability_burden", 0.0)
                ),
                "home_availability_goal_multiplier": float(
                    getattr(row, "home_availability_goal_multiplier", 1.0)
                ),
                "away_availability_goal_multiplier": float(
                    getattr(row, "away_availability_goal_multiplier", 1.0)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "match_number"], ignore_index=True)


def build_match_driver_table(
    results: pd.DataFrame,
    forecasts: pd.DataFrame,
    configs: BaselineModelConfigs | None = None,
) -> pd.DataFrame:
    """Decompose forecast expected goals into interpretable log-scale drivers."""

    required = {
        "match_number",
        "home_team",
        "away_team",
        "home_availability_goal_multiplier",
        "away_availability_goal_multiplier",
    }
    _require_columns(forecasts, required, "match details")
    validated_results = validate_results_frame(results)
    model_configs = configs or default_model_configs()
    model = FormAdjustedPoissonModel(config=model_configs.form_adjusted_poisson).fit(
        validated_results
    )
    base = model.base_model
    if base.global_goal_rate_ is None or base.home_goal_multiplier_ is None:
        raise RuntimeError("Form-adjusted Poisson model did not fit base rates.")
    if model.form_table is None:
        raise RuntimeError("Form-adjusted Poisson model did not fit recent-form table.")

    rows = []
    for forecast in forecasts.itertuples(index=False):
        fixture = Fixture(home_team=forecast.home_team, away_team=forecast.away_team, neutral=True)
        home_base, away_base = base.expected_goals(fixture)
        home_form = model.form_table.get(forecast.home_team)
        away_form = model.form_table.get(forecast.away_team)
        rows.extend(
            _team_driver_rows(
                match_number=int(forecast.match_number),
                team=str(forecast.home_team),
                opponent=str(forecast.away_team),
                goal_side="home",
                base_expected_goals=home_base,
                attack_multiplier=base.attack_.get(str(forecast.home_team), 1.0),
                opponent_defense_multiplier=base.defense_.get(
                    str(forecast.away_team),
                    1.0,
                ),
                own_form_attack=home_form.attack_index,
                opponent_form_defense=away_form.defensive_vulnerability_index,
                availability_multiplier=float(forecast.home_availability_goal_multiplier),
                form_strength=model_configs.form_adjusted_poisson.form_strength,
            )
        )
        rows.extend(
            _team_driver_rows(
                match_number=int(forecast.match_number),
                team=str(forecast.away_team),
                opponent=str(forecast.home_team),
                goal_side="away",
                base_expected_goals=away_base,
                attack_multiplier=base.attack_.get(str(forecast.away_team), 1.0),
                opponent_defense_multiplier=base.defense_.get(
                    str(forecast.home_team),
                    1.0,
                ),
                own_form_attack=away_form.attack_index,
                opponent_form_defense=home_form.defensive_vulnerability_index,
                availability_multiplier=float(forecast.away_availability_goal_multiplier),
                form_strength=model_configs.form_adjusted_poisson.form_strength,
            )
        )
    output = pd.DataFrame(rows)
    output["abs_contribution_log_expected_goals"] = output[
        "contribution_log_expected_goals"
    ].abs()
    return output.sort_values(
        ["match_number", "abs_contribution_log_expected_goals"],
        ascending=[True, False],
        ignore_index=True,
    )


def build_final_stage_match_impacts(
    results: pd.DataFrame,
    forecasts: pd.DataFrame,
    advancement_probabilities: dict[tuple[str, str], float],
    config: UpcomingForecastConfig | None = None,
) -> pd.DataFrame:
    """Estimate how much each group match's directional forecast moves advancement."""

    required = {
        "match_number",
        "group",
        "home_team",
        "away_team",
        "form_poisson_home_expected_goals",
        "form_poisson_away_expected_goals",
    }
    _require_columns(forecasts, required, "forecasts")
    impact_config = config or UpcomingForecastConfig()
    baseline = simulate_official_tournament(
        results,
        forecasts,
        n_simulations=impact_config.impact_simulations,
        seed=impact_config.random_seed,
        advancement_probabilities=advancement_probabilities,
    )
    rows = []
    for forecast in forecasts.itertuples(index=False):
        counterfactual = forecasts.copy()
        row_selector = counterfactual["match_number"] == int(forecast.match_number)
        neutral_lambda = (
            float(forecast.form_poisson_home_expected_goals)
            + float(forecast.form_poisson_away_expected_goals)
        ) / 2.0
        counterfactual.loc[row_selector, "form_poisson_home_expected_goals"] = neutral_lambda
        counterfactual.loc[row_selector, "form_poisson_away_expected_goals"] = neutral_lambda
        comparison = simulate_official_tournament(
            results,
            counterfactual,
            n_simulations=impact_config.impact_simulations,
            seed=impact_config.random_seed,
            advancement_probabilities=advancement_probabilities,
        )
        delta = baseline.merge(
            comparison,
            on="team",
            suffixes=("_baseline", "_neutralized"),
            validate="one_to_one",
        )
        for column in [
            "round_of_32_probability",
            "round_of_16_probability",
            "quarter_final_probability",
            "semi_final_probability",
            "final_probability",
            "winner_probability",
        ]:
            delta[f"{column}_delta"] = (
                delta[f"{column}_baseline"] - delta[f"{column}_neutralized"]
            )
        winner_delta = delta.loc[delta["winner_probability_delta"].abs().idxmax()]
        rows.append(
            {
                "match_number": int(forecast.match_number),
                "group": forecast.group,
                "home_team": forecast.home_team,
                "away_team": forecast.away_team,
                "round_of_32_l1_impact": float(
                    delta["round_of_32_probability_delta"].abs().sum()
                ),
                "round_of_16_l1_impact": float(
                    delta["round_of_16_probability_delta"].abs().sum()
                ),
                "quarter_final_l1_impact": float(
                    delta["quarter_final_probability_delta"].abs().sum()
                ),
                "semi_final_l1_impact": float(
                    delta["semi_final_probability_delta"].abs().sum()
                ),
                "final_l1_impact": float(delta["final_probability_delta"].abs().sum()),
                "winner_l1_impact": float(delta["winner_probability_delta"].abs().sum()),
                "total_final_stage_impact": float(
                    delta[
                        [
                            "round_of_32_probability_delta",
                            "round_of_16_probability_delta",
                            "quarter_final_probability_delta",
                            "semi_final_probability_delta",
                            "final_probability_delta",
                            "winner_probability_delta",
                        ]
                    ]
                    .abs()
                    .sum()
                    .sum()
                ),
                "largest_winner_delta_team": winner_delta["team"],
                "largest_winner_probability_delta": float(
                    winner_delta["winner_probability_delta"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "total_final_stage_impact",
        ascending=False,
        ignore_index=True,
    )


def _team_driver_rows(
    match_number: int,
    team: str,
    opponent: str,
    goal_side: str,
    base_expected_goals: float,
    attack_multiplier: float,
    opponent_defense_multiplier: float,
    own_form_attack: float,
    opponent_form_defense: float,
    availability_multiplier: float,
    form_strength: float,
) -> list[dict[str, object]]:
    form_attack = form_strength * np.log(own_form_attack)
    form_defense = form_strength * np.log(opponent_form_defense)
    rows = [
        _driver_row(
            match_number,
            team,
            opponent,
            goal_side,
            "team_attack_strength",
            np.log(attack_multiplier),
            f"{team} attack multiplier from historical scoring.",
        ),
        _driver_row(
            match_number,
            team,
            opponent,
            goal_side,
            "opponent_defensive_vulnerability",
            np.log(opponent_defense_multiplier),
            f"{opponent} defensive goals-against multiplier.",
        ),
        _driver_row(
            match_number,
            team,
            opponent,
            goal_side,
            "recent_attack_form",
            form_attack,
            f"{team} recent attack index with form strength {form_strength:.2f}.",
        ),
        _driver_row(
            match_number,
            team,
            opponent,
            goal_side,
            "opponent_recent_defensive_form",
            form_defense,
            f"{opponent} recent defensive vulnerability with form strength {form_strength:.2f}.",
        ),
        _driver_row(
            match_number,
            team,
            opponent,
            goal_side,
            "availability_adjustment",
            np.log(availability_multiplier),
            f"{team} expected-goal multiplier from current injury/suspension burden.",
        ),
    ]
    residual = (
        np.log(base_expected_goals)
        - np.log(attack_multiplier)
        - np.log(opponent_defense_multiplier)
    )
    rows.append(
        _driver_row(
            match_number,
            team,
            opponent,
            goal_side,
            "base_goal_environment",
            residual,
            "Global scoring rate, venue treatment, and shrinkage residual.",
        )
    )
    return rows


def _driver_row(
    match_number: int,
    team: str,
    opponent: str,
    goal_side: str,
    driver: str,
    contribution: float,
    explanation: str,
) -> dict[str, object]:
    return {
        "match_number": match_number,
        "team": team,
        "opponent": opponent,
        "goal_side": goal_side,
        "driver": driver,
        "contribution_log_expected_goals": float(contribution),
        "multiplier": float(np.exp(contribution)),
        "direction": "raises expected goals" if contribution >= 0 else "lowers expected goals",
        "explanation": explanation,
    }


def _format_top_scorelines(score_matrix: np.ndarray, n: int = 5) -> str:
    return "; ".join(
        f"{scoreline.label} ({scoreline.probability:.3f})"
        for scoreline in top_scorelines(score_matrix, n=n)
    )


def _scoreline_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {label}: {sorted(missing)}")
