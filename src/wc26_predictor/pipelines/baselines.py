"""Baseline fitting, backtesting, and fixture forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from wc26_predictor.data.schema import Fixture, OutcomeProbabilities, validate_results_frame
from wc26_predictor.evaluation.metrics import brier_score, multiclass_log_loss, observed_outcome
from wc26_predictor.evaluation.score_metrics import (
    exact_score_accuracy,
    goal_mae,
    goal_rmse,
    total_goal_mae,
)
from wc26_predictor.models.elo import EloConfig, EloRatings
from wc26_predictor.models.ensemble import (
    MODEL_NAMES,
    EnsembleWeights,
    add_weighted_ensemble_columns,
    fit_ensemble_weights,
    prediction_frame_to_probabilities,
    weighted_average_probabilities,
)
from wc26_predictor.models.form_adjusted_poisson import (
    FormAdjustedPoissonConfig,
    FormAdjustedPoissonModel,
)
from wc26_predictor.models.poisson import IndependentPoissonModel, PoissonConfig, ScorePrediction
from wc26_predictor.simulation.abstract_tournament import simulate_abstract_tournament
from wc26_predictor.simulation.official_tournament import simulate_official_tournament


@dataclass(frozen=True, slots=True)
class EvaluationWindow:
    """A temporal holdout window."""

    name: str
    train_until: date
    test_start: date
    test_end: date
    tournament: str | None = None


@dataclass(frozen=True, slots=True)
class BaselineModelConfigs:
    """Model hyperparameters selected for the baseline workflow."""

    elo: EloConfig
    poisson: PoissonConfig
    form_adjusted_poisson: FormAdjustedPoissonConfig


WORLD_CUP_WINDOWS = [
    EvaluationWindow(
        name="2014_fifa_world_cup",
        train_until=date(2014, 6, 11),
        test_start=date(2014, 6, 12),
        test_end=date(2014, 7, 13),
        tournament="FIFA World Cup",
    ),
    EvaluationWindow(
        name="2018_fifa_world_cup",
        train_until=date(2018, 6, 13),
        test_start=date(2018, 6, 14),
        test_end=date(2018, 7, 15),
        tournament="FIFA World Cup",
    ),
    EvaluationWindow(
        name="2022_fifa_world_cup",
        train_until=date(2022, 11, 19),
        test_start=date(2022, 11, 20),
        test_end=date(2022, 12, 18),
        tournament="FIFA World Cup",
    ),
]


def default_model_configs() -> BaselineModelConfigs:
    """Return default model configurations."""

    poisson = PoissonConfig()
    return BaselineModelConfigs(
        elo=EloConfig(),
        poisson=poisson,
        form_adjusted_poisson=FormAdjustedPoissonConfig(poisson=poisson),
    )


def tune_baseline_model_configs(results: pd.DataFrame) -> tuple[BaselineModelConfigs, pd.DataFrame]:
    """Tune baseline hyperparameters on historical World Cup validation windows."""

    validated = validate_results_frame(results)
    tuning_rows = []

    best_elo, elo_rows = _tune_elo_config(validated)
    tuning_rows.extend(elo_rows)

    best_poisson, poisson_rows = _tune_poisson_config(validated)
    tuning_rows.extend(poisson_rows)

    best_form, form_rows = _tune_form_config(validated, best_poisson)
    tuning_rows.extend(form_rows)

    configs = BaselineModelConfigs(
        elo=best_elo,
        poisson=best_poisson,
        form_adjusted_poisson=best_form,
    )
    return configs, pd.DataFrame(tuning_rows).sort_values(
        ["model", "rank", "log_loss"],
        ignore_index=True,
    )


def model_configs_to_frame(configs: BaselineModelConfigs) -> pd.DataFrame:
    """Convert selected configs to a tidy dataframe."""

    return pd.DataFrame(
        [
            {
                "model": "elo",
                "parameter": "k_factor",
                "value": configs.elo.k_factor,
            },
            {
                "model": "elo",
                "parameter": "draw_probability",
                "value": configs.elo.draw_probability,
            },
            {
                "model": "elo",
                "parameter": "home_advantage",
                "value": configs.elo.home_advantage,
            },
            {
                "model": "elo",
                "parameter": "neutral_home_advantage",
                "value": configs.elo.neutral_home_advantage,
            },
            {
                "model": "poisson",
                "parameter": "prior_strength",
                "value": configs.poisson.prior_strength,
            },
            {
                "model": "poisson",
                "parameter": "recency_half_life_days",
                "value": configs.poisson.recency_half_life_days,
            },
            {
                "model": "poisson",
                "parameter": "use_tournament_importance",
                "value": configs.poisson.use_tournament_importance,
            },
            {
                "model": "form_adjusted_poisson",
                "parameter": "form_window_matches",
                "value": configs.form_adjusted_poisson.form_window_matches,
            },
            {
                "model": "form_adjusted_poisson",
                "parameter": "form_prior_matches",
                "value": configs.form_adjusted_poisson.form_prior_matches,
            },
            {
                "model": "form_adjusted_poisson",
                "parameter": "form_strength",
                "value": configs.form_adjusted_poisson.form_strength,
            },
        ]
    )


def collect_world_cup_validation_predictions(
    results: pd.DataFrame,
    configs: BaselineModelConfigs | None = None,
    windows: list[EvaluationWindow] | None = None,
) -> pd.DataFrame:
    """Collect model predictions on historical World Cup holdout windows."""

    validated = validate_results_frame(results)
    model_configs = configs or default_model_configs()
    validation_windows = windows or WORLD_CUP_WINDOWS
    frames = []
    for window in validation_windows:
        frames.append(_window_prediction_frame(validated, window, model_configs))
    return pd.concat(frames, ignore_index=True)


def fit_world_cup_ensemble_weights(
    results: pd.DataFrame,
    configs: BaselineModelConfigs | None = None,
    step: float = 0.01,
) -> EnsembleWeights:
    """Fit ensemble weights on historical World Cup holdout predictions."""

    validation_predictions = collect_world_cup_validation_predictions(results, configs)
    return fit_ensemble_weights(validation_predictions, step=step)


def evaluate_world_cup_baselines(
    results: pd.DataFrame,
    configs: BaselineModelConfigs | None = None,
) -> pd.DataFrame:
    """Evaluate baseline models and leave-one-window-out calibrated ensembles."""

    validation_predictions = collect_world_cup_validation_predictions(results, configs)
    rows = []
    for window_name, frame in validation_predictions.groupby("window"):
        window = _window_by_name(str(window_name))
        observed = frame["observed"].tolist()
        for model in MODEL_NAMES:
            row = _score_predictions(
                window,
                model,
                prediction_frame_to_probabilities(frame, model),
                observed,
            )
            if model in {"poisson", "form_adjusted_poisson"}:
                row.update(_score_metric_summary(frame, model))
            rows.append(row)
        rows.append(
            _score_predictions(
                window,
                "ensemble_equal_weight",
                prediction_frame_to_probabilities(frame, "ensemble_equal_weight"),
                observed,
            )
        )

        training = validation_predictions[validation_predictions["window"] != window_name]
        calibrated_weights = fit_ensemble_weights(training)
        calibrated = add_weighted_ensemble_columns(frame, calibrated_weights, "ensemble_calibrated")
        calibrated_row = _score_predictions(
            window,
            "ensemble_calibrated_lowo",
            prediction_frame_to_probabilities(calibrated, "ensemble_calibrated"),
            observed,
        )
        calibrated_row.update(
            {
                "elo_weight": calibrated_weights.elo,
                "poisson_weight": calibrated_weights.poisson,
                "form_adjusted_poisson_weight": calibrated_weights.form_adjusted_poisson,
            }
        )
        rows.append(calibrated_row)

    return pd.DataFrame(rows).sort_values(["window", "model"]).reset_index(drop=True)


def forecast_2026_group_fixtures(
    results: pd.DataFrame,
    fixtures: pd.DataFrame,
    ensemble_weights: EnsembleWeights | None = None,
    configs: BaselineModelConfigs | None = None,
) -> pd.DataFrame:
    """Fit baselines on all completed results and forecast 2026 group fixtures."""

    validated_results = validate_results_frame(results)
    validated_fixtures = _validate_fixture_frame(fixtures)
    model_configs = configs or default_model_configs()

    elo = EloRatings(config=model_configs.elo).fit(validated_results)
    poisson = IndependentPoissonModel(config=model_configs.poisson).fit(validated_results)
    form_poisson = FormAdjustedPoissonModel(
        config=model_configs.form_adjusted_poisson
    ).fit(validated_results)
    weights = ensemble_weights or EnsembleWeights.equal()

    rows = []
    for row in validated_fixtures.itertuples(index=False):
        fixture = Fixture(home_team=row.home_team, away_team=row.away_team, neutral=True)
        elo_probs = elo.predict_outcome(fixture)
        poisson_prediction = poisson.predict(fixture)
        poisson_probs = poisson_prediction.outcome_probabilities
        form_prediction = form_poisson.predict(fixture)
        form_probs = form_prediction.outcome_probabilities
        equal_weight_probs = _average_probabilities([elo_probs, poisson_probs, form_probs])
        ensemble_probs = weighted_average_probabilities(
            {
                "elo": elo_probs,
                "poisson": poisson_probs,
                "form_adjusted_poisson": form_probs,
            },
            weights,
        )

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
                "elo_home_win": elo_probs.home_win,
                "elo_draw": elo_probs.draw,
                "elo_away_win": elo_probs.away_win,
                "poisson_home_expected_goals": poisson_prediction.home_expected_goals,
                "poisson_away_expected_goals": poisson_prediction.away_expected_goals,
                **_score_prediction_columns("poisson", poisson_prediction),
                "poisson_home_win": poisson_probs.home_win,
                "poisson_draw": poisson_probs.draw,
                "poisson_away_win": poisson_probs.away_win,
                "form_poisson_home_expected_goals": form_prediction.home_expected_goals,
                "form_poisson_away_expected_goals": form_prediction.away_expected_goals,
                **_score_prediction_columns("form_poisson", form_prediction),
                **_primary_score_columns(form_prediction),
                "form_poisson_home_win": form_probs.home_win,
                "form_poisson_draw": form_probs.draw,
                "form_poisson_away_win": form_probs.away_win,
                "ensemble_equal_weight_home_win": equal_weight_probs.home_win,
                "ensemble_equal_weight_draw": equal_weight_probs.draw,
                "ensemble_equal_weight_away_win": equal_weight_probs.away_win,
                "ensemble_home_win": ensemble_probs.home_win,
                "ensemble_draw": ensemble_probs.draw,
                "ensemble_away_win": ensemble_probs.away_win,
            }
        )

    return pd.DataFrame(rows).sort_values("match_number").reset_index(drop=True)


def write_baseline_summary(
    backtest: pd.DataFrame,
    forecasts: pd.DataFrame,
    destination: str | Path,
    tournament_probabilities: pd.DataFrame | None = None,
    top_scorers: pd.DataFrame | None = None,
    availability_adjusted_top_scorers: pd.DataFrame | None = None,
    team_availability: pd.DataFrame | None = None,
    availability_impact: pd.DataFrame | None = None,
    transfermarkt_team_injury_burden: pd.DataFrame | None = None,
    club_adjusted_top_scorers: pd.DataFrame | None = None,
    transfermarkt_adjusted_top_scorers: pd.DataFrame | None = None,
    ensemble_weights: EnsembleWeights | None = None,
    selected_configs: pd.DataFrame | None = None,
    market_metrics: pd.DataFrame | None = None,
    market_decision: dict[str, object] | None = None,
) -> None:
    """Write a compact markdown summary of baseline outputs."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    average_metrics = (
        backtest.groupby("model", as_index=False)
        .agg(log_loss=("log_loss", "mean"), brier_score=("brier_score", "mean"))
        .sort_values("log_loss")
    )

    most_decisive = forecasts.assign(
        max_probability=forecasts[
            ["ensemble_home_win", "ensemble_draw", "ensemble_away_win"]
        ].max(axis=1)
    ).sort_values("max_probability", ascending=False)

    lines = [
        "# Baseline Model Summary",
        "",
        "## Backtest Averages",
        "",
        _markdown_table(average_metrics, float_digits=4),
        "",
        "## Most Decisive 2026 Group Forecasts",
        "",
        _markdown_table(
            most_decisive[
                [
                    "match_number",
                    "group",
                    "home_team",
                    "away_team",
                    "predicted_score",
                    "predicted_score_outcome",
                    "predicted_score_probability",
                    "ensemble_home_win",
                    "ensemble_draw",
                    "ensemble_away_win",
                ]
            ].head(10),
            float_digits=3,
        ),
        "",
        "Notes:",
        "",
        "- World Cup 2026 fixtures are treated as venue-neutral in this first baseline.",
        (
            "- Host-country advantage is not hard-coded; it should be retained only if it "
            "improves backtests."
        ),
        (
            "- The primary ensemble uses weights learned from historical World Cup holdout "
            "predictions."
        ),
        "",
    ]
    if ensemble_weights is not None:
        lines.extend(
            [
                "## Calibrated Ensemble Weights",
                "",
                _markdown_table(
                    pd.DataFrame(
                        {
                            "model": list(ensemble_weights.as_dict().keys()),
                            "weight": list(ensemble_weights.as_dict().values()),
                        }
                    ),
                    float_digits=3,
                ),
                "",
                (
                    "Weights are fit by minimizing multiclass log loss over the 2014, 2018, "
                    "and 2022 World Cup validation predictions."
                ),
                "",
            ]
        )
    if selected_configs is not None:
        lines.extend(
            [
                "## Tuned Hyperparameters",
                "",
                _markdown_table(selected_configs, float_digits=3),
                "",
                (
                    "Hyperparameters are selected by minimizing validation log loss across the "
                    "2014, 2018, and 2022 World Cup holdout windows."
                ),
                "",
            ]
        )
    if market_metrics is not None and market_decision is not None:
        average_market_metrics = (
            market_metrics.groupby("model", as_index=False)
            .agg(log_loss=("log_loss", "mean"), brier_score=("brier_score", "mean"))
            .sort_values("log_loss")
        )
        lines.extend(
            [
                "## Market Benchmark",
                "",
                _markdown_table(average_market_metrics, float_digits=4),
                "",
                f"Gate decision: {market_decision['reason']}",
                "",
            ]
        )
    if team_availability is not None:
        burdened = team_availability.sort_values(
            ["team_availability_burden", "unavailable_players"],
            ascending=[False, False],
        ).head(12)
        lines.extend(
            [
                "## Availability Burden",
                "",
                _markdown_table(burdened, float_digits=3),
                "",
                _markdown_table(availability_impact, float_digits=3)
                if availability_impact is not None
                else "",
                "",
                (
                    "Availability burden is neutral unless current injury/suspension overrides "
                    "are provided in `data/raw/world_cup_2026_player_availability.csv`."
                ),
                "",
            ]
        )
    if transfermarkt_team_injury_burden is not None:
        injury_burden = transfermarkt_team_injury_burden.sort_values(
            ["team_current_open_injury_burden", "team_historical_injury_burden"],
            ascending=[False, False],
        ).head(12)
        lines.extend(
            [
                "## Transfermarkt Injury Burden",
                "",
                _markdown_table(injury_burden, float_digits=3),
                "",
                (
                    "Transfermarkt injury history is joined by player ID where possible and "
                    "otherwise by normalized player, country, and club signals. Historical "
                    "recurrence burden is diagnostic; verified current overrides should still "
                    "be supplied through `data/raw/world_cup_2026_player_availability.csv` "
                    "before the tournament."
                ),
                "",
            ]
        )

    if top_scorers is not None:
        lines.extend(
            [
                "## National-Team Top-Scorer Baseline",
                "",
                _markdown_table(
                    top_scorers[
                        [
                            "team",
                            "scorer",
                            "weighted_goals",
                            "goal_share",
                            "expected_team_group_goals",
                            "expected_team_matches",
                            "expected_group_goals",
                            "expected_tournament_goals",
                        ]
                    ].head(15),
                    float_digits=3,
                ),
                "",
                (
                    "This scorer table uses national-team goalscorer history, 2026 squad "
                    "filtering, and official-bracket expected-match exposure."
                ),
                "",
            ]
        )
    if tournament_probabilities is not None:
        lines.extend(
            [
                "## Official-Bracket Tournament Winner Probabilities",
                "",
                _markdown_table(
                    tournament_probabilities[
                        [
                            "team",
                            "round_of_32_probability",
                            "quarter_final_probability",
                            "semi_final_probability",
                            "final_probability",
                            "winner_probability",
                        ]
                    ].head(12),
                    float_digits=3,
                ),
                "",
                (
                    "These probabilities use group-stage simulation, FIFA Annex C third-place "
                    "assignments, and the official 2026 knockout bracket."
                ),
                "",
            ]
        )
    if availability_adjusted_top_scorers is not None:
        lines.extend(
            [
                "## Availability-Adjusted Top-Scorer Baseline",
                "",
                _markdown_table(
                    availability_adjusted_top_scorers[
                        [
                            "team",
                            "scorer",
                            "status",
                            "expected_minutes_share",
                            "penalty_taker_rank",
                            "expected_tournament_goals",
                            "expected_tournament_goals_availability_adjusted",
                        ]
                    ].head(15),
                    float_digits=3,
                ),
                "",
                (
                    "This table applies expected-minutes, penalty-role, and injury/suspension "
                    "status adjustments before club-form enrichment."
                ),
                "",
            ]
        )
    if club_adjusted_top_scorers is not None:
        lines.extend(
            [
                "## Club-Form Adjusted Top 100",
                "",
                _markdown_table(
                    club_adjusted_top_scorers[
                        [
                            "team",
                            "scorer",
                            "club",
                            "club_form_match_quality",
                            "club_form_goals",
                            "expected_tournament_goals",
                            "expected_tournament_goals_club_adjusted",
                        ]
                    ].head(15),
                    float_digits=3,
                ),
                "",
                (
                    "Club form is sourced from curated public top-scorer tables and currently "
                    "covers only matched top-100 candidates."
                ),
                "",
            ]
        )
    if transfermarkt_adjusted_top_scorers is not None:
        coverage = (
            transfermarkt_adjusted_top_scorers["transfermarkt_match_quality"]
            .value_counts(dropna=False)
            .rename_axis("match_quality")
            .reset_index(name="players")
        )
        lines.extend(
            [
                "## Transfermarkt-Adjusted Top 100",
                "",
                _markdown_table(
                    transfermarkt_adjusted_top_scorers[
                        [
                            "team",
                            "scorer",
                            "transfermarkt_match_quality",
                            "club_goals_source",
                            "club_goals_model",
                            "club_weighted_goals_model",
                            "club_minutes_model",
                            "club_weighted_goals_per90",
                            "transfermarkt_multiplier",
                            "latest_market_value_in_eur",
                            "expected_tournament_goals",
                            "expected_tournament_goals_transfermarkt_adjusted",
                        ]
                    ].head(15),
                    float_digits=3,
                ),
                "",
                "### Transfermarkt Match Coverage",
                "",
                _markdown_table(coverage, float_digits=0),
                "",
                (
                    "Transfermarkt features are preferred over sparse public top-scorer tables "
                    "when the local Kaggle dump is available."
                ),
                "",
            ]
        )
    destination_path.write_text("\n".join(lines), encoding="utf-8")


def estimate_abstract_expected_matches(
    results: pd.DataFrame,
    forecasts: pd.DataFrame,
    n_simulations: int = 2000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Estimate team tournament matches with the optimized abstract simulator."""

    tournament = simulate_abstract_tournament(
        results=results,
        forecasts=forecasts,
        n_simulations=n_simulations,
        seed=seed,
    )
    return tournament[
        [
            "team",
            "round_of_32_probability",
            "expected_knockout_matches",
            "expected_team_matches",
        ]
    ].sort_values("expected_team_matches", ascending=False, ignore_index=True)


def estimate_official_expected_matches(
    results: pd.DataFrame,
    forecasts: pd.DataFrame,
    n_simulations: int = 2000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Estimate team tournament matches with the official 2026 bracket simulator."""

    tournament = simulate_official_tournament(
        results=results,
        forecasts=forecasts,
        n_simulations=n_simulations,
        seed=seed,
    )
    return tournament[
        [
            "team",
            "round_of_32_probability",
            "expected_knockout_matches",
            "expected_team_matches",
        ]
    ].sort_values("expected_team_matches", ascending=False, ignore_index=True)


def build_pairwise_advancement_probabilities(
    results: pd.DataFrame,
    teams: set[str] | list[str],
    ensemble_weights: EnsembleWeights,
    configs: BaselineModelConfigs | None = None,
) -> dict[tuple[str, str], float]:
    """Build neutral knockout advancement probabilities for ordered team pairs."""

    validated_results = validate_results_frame(results)
    weights = ensemble_weights
    weights.validate()
    model_configs = configs or default_model_configs()
    elo = EloRatings(config=model_configs.elo).fit(validated_results)
    poisson = IndependentPoissonModel(config=model_configs.poisson).fit(validated_results)
    form_poisson = FormAdjustedPoissonModel(
        config=model_configs.form_adjusted_poisson
    ).fit(validated_results)

    probabilities = {}
    team_list = sorted(teams)
    for first in team_list:
        for second in team_list:
            if first == second:
                continue
            fixture = Fixture(home_team=first, away_team=second, neutral=True)
            ensemble = weighted_average_probabilities(
                {
                    "elo": elo.predict_outcome(fixture),
                    "poisson": poisson.predict_outcome(fixture),
                    "form_adjusted_poisson": form_poisson.predict_outcome(fixture),
                },
                weights,
            )
            decisive_probability = ensemble.home_win + ensemble.away_win
            if decisive_probability <= 0:
                raise ValueError(f"Non-positive decisive probability for {first} vs {second}.")
            probabilities[(first, second)] = ensemble.home_win / decisive_probability
    return probabilities


def attach_completed_fixture_results(
    forecasts: pd.DataFrame,
    results: pd.DataFrame,
    tournament: str = "FIFA World Cup",
) -> pd.DataFrame:
    """Mark group fixtures that already have completed results.

    Completed scores are used by the tournament simulators as fixed group-stage
    outcomes, while still allowing the same forecast table to carry predictions
    for matches that have not been played yet.
    """

    required_forecast_columns = {"date", "home_team", "away_team"}
    missing = required_forecast_columns.difference(forecasts.columns)
    if missing:
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")

    validated_results = validate_results_frame(results)
    completed = validated_results[validated_results["tournament"] == tournament].copy()
    completed["date"] = pd.to_datetime(completed["date"], errors="raise").dt.date
    completed = completed.loc[
        :,
        ["date", "home_team", "away_team", "home_score", "away_score"],
    ].rename(
        columns={
            "home_score": "completed_home_score",
            "away_score": "completed_away_score",
        }
    )

    output = forecasts.copy()
    output["date"] = pd.to_datetime(output["date"], errors="raise").dt.date
    output = output.merge(
        completed,
        on=["date", "home_team", "away_team"],
        how="left",
        validate="many_to_one",
    )
    output["is_completed"] = output["completed_home_score"].notna()
    return output


def build_full_match_forecast_table(
    group_forecasts: pd.DataFrame,
    knockout_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Combine fixed group forecasts with conditional knockout forecasts."""

    required_group_columns = {
        "match_number",
        "group",
        "home_team",
        "away_team",
        "predicted_score",
        "predicted_score_outcome",
        "predicted_score_probability",
        "top_scorelines",
        "ensemble_home_win",
        "ensemble_draw",
        "ensemble_away_win",
    }
    missing_group = required_group_columns.difference(group_forecasts.columns)
    if missing_group:
        raise ValueError(f"Missing group forecast columns: {sorted(missing_group)}")

    required_knockout_columns = {
        "match_number",
        "round",
        "first_team",
        "second_team",
        "pairing_probability",
        "first_advancement_probability",
        "second_advancement_probability",
        "simulation_count",
    }
    missing_knockout = required_knockout_columns.difference(knockout_forecasts.columns)
    if missing_knockout:
        raise ValueError(f"Missing knockout forecast columns: {sorted(missing_knockout)}")

    group_rows = pd.DataFrame(
        {
            "match_number": group_forecasts["match_number"],
            "round": "group",
            "group": group_forecasts["group"],
            "first_team": group_forecasts["home_team"],
            "second_team": group_forecasts["away_team"],
            "predicted_score": group_forecasts["predicted_score"],
            "predicted_score_outcome": group_forecasts["predicted_score_outcome"],
            "predicted_score_probability": group_forecasts["predicted_score_probability"],
            "top_scorelines": group_forecasts["top_scorelines"],
            "pairing_probability": 1.0,
            "first_win_probability": group_forecasts["ensemble_home_win"],
            "draw_probability": group_forecasts["ensemble_draw"],
            "second_win_probability": group_forecasts["ensemble_away_win"],
            "first_advancement_probability": float("nan"),
            "second_advancement_probability": float("nan"),
            "simulation_count": float("nan"),
        }
    )
    knockout_rows = pd.DataFrame(
        {
            "match_number": knockout_forecasts["match_number"],
            "round": knockout_forecasts["round"],
            "group": "",
            "first_team": knockout_forecasts["first_team"],
            "second_team": knockout_forecasts["second_team"],
            "predicted_score": "",
            "predicted_score_outcome": "",
            "predicted_score_probability": float("nan"),
            "top_scorelines": "",
            "pairing_probability": knockout_forecasts["pairing_probability"],
            "first_win_probability": float("nan"),
            "draw_probability": float("nan"),
            "second_win_probability": float("nan"),
            "first_advancement_probability": knockout_forecasts[
                "first_advancement_probability"
            ],
            "second_advancement_probability": knockout_forecasts[
                "second_advancement_probability"
            ],
            "simulation_count": knockout_forecasts["simulation_count"],
        }
    )
    return pd.concat([group_rows, knockout_rows], ignore_index=True).sort_values(
        ["match_number", "pairing_probability", "first_team", "second_team"],
        ascending=[True, False, True, True],
        ignore_index=True,
    )


def _tune_elo_config(results: pd.DataFrame) -> tuple[EloConfig, list[dict[str, object]]]:
    candidates = [
        EloConfig(
            k_factor=k_factor,
            draw_probability=draw_probability,
            home_advantage=home_advantage,
            neutral_home_advantage=neutral_home_advantage,
        )
        for k_factor, draw_probability, home_advantage, neutral_home_advantage in product(
            [20.0, 30.0, 40.0],
            [0.22, 0.26, 0.30],
            [40.0, 60.0, 80.0],
            [0.0, 10.0],
        )
    ]
    rows = [
        {
            "model": "elo",
            "config": config,
            "log_loss": _evaluate_single_model_config(results, "elo", config),
            "k_factor": config.k_factor,
            "draw_probability": config.draw_probability,
            "home_advantage": config.home_advantage,
            "neutral_home_advantage": config.neutral_home_advantage,
        }
        for config in candidates
    ]
    return _select_config(rows)


def _tune_poisson_config(results: pd.DataFrame) -> tuple[PoissonConfig, list[dict[str, object]]]:
    candidates = [
        PoissonConfig(
            prior_strength=prior_strength,
            recency_half_life_days=recency_half_life_days,
            use_tournament_importance=use_tournament_importance,
        )
        for prior_strength, recency_half_life_days, use_tournament_importance in product(
            [3.0, 6.0, 10.0, 16.0],
            [365, 730, 1095, None],
            [True, False],
        )
    ]
    rows = [
        {
            "model": "poisson",
            "config": config,
            "log_loss": _evaluate_single_model_config(results, "poisson", config),
            "prior_strength": config.prior_strength,
            "recency_half_life_days": config.recency_half_life_days,
            "use_tournament_importance": config.use_tournament_importance,
        }
        for config in candidates
    ]
    return _select_config(rows)


def _tune_form_config(
    results: pd.DataFrame,
    poisson_config: PoissonConfig,
) -> tuple[FormAdjustedPoissonConfig, list[dict[str, object]]]:
    candidates = [
        FormAdjustedPoissonConfig(
            poisson=poisson_config,
            form_window_matches=form_window_matches,
            form_prior_matches=form_prior_matches,
            form_strength=form_strength,
        )
        for form_window_matches, form_prior_matches, form_strength in product(
            [6, 10, 14],
            [4.0, 6.0, 10.0],
            [0.15, 0.35, 0.55],
        )
    ]
    rows = [
        {
            "model": "form_adjusted_poisson",
            "config": config,
            "log_loss": _evaluate_single_model_config(
                results,
                "form_adjusted_poisson",
                config,
            ),
            "poisson_prior_strength": config.poisson.prior_strength,
            "poisson_recency_half_life_days": config.poisson.recency_half_life_days,
            "poisson_use_tournament_importance": config.poisson.use_tournament_importance,
            "form_window_matches": config.form_window_matches,
            "form_prior_matches": config.form_prior_matches,
            "form_strength": config.form_strength,
        }
        for config in candidates
    ]
    return _select_config(rows)


def _select_config(rows: list[dict[str, object]]):
    ranked = sorted(rows, key=lambda row: float(row["log_loss"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    best = ranked[0]["config"]
    public_rows = [{key: value for key, value in row.items() if key != "config"} for row in ranked]
    return best, public_rows


def _evaluate_single_model_config(
    results: pd.DataFrame,
    model_name: str,
    config: EloConfig | PoissonConfig | FormAdjustedPoissonConfig,
) -> float:
    predictions = []
    observed = []
    for window in WORLD_CUP_WINDOWS:
        train = results[results["date"] <= window.train_until].copy()
        test = results[
            (results["date"] >= window.test_start) & (results["date"] <= window.test_end)
        ].copy()
        if window.tournament is not None:
            test = test[test["tournament"] == window.tournament].copy()
        if train.empty:
            raise ValueError(f"Training split is empty for {window.name}.")
        if test.empty:
            raise ValueError(f"Test split is empty for {window.name}.")

        if model_name == "elo":
            if not isinstance(config, EloConfig):
                raise TypeError("Expected EloConfig for Elo tuning.")
            model = EloRatings(config=config).fit(train)
        elif model_name == "poisson":
            if not isinstance(config, PoissonConfig):
                raise TypeError("Expected PoissonConfig for Poisson tuning.")
            model = IndependentPoissonModel(config=config).fit(train)
        elif model_name == "form_adjusted_poisson":
            if not isinstance(config, FormAdjustedPoissonConfig):
                raise TypeError("Expected FormAdjustedPoissonConfig for form tuning.")
            model = FormAdjustedPoissonModel(config=config).fit(train)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        for match in test.itertuples(index=False):
            fixture = Fixture(
                home_team=match.home_team,
                away_team=match.away_team,
                neutral=bool(match.neutral),
                tournament=match.tournament,
                city=match.city,
                country=match.country,
            )
            predictions.append(model.predict_outcome(fixture))
            observed.append(observed_outcome(int(match.home_score), int(match.away_score)))
    return multiclass_log_loss(predictions, observed)


def _window_prediction_frame(
    results: pd.DataFrame,
    window: EvaluationWindow,
    configs: BaselineModelConfigs,
) -> pd.DataFrame:
    train = results[results["date"] <= window.train_until].copy()
    test = results[
        (results["date"] >= window.test_start) & (results["date"] <= window.test_end)
    ].copy()
    if window.tournament is not None:
        test = test[test["tournament"] == window.tournament].copy()

    if train.empty:
        raise ValueError(f"Training split is empty for {window.name}.")
    if test.empty:
        raise ValueError(f"Test split is empty for {window.name}.")

    elo = EloRatings(config=configs.elo).fit(train)
    poisson = IndependentPoissonModel(config=configs.poisson).fit(train)
    form_poisson = FormAdjustedPoissonModel(config=configs.form_adjusted_poisson).fit(train)

    rows = []
    for match in test.itertuples(index=False):
        fixture = Fixture(
            home_team=match.home_team,
            away_team=match.away_team,
            neutral=bool(match.neutral),
            tournament=match.tournament,
            city=match.city,
            country=match.country,
        )
        elo_probs = elo.predict_outcome(fixture)
        poisson_prediction = poisson.predict(fixture)
        poisson_probs = poisson_prediction.outcome_probabilities
        form_poisson_prediction = form_poisson.predict(fixture)
        form_poisson_probs = form_poisson_prediction.outcome_probabilities
        equal_weight_probs = _average_probabilities([elo_probs, poisson_probs, form_poisson_probs])
        rows.append(
            {
                "window": window.name,
                "date": match.date,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "home_score": int(match.home_score),
                "away_score": int(match.away_score),
                "observed_score": f"{int(match.home_score)}-{int(match.away_score)}",
                "observed": observed_outcome(int(match.home_score), int(match.away_score)),
                "elo_home_win": elo_probs.home_win,
                "elo_draw": elo_probs.draw,
                "elo_away_win": elo_probs.away_win,
                **_score_prediction_columns(
                    "poisson",
                    poisson_prediction,
                    observed_score=(int(match.home_score), int(match.away_score)),
                ),
                "poisson_home_win": poisson_probs.home_win,
                "poisson_draw": poisson_probs.draw,
                "poisson_away_win": poisson_probs.away_win,
                **_score_prediction_columns(
                    "form_adjusted_poisson",
                    form_poisson_prediction,
                    observed_score=(int(match.home_score), int(match.away_score)),
                ),
                "form_adjusted_poisson_home_win": form_poisson_probs.home_win,
                "form_adjusted_poisson_draw": form_poisson_probs.draw,
                "form_adjusted_poisson_away_win": form_poisson_probs.away_win,
                "ensemble_equal_weight_home_win": equal_weight_probs.home_win,
                "ensemble_equal_weight_draw": equal_weight_probs.draw,
                "ensemble_equal_weight_away_win": equal_weight_probs.away_win,
            }
        )
    return pd.DataFrame(rows)


def _window_by_name(name: str) -> EvaluationWindow:
    for window in WORLD_CUP_WINDOWS:
        if window.name == name:
            return window
    raise ValueError(f"Unknown evaluation window: {name}")


def _evaluate_window(results: pd.DataFrame, window: EvaluationWindow) -> list[dict[str, object]]:
    train = results[results["date"] <= window.train_until].copy()
    test = results[
        (results["date"] >= window.test_start) & (results["date"] <= window.test_end)
    ].copy()
    if window.tournament is not None:
        test = test[test["tournament"] == window.tournament].copy()

    if train.empty:
        raise ValueError(f"Training split is empty for {window.name}.")
    if test.empty:
        raise ValueError(f"Test split is empty for {window.name}.")

    elo = EloRatings(config=EloConfig()).fit(train)
    poisson = IndependentPoissonModel(config=PoissonConfig()).fit(train)
    form_poisson = FormAdjustedPoissonModel().fit(train)

    elo_predictions = []
    poisson_predictions = []
    form_poisson_predictions = []
    ensemble_predictions = []
    observed = []

    for row in test.itertuples(index=False):
        fixture = Fixture(
            home_team=row.home_team,
            away_team=row.away_team,
            neutral=bool(row.neutral),
            tournament=row.tournament,
            city=row.city,
            country=row.country,
        )
        elo_probs = elo.predict_outcome(fixture)
        poisson_probs = poisson.predict_outcome(fixture)
        form_poisson_probs = form_poisson.predict_outcome(fixture)
        ensemble_probs = _average_probabilities([elo_probs, poisson_probs, form_poisson_probs])

        elo_predictions.append(elo_probs)
        poisson_predictions.append(poisson_probs)
        form_poisson_predictions.append(form_poisson_probs)
        ensemble_predictions.append(ensemble_probs)
        observed.append(observed_outcome(int(row.home_score), int(row.away_score)))

    return [
        _score_predictions(window, "elo", elo_predictions, observed),
        _score_predictions(window, "poisson", poisson_predictions, observed),
        _score_predictions(window, "form_adjusted_poisson", form_poisson_predictions, observed),
        _score_predictions(window, "ensemble_equal_weight", ensemble_predictions, observed),
    ]


def _score_predictions(
    window: EvaluationWindow,
    model_name: str,
    predictions: list[OutcomeProbabilities],
    observed: list[str],
) -> dict[str, object]:
    return {
        "window": window.name,
        "model": model_name,
        "train_until": window.train_until.isoformat(),
        "test_start": window.test_start.isoformat(),
        "test_end": window.test_end.isoformat(),
        "n_matches": len(observed),
        "log_loss": multiclass_log_loss(predictions, observed),
        "brier_score": brier_score(predictions, observed),
        "exact_score_log_loss": float("nan"),
        "exact_score_accuracy": float("nan"),
        "goal_mae": float("nan"),
        "goal_rmse": float("nan"),
        "total_goal_mae": float("nan"),
    }


def _score_prediction_columns(
    prefix: str,
    prediction: ScorePrediction,
    observed_score: tuple[int, int] | None = None,
) -> dict[str, object]:
    modal = prediction.most_likely_scoreline()
    columns: dict[str, object] = {
        f"{prefix}_predicted_home_score": modal.home_score,
        f"{prefix}_predicted_away_score": modal.away_score,
        f"{prefix}_predicted_score": modal.label,
        f"{prefix}_predicted_score_outcome": _scoreline_outcome(
            modal.home_score,
            modal.away_score,
        ),
        f"{prefix}_predicted_score_probability": modal.probability,
        f"{prefix}_top_scorelines": _format_top_scorelines(prediction),
    }
    if observed_score is None:
        return columns

    home_score, away_score = observed_score
    probability = (
        float(prediction.score_matrix[home_score, away_score])
        if home_score < prediction.score_matrix.shape[0]
        and away_score < prediction.score_matrix.shape[1]
        else 0.0
    )
    columns.update(
        {
            f"{prefix}_observed_score_probability": probability,
            f"{prefix}_home_goal_error": modal.home_score - home_score,
            f"{prefix}_away_goal_error": modal.away_score - away_score,
            f"{prefix}_total_goal_error": (
                modal.home_score + modal.away_score - home_score - away_score
            ),
        }
    )
    return columns


def _primary_score_columns(prediction: ScorePrediction) -> dict[str, object]:
    modal = prediction.most_likely_scoreline()
    return {
        "predicted_home_score": modal.home_score,
        "predicted_away_score": modal.away_score,
        "predicted_score": modal.label,
        "predicted_score_outcome": _scoreline_outcome(modal.home_score, modal.away_score),
        "predicted_score_probability": modal.probability,
        "top_scorelines": _format_top_scorelines(prediction),
    }


def _format_top_scorelines(prediction: ScorePrediction, n: int = 5) -> str:
    return "; ".join(
        f"{scoreline.label} ({scoreline.probability:.3f})"
        for scoreline in prediction.top_scorelines(n=n)
    )


def _scoreline_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"


def _score_metric_summary(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    required = {
        "home_score",
        "away_score",
        f"{prefix}_predicted_home_score",
        f"{prefix}_predicted_away_score",
        f"{prefix}_observed_score_probability",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing score metric columns: {sorted(missing)}")

    predicted_scores = list(
        frame[
            [f"{prefix}_predicted_home_score", f"{prefix}_predicted_away_score"]
        ].itertuples(index=False, name=None)
    )
    observed_scores = list(frame[["home_score", "away_score"]].itertuples(index=False, name=None))
    observed_probabilities = frame[f"{prefix}_observed_score_probability"].to_numpy(dtype=float)
    return {
        "exact_score_log_loss": float(-np.log(np.clip(observed_probabilities, 1e-15, 1.0)).mean()),
        "exact_score_accuracy": exact_score_accuracy(predicted_scores, observed_scores),
        "goal_mae": goal_mae(predicted_scores, observed_scores),
        "goal_rmse": goal_rmse(predicted_scores, observed_scores),
        "total_goal_mae": total_goal_mae(predicted_scores, observed_scores),
    }


def _average_probabilities(predictions: list[OutcomeProbabilities]) -> OutcomeProbabilities:
    if not predictions:
        raise ValueError("At least one prediction is required.")

    n_predictions = len(predictions)
    return OutcomeProbabilities(
        home_win=sum(prediction.home_win for prediction in predictions) / n_predictions,
        draw=sum(prediction.draw for prediction in predictions) / n_predictions,
        away_win=sum(prediction.away_win for prediction in predictions) / n_predictions,
    )


def _validate_fixture_frame(fixtures: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "date",
        "time_local",
        "utc_offset",
        "group",
        "match_number",
        "home_team",
        "away_team",
        "stadium",
        "city",
    }
    missing = required_columns.difference(fixtures.columns)
    if missing:
        raise ValueError(f"Missing fixture columns: {sorted(missing)}")

    normalized = fixtures.loc[:, sorted(required_columns)].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.date
    normalized["match_number"] = pd.to_numeric(
        normalized["match_number"],
        errors="raise",
    ).astype(int)

    string_columns = [
        "time_local",
        "utc_offset",
        "group",
        "home_team",
        "away_team",
        "stadium",
        "city",
    ]
    for column in string_columns:
        normalized[column] = normalized[column].astype("string").str.strip()
        if normalized[column].isna().any() or (normalized[column] == "").any():
            raise ValueError(f"Column {column!r} contains missing or empty values.")

    if normalized["match_number"].duplicated().any():
        raise ValueError("Fixture match numbers must be unique.")
    if (normalized["home_team"] == normalized["away_team"]).any():
        raise ValueError("Fixture table contains identical home and away teams.")

    return normalized.sort_values("match_number").reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame, float_digits: int) -> str:
    if frame.empty:
        return "_No rows._"

    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "NA" if pd.isna(value) else f"{value:.{float_digits}f}"
            )
        else:
            formatted[column] = formatted[column].map(
                lambda value: "NA" if pd.isna(value) else str(value)
            )

    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in formatted.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])
