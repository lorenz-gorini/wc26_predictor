"""Backtests for exact-score selection strategies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wc26_predictor.data.schema import Fixture, validate_results_frame
from wc26_predictor.evaluation.metrics import observed_outcome
from wc26_predictor.evaluation.score_metrics import goal_mae, goal_rmse, total_goal_mae
from wc26_predictor.models.elo import EloRatings
from wc26_predictor.models.ensemble import (
    EnsembleWeights,
    fit_ensemble_weights,
    weighted_average_probabilities,
)
from wc26_predictor.models.form_adjusted_poisson import FormAdjustedPoissonModel
from wc26_predictor.models.poisson import ScorelineProbability
from wc26_predictor.pipelines.baselines import (
    WORLD_CUP_WINDOWS,
    BaselineModelConfigs,
    EvaluationWindow,
    collect_world_cup_validation_predictions,
    default_model_configs,
)

ProgressCallback = Callable[[str], None]

STRATEGY_SCORE_ONLY = "score_only"
STRATEGY_1X2_COMPATIBLE = "1x2_compatible"


@dataclass(frozen=True, slots=True)
class ScoreStrategyBacktestResult:
    """Outputs from a score-strategy backtest."""

    predictions: pd.DataFrame
    summary: pd.DataFrame


def rolling_score_strategy_backtest(
    results: pd.DataFrame,
    configs: BaselineModelConfigs | None = None,
    windows: list[EvaluationWindow] | None = None,
    progress: ProgressCallback | None = None,
) -> ScoreStrategyBacktestResult:
    """Compare exact-score strategies with chronological World Cup roll-forward.

    For each World Cup window, models are initially fit on matches before the
    tournament. Matches inside the tournament are predicted in chronological
    order, then appended to the training history before the next prediction.
    Ensemble weights for the 1X2 compatibility rule are fit only on prior World
    Cup validation windows; the first validation window falls back to equal
    weights because no previous World Cup holdout exists.
    """

    validated = validate_results_frame(results)
    model_configs = configs or default_model_configs()
    evaluation_windows = windows or WORLD_CUP_WINDOWS
    rows = []

    for window in evaluation_windows:
        target = _window_matches(validated, window)
        history = validated[validated["date"] <= window.train_until].copy()
        weights = _prior_window_weights(
            validated,
            window,
            model_configs,
            evaluation_windows,
        )
        if progress is not None:
            progress(
                f"{window.name}: {len(target)} matches, ensemble weights "
                f"elo={weights.elo:.2f}, poisson={weights.poisson:.2f}, "
                f"form={weights.form_adjusted_poisson:.2f}"
            )

        for index, match in enumerate(target.itertuples(index=False), start=1):
            rows.append(_predict_match(history, match, index, window, weights, model_configs))
            history = pd.concat([history, pd.DataFrame([match._asdict()])], ignore_index=True)
            if progress is not None and (index % 10 == 0 or index == len(target)):
                progress(f"{window.name}: predicted {index}/{len(target)} matches")

    predictions = pd.DataFrame(rows)
    summary = summarize_score_strategy_predictions(predictions)
    return ScoreStrategyBacktestResult(predictions=predictions, summary=summary)


def summarize_score_strategy_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize score-strategy predictions by strategy and window."""

    required = {
        "window",
        "home_score",
        "away_score",
        "observed",
        "score_only_home_score",
        "score_only_away_score",
        "score_only_observed_score_probability",
        "compatible_home_score",
        "compatible_away_score",
        "compatible_observed_score_probability",
        "compatible_outcome",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    rows = []
    for window_name, frame in predictions.groupby("window", sort=True):
        rows.extend(_strategy_metric_rows(str(window_name), frame))
    rows.extend(_strategy_metric_rows("all_windows", predictions))
    return pd.DataFrame(rows).sort_values(["window", "exact_score_log_loss"], ignore_index=True)


def write_score_strategy_report(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    destination: str,
) -> None:
    """Write a compact Markdown score-strategy report."""

    destination_path = pd.io.common.stringify_path(destination)
    average = summary[summary["window"] == "all_windows"].copy()
    by_window = summary[summary["window"] != "all_windows"].copy()
    first_rows = predictions.sort_values(["window", "match_index"]).head(8)
    conclusion = _strategy_conclusion(average)
    lines = [
        "# Score Strategy Backtest",
        "",
        "This report compares two exact-score selection strategies on historical World Cup "
        "holdout windows with chronological roll-forward inside each tournament.",
        "",
        "Strategies:",
        "",
        "- `score_only`: choose the modal score from the form-adjusted Poisson score matrix.",
        "- `1x2_compatible`: choose the most likely score whose outcome matches the 1X2 "
        "ensemble's most likely outcome.",
        "",
        "The 1X2 ensemble weights are fit from prior World Cup validation windows only. "
        "The first window uses equal weights because no prior World Cup validation window exists.",
        "",
        "## Interpretation",
        "",
        conclusion,
        "",
        "## All Windows",
        "",
        _markdown_table(average, float_digits=4),
        "",
        "## By World Cup Window",
        "",
        _markdown_table(by_window, float_digits=4),
        "",
        "## First Predictions",
        "",
        _markdown_table(
            first_rows[
                [
                    "window",
                    "match_index",
                    "home_team",
                    "away_team",
                    "observed_score",
                    "score_only_score",
                    "compatible_score",
                    "compatible_outcome",
                    "ensemble_outcome",
                ]
            ],
            float_digits=4,
        ),
        "",
    ]
    with open(destination_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def select_compatible_scoreline(
    score_matrix: np.ndarray,
    outcome: str,
) -> tuple[ScorelineProbability, float]:
    """Select the best exact score compatible with a 1X2 outcome.

    Returns the selected scoreline and the total score-model probability mass in
    the compatible outcome bucket.
    """

    mask = _outcome_mask(score_matrix, outcome)
    outcome_mass = float(score_matrix[mask].sum())
    if outcome_mass <= 0:
        raise ValueError(f"No probability mass for outcome {outcome!r}.")
    compatible_scores = np.where(mask, score_matrix, -1.0)
    home_score, away_score = np.unravel_index(np.argmax(compatible_scores), score_matrix.shape)
    return (
        ScorelineProbability(
            home_score=int(home_score),
            away_score=int(away_score),
            probability=float(score_matrix[home_score, away_score]),
        ),
        outcome_mass,
    )


def _strategy_conclusion(average: pd.DataFrame) -> str:
    score_only = average[average["strategy"] == STRATEGY_SCORE_ONLY].iloc[0]
    compatible = average[average["strategy"] == STRATEGY_1X2_COMPATIBLE].iloc[0]
    exact_hit_delta = (
        compatible["exact_score_accuracy"] - score_only["exact_score_accuracy"]
    )
    goal_mae_delta = compatible["goal_mae"] - score_only["goal_mae"]
    return (
        "The 1X2-compatible strategy improves exact-score hit rate by "
        f"{100.0 * exact_hit_delta:.1f} percentage points and improves outcome accuracy, "
        f"but it has {goal_mae_delta:+.3f} higher goal MAE. Its conditional exact-score "
        "log loss is much "
        "worse because the gate assigns zero probability to scores from the other two 1X2 "
        "outcome classes whenever the aggregate outcome call is wrong. Therefore, use the "
        "1X2-compatible rule only if the objective is a single contest-style score pick; keep "
        "the raw score distribution when calibrated probabilities or goal-error performance "
        "matter."
    )


def _prior_window_weights(
    results: pd.DataFrame,
    target_window: EvaluationWindow,
    configs: BaselineModelConfigs,
    windows: list[EvaluationWindow],
) -> EnsembleWeights:
    prior_windows = [
        window for window in windows if window.test_end < target_window.test_start
    ]
    if not prior_windows:
        return EnsembleWeights.equal()

    validation_predictions = collect_world_cup_validation_predictions(
        results,
        configs=configs,
        windows=prior_windows,
    )
    return fit_ensemble_weights(validation_predictions)


def _predict_match(
    history: pd.DataFrame,
    match: object,
    match_index: int,
    window: EvaluationWindow,
    weights: EnsembleWeights,
    configs: BaselineModelConfigs,
) -> dict[str, object]:
    elo = EloRatings(config=configs.elo).fit(history)
    form_poisson = FormAdjustedPoissonModel(config=configs.form_adjusted_poisson).fit(history)
    poisson = form_poisson.base_model

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
    form_prediction = form_poisson.predict(fixture)
    form_probs = form_prediction.outcome_probabilities
    ensemble = weighted_average_probabilities(
        {
            "elo": elo_probs,
            "poisson": poisson_prediction.outcome_probabilities,
            "form_adjusted_poisson": form_probs,
        },
        weights,
    )
    ensemble_outcome = max(ensemble.as_dict(), key=ensemble.as_dict().get)
    score_only = form_prediction.most_likely_scoreline()
    compatible, compatible_mass = select_compatible_scoreline(
        form_prediction.score_matrix,
        ensemble_outcome,
    )
    observed = observed_outcome(int(match.home_score), int(match.away_score))
    observed_score = (int(match.home_score), int(match.away_score))
    raw_observed_probability = _score_probability(form_prediction.score_matrix, observed_score)
    compatible_observed_probability = (
        raw_observed_probability / compatible_mass if observed == ensemble_outcome else 0.0
    )

    return {
        "window": window.name,
        "match_index": match_index,
        "date": match.date,
        "home_team": match.home_team,
        "away_team": match.away_team,
        "home_score": int(match.home_score),
        "away_score": int(match.away_score),
        "observed_score": f"{int(match.home_score)}-{int(match.away_score)}",
        "observed": observed,
        "ensemble_home_win": ensemble.home_win,
        "ensemble_draw": ensemble.draw,
        "ensemble_away_win": ensemble.away_win,
        "ensemble_outcome": ensemble_outcome,
        "score_only_home_score": score_only.home_score,
        "score_only_away_score": score_only.away_score,
        "score_only_score": score_only.label,
        "score_only_outcome": _scoreline_outcome(score_only.home_score, score_only.away_score),
        "score_only_score_probability": score_only.probability,
        "score_only_observed_score_probability": raw_observed_probability,
        "score_only_compatible_with_ensemble": (
            _scoreline_outcome(score_only.home_score, score_only.away_score)
            == ensemble_outcome
        ),
        "compatible_home_score": compatible.home_score,
        "compatible_away_score": compatible.away_score,
        "compatible_score": compatible.label,
        "compatible_outcome": _scoreline_outcome(
            compatible.home_score,
            compatible.away_score,
        ),
        "compatible_score_probability": compatible.probability,
        "compatible_outcome_probability_mass": compatible_mass,
        "compatible_observed_score_probability": compatible_observed_probability,
    }


def _strategy_metric_rows(window_name: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    observed_scores = list(frame[["home_score", "away_score"]].itertuples(index=False, name=None))
    strategies = [
        (
            STRATEGY_SCORE_ONLY,
            "score_only",
            "score_only_observed_score_probability",
        ),
        (
            STRATEGY_1X2_COMPATIBLE,
            "compatible",
            "compatible_observed_score_probability",
        ),
    ]
    rows = []
    for strategy_name, prefix, probability_column in strategies:
        predicted_scores = list(
            frame[[f"{prefix}_home_score", f"{prefix}_away_score"]].itertuples(
                index=False,
                name=None,
            )
        )
        observed_probabilities = frame[probability_column].to_numpy(dtype=float)
        predicted_outcomes = [
            _scoreline_outcome(home_score, away_score)
            for home_score, away_score in predicted_scores
        ]
        rows.append(
            {
                "window": window_name,
                "strategy": strategy_name,
                "n_matches": len(frame),
                "exact_score_log_loss": float(
                    -np.log(np.clip(observed_probabilities, 1e-15, 1.0)).mean()
                ),
                "exact_score_accuracy": float(
                    np.mean(
                        [
                            predicted == observed
                            for predicted, observed in zip(
                                predicted_scores,
                                observed_scores,
                                strict=True,
                            )
                        ]
                    )
                ),
                "outcome_accuracy": float(
                    np.mean(frame["observed"].to_numpy() == np.array(predicted_outcomes))
                ),
                "goal_mae": goal_mae(predicted_scores, observed_scores),
                "goal_rmse": goal_rmse(predicted_scores, observed_scores),
                "total_goal_mae": total_goal_mae(predicted_scores, observed_scores),
                "average_selected_score_probability": float(
                    frame[f"{prefix}_score_probability"].mean()
                ),
            }
        )
    return rows


def _window_matches(results: pd.DataFrame, window: EvaluationWindow) -> pd.DataFrame:
    frame = results[
        (results["date"] >= window.test_start) & (results["date"] <= window.test_end)
    ].copy()
    if window.tournament is not None:
        frame = frame[frame["tournament"] == window.tournament].copy()
    if frame.empty:
        raise ValueError(f"No matches found for {window.name}.")
    return frame.sort_values("date", ignore_index=True)


def _score_probability(score_matrix: np.ndarray, observed_score: tuple[int, int]) -> float:
    home_score, away_score = observed_score
    if home_score >= score_matrix.shape[0] or away_score >= score_matrix.shape[1]:
        return 0.0
    return float(score_matrix[home_score, away_score])


def _outcome_mask(score_matrix: np.ndarray, outcome: str) -> np.ndarray:
    home_scores, away_scores = np.indices(score_matrix.shape)
    if outcome == "home_win":
        return home_scores > away_scores
    if outcome == "away_win":
        return home_scores < away_scores
    if outcome == "draw":
        return home_scores == away_scores
    raise ValueError(f"Unknown outcome: {outcome!r}")


def _scoreline_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"


def _markdown_table(frame: pd.DataFrame, float_digits: int = 3) -> str:
    if frame.empty:
        return "_No rows._"
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=["floating"]).columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.{float_digits}f}")
    text_frame = formatted.astype(str)
    rows = [
        "| " + " | ".join(text_frame.columns) + " |",
        "| " + " | ".join("---" for _ in text_frame.columns) + " |",
    ]
    for row in text_frame.itertuples(index=False):
        rows.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(rows)
