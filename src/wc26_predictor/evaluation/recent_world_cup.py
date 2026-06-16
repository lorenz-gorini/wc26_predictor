"""Recent World Cup holdout evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wc26_predictor.data.odds import team_key
from wc26_predictor.data.schema import Fixture, OutcomeProbabilities, validate_results_frame
from wc26_predictor.evaluation.metrics import brier_score, multiclass_log_loss, observed_outcome
from wc26_predictor.models.elo import EloRatings
from wc26_predictor.models.ensemble import (
    EnsembleWeights,
    add_weighted_ensemble_columns,
    prediction_frame_to_probabilities,
    weighted_average_probabilities,
)
from wc26_predictor.models.form_adjusted_poisson import FormAdjustedPoissonModel
from wc26_predictor.models.market import (
    fit_model_market_weight,
    match_validation_predictions_to_odds,
)
from wc26_predictor.models.poisson import IndependentPoissonModel
from wc26_predictor.pipelines.baselines import (
    BaselineModelConfigs,
    collect_world_cup_validation_predictions,
    default_model_configs,
    fit_world_cup_ensemble_weights,
)

OUTCOMES = ("home_win", "draw", "away_win")
PROBABILITY_COLUMNS = {
    "home_win": "market_home_win",
    "draw": "market_draw",
    "away_win": "market_away_win",
}


@dataclass(frozen=True, slots=True)
class RecentHoldoutResult:
    """Recent holdout predictions and aggregate metrics."""

    predictions: pd.DataFrame
    summary: pd.DataFrame
    market_weight: float | None


def evaluate_recent_world_cup_holdouts(
    results: pd.DataFrame,
    fixtures: pd.DataFrame | None = None,
    market_probabilities: pd.DataFrame | None = None,
    historical_odds: pd.DataFrame | None = None,
    holdout_sizes: Iterable[int] = (5, 10, 20),
    configs: BaselineModelConfigs | None = None,
    tournament: str = "FIFA World Cup",
    tournament_start: str = "2026-06-01",
) -> RecentHoldoutResult:
    """Evaluate models on the latest completed World Cup matches.

    For each holdout size, the models are fit on all data before the holdout
    block. The target block is the last ``k`` completed 2026 World Cup matches,
    ordered by fixture match number when fixtures are available.
    """

    validated_results = validate_results_frame(results)
    model_configs = configs or default_model_configs()
    completed = _completed_current_world_cup(
        validated_results,
        fixtures=fixtures,
        tournament=tournament,
        tournament_start=tournament_start,
    )
    if completed.empty:
        raise ValueError("No completed current World Cup matches were found.")

    current_market = (
        validate_market_probability_frame(market_probabilities)
        if market_probabilities is not None
        else None
    )
    market_weight = _historical_market_weight(
        validated_results,
        historical_odds,
        model_configs,
    )

    rows = []
    for holdout_size in _normalized_holdout_sizes(holdout_sizes):
        target = completed.tail(min(holdout_size, len(completed))).copy()
        train = _training_results_before_target(validated_results, completed, target)
        rows.extend(
            _predict_target_block(
                train=train,
                target=target,
                holdout_size=holdout_size,
                configs=model_configs,
                current_market=current_market,
                market_weight=market_weight,
            )
        )

    predictions = pd.DataFrame(rows)
    summary = summarize_recent_holdout_predictions(predictions)
    return RecentHoldoutResult(
        predictions=predictions,
        summary=summary,
        market_weight=market_weight,
    )


def summarize_recent_holdout_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize recent holdout predictions by holdout size and model."""

    required = {"holdout_size", "observed", *(f"ensemble_{outcome}" for outcome in OUTCOMES)}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing recent holdout prediction columns: {sorted(missing)}")

    rows = []
    for holdout_size, frame in predictions.groupby("holdout_size", sort=True):
        models = ["elo", "poisson", "form_adjusted_poisson", "ensemble"]
        if _has_probability_columns(frame, "market"):
            models.append("market")
        if _has_probability_columns(frame, "model_market"):
            models.append("model_market")
        for model in models:
            rows.append(_metric_row(int(holdout_size), model, frame))
    return pd.DataFrame(rows).sort_values(
        ["holdout_size", "log_loss", "model"],
        ignore_index=True,
    )


def write_recent_holdout_report(
    result: RecentHoldoutResult,
    destination: str | Path,
) -> None:
    """Write a compact Markdown report for recent World Cup holdouts."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)

    latest_predictions = result.predictions.sort_values(
        ["holdout_size", "match_number"],
        ascending=[True, True],
    )
    max_size = int(latest_predictions["holdout_size"].max())
    latest_block = latest_predictions[latest_predictions["holdout_size"] == max_size]
    market_note = (
        f"Historical model-market weight: model={result.market_weight:.2f}, "
        f"market={1.0 - result.market_weight:.2f}."
        if result.market_weight is not None
        else "No historical market weight was available."
    )
    if not _has_probability_columns(result.predictions, "market"):
        market_note += " No current 2026 market probabilities were matched."

    lines = [
        "# Recent World Cup Holdout Backtest",
        "",
        "Models are fit only on matches before each holdout block. The target blocks are "
        "the latest completed 2026 World Cup matches available in local data.",
        "",
        market_note,
        "",
        "## Summary",
        "",
        _markdown_table(result.summary, float_digits=4),
        "",
        f"## Last {max_size} Match Predictions",
        "",
        _markdown_table(
            latest_block[
                [
                    "match_number",
                    "date",
                    "home_team",
                    "away_team",
                    "observed_score",
                    "observed",
                    "ensemble_predicted_outcome",
                    "ensemble_observed_probability",
                ]
            ],
            float_digits=4,
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_market_probability_frame(probabilities: pd.DataFrame) -> pd.DataFrame:
    """Validate current-match market probabilities keyed by fixture match number."""

    required = {"match_number", *PROBABILITY_COLUMNS.values()}
    missing = required.difference(probabilities.columns)
    if missing:
        raise ValueError(f"Missing market probability columns: {sorted(missing)}")

    optional = [column for column in ["market_source"] if column in probabilities.columns]
    normalized = probabilities.loc[
        :,
        ["match_number", *PROBABILITY_COLUMNS.values(), *optional],
    ].copy()
    normalized["match_number"] = pd.to_numeric(
        normalized["match_number"],
        errors="raise",
    )
    if (normalized["match_number"] % 1 != 0).any():
        raise ValueError("Market probability match numbers must be integers.")
    normalized["match_number"] = normalized["match_number"].astype(int)
    if normalized["match_number"].duplicated().any():
        duplicated = sorted(
            normalized.loc[normalized["match_number"].duplicated(), "match_number"].unique()
        )
        raise ValueError(f"Duplicate market probability match numbers: {duplicated}")

    for column in PROBABILITY_COLUMNS.values():
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        if ((normalized[column] < 0.0) | (normalized[column] > 1.0)).any():
            raise ValueError(f"Column {column!r} must contain probabilities in [0, 1].")

    probability_sum = normalized[list(PROBABILITY_COLUMNS.values())].sum(axis=1)
    if not np.allclose(probability_sum, 1.0, atol=1e-3):
        raise ValueError("Market probabilities must sum to one.")
    return normalized.sort_values("match_number", ignore_index=True)


def load_market_probabilities_csv(path: str | Path) -> pd.DataFrame:
    """Load current-match market probabilities from CSV."""

    return validate_market_probability_frame(pd.read_csv(path))


def _completed_current_world_cup(
    results: pd.DataFrame,
    fixtures: pd.DataFrame | None,
    tournament: str,
    tournament_start: str,
) -> pd.DataFrame:
    start_date = pd.to_datetime(tournament_start, errors="raise").date()
    completed = results[
        (results["tournament"] == tournament) & (results["date"] >= start_date)
    ].copy()
    if fixtures is not None:
        completed = _attach_fixture_match_numbers(completed, fixtures)
        return completed.sort_values("match_number", ignore_index=True)

    completed["match_number"] = np.arange(1, len(completed) + 1)
    return completed.sort_values(["date", "home_team", "away_team"], ignore_index=True)


def _attach_fixture_match_numbers(results: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "match_number", "home_team", "away_team"}
    missing = required.difference(fixtures.columns)
    if missing:
        raise ValueError(f"Missing fixture columns: {sorted(missing)}")

    fixture_frame = fixtures.loc[:, list(required)].copy()
    fixture_frame["date"] = pd.to_datetime(fixture_frame["date"], errors="raise").dt.date
    fixture_frame["match_number"] = pd.to_numeric(
        fixture_frame["match_number"],
        errors="raise",
    ).astype(int)
    fixture_frame["home_team_key"] = fixture_frame["home_team"].map(team_key)
    fixture_frame["away_team_key"] = fixture_frame["away_team"].map(team_key)

    completed = results.copy()
    completed["home_team_key"] = completed["home_team"].map(team_key)
    completed["away_team_key"] = completed["away_team"].map(team_key)
    matched = completed.merge(
        fixture_frame.loc[
            :,
            ["date", "match_number", "home_team_key", "away_team_key"],
        ],
        on=["date", "home_team_key", "away_team_key"],
        how="left",
        validate="one_to_one",
    )
    if matched["match_number"].isna().any():
        missing_rows = matched.loc[
            matched["match_number"].isna(),
            ["date", "home_team", "away_team"],
        ]
        raise ValueError(
            "Could not match completed results to fixture match numbers: "
            f"{missing_rows.to_dict(orient='records')}"
        )
    return matched.drop(columns=["home_team_key", "away_team_key"])


def _normalized_holdout_sizes(holdout_sizes: Iterable[int]) -> list[int]:
    values = sorted({int(size) for size in holdout_sizes})
    if not values or any(size <= 0 for size in values):
        raise ValueError("holdout_sizes must contain positive integers.")
    return values


def _training_results_before_target(
    results: pd.DataFrame,
    completed: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    first_target_match = int(target["match_number"].min())
    future_current_keys = _match_keys(completed[completed["match_number"] >= first_target_match])
    train = results[~results.apply(_row_key, axis=1).isin(future_current_keys)].copy()
    if train.empty:
        raise ValueError("Recent holdout training split is empty.")
    return train


def _predict_target_block(
    train: pd.DataFrame,
    target: pd.DataFrame,
    holdout_size: int,
    configs: BaselineModelConfigs,
    current_market: pd.DataFrame | None,
    market_weight: float | None,
) -> list[dict[str, object]]:
    ensemble_weights = _fit_ensemble_weights_or_equal(train, configs)
    elo = EloRatings(config=configs.elo).fit(train)
    poisson = IndependentPoissonModel(config=configs.poisson).fit(train)
    form_poisson = FormAdjustedPoissonModel(config=configs.form_adjusted_poisson).fit(train)

    market_by_match = (
        current_market.set_index("match_number") if current_market is not None else None
    )

    rows = []
    for match in target.itertuples(index=False):
        fixture = Fixture(
            home_team=match.home_team,
            away_team=match.away_team,
            neutral=bool(match.neutral),
            tournament=match.tournament,
            city=match.city,
            country=match.country,
        )
        elo_probs = elo.predict_outcome(fixture)
        poisson_probs = poisson.predict_outcome(fixture)
        form_probs = form_poisson.predict_outcome(fixture)
        ensemble = weighted_average_probabilities(
            {
                "elo": elo_probs,
                "poisson": poisson_probs,
                "form_adjusted_poisson": form_probs,
            },
            ensemble_weights,
        )
        observed = observed_outcome(int(match.home_score), int(match.away_score))
        row = {
            "holdout_size": holdout_size,
            "training_matches": len(train),
            "match_number": int(match.match_number),
            "date": match.date,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "home_score": int(match.home_score),
            "away_score": int(match.away_score),
            "observed_score": f"{int(match.home_score)}-{int(match.away_score)}",
            "observed": observed,
            "ensemble_weight_elo": ensemble_weights.elo,
            "ensemble_weight_poisson": ensemble_weights.poisson,
            "ensemble_weight_form_adjusted_poisson": (
                ensemble_weights.form_adjusted_poisson
            ),
            **_probability_columns("elo", elo_probs, observed),
            **_probability_columns("poisson", poisson_probs, observed),
            **_probability_columns("form_adjusted_poisson", form_probs, observed),
            **_probability_columns("ensemble", ensemble, observed),
        }
        if market_by_match is not None and int(match.match_number) in market_by_match.index:
            market_row = market_by_match.loc[int(match.match_number)]
            market = OutcomeProbabilities(
                home_win=float(market_row.market_home_win),
                draw=float(market_row.market_draw),
                away_win=float(market_row.market_away_win),
            )
            row.update(_probability_columns("market", market, observed))
            if market_weight is not None:
                combined = _combine_model_market(ensemble, market, market_weight)
                row["model_market_model_weight"] = market_weight
                row["model_market_market_weight"] = 1.0 - market_weight
                row.update(_probability_columns("model_market", combined, observed))
        rows.append(row)
    return rows


def _historical_market_weight(
    results: pd.DataFrame,
    odds: pd.DataFrame | None,
    configs: BaselineModelConfigs,
) -> float | None:
    if odds is None:
        return None
    try:
        weights = _fit_ensemble_weights_or_equal(results, configs)
        validation = collect_world_cup_validation_predictions(results, configs=configs)
        validation = add_weighted_ensemble_columns(validation, weights)
        matched = match_validation_predictions_to_odds(validation, odds)
    except ValueError:
        return None
    if matched.empty:
        return None
    return fit_model_market_weight(matched)


def _fit_ensemble_weights_or_equal(
    results: pd.DataFrame,
    configs: BaselineModelConfigs,
) -> EnsembleWeights:
    try:
        return fit_world_cup_ensemble_weights(results, configs=configs)
    except ValueError:
        return EnsembleWeights.equal()


def _probability_columns(
    prefix: str,
    probabilities: OutcomeProbabilities,
    observed: str,
) -> dict[str, object]:
    values = probabilities.as_dict()
    predicted = max(values, key=values.get)
    return {
        f"{prefix}_home_win": probabilities.home_win,
        f"{prefix}_draw": probabilities.draw,
        f"{prefix}_away_win": probabilities.away_win,
        f"{prefix}_predicted_outcome": predicted,
        f"{prefix}_observed_probability": values[observed],
    }


def _combine_model_market(
    model: OutcomeProbabilities,
    market: OutcomeProbabilities,
    model_weight: float,
) -> OutcomeProbabilities:
    market_weight = 1.0 - model_weight
    return OutcomeProbabilities(
        home_win=model_weight * model.home_win + market_weight * market.home_win,
        draw=model_weight * model.draw + market_weight * market.draw,
        away_win=model_weight * model.away_win + market_weight * market.away_win,
    )


def _metric_row(holdout_size: int, model: str, frame: pd.DataFrame) -> dict[str, object]:
    probabilities = prediction_frame_to_probabilities(frame, model)
    observed = frame["observed"].tolist()
    predicted = frame[f"{model}_predicted_outcome"].tolist()
    return {
        "holdout_size": holdout_size,
        "model": model,
        "n_matches": len(frame),
        "log_loss": multiclass_log_loss(probabilities, observed),
        "brier_score": brier_score(probabilities, observed),
        "outcome_accuracy": float(np.mean(np.array(predicted) == np.array(observed))),
        "average_observed_probability": float(frame[f"{model}_observed_probability"].mean()),
    }


def _has_probability_columns(frame: pd.DataFrame, prefix: str) -> bool:
    return {f"{prefix}_{outcome}" for outcome in OUTCOMES}.issubset(frame.columns)


def _match_keys(frame: pd.DataFrame) -> set[tuple[object, str, str]]:
    return set(frame.apply(_row_key, axis=1))


def _row_key(row: pd.Series) -> tuple[object, str, str]:
    return (row["date"], str(row["home_team"]), str(row["away_team"]))


def _markdown_table(frame: pd.DataFrame, float_digits: int) -> str:
    if frame.empty:
        return "_No rows._"
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=["floating"]).columns:
        formatted[column] = formatted[column].map(
            lambda value: "NA" if pd.isna(value) else f"{value:.{float_digits}f}"
        )
    text_frame = formatted.astype(str)
    rows = [
        "| " + " | ".join(text_frame.columns) + " |",
        "| " + " | ".join("---" for _ in text_frame.columns) + " |",
    ]
    for row in text_frame.itertuples(index=False):
        rows.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(rows)
