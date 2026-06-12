"""Market-implied probability benchmarking and ensembles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wc26_predictor.data.odds import team_key
from wc26_predictor.data.schema import OutcomeProbabilities
from wc26_predictor.evaluation.metrics import brier_score, multiclass_log_loss

OUTCOMES = ("home_win", "draw", "away_win")


def match_validation_predictions_to_odds(
    validation_predictions: pd.DataFrame,
    odds: pd.DataFrame,
    model_prefix: str = "ensemble",
) -> pd.DataFrame:
    """Join validation predictions to market probabilities by date and teams."""

    required_predictions = {
        "window",
        "date",
        "home_team",
        "away_team",
        "observed",
        *(f"{model_prefix}_{outcome}" for outcome in OUTCOMES),
    }
    missing_predictions = required_predictions.difference(
        validation_predictions.columns
    )
    if missing_predictions:
        raise ValueError(f"Missing prediction columns: {sorted(missing_predictions)}")

    required_odds = {
        "date",
        "home_team_key",
        "away_team_key",
        "market_home_win",
        "market_draw",
        "market_away_win",
        "market_overround",
    }
    missing_odds = required_odds.difference(odds.columns)
    if missing_odds:
        raise ValueError(f"Missing odds columns: {sorted(missing_odds)}")

    predictions = validation_predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"], errors="raise").dt.date
    predictions["prediction_id"] = np.arange(len(predictions))
    predictions["home_team_key"] = predictions["home_team"].map(team_key)
    predictions["away_team_key"] = predictions["away_team"].map(team_key)

    odds_direct = odds.loc[
        :,
        [
            "date",
            "home_team_key",
            "away_team_key",
            "market_home_win",
            "market_draw",
            "market_away_win",
            "market_overround",
        ],
    ].copy()
    odds_direct = _expand_match_dates(odds_direct)
    odds_direct["odds_match_direction"] = "direct"

    odds_reversed = odds_direct.rename(
        columns={
            "home_team_key": "away_team_key",
            "away_team_key": "home_team_key",
            "market_home_win": "market_away_win",
            "market_away_win": "market_home_win",
        }
    )
    odds_reversed["odds_match_direction"] = "reversed"

    matched = predictions.merge(
        pd.concat([odds_direct, odds_reversed], ignore_index=True),
        left_on=["date", "home_team_key", "away_team_key"],
        right_on=["match_date", "home_team_key", "away_team_key"],
        how="inner",
    )
    if matched.empty:
        return matched
    matched = (
        matched.sort_values(
            ["prediction_id", "odds_date_delta_abs", "odds_match_direction"],
            ascending=[True, True, True],
        )
        .drop_duplicates("prediction_id")
        .drop(columns=["prediction_id", "match_date", "odds_date_delta_abs"])
    )

    model_columns = {
        f"{model_prefix}_{outcome}": f"model_{outcome}" for outcome in OUTCOMES
    }
    matched = matched.rename(columns=model_columns)
    return matched.sort_values(
        ["window", "date", "home_team", "away_team"]
    ).reset_index(drop=True)


def evaluate_market_models(
    matched: pd.DataFrame,
    step: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate model-only, market-only, and validation-gated model+market."""

    if matched.empty:
        empty_metrics = pd.DataFrame(
            columns=[
                "window",
                "model",
                "n_matches",
                "log_loss",
                "brier_score",
                "model_weight",
                "market_weight",
            ]
        )
        empty_weights = pd.DataFrame(
            columns=["window", "model_weight", "market_weight"]
        )
        return empty_metrics, empty_weights

    rows = []
    weight_rows = []
    for window_name, frame in matched.groupby("window", sort=True):
        training = matched[matched["window"] != window_name]
        if training.empty:
            model_weight = 1.0
        else:
            model_weight = fit_model_market_weight(training, step=step)
        market_weight = 1.0 - model_weight
        weight_rows.append(
            {
                "window": window_name,
                "model_weight": model_weight,
                "market_weight": market_weight,
            }
        )

        observed = frame["observed"].tolist()
        rows.append(
            _metric_row(
                window_name, "model_only", _probabilities(frame, "model"), observed
            )
        )
        rows.append(
            _metric_row(
                window_name, "market_only", _probabilities(frame, "market"), observed
            )
        )

        combined = _combined_probabilities(frame, model_weight)
        row = _metric_row(window_name, "model_market_lowo", combined, observed)
        row["model_weight"] = model_weight
        row["market_weight"] = market_weight
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def market_gate_decision(metrics: pd.DataFrame) -> dict[str, object]:
    """Return whether model+market improves LOWO validation over model-only."""

    if metrics.empty:
        return {
            "use_model_market": False,
            "reason": "No matched market odds were available.",
            "model_log_loss": np.nan,
            "model_market_log_loss": np.nan,
        }

    averages = metrics.groupby("model")["log_loss"].mean()
    model_log_loss = float(averages["model_only"])
    market_log_loss = float(averages["market_only"])
    model_market_log_loss = float(averages["model_market_lowo"])
    use_model_market = model_market_log_loss < min(model_log_loss, market_log_loss)
    return {
        "use_model_market": use_model_market,
        "reason": (
            "LOWO model+market improves both model-only and market-only log loss."
            if use_model_market
            else "LOWO model+market does not improve over both model-only and market-only log loss."
        ),
        "model_log_loss": model_log_loss,
        "market_log_loss": market_log_loss,
        "model_market_log_loss": model_market_log_loss,
    }


def fit_model_market_weight(matched: pd.DataFrame, step: float = 0.01) -> float:
    """Fit the model probability weight in a convex model/market ensemble."""

    if matched.empty:
        raise ValueError("Cannot fit a model/market weight on an empty frame.")
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1].")

    observed = matched["observed"].tolist()
    best_loss = np.inf
    best_weight = 1.0
    grid_size = int(round(1.0 / step))
    for index in range(grid_size + 1):
        model_weight = index * step
        predictions = _combined_probabilities(matched, model_weight)
        loss = multiclass_log_loss(predictions, observed)
        if loss < best_loss:
            best_loss = loss
            best_weight = float(model_weight)
    return best_weight


def _metric_row(
    window: str,
    model_name: str,
    predictions: list[OutcomeProbabilities],
    observed: list[str],
) -> dict[str, object]:
    return {
        "window": window,
        "model": model_name,
        "n_matches": len(observed),
        "log_loss": multiclass_log_loss(predictions, observed),
        "brier_score": brier_score(predictions, observed),
        "model_weight": np.nan,
        "market_weight": np.nan,
    }


def _probabilities(frame: pd.DataFrame, prefix: str) -> list[OutcomeProbabilities]:
    columns = [f"{prefix}_{outcome}" for outcome in OUTCOMES]
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing probability columns: {sorted(missing)}")
    return [
        OutcomeProbabilities(home_win=row[0], draw=row[1], away_win=row[2])
        for row in frame.loc[:, columns].itertuples(index=False, name=None)
    ]


def _combined_probabilities(
    frame: pd.DataFrame, model_weight: float
) -> list[OutcomeProbabilities]:
    if model_weight < 0 or model_weight > 1:
        raise ValueError("model_weight must be in [0, 1].")
    market_weight = 1.0 - model_weight
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            OutcomeProbabilities(
                home_win=model_weight * row.model_home_win
                + market_weight * row.market_home_win,
                draw=model_weight * row.model_draw + market_weight * row.market_draw,
                away_win=model_weight * row.model_away_win
                + market_weight * row.market_away_win,
            )
        )
    return rows


def _expand_match_dates(odds: pd.DataFrame) -> pd.DataFrame:
    expanded = []
    odds_dates = pd.to_datetime(odds["date"], errors="raise").dt.date
    base = odds.drop(columns=["date"])
    for delta in (-1, 0, 1):
        shifted = base.copy()
        shifted["odds_date"] = odds_dates
        shifted["match_date"] = (
            pd.to_datetime(shifted["odds_date"]) + pd.to_timedelta(delta, unit="D")
        ).dt.date
        shifted["odds_date_delta_days"] = (
            pd.to_datetime(shifted["odds_date"]) - pd.to_datetime(shifted["match_date"])
        ).dt.days
        shifted["odds_date_delta_abs"] = shifted["odds_date_delta_days"].abs()
        expanded.append(shifted)
    return pd.concat(expanded, ignore_index=True)
