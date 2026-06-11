"""Generate a standalone model-validation dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from wc26_predictor.data.ingest_results import load_results_csv
from wc26_predictor.evaluation.metrics import OUTCOME_ORDER
from wc26_predictor.models.elo import EloConfig
from wc26_predictor.models.ensemble import (
    EnsembleWeights,
    add_weighted_ensemble_columns,
    fit_ensemble_weights,
)
from wc26_predictor.models.form_adjusted_poisson import FormAdjustedPoissonConfig
from wc26_predictor.models.poisson import PoissonConfig
from wc26_predictor.pipelines.baselines import (
    BaselineModelConfigs,
    collect_world_cup_validation_predictions,
)

INTERNAL_MODELS = [
    "elo",
    "poisson",
    "form_adjusted_poisson",
    "ensemble_equal_weight",
    "ensemble_calibrated_lowo",
]

MODEL_LABELS = {
    "elo": "Elo",
    "poisson": "Poisson",
    "form_adjusted_poisson": "Form Poisson",
    "ensemble_equal_weight": "Equal ensemble",
    "ensemble_calibrated_lowo": "LOWO ensemble",
    "ensemble_final_weight": "Final ensemble",
    "model_only": "Model only",
    "market_only": "Market only",
    "model_market_lowo": "Model + market",
}

MODEL_COLORS = {
    "elo": "#2f6f73",
    "poisson": "#b75d3b",
    "form_adjusted_poisson": "#7f5aa2",
    "ensemble_equal_weight": "#607d8b",
    "ensemble_calibrated_lowo": "#bf8f2f",
    "ensemble_final_weight": "#1f5fbf",
    "model_only": "#1f5fbf",
    "market_only": "#2f6f73",
    "model_market_lowo": "#bf8f2f",
}


@dataclass(frozen=True, slots=True)
class ValidationDashboardConfig:
    """Configuration for validation dashboard generation."""

    bootstrap_samples: int = 2_000
    random_seed: int = 2026
    confidence_level: float = 0.95


def generate_validation_dashboard(
    project_root: Path,
    destination: Path | None = None,
    config: ValidationDashboardConfig | None = None,
) -> Path:
    """Generate a standalone HTML dashboard with validation diagnostics."""

    root = Path(project_root)
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination or reports_dir / "model_performance_dashboard.html"
    dashboard_config = config or ValidationDashboardConfig()

    selected_configs = _load_model_configs(
        processed_dir / "world_cup_2026_selected_hyperparameters.csv"
    )
    ensemble_weights = _load_ensemble_weights(
        processed_dir / "world_cup_2026_ensemble_weights.csv"
    )
    validation_predictions = _validation_predictions(root, selected_configs, ensemble_weights)
    per_match = _per_match_internal_scores(validation_predictions)

    metric_summary = _metric_summary(per_match, dashboard_config)
    score_metric_summary = _score_metric_summary(per_match, dashboard_config)
    delta_summary = _paired_delta_summary(
        per_match=per_match,
        baseline_model="elo",
        models=[
            "poisson",
            "form_adjusted_poisson",
            "ensemble_equal_weight",
            "ensemble_calibrated_lowo",
        ],
        config=dashboard_config,
    )
    window_summary = _window_metric_summary(per_match)

    market_matches = _read_optional_csv(processed_dir / "world_cup_market_validation_matches.csv")
    market_metrics = _read_optional_csv(processed_dir / "world_cup_market_validation_metrics.csv")
    market_per_match = (
        _per_match_market_scores(market_matches) if market_matches is not None else pd.DataFrame()
    )
    market_delta_summary = (
        _paired_delta_summary(
            per_match=market_per_match,
            baseline_model="market_only",
            models=["model_only", "model_market_lowo"],
            config=dashboard_config,
        )
        if not market_per_match.empty
        else pd.DataFrame()
    )

    forecasts = pd.read_csv(processed_dir / "world_cup_2026_baseline_forecasts.csv")
    future_uncertainty = _future_prediction_uncertainty(
        forecasts=forecasts,
        validation_match_count=validation_predictions["match_id"].nunique(),
    )

    html = _build_dashboard_html(
        metric_summary=metric_summary,
        score_metric_summary=score_metric_summary,
        delta_summary=delta_summary,
        window_summary=window_summary,
        market_metrics=market_metrics,
        market_delta_summary=market_delta_summary,
        future_uncertainty=future_uncertainty,
        validation_match_count=validation_predictions["match_id"].nunique(),
        config=dashboard_config,
    )
    destination_path.write_text(html, encoding="utf-8")
    return destination_path


def _load_model_configs(path: Path) -> BaselineModelConfigs:
    frame = pd.read_csv(path)
    required = {"model", "parameter", "value"}
    _require_columns(frame, required, "selected hyperparameters")
    values = {
        (str(row.model), str(row.parameter)): row.value
        for row in frame.itertuples(index=False)
    }
    poisson = PoissonConfig(
        prior_strength=float(values[("poisson", "prior_strength")]),
        recency_half_life_days=int(values[("poisson", "recency_half_life_days")]),
        use_tournament_importance=_parse_bool(
            values[("poisson", "use_tournament_importance")]
        ),
    )
    return BaselineModelConfigs(
        elo=EloConfig(
            k_factor=float(values[("elo", "k_factor")]),
            draw_probability=float(values[("elo", "draw_probability")]),
            home_advantage=float(values[("elo", "home_advantage")]),
            neutral_home_advantage=float(values[("elo", "neutral_home_advantage")]),
        ),
        poisson=poisson,
        form_adjusted_poisson=FormAdjustedPoissonConfig(
            poisson=poisson,
            form_window_matches=int(values[("form_adjusted_poisson", "form_window_matches")]),
            form_prior_matches=float(values[("form_adjusted_poisson", "form_prior_matches")]),
            form_strength=float(values[("form_adjusted_poisson", "form_strength")]),
        ),
    )


def _load_ensemble_weights(path: Path) -> EnsembleWeights:
    frame = pd.read_csv(path)
    _require_columns(frame, {"model", "weight"}, "ensemble weights")
    weights = {str(row.model): float(row.weight) for row in frame.itertuples(index=False)}
    return EnsembleWeights(
        elo=weights["elo"],
        poisson=weights["poisson"],
        form_adjusted_poisson=weights["form_adjusted_poisson"],
    )


def _validation_predictions(
    project_root: Path,
    configs: BaselineModelConfigs,
    ensemble_weights: EnsembleWeights,
) -> pd.DataFrame:
    results = load_results_csv(project_root / "data" / "processed" / "international_results.csv")
    predictions = collect_world_cup_validation_predictions(results, configs=configs)
    predictions = add_weighted_ensemble_columns(
        predictions,
        ensemble_weights,
        output_prefix="ensemble_final_weight",
    )
    frames = []
    for window_name, frame in predictions.groupby("window", sort=True):
        training = predictions[predictions["window"] != window_name]
        weights = fit_ensemble_weights(training)
        calibrated = add_weighted_ensemble_columns(
            frame,
            weights,
            output_prefix="ensemble_calibrated_lowo",
        )
        frames.append(calibrated)
    output = pd.concat(frames, ignore_index=True)
    output["match_id"] = _match_id(output)
    return output


def _per_match_internal_scores(validation_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in INTERNAL_MODELS:
        rows.extend(_score_rows(validation_predictions, model))
    return pd.DataFrame(rows)


def _per_match_market_scores(market_matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = market_matches.copy()
    frame["match_id"] = _match_id(frame)
    for model in ["model_only", "market_only"]:
        prefix = "model" if model == "model_only" else "market"
        rows.extend(_score_rows(frame, prefix, model_name=model))

    combined = frame.copy()
    weight_frame = _market_lowo_weights(frame)
    combined = combined.merge(weight_frame, on="window", how="left", validate="many_to_one")
    for outcome in OUTCOME_ORDER:
        combined[f"model_market_lowo_{outcome}"] = (
            combined["model_weight"] * combined[f"model_{outcome}"]
            + combined["market_weight"] * combined[f"market_{outcome}"]
        )
    rows.extend(_score_rows(combined, "model_market_lowo", model_name="model_market_lowo"))
    return pd.DataFrame(rows)


def _market_lowo_weights(matched: pd.DataFrame) -> pd.DataFrame:
    from wc26_predictor.models.market import fit_model_market_weight

    rows = []
    for window_name in matched["window"].unique():
        training = matched[matched["window"] != window_name]
        model_weight = 1.0 if training.empty else fit_model_market_weight(training)
        rows.append(
            {
                "window": window_name,
                "model_weight": model_weight,
                "market_weight": 1.0 - model_weight,
            }
        )
    return pd.DataFrame(rows)


def _score_rows(
    frame: pd.DataFrame,
    prefix: str,
    model_name: str | None = None,
) -> list[dict[str, object]]:
    probabilities = frame[[f"{prefix}_{outcome}" for outcome in OUTCOME_ORDER]].to_numpy(
        dtype=float
    )
    observed = frame["observed"].map({name: index for index, name in enumerate(OUTCOME_ORDER)})
    if observed.isna().any():
        bad_values = sorted(frame.loc[observed.isna(), "observed"].unique())
        raise ValueError(f"Invalid observed outcomes: {bad_values}")
    observed_indices = observed.to_numpy(dtype=int)
    observed_matrix = np.zeros_like(probabilities)
    observed_matrix[np.arange(len(observed_indices)), observed_indices] = 1.0
    log_losses = -np.log(
        np.clip(probabilities[np.arange(len(observed_indices)), observed_indices], 1e-15, 1.0)
    )
    brier_scores = np.sum((probabilities - observed_matrix) ** 2, axis=1)
    output_model = model_name or prefix
    rows = []
    for position, row in enumerate(frame.itertuples(index=False)):
        item = {
            "match_id": row.match_id,
            "window": row.window,
            "date": row.date,
            "home_team": row.home_team,
            "away_team": row.away_team,
            "model": output_model,
            "log_loss": float(log_losses[position]),
            "brier_score": float(brier_scores[position]),
        }
        if f"{prefix}_observed_score_probability" in frame.columns:
            probability = float(getattr(row, f"{prefix}_observed_score_probability"))
            predicted_home = int(getattr(row, f"{prefix}_predicted_home_score"))
            predicted_away = int(getattr(row, f"{prefix}_predicted_away_score"))
            observed_home = int(row.home_score)
            observed_away = int(row.away_score)
            item.update(
                {
                    "exact_score_log_loss": float(-np.log(np.clip(probability, 1e-15, 1.0))),
                    "exact_score_hit": float(
                        predicted_home == observed_home and predicted_away == observed_away
                    ),
                    "goal_abs_error": float(
                        (
                            abs(predicted_home - observed_home)
                            + abs(predicted_away - observed_away)
                        )
                        / 2.0
                    ),
                    "total_goal_abs_error": float(
                        abs(predicted_home + predicted_away - observed_home - observed_away)
                    ),
                }
            )
        else:
            item.update(
                {
                    "exact_score_log_loss": np.nan,
                    "exact_score_hit": np.nan,
                    "goal_abs_error": np.nan,
                    "total_goal_abs_error": np.nan,
                }
            )
        rows.append(item)
    return rows


def _metric_summary(
    per_match: pd.DataFrame,
    config: ValidationDashboardConfig,
) -> pd.DataFrame:
    rows = []
    for model, frame in per_match.groupby("model", sort=False):
        row = {
            "model": model,
            "label": MODEL_LABELS.get(model, model),
            "n_matches": len(frame),
            "color": MODEL_COLORS.get(model, "#555555"),
        }
        for metric in ["log_loss", "brier_score"]:
            lower, upper = _bootstrap_ci(frame[metric].to_numpy(dtype=float), config)
            row[metric] = float(frame[metric].mean())
            row[f"{metric}_lower"] = lower
            row[f"{metric}_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows).sort_values("log_loss", ignore_index=True)


def _window_metric_summary(per_match: pd.DataFrame) -> pd.DataFrame:
    return (
        per_match.groupby(["window", "model"], as_index=False)
        .agg(
            n_matches=("match_id", "size"),
            log_loss=("log_loss", "mean"),
            brier_score=("brier_score", "mean"),
        )
        .sort_values(["window", "log_loss"], ignore_index=True)
    )


def _score_metric_summary(
    per_match: pd.DataFrame,
    config: ValidationDashboardConfig,
) -> pd.DataFrame:
    score_frame = per_match.dropna(subset=["exact_score_log_loss"]).copy()
    rows = []
    for model, frame in score_frame.groupby("model", sort=False):
        row = {
            "model": model,
            "label": MODEL_LABELS.get(model, model),
            "n_matches": len(frame),
            "color": MODEL_COLORS.get(model, "#555555"),
            "exact_score_accuracy": float(frame["exact_score_hit"].mean()),
            "goal_mae": float(frame["goal_abs_error"].mean()),
            "total_goal_mae": float(frame["total_goal_abs_error"].mean()),
        }
        lower, upper = _bootstrap_ci(frame["exact_score_log_loss"].to_numpy(dtype=float), config)
        row["exact_score_log_loss"] = float(frame["exact_score_log_loss"].mean())
        row["exact_score_log_loss_lower"] = lower
        row["exact_score_log_loss_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows).sort_values("exact_score_log_loss", ignore_index=True)


def _paired_delta_summary(
    per_match: pd.DataFrame,
    baseline_model: str,
    models: list[str],
    config: ValidationDashboardConfig,
) -> pd.DataFrame:
    if per_match.empty:
        return pd.DataFrame()
    pivot = per_match.pivot(index="match_id", columns="model", values="log_loss")
    missing = [model for model in [baseline_model, *models] if model not in pivot.columns]
    if missing:
        raise ValueError(f"Missing models for paired comparison: {missing}")
    rows = []
    for model in models:
        delta = (pivot[model] - pivot[baseline_model]).dropna().to_numpy(dtype=float)
        lower, upper = _bootstrap_ci(delta, config)
        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "baseline": baseline_model,
                "delta_log_loss": float(delta.mean()),
                "delta_log_loss_lower": lower,
                "delta_log_loss_upper": upper,
                "n_matches": len(delta),
                "color": MODEL_COLORS.get(model, "#555555"),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_log_loss", ignore_index=True)


def _bootstrap_ci(
    values: np.ndarray,
    config: ValidationDashboardConfig,
) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty array.")
    rng = np.random.default_rng(config.random_seed)
    samples = rng.choice(values, size=(config.bootstrap_samples, values.size), replace=True)
    means = samples.mean(axis=1)
    alpha = 1.0 - config.confidence_level
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def _future_prediction_uncertainty(
    forecasts: pd.DataFrame,
    validation_match_count: int,
) -> pd.DataFrame:
    required = {
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
    }
    _require_columns(forecasts, required, "future forecasts")

    model_prefixes = [
        "elo",
        "poisson",
        "form_poisson",
        "ensemble_equal_weight",
        "ensemble",
    ]
    rows = []
    for row in forecasts.itertuples(index=False):
        probabilities = {
            "home_win": float(row.ensemble_home_win),
            "draw": float(row.ensemble_draw),
            "away_win": float(row.ensemble_away_win),
        }
        most_likely_outcome = max(probabilities, key=probabilities.get)
        most_likely_probability = probabilities[most_likely_outcome]
        model_values = np.array(
            [
                getattr(row, f"{prefix}_{most_likely_outcome}")
                for prefix in model_prefixes
            ],
            dtype=float,
        )
        disagreement_sd = float(model_values.std(ddof=1))
        backtest_sampling_se = float(
            np.sqrt(
                most_likely_probability
                * (1.0 - most_likely_probability)
                / validation_match_count
            )
        )
        total_sd = float(np.sqrt(disagreement_sd**2 + backtest_sampling_se**2))
        rows.append(
            {
                "match_number": int(row.match_number),
                "group": row.group,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "predicted_score": row.predicted_score,
                "predicted_score_outcome": row.predicted_score_outcome,
                "predicted_score_probability": float(row.predicted_score_probability),
                "most_likely_outcome": most_likely_outcome,
                "most_likely_probability": most_likely_probability,
                "model_disagreement_sd": disagreement_sd,
                "backtest_sampling_se": backtest_sampling_se,
                "total_probability_sd": total_sd,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "total_probability_sd",
        ascending=False,
        ignore_index=True,
    )


def _build_dashboard_html(
    metric_summary: pd.DataFrame,
    score_metric_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
    market_metrics: pd.DataFrame | None,
    market_delta_summary: pd.DataFrame,
    future_uncertainty: pd.DataFrame,
    validation_match_count: int,
    config: ValidationDashboardConfig,
) -> str:
    best_internal = metric_summary.iloc[0]
    market_average = _market_average(market_metrics)
    market_card = (
        f"{market_average.iloc[0]['label']} ({market_average.iloc[0]['log_loss']:.3f})"
        if market_average is not None and not market_average.empty
        else "No matched market data"
    )
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>World Cup 2026 Model Performance Dashboard</title>",
        f"<style>{_dashboard_css()}</style>",
        "</head>",
        "<body>",
        '<main class="page">',
        '<section class="hero">',
        "<div>",
        "<p class=\"eyebrow\">World Cup 2026 predictor</p>",
        "<h1>Model performance dashboard</h1>",
        "<p class=\"lede\">Backtesting, bookmaker benchmarking, model contribution "
        "diagnostics, and future forecast uncertainty for the current forecasting "
        "pipeline.</p>",
        "</div>",
        '<div class="hero-note">',
        f"<strong>{validation_match_count}</strong><span>historical World Cup holdout "
        "matches</span>",
        "</div>",
        "</section>",
        '<section class="kpi-grid">',
        _kpi_card(
            "Best internal model",
            str(best_internal["label"]),
            f"log loss {best_internal['log_loss']:.3f}",
        ),
        _kpi_card(
            "Market benchmark",
            market_card,
            "lower log loss is better",
        ),
        _kpi_card(
            "Bootstrap",
            f"{config.confidence_level:.0%} intervals",
            f"{config.bootstrap_samples:,} resamples, fixed seed {config.random_seed}",
        ),
        "</section>",
        '<section class="grid two">',
        _panel(
            "Average backtest performance",
            "Bootstrapped intervals use match-level resampling. Lower values are better.",
            _horizontal_bar_chart(
                metric_summary,
                value_column="log_loss",
                lower_column="log_loss_lower",
                upper_column="log_loss_upper",
                title="Log loss",
                x_format="{:.3f}",
            ),
        ),
        _panel(
            "Brier score",
            "The Brier score gives a smoother penalty for probability mass placed on "
            "wrong outcomes.",
            _horizontal_bar_chart(
                metric_summary,
                value_column="brier_score",
                lower_column="brier_score_lower",
                upper_column="brier_score_upper",
                title="Brier score",
                x_format="{:.3f}",
            ),
        ),
        "</section>",
        '<section class="grid two">',
        _panel(
            "Exact-score backtest performance",
            "Only score-distribution models are shown here. Lower exact-score log loss "
            "is better; intervals use match-level bootstrap resampling.",
            _score_metric_panel(score_metric_summary),
        ),
        _panel(
            "Feature and approach contribution",
            "Paired log-loss deltas versus Elo on the same matches. Negative means the "
            "approach improved on Elo.",
            _delta_chart(delta_summary),
        ),
        _panel(
            "Bookmaker comparison",
            "Historical bookmaker odds are margin-adjusted and matched to the same World "
            "Cup validation matches.",
            _market_section(market_average, market_delta_summary),
        ),
        "</section>",
        _panel(
            "Performance by World Cup holdout window",
            "The model ranking is not perfectly stable across tournaments, which is "
            "expected with only 64 matches per window.",
            _window_table(window_summary),
            wide=True,
        ),
        '<section class="grid two">',
        _panel(
            "Future match uncertainty",
            "The uncertainty score combines model-family disagreement with a simple "
            "backtest-calibrated sampling term. It is a diagnostic standard deviation, "
            "not a full Bayesian posterior.",
            _uncertainty_chart(future_uncertainty.head(12)),
        ),
        _panel(
            "Most uncertain future match probabilities",
            "These are the group fixtures where current model-family disagreement is "
            "largest after adding the validation-sample term.",
            _future_uncertainty_table(future_uncertainty.head(12)),
        ),
        "</section>",
        '<section class="notes">',
        "<h2>Interpretation</h2>",
        "<p>The current evidence supports the pipeline as a transparent scenario engine. "
        "It does not support treating small probability differences as precise. Elo remains "
        "the strongest internal outcome baseline, while bookmaker-implied probabilities are "
        "the strongest external benchmark on the matched historical sample.</p>",
        "<p>Future score predictions can be added to this dashboard once the scoreline layer "
        "is promoted from diagnostics to report output.</p>",
        "</section>",
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def _market_average(market_metrics: pd.DataFrame | None) -> pd.DataFrame | None:
    if market_metrics is None or market_metrics.empty:
        return None
    rows = (
        market_metrics.groupby("model", as_index=False)
        .agg(log_loss=("log_loss", "mean"), brier_score=("brier_score", "mean"))
        .sort_values("log_loss", ignore_index=True)
    )
    rows["label"] = rows["model"].map(lambda value: MODEL_LABELS.get(value, value))
    rows["color"] = rows["model"].map(lambda value: MODEL_COLORS.get(value, "#555555"))
    return rows


def _kpi_card(title: str, value: str, detail: str) -> str:
    return (
        '<article class="kpi">'
        f"<span>{escape(title)}</span>"
        f"<strong>{escape(value)}</strong>"
        f"<small>{escape(detail)}</small>"
        "</article>"
    )


def _panel(title: str, subtitle: str, body: str, wide: bool = False) -> str:
    class_name = "panel wide" if wide else "panel"
    return (
        f'<section class="{class_name}">'
        f"<h2>{escape(title)}</h2>"
        f"<p>{escape(subtitle)}</p>"
        f"{body}"
        "</section>"
    )


def _horizontal_bar_chart(
    frame: pd.DataFrame,
    value_column: str,
    lower_column: str,
    upper_column: str,
    title: str,
    x_format: str,
) -> str:
    chart = frame.sort_values(value_column, ascending=True).reset_index(drop=True)
    width = 760
    left = 190
    right = 70
    top = 36
    row_height = 34
    height = top + row_height * len(chart) + 28
    values = chart[[value_column, lower_column, upper_column]].to_numpy(dtype=float).ravel()
    x_min = max(0.0, float(np.nanmin(values)) * 0.96)
    x_max = float(np.nanmax(values)) * 1.03

    def x_pos(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    rows = [
        _svg_open(width, height),
        f'<text x="{left}" y="18" class="chart-title">{escape(title)}</text>',
    ]
    for index, row in enumerate(chart.itertuples(index=False)):
        y = top + index * row_height
        value = float(getattr(row, value_column))
        lower = float(getattr(row, lower_column))
        upper = float(getattr(row, upper_column))
        label = str(row.label)
        color = str(row.color)
        rows.extend(
            [
                f'<text x="8" y="{y + 18}" class="axis-label">{escape(label)}</text>',
                f'<line x1="{left}" y1="{y + 13}" x2="{x_pos(upper):.2f}" '
                f'y2="{y + 13}" class="ci-line" />',
                f'<circle cx="{x_pos(lower):.2f}" cy="{y + 13}" r="3" class="ci-dot" />',
                f'<circle cx="{x_pos(upper):.2f}" cy="{y + 13}" r="3" class="ci-dot" />',
                f'<rect x="{left}" y="{y + 4}" width="{x_pos(value) - left:.2f}" '
                f'height="18" rx="4" fill="{color}" />',
                f'<text x="{x_pos(value) + 8:.2f}" y="{y + 18}" class="value-label">'
                f"{x_format.format(value)}</text>",
            ]
        )
    rows.append("</svg>")
    return "\n".join(rows)


def _delta_chart(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<p class="empty">No paired deltas available.</p>'
    chart = frame.sort_values("delta_log_loss", ascending=True).reset_index(drop=True)
    width = 760
    left = 190
    right = 70
    top = 36
    row_height = 36
    height = top + row_height * len(chart) + 30
    values = chart[
        ["delta_log_loss", "delta_log_loss_lower", "delta_log_loss_upper"]
    ].to_numpy(dtype=float).ravel()
    bound = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.001) * 1.12
    x_min, x_max = -bound, bound

    def x_pos(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    zero = x_pos(0.0)
    rows = [
        _svg_open(width, height),
        f'<line x1="{zero:.2f}" y1="24" x2="{zero:.2f}" y2="{height - 18}" '
        'class="zero-line" />',
        '<text x="8" y="18" class="chart-title">Delta log loss vs Elo</text>',
    ]
    for index, row in enumerate(chart.itertuples(index=False)):
        y = top + index * row_height
        value = float(row.delta_log_loss)
        lower = float(row.delta_log_loss_lower)
        upper = float(row.delta_log_loss_upper)
        bar_x = min(zero, x_pos(value))
        bar_width = abs(x_pos(value) - zero)
        color = "#2f6f73" if value < 0 else "#b75d3b"
        rows.extend(
            [
                f'<text x="8" y="{y + 18}" class="axis-label">{escape(str(row.label))}</text>',
                f'<line x1="{x_pos(lower):.2f}" y1="{y + 13}" '
                f'x2="{x_pos(upper):.2f}" y2="{y + 13}" class="ci-line" />',
                f'<rect x="{bar_x:.2f}" y="{y + 4}" width="{bar_width:.2f}" '
                f'height="18" rx="4" fill="{color}" />',
                f'<text x="{x_pos(value) + 8:.2f}" y="{y + 18}" class="value-label">'
                f"{value:+.4f}</text>",
            ]
        )
    rows.append("</svg>")
    return "\n".join(rows)


def _market_section(
    market_average: pd.DataFrame | None,
    market_delta_summary: pd.DataFrame,
) -> str:
    if market_average is None or market_average.empty:
        return '<p class="empty">No market validation files were available.</p>'
    chart = _horizontal_bar_chart(
        market_average,
        value_column="log_loss",
        lower_column="log_loss",
        upper_column="log_loss",
        title="Market validation log loss",
        x_format="{:.3f}",
    )
    if market_delta_summary.empty:
        return chart
    return chart + _delta_chart(market_delta_summary)


def _score_metric_panel(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<p class="empty">No exact-score validation data available.</p>'
    chart = _horizontal_bar_chart(
        frame,
        value_column="exact_score_log_loss",
        lower_column="exact_score_log_loss_lower",
        upper_column="exact_score_log_loss_upper",
        title="Exact-score log loss",
        x_format="{:.3f}",
    )
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.label))}</td>"
            f"<td>{row.exact_score_accuracy:.1%}</td>"
            f"<td>{row.goal_mae:.3f}</td>"
            f"<td>{row.total_goal_mae:.3f}</td>"
            "</tr>"
        )
    table = (
        '<div class="table-wrap compact score-table"><table>'
        "<thead><tr><th>Model</th><th>Exact hit</th><th>Goal MAE</th>"
        "<th>Total-goal MAE</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
    return chart + table


def _window_table(window_summary: pd.DataFrame) -> str:
    rows = []
    display = window_summary.copy()
    display["model"] = display["model"].map(lambda value: MODEL_LABELS.get(value, value))
    display = display.sort_values(["window", "log_loss"], ignore_index=True)
    for row in display.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.window))}</td>"
            f"<td>{escape(str(row.model))}</td>"
            f"<td>{row.n_matches}</td>"
            f"<td>{row.log_loss:.3f}</td>"
            f"<td>{row.brier_score:.3f}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table>'
        "<thead><tr><th>Window</th><th>Model</th><th>Matches</th>"
        "<th>Log loss</th><th>Brier</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _uncertainty_chart(frame: pd.DataFrame) -> str:
    chart = frame.sort_values("total_probability_sd", ascending=True).reset_index(drop=True)
    plot = chart.copy()
    plot["label"] = plot["home_team"] + " vs " + plot["away_team"]
    plot["color"] = "#1f5fbf"
    plot["lower"] = 0.0
    plot["upper"] = plot["total_probability_sd"]
    return _horizontal_bar_chart(
        plot,
        value_column="total_probability_sd",
        lower_column="lower",
        upper_column="upper",
        title="Diagnostic probability standard deviation",
        x_format="{:.3f}",
    )


def _future_uncertainty_table(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{row.match_number}</td>"
            f"<td>{escape(str(row.group))}</td>"
            f"<td>{escape(str(row.home_team))}</td>"
            f"<td>{escape(str(row.away_team))}</td>"
            f"<td>{escape(str(row.predicted_score))}</td>"
            f"<td>{escape(str(row.predicted_score_outcome))}</td>"
            f"<td>{row.predicted_score_probability:.1%}</td>"
            f"<td>{escape(str(row.most_likely_outcome))}</td>"
            f"<td>{row.most_likely_probability:.1%}</td>"
            f"<td>{row.model_disagreement_sd:.3f}</td>"
            f"<td>{row.backtest_sampling_se:.3f}</td>"
            f"<td>{row.total_probability_sd:.3f}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap compact"><table>'
        "<thead><tr><th>#</th><th>Group</th><th>Home</th><th>Away</th>"
        "<th>Score</th><th>Score outcome</th><th>Score prob.</th>"
        "<th>Aggregate outcome</th><th>Outcome prob.</th>"
        "<th>Model SD</th><th>Backtest SE</th>"
        "<th>Total SD</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _svg_open(width: int, height: int) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'xmlns="http://www.w3.org/2000/svg">'
    )


def _dashboard_css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f5f1e8;
  --panel: #fffdf8;
  --ink: #152223;
  --muted: #66706b;
  --line: #ded7ca;
  --accent: #1f5fbf;
  --shadow: 0 18px 50px rgba(45, 38, 24, 0.12);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
    sans-serif;
}
.page {
  width: min(1440px, 100%);
  margin: 0 auto;
  padding: 32px;
}
.hero {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 28px;
  align-items: end;
  padding: 40px 0 28px;
  border-bottom: 1px solid var(--line);
}
.eyebrow {
  margin: 0 0 10px;
  color: var(--accent);
  font-size: 0.82rem;
  font-weight: 800;
  text-transform: uppercase;
}
h1 {
  margin: 0;
  font-size: clamp(2.2rem, 5vw, 5.2rem);
  line-height: 0.96;
  letter-spacing: 0;
}
.lede {
  max-width: 820px;
  margin: 20px 0 0;
  color: var(--muted);
  font-size: 1.06rem;
  line-height: 1.55;
}
.hero-note {
  min-width: 190px;
  padding: 20px;
  border-left: 4px solid var(--accent);
  background: rgba(255, 253, 248, 0.7);
}
.hero-note strong {
  display: block;
  font-size: 2.4rem;
}
.hero-note span {
  color: var(--muted);
}
.kpi-grid,
.grid {
  display: grid;
  gap: 18px;
  margin-top: 22px;
}
.kpi-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}
.grid.two {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.kpi,
.panel,
.notes {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}
.kpi {
  padding: 22px;
}
.kpi span,
.kpi small,
.panel p,
.notes p {
  color: var(--muted);
}
.kpi strong {
  display: block;
  margin: 8px 0;
  font-size: 1.55rem;
}
.panel,
.notes {
  padding: 24px;
}
.panel.wide {
  margin-top: 22px;
}
h2 {
  margin: 0 0 8px;
  font-size: 1.28rem;
}
.panel p,
.notes p {
  margin: 0 0 18px;
  line-height: 1.5;
}
svg {
  display: block;
  width: 100%;
  height: auto;
}
.chart-title {
  fill: var(--muted);
  font-size: 13px;
  font-weight: 700;
}
.axis-label {
  fill: var(--ink);
  font-size: 13px;
}
.value-label {
  fill: var(--muted);
  font-size: 12px;
}
.ci-line {
  stroke: #3b4440;
  stroke-width: 2;
  opacity: 0.35;
}
.ci-dot {
  fill: #3b4440;
  opacity: 0.45;
}
.zero-line {
  stroke: #222;
  stroke-width: 1.5;
  stroke-dasharray: 4 5;
  opacity: 0.35;
}
.table-wrap {
  max-height: 520px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
}
.table-wrap.compact {
  max-height: 430px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}
th,
td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  white-space: nowrap;
}
th {
  position: sticky;
  top: 0;
  background: #f7f2e8;
  color: #3a413f;
  z-index: 1;
}
tr:nth-child(even) td {
  background: rgba(245, 241, 232, 0.45);
}
.notes {
  margin-top: 22px;
}
.empty {
  padding: 20px;
  border: 1px dashed var(--line);
  border-radius: 8px;
}
@media (max-width: 920px) {
  .page { padding: 18px; }
  .hero,
  .kpi-grid,
  .grid.two {
    grid-template-columns: 1fr;
  }
}
"""


def _match_id(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["window"].astype(str)
        + "|"
        + frame["date"].astype(str)
        + "|"
        + frame["home_team"].astype(str)
        + "|"
        + frame["away_team"].astype(str)
    )


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {label}: {sorted(missing)}")


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")
