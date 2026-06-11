# Model Validation Review

Generated on 2026-06-11 after regenerating the baseline pipeline and final reports.

## Executive Assessment

The project improvements mostly make sense. The code now has a coherent validation loop:
historical World Cup holdout windows, proper probabilistic metrics, tuned baseline
hyperparameters, leave-one-window-out ensemble fitting, bookmaker-implied probabilities,
official-bracket simulation, player availability adjustments, and top-scorer enrichment.

The primary match-level target is now the final exact score. Win/draw/loss remains useful for
secondary evaluation, but the scoreline distribution is the object that should drive match
predictions. The evidence does not support treating the current forecasts as highly precise
probabilities. It supports using them as ranked probabilistic scenarios with useful relative
strength signals. The strongest external 1X2 check is the bookmaker benchmark, and it remains
slightly better than the internal model on the matched historical World Cup sample.

## Backtesting Evidence

The internal match-outcome validation uses the 2014, 2018, and 2022 FIFA World Cups as temporal
holdout windows. Each window has 64 matches, so the validation sample is 192 matches in total.

Average validation scores:

| model | log_loss | brier_score |
| --- | ---: | ---: |
| elo | 0.9723 | 0.5716 |
| ensemble_calibrated_lowo | 0.9786 | 0.5762 |
| ensemble_equal_weight | 0.9880 | 0.5884 |
| form_adjusted_poisson | 1.0155 | 0.6090 |
| poisson | 1.0183 | 0.6110 |

Interpretation:

- Elo is still the best internal baseline on average.
- The calibrated ensemble is close, but it does not beat Elo out of sample.
- The Poisson and form-adjusted Poisson models add useful diagnostics and expected-goals
  structure, but the current validation does not justify giving them large outcome-probability
  weight.
- The fitted 2026 ensemble weights reflect this: 0.85 Elo, 0.00 Poisson, 0.15 form-adjusted
  Poisson.

The ensemble improvement is therefore mostly architectural rather than empirical at this stage:
it gives a clean way to include model families, but the data currently says the simple Elo signal
should dominate.

## Exact-Score Evidence

Exact-score validation is available only for score-distribution models. Elo and bookmaker 1X2 odds
do not produce exact-score matrices without an additional score model.

Average score validation scores:

| model | exact_score_log_loss | exact_score_accuracy | goal_mae | total_goal_mae |
| --- | ---: | ---: | ---: | ---: |
| poisson | 2.9634 | 0.0365 | 0.9792 | 1.6771 |
| form_adjusted_poisson | 2.9704 | 0.0469 | 1.0130 | 1.7240 |

Interpretation:

- Exact-score prediction is a materially harder target than 1X2 prediction.
- The modal exact score is often a draw or low-scoring result even when the aggregate 1X2
  probability favors one team, because many separate winning scorelines can jointly dominate one
  individual draw scoreline.
- The form-adjusted Poisson model has a slightly higher exact-score hit rate, while the simpler
  Poisson model has slightly better exact-score log loss and goal-error metrics on this sample.
- Current exact-score forecasts should be read as modal scoreline probabilities plus nearby
  alternatives, not as high-confidence point predictions.

## Bookmaker Odds Benchmark

The bookmaker comparison uses Football-Data World Cup 1X2 odds, margin-adjusted to probabilities,
matched to the same 192 validation matches.

Average validation scores:

| model | log_loss | brier_score |
| --- | ---: | ---: |
| market_only | 0.9613 | 0.5667 |
| model_market_lowo | 0.9616 | 0.5671 |
| model_only | 0.9707 | 0.5722 |

By window:

| window | model_only_log_loss | market_only_log_loss | model_market_log_loss |
| --- | ---: | ---: | ---: |
| 2014_fifa_world_cup | 0.9252 | 0.9399 | 0.9320 |
| 2018_fifa_world_cup | 0.9458 | 0.9454 | 0.9389 |
| 2022_fifa_world_cup | 1.0411 | 0.9984 | 1.0138 |

Interpretation:

- The internal model beat the market in 2014.
- The internal model and market were almost tied in 2018.
- The market was materially better in 2022.
- Averaged over all three tournaments, market-only is best.
- The current gate correctly refuses to use the model+market blend because it does not improve
  over both model-only and market-only log loss.

This is the most important trust signal. If current 2026 bookmaker odds are later available for
all fixtures, they should be treated as the primary benchmark and possibly the primary forecast
input unless new validation shows the model adds independent information.

## What We Can Trust

High confidence:

- The pipeline is internally reproducible: tests pass, validation artifacts regenerate, and final
  reports are produced from processed outputs.
- The direction of the model-selection result is credible: Elo dominates the internal match
  outcome baselines.
- The bookmaker benchmark is correctly integrated as a margin-adjusted probabilistic baseline.

Medium confidence:

- Broad team tiers and favorites are useful. Large gaps such as Spain or Argentina versus clear
  underdogs are meaningful.
- Group advancement and winner rankings are useful for scenario analysis, especially near the top
  of the distribution.
- The official 2026 bracket simulation is a better representation than an abstract tournament
  simulator for final report outputs.

Low confidence:

- Small probability gaps should not be over-interpreted. A difference such as 3.8% versus 3.5%
  winner probability is not decision-grade.
- Exact-score predictions are sensitive to the Poisson goal-rate assumptions and should always be
  shown with top alternative scorelines.
- Top-scorer probabilities are sensitive to squad assumptions, player availability, penalty roles,
  and the Transfermarkt multiplier. They are useful rankings, not calibrated betting probabilities.
- Current injury and availability adjustments are incomplete unless verified current overrides are
  maintained close to the tournament.
- The model has only three World Cup holdout windows. That is a small sample for estimating robust
  calibration or judging narrow model differences.

## Practical Conclusion

The improvements made the project substantially more credible, especially because model outputs
are now compared against both historical outcomes and bookmaker-implied probabilities. The current
forecasts should be trusted as a structured baseline and scenario engine, not as final market-grade
probabilities.

For decision use, the hierarchy should be:

1. Use bookmaker odds as the benchmark whenever available.
2. Use the internal model where market odds are unavailable or where a transparent counterfactual
   scenario is needed.
3. Treat the internal model as adding value only if future backtests show it improves log loss
   beyond the market or beyond the Elo-only baseline.

## Recommended Next Work

- Add calibration tables or reliability plots for the historical validation predictions.
- Add confidence intervals or bootstrap uncertainty for winner probabilities and top-scorer
  probabilities.
- Backtest top-scorer outputs separately; match-outcome validation does not validate scorer
  probabilities.
- Compare forecasts with current 2026 bookmaker odds when those fixture-level markets become
  available.
- Consider reporting Elo-only alongside the ensemble in final reports, since Elo is the strongest
  validated internal baseline.
