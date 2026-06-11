# Project Overview

## Objective

The project estimates probabilistic forecasts for the FIFA World Cup 2026. The
main outputs should eventually be:

- match scoreline probabilities;
- home/draw/away probabilities;
- group-stage qualification probabilities;
- knockout progression probabilities;
- tournament winner probabilities;
- model evaluation reports against historical data and betting-market baselines.

The project is designed for research-grade analysis: transparent assumptions,
explicit validation, reproducible data pipelines, and proper scoring rules.

## Preliminary Model

The preliminary implementation is the baseline that every later model must beat.
It contains two complementary components.

### Elo

The Elo model tracks team strength through time. It updates ratings sequentially
after each match using:

- pre-match rating difference;
- home advantage for non-neutral matches;
- match result;
- optional goal-margin weighting.

Elo is useful because it is interpretable, hard to overfit, and a strong
benchmark for international football.

### Independent Poisson

The Poisson model estimates expected goals for both teams. It uses:

- global international goal rate;
- team attack multipliers;
- team defense multipliers;
- shrinkage toward the global mean;
- optional recency weighting;
- home-goal adjustment for non-neutral matches.

The model returns a full score matrix. Match outcome probabilities are derived
from that matrix, not fitted as a separate classifier.

## Full Target Model

The full model should become an ensemble of:

1. **Elo and ranking features**
   Dynamic ratings, FIFA rankings, rank changes, confederation indicators, match
   importance, neutral venue, and host advantage.

2. **Hierarchical score model**
   A Bayesian bivariate Poisson or Dixon-Coles-style model with attack and
   defense latent strengths, team-level partial pooling, time decay, and
   uncertainty intervals.

3. **Gradient boosting**
   A tabular model trained on features such as squad value, player minutes at
   elite clubs, recent form, goals for/against, rest days, travel, venue,
   temperature, and altitude.

4. **Market benchmark**
   Odds-implied probabilities adjusted for bookmaker margin. These should be
   compared against the data-only model and optionally used in an ensemble.

5. **Structured news layer**
   LLMs can help extract injuries, suspensions, coach changes, and lineup
   uncertainty from curated sources. Raw fan sentiment should not be treated as
   independent signal.

## Evaluation Principles

Use rolling-origin backtests. At every prediction date, train only on information
that would have been available at that date.

Primary metrics:

- log loss;
- Brier score;
- calibration by probability bucket;
- scoreline likelihood;
- tournament simulation calibration where historical brackets are available.

The baseline to beat is not random choice. It is Elo, FIFA ranking features, and
bookmaker-implied probabilities.

