# Baseline Models

The project now has a reproducible baseline run:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/run_baselines.py
```

## Models

### Elo

The Elo baseline updates team strength sequentially through completed historical
matches. It uses home advantage for non-neutral historical games and a fixed draw
probability when converting expected score to home/draw/away probabilities.

### Poisson

The independent-Poisson model estimates team attack and defense multipliers with
shrinkage toward the global goal rate. It outputs expected goals and a full
scoreline probability matrix.

### Form-Adjusted Poisson

This model starts from the Poisson model, then adjusts expected goals using each
team's recent attack index and defensive vulnerability index. Recent form is
computed from the last 10 matches with shrinkage toward the global goal rate.

### Ensembles

The project keeps two ensemble variants:

- `ensemble_equal_weight_*`: simple average of Elo, Poisson, and form-adjusted
  Poisson probabilities;
- `ensemble_*`: calibrated convex weights fit by minimizing log loss on the
  2014, 2018, and 2022 World Cup holdout predictions.

The calibrated weights are written to:

```text
data/processed/world_cup_2026_ensemble_weights.csv
```

The backtest table includes leave-one-window-out calibrated ensemble metrics so
the report shows a validation estimate that does not score each window with
weights fit on that same window.

### Hyperparameter Tuning

The baseline workflow tunes model hyperparameters before fitting final forecasts.
The tuning objective is multiclass log loss over the 2014, 2018, and 2022 World
Cup validation predictions.

Tuned parameter grids currently cover:

- Elo `k_factor`, draw probability, home advantage, and neutral-home handling;
- Poisson prior strength, recency half-life, and tournament-importance weighting;
- form-adjusted Poisson window length, form prior, and form strength.

Outputs:

```text
data/processed/world_cup_2026_hyperparameter_tuning.csv
data/processed/world_cup_2026_selected_hyperparameters.csv
```

This is still a compact validation design. The next statistically stronger
version should add more rolling-origin tournament and non-tournament windows to
reduce the risk of overfitting to three World Cups.

### Market Benchmark

The workflow ingests Football-Data World Cup odds from:

```text
data/raw/odds/WorldCup2026.xlsx
```

Despite the filename, this workbook currently contains historical World Cup
finals sheets for 2014, 2018, and 2022, plus World Cup 2026 qualifiers. It does
not yet contain bookmaker odds for the 2026 finals group-stage fixtures.

The market layer:

- reads the historical World Cup sheets;
- removes bookmaker margin from average home/draw/away decimal odds;
- matches market probabilities to the World Cup validation windows;
- evaluates model-only, market-only, and model+market ensembles;
- fits model+market weights leave-one-window-out;
- writes a gate decision that allows market blending only when LOWO log loss
  improves over model-only predictions.

Outputs:

```text
data/processed/world_cup_market_validation_matches.csv
data/processed/world_cup_market_validation_metrics.csv
data/processed/world_cup_market_ensemble_weights_lowo.csv
data/processed/world_cup_market_gate_decision.csv
```

When 2026 finals odds become available, the same gate should be used before
applying market adjustment to current match forecasts.

## Backtesting

The script evaluates the baselines on the 2014, 2018, and 2022 World Cups:

- train on matches before the tournament;
- test only on that World Cup's completed matches;
- report log loss and Brier score.

This is a preliminary backtest. The next version should add rolling-origin
monthly or quarterly windows and compare against bookmaker-implied probabilities.

## Forecasting

The script forecasts all 72 World Cup 2026 group-stage fixtures from
`data/raw/world_cup_2026_fixtures.csv`.

Output columns include:

- Elo home/draw/away probabilities;
- Poisson expected goals;
- Poisson home/draw/away probabilities;
- form-adjusted Poisson expected goals and probabilities;
- equal-weight ensemble probabilities;
- calibrated ensemble probabilities.

The workflow also writes conditional knockout match forecasts:

```text
data/processed/world_cup_2026_knockout_match_forecasts.csv
data/processed/world_cup_2026_full_match_forecasts.csv
```

`world_cup_2026_knockout_match_forecasts.csv` contains official-bracket match
numbers, possible pairings, pairing probabilities from simulation, and calibrated
conditional advancement probabilities. `world_cup_2026_full_match_forecasts.csv`
combines the fixed group-stage forecast rows with those conditional knockout
rows.

World Cup 2026 fixtures are currently treated as venue-neutral. This avoids
misassigning host advantage when the listed `home_team` is not the actual host.

## Top-Scorer Baseline

If `data/raw/goalscorers.csv` exists, `scripts/run_baselines.py` also produces:

```text
data/processed/world_cup_2026_top_scorer_baseline.csv
data/processed/world_cup_2026_top_scorer_availability_adjusted.csv
data/processed/world_cup_2026_team_expected_matches.csv
data/processed/world_cup_2026_official_tournament_probabilities.csv
data/processed/world_cup_2026_knockout_match_forecasts.csv
data/processed/world_cup_2026_full_match_forecasts.csv
data/processed/world_cup_2026_top_scorer_club_adjusted_top100.csv
data/processed/world_cup_2026_top_scorer_transfermarkt_adjusted_top100.csv
```

This baseline uses national-team goalscorer history, filters to listed 2026
squad players when squad data are available, and adjusts for team tournament
exposure. Expected matches are estimated with group-stage Monte Carlo simulation
and the official 2026 knockout bracket. Third-place pairings use the FIFA Annex C
assignment table. The availability-adjusted layer adds expected-minutes,
penalty-role, and injury/suspension status adjustments before club-form
enrichment.

Availability-related outputs are:

```text
data/processed/world_cup_2026_player_availability_features.csv
data/processed/world_cup_2026_team_availability_burden.csv
data/processed/world_cup_2026_availability_impact.csv
data/processed/world_cup_2026_availability_adjusted_forecasts.csv
```

Without a current injury override file, team burden is zero and match forecasts
remain unchanged. Expected-minutes and penalty-role adjustments still affect the
availability-adjusted scorer table.
