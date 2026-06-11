# World Cup 2026 Predictor

This project is a reproducible research codebase for forecasting FIFA World Cup
2026 matches and tournament paths. The goal is not to produce a single dramatic
winner pick, but to estimate calibrated probabilities for scorelines, match
outcomes, group advancement, and later full tournament progression.

The project starts with transparent models that can be audited and backtested,
then leaves a clean path toward richer state-of-the-art layers.

## Current Implementation

The implemented preliminary model contains:

- strict data validation for international results;
- dynamic Elo ratings for interpretable team strength;
- a regularized independent-Poisson score model;
- proper scoring-rule evaluation with log loss, Brier score, and calibration tables;
- Monte Carlo simulation for matches and group-stage advancement;
- a CLI demo and pytest test suite.

The public historical result archive can include scheduled future fixtures with
missing scores. Those rows are kept in `data/raw/`; model-ready completed matches
are written to `data/processed/`.

This is intentionally a strong baseline rather than a large black box. For
international football, data are sparse, teams change slowly, and a calibrated
simple model is often more reliable than an overfit complex model.

## Full Modeling Approach

The intended full system has four layers:

1. **Transparent strength layer**
   Dynamic Elo, FIFA ranking features, confederation effects, host advantage,
   time decay, and match importance.

2. **Score-distribution layer**
   A Poisson-family model for home and away goals. The current implementation is
   a regularized moment estimator. A later version should add a hierarchical
   Bayesian model with attack/defense latent strengths and partial pooling by
   confederation.

3. **Feature-rich machine learning layer**
   Gradient boosting on engineered features such as recent form, squad market
   value, player club strength, rest days, travel distance, venue, climate,
   ranking momentum, and injury/suspension indicators.

4. **Market and information layer**
   Bookmaker odds should be treated as a benchmark and possible ensemble input,
   not as ground truth. News and sentiment should be converted into structured
   variables, for example injuries or lineup uncertainty, rather than raw fan
   opinions.

## Quick Start

Run the tests:

```bash
python3 -m pytest
```

Run the demo forecast:

```bash
PYTHONPATH=src python3 -m wc26_predictor.cli
```

The demo trains on `data/raw/sample_results.csv` and forecasts a neutral
Argentina vs Brazil fixture.

Download public data:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/download_data.py
```

This writes:

- `data/raw/international_results.csv`;
- `data/raw/goalscorers.csv`;
- `data/processed/international_results.csv`;
- `data/raw/world_cup_2026_fixtures.csv`;
- `data/raw/world_cup_2026_squads.csv`;
- `data/raw/club_top_scorers.csv`;
- `data/raw/download_metadata.json`.

Fit and evaluate the baseline models:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/run_baselines.py
```

This writes:

- `data/processed/baseline_backtest.csv`;
- `data/processed/world_cup_2026_hyperparameter_tuning.csv`;
- `data/processed/world_cup_2026_selected_hyperparameters.csv`;
- `data/processed/world_cup_2026_ensemble_weights.csv`;
- `data/processed/world_cup_market_validation_matches.csv`;
- `data/processed/world_cup_market_validation_metrics.csv`;
- `data/processed/world_cup_market_ensemble_weights_lowo.csv`;
- `data/processed/world_cup_market_gate_decision.csv`;
- `data/processed/world_cup_2026_baseline_forecasts.csv`;
- `data/processed/world_cup_2026_player_availability_features.csv`;
- `data/processed/world_cup_2026_team_availability_burden.csv`;
- `data/processed/world_cup_2026_transfermarkt_squad_injury_features.csv`;
- `data/processed/world_cup_2026_transfermarkt_team_injury_burden.csv`;
- `data/processed/world_cup_2026_availability_impact.csv`;
- `data/processed/world_cup_2026_availability_adjusted_forecasts.csv`;
- `data/processed/world_cup_2026_knockout_match_forecasts.csv`;
- `data/processed/world_cup_2026_full_match_forecasts.csv`;
- `data/processed/world_cup_2026_team_expected_matches.csv`;
- `data/processed/world_cup_2026_official_tournament_probabilities.csv`;
- `data/processed/world_cup_2026_top_scorer_baseline.csv`;
- `data/processed/world_cup_2026_top_scorer_availability_adjusted.csv`;
- `data/processed/world_cup_2026_top_scorer_club_adjusted_top100.csv`;
- `data/processed/world_cup_2026_top_scorer_transfermarkt_adjusted_top100.csv`;
- `reports/baseline_summary.md`.

Generate final report tables after the processed outputs exist:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/generate_final_reports.py
```

This writes final group-advancement, round-by-round, winner, top-scorer, and
match-forecast tables under `reports/`, plus `reports/final_report.md`.

Generate the local model-performance dashboard:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/generate_validation_dashboard.py
```

This writes `reports/model_performance_dashboard.html`, a standalone dashboard
with backtest plots, bookmaker comparisons, model contribution diagnostics, and
future-match probability uncertainty. Open it directly in a browser or serve the
reports directory locally:

```bash
/Users/lorenzogorini/anaconda3/envs/general/bin/python -m http.server 8000 --directory reports
```

## Local Secrets

Put API keys in a local `.env` file at the project root. The file is ignored by
Git:

```text
SOCCERDATA_API_KEY=your_key_here
```

Do not paste API keys into notebooks, scripts, CSV files, or chat transcripts.
Because SoccerDataAPI has a low daily request budget, API-backed ingestion
should cache raw responses under `data/raw/` and avoid repeated calls during
model experimentation.

Fetch SoccerDataAPI current sidelined-player data cautiously:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/fetch_soccerdata_injuries.py --max-match-requests 0
```

This spends at most one request for the upcoming match-preview index, writes a
cached copy under `data/raw/soccerdata_api/`, and writes preview CSVs under
`data/raw/`. Increase `--max-match-requests` only after confirming that the
filtered previews contain relevant World Cup matches.

Fetch public club-level injury rows and map them to World Cup squad players:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/fetch_sportsgambler_injuries.py
```

This writes `data/raw/sportsgambler_player_availability_overrides.csv`, which
the baseline pipeline consumes automatically before manual overrides.

## Data Strategy

Recommended raw sources:

- `martj42/international_results`: men’s full international results with date,
  teams, score, tournament, location, and neutral venue.
- Wikipedia World Cup 2026 group pages as machine-readable fixture mirrors of
  FIFA-cited match tables.
- Football-Data World Cup workbook for market-implied probabilities.
- FIFA rankings and Elo ratings as benchmark strength features.
- Squad features from public APIs or reproducible scrapers.

Keep raw files immutable in `data/raw/`. Write normalized files to
`data/processed/`. The project should raise on malformed input instead of
guessing silently.

## Example Python Usage

```python
from wc26_predictor.data.ingest_results import load_results_csv
from wc26_predictor.data.schema import Fixture
from wc26_predictor.models.poisson import IndependentPoissonModel

results = load_results_csv("data/raw/sample_results.csv")
model = IndependentPoissonModel().fit(results)

prediction = model.predict(Fixture("Argentina", "Brazil", neutral=True))
print(prediction.home_expected_goals, prediction.away_expected_goals)
print(prediction.outcome_probabilities.as_dict())
```

## Next Steps

1. Add World Cup 2026 finals bookmaker odds when those markets are available.
2. Add verified current injury/suspension overrides when reliable squad news is available.
3. Improve group-stage tie-breakers and knockout extra-time/shootout treatment.
4. Extend model validation beyond the three World Cup holdout windows.
5. Produce final model cards and richer match/tournament reports.

See [docs/project_overview.md](docs/project_overview.md) and
[docs/usage.md](docs/usage.md) for more detail. Source URLs and manual download
instructions are in [docs/data_sources.md](docs/data_sources.md). The separate
player scorer baseline is described in [docs/scorer_model.md](docs/scorer_model.md).

## Project Layout

```text
src/wc26_predictor/
  data/         ingestion, schemas, validation
  features/     feature transforms
  models/       Elo and score models
  evaluation/   log loss, Brier score, calibration
  simulation/   match and group-stage simulation
```
