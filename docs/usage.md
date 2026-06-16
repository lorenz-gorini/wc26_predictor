# Usage

## Install Locally

From the project root:

```bash
python3 -m pip install -e .
```

If you do not want to install the package, tests and demos still work because
`pyproject.toml` sets `src` on the pytest path.

## Run Tests

```bash
python3 -m pytest
```

## Run the Demo

```bash
PYTHONPATH=src python3 -m wc26_predictor.cli
```

Expected output includes:

- Elo home/draw/away probabilities;
- Poisson expected goals;
- Poisson home/draw/away probabilities.

## Download Public Data

Use the `general` conda environment:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/download_data.py
```

The script downloads:

- `data/raw/international_results.csv` from
  `https://raw.githubusercontent.com/martj42/international_results/master/results.csv`;
- `data/raw/goalscorers.csv` from
  `https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv`;
- `data/raw/world_cup_2026_fixtures.csv` from the Wikipedia pages
  `https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_A` through Group L;
- `data/raw/world_cup_2026_squads.csv` from
  `https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads`;
- `data/raw/download_metadata.json` with source URLs and timestamps;
- `data/processed/international_results.csv`, filtering to completed matches.

If network access is unavailable, download the historical CSV manually from the
raw GitHub URL above and place it at `data/raw/international_results.csv`.

## Load Data

```python
from wc26_predictor.data.ingest_results import load_results_csv

results = load_results_csv("data/raw/sample_results.csv")
```

The loader validates required columns, dates, team names, scores, and neutral
venue flags. Invalid input raises an error.

## Fit Elo

```python
from wc26_predictor.data.schema import Fixture
from wc26_predictor.models.elo import EloRatings

elo = EloRatings().fit(results)
probabilities = elo.predict_outcome(Fixture("Argentina", "Brazil", neutral=True))
print(probabilities.as_dict())
```

## Fit the Poisson Score Model

```python
from wc26_predictor.data.schema import Fixture
from wc26_predictor.models.poisson import IndependentPoissonModel

model = IndependentPoissonModel().fit(results)
prediction = model.predict(Fixture("Argentina", "Brazil", neutral=True))

print(prediction.home_expected_goals)
print(prediction.away_expected_goals)
print(prediction.outcome_probabilities.as_dict())
```

## Simulate a Group

```python
import numpy as np

from wc26_predictor.data.schema import Fixture
from wc26_predictor.simulation.group_stage import simulate_group

fixtures = [
    Fixture("Argentina", "Brazil"),
    Fixture("Argentina", "Germany"),
    Fixture("Argentina", "Japan"),
    Fixture("Brazil", "Germany"),
    Fixture("Brazil", "Japan"),
    Fixture("Germany", "Japan"),
]

simulation = simulate_group(fixtures, model, np.random.default_rng(2026))
print(simulation.table)
```

## Recommended Workflow

1. Put raw downloaded data in `data/raw/`.
2. Validate and normalize it through `wc26_predictor.data`.
3. Fit baselines.
4. Backtest against historical holdout windows.
5. Add richer features only after the baseline metrics are known.

## Run Baselines

After downloading the public data, run:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/run_baselines.py
```

The script fits Elo and Poisson baselines on completed historical matches,
evaluates them on the 2014, 2018, and 2022 World Cups, and forecasts all 72
World Cup 2026 group-stage fixtures.

Outputs:

```text
data/processed/baseline_backtest.csv
data/processed/world_cup_2026_hyperparameter_tuning.csv
data/processed/world_cup_2026_selected_hyperparameters.csv
data/processed/world_cup_2026_ensemble_weights.csv
data/processed/world_cup_market_validation_matches.csv
data/processed/world_cup_market_validation_metrics.csv
data/processed/world_cup_market_ensemble_weights_lowo.csv
data/processed/world_cup_market_gate_decision.csv
data/processed/world_cup_2026_baseline_forecasts.csv
data/processed/world_cup_2026_player_availability_features.csv
data/processed/world_cup_2026_team_availability_burden.csv
data/processed/world_cup_2026_transfermarkt_squad_injury_features.csv
data/processed/world_cup_2026_transfermarkt_team_injury_burden.csv
data/processed/world_cup_2026_availability_impact.csv
data/processed/world_cup_2026_availability_adjusted_forecasts.csv
data/processed/world_cup_2026_knockout_match_forecasts.csv
data/processed/world_cup_2026_full_match_forecasts.csv
data/processed/world_cup_2026_upcoming_match_details.csv
data/processed/world_cup_2026_latest_prediction_snapshot.csv
data/processed/world_cup_2026_played_match_prediction_checks.csv
data/processed/world_cup_2026_match_prediction_drivers.csv
data/processed/world_cup_2026_match_final_stage_impacts.csv
data/processed/world_cup_2026_team_expected_matches.csv
data/processed/world_cup_2026_official_tournament_probabilities.csv
data/processed/world_cup_2026_top_scorer_baseline.csv
data/processed/world_cup_2026_top_scorer_availability_adjusted.csv
data/processed/world_cup_2026_top_scorer_club_adjusted_top100.csv
data/processed/world_cup_2026_top_scorer_transfermarkt_adjusted_top100.csv
reports/baseline_summary.md
```

For a more stable but slower top-scorer exposure estimate, increase the
simulation count:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/run_baselines.py --n-simulations 10000
```

The first baseline treats World Cup 2026 fixtures as venue-neutral. Host-country
advantage should be added explicitly in the next modeling layer because FIFA
fixture ordering is not the same as true home advantage.

## Update Predictions After Matches

After the tournament starts, refresh completed results and regenerate the
forecast artifacts with:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/update_predictions.py
```

By default this refreshes the public international-results and goalscorers CSVs,
keeps the existing static fixture/squad inputs, reruns the baseline pipeline,
regenerates final reports, and rebuilds `reports/model_performance_dashboard.html`.
Completed World Cup group matches found in `data/processed/international_results.csv`
are held fixed inside tournament simulations, while unplayed fixtures are
forecast with the latest fitted model state.

At the start of `scripts/update_predictions.py`, the previous
`world_cup_2026_latest_prediction_snapshot.csv` is archived into
`world_cup_2026_prediction_snapshots.csv` before new results are downloaded.
This makes completed-match diagnostics compare results against the last saved
pre-match prediction, not against a model refit after the result is known.

Use `--refresh-static-data` when you also want to redownload fixtures, squads,
and club-form tables. Use `--impact-simulations` to trade off speed versus
Monte Carlo stability for the per-match final-stage impact plot.

## Open the Dashboard

Generate the split static dashboard with:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/generate_validation_dashboard.py
```

Serve the report folder locally:

```bash
/Users/lorenzogorini/anaconda3/envs/general/bin/python -m http.server 8000 --bind 127.0.0.1 --directory reports
```

Then open:

```text
http://127.0.0.1:8000/model_performance_dashboard.html
```

The root page links to:

- `dashboard/future_matches.html`: next unplayed match predictions and drivers;
- `dashboard/group_stage.html`: group pools, played matches, prediction checks, and past fixtures awaiting downloaded results;
- `dashboard/next_phases.html`: round-by-round, winner, and knockout-pairing forecasts;
- `dashboard/model_performance.html`: historical validation diagnostics.

## Generate Final Reports

After `scripts/run_baselines.py` has produced the processed outputs, run:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/generate_final_reports.py
```

This writes:

```text
reports/final_group_advancement_probabilities.csv
reports/final_round_by_round_probabilities.csv
reports/final_winner_probabilities.csv
reports/final_top_scorer_probabilities.csv
reports/final_group_match_forecasts.csv
reports/final_knockout_match_forecasts.csv
reports/final_full_match_forecasts.csv
reports/final_report.md
```

Top-scorer probabilities are computed from the Transfermarkt-adjusted top-100
expected-goals table using an independent Poisson approximation. Tied top-scorer
outcomes are split evenly. Group advancement currently means probability of
reaching the round of 32, not exact group-position probabilities.
