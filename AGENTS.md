# World Cup 2026 Predictor: Agent Instructions

## Project Purpose

This is a reproducible research codebase for forecasting FIFA World Cup 2026
matches, group advancement, tournament paths, and top-scorer probabilities.
The primary match-level prediction target is the final exact score. Win/draw/loss
probabilities remain important secondary diagnostics and evaluation targets, but
do not replace the exact-score forecast. The project values transparent,
backtestable probabilistic models over opaque winner picks. Forecasts should be
treated as calibrated scenarios and model diagnostics, not as betting advice.

The owner has a computer vision and PyTorch background and is now a PhD student
in Economics. Be precise, consistent, readable, and explicit about assumptions.
It is acceptable for code to raise an error on malformed inputs instead of
silently continuing with unreliable data.

## Runtime Environment

Use the `general` conda environment unless there is a concrete reason not to:

```bash
/Users/lorenzogorini/anaconda3/envs/general/bin/python
```

The project is installed there in editable mode. For scripts that import
`wc26_predictor`, prefer:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/<script>.py
```

Run tests with:

```bash
/Users/lorenzogorini/anaconda3/envs/general/bin/python -m pytest
```

## Repository Structure

- `src/wc26_predictor/data/`: ingestion, source-specific loaders, schemas, validation.
- `src/wc26_predictor/features/`: feature transforms such as recent form and importance.
- `src/wc26_predictor/models/`: Elo, Poisson, form-adjusted Poisson, ensembles, market logic,
  availability impact, and top-scorer models.
- `src/wc26_predictor/evaluation/`: proper scoring rules and backtesting helpers.
- `src/wc26_predictor/simulation/`: match, group-stage, abstract tournament, and official
  2026 bracket simulation.
- `src/wc26_predictor/reporting/`: generated final reports and validation dashboards.
- `scripts/`: reproducible command-line workflows.
- `tests/`: pytest coverage for data, models, simulation, and reporting.
- `data/raw/`: raw or downloaded inputs. This directory is mostly gitignored.
- `data/processed/`: generated processed artifacts. This directory is mostly gitignored.
- `reports/`: committed human-readable outputs and final CSV reports.

## Main Workflows

Download public data:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/download_data.py
```

Run baseline modeling, backtests, market comparison, availability adjustments, and tournament
simulation:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/run_baselines.py
```

Generate final report tables:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/generate_final_reports.py
```

Generate the local validation dashboard:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/generate_validation_dashboard.py
```

Open `reports/model_performance_dashboard.html` directly in a browser, or serve reports locally:

```bash
/Users/lorenzogorini/anaconda3/envs/general/bin/python -m http.server 8000 --directory reports
```

## Modeling Context

Current validation uses 2014, 2018, and 2022 FIFA World Cup holdout windows.
For exact scores, use exact-score log loss, exact-score accuracy, goal MAE, goal
RMSE, and total-goal MAE. For 1X2 probabilities, use multiclass log loss and
multiclass Brier score. Lower is better for loss/error metrics.

Important current conclusions:

- Elo is the strongest internal baseline on average.
- Poisson and form-adjusted Poisson add useful score and feature diagnostics, but current
  backtests do not justify large outcome-probability weights.
- The calibrated internal ensemble is close to Elo but does not beat Elo out of sample.
- Bookmaker-implied probabilities are the strongest benchmark on the matched historical
  World Cup sample.
- The model+market gate should only use a combined model if leave-one-window-out validation
  improves over both model-only and market-only log loss.

## Coding Standards

- Prefer small, explicit functions with typed dataclasses where useful.
- Keep code Pythonic, readable, and non-redundant.
- Prefer structured parsers and validators over ad hoc string manipulation.
- Avoid repeated defensive checks after inputs have already been validated.
- Raise clear `ValueError` or `RuntimeError` exceptions for malformed or impossible states.
- Keep generated artifacts reproducible: use fixed seeds for simulations and bootstraps.
- Add or update focused tests when behavior changes.
- Use `rg` for searching.
- Do not commit secrets. `.env` is local and ignored.

## Git Workflow

After completing the changes requested in a prompt, run the relevant tests and commit the
finished work on `main`. Do not leave completed work uncommitted.

Use concise commit messages that describe the user-visible change.
