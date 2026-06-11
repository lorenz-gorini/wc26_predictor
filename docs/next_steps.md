# Next Steps

## Data

1. Audit the downloaded `world_cup_2026_fixtures.csv` against FIFA's official
   fixture page before final forecasts.
2. Add World Cup 2026 finals bookmaker odds when markets become available.
3. Add squad snapshots with the date at which each squad was known.
4. Add a country/team-name alias table for sources that differ on names such as
   Czechia/Czech Republic, Türkiye/Turkey, Iran/IR Iran, and Congo DR/DR Congo.

## Modeling

1. Extend the current World Cup holdout backtests to rolling-origin monthly or
   quarterly windows.
2. Extend the Poisson model to a Dixon-Coles correction for low-score dependence.
3. Add Bayesian hierarchical modeling as an optional module.
4. Add gradient boosting once the feature table is stable.
5. Calibrate model outputs using validation-period isotonic or temperature
   scaling if needed.

## Simulation

1. Implement the remaining official group-stage tie-breakers beyond points,
   goal difference, and goals scored.
2. Add explicit extra-time and penalty shootout assumptions for knockout matches.
3. Report uncertainty intervals over Monte Carlo simulations.
4. Keep the packaged FIFA Annex C third-place assignment table synced if FIFA
   revises the tournament regulations.

## Reporting

1. Produce a model card documenting assumptions and known weaknesses.
2. Generate a richer final report with group tables, bracket-path summaries, and
   top-scorer probabilities.
3. Compare model probabilities to market probabilities before kickoff.
