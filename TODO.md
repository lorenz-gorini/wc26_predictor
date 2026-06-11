# TODO

## Completed

1. Optimize tournament simulation.
   - Added a reusable abstract tournament simulator with cached group fixture arrays and Elo pairwise probabilities.
   - Writes team advancement, winner, knockout-match, and expected-match outputs to CSV.

2. Implement the official 2026 knockout bracket.
   - Encoded the official round-of-32, round-of-16, quarterfinal, semifinal, final, and third-place match tree.
   - Added the FIFA Annex C third-place assignment table as a local package resource.
   - Replaced the default abstract random knockout pairing model in the baseline workflow.

3. Produce full match forecasts.
   - Fixed group-stage forecasts are written to `world_cup_2026_baseline_forecasts.csv`.
   - Conditional official-bracket knockout pairings are written to `world_cup_2026_knockout_match_forecasts.csv`.
   - Combined group and knockout rows are written to `world_cup_2026_full_match_forecasts.csv`.

4. Learn calibrated ensemble weights.
   - Added grid-search log-loss fitting over historical World Cup holdout predictions.
   - The default group forecast and knockout advancement model now use calibrated weights.
   - Leave-one-window-out calibrated ensemble metrics are reported in the backtest table.

5. Tune model hyperparameters.
   - Added validation-grid tuning for Elo `k_factor`, draw probability, home advantage, and neutral-home handling.
   - Added validation-grid tuning for Poisson prior strength, recency half-life, and tournament-importance weighting.
   - Added validation-grid tuning for form window length, form prior, and form strength.
   - Selected configs are written to `world_cup_2026_selected_hyperparameters.csv`; full tuning results are written to `world_cup_2026_hyperparameter_tuning.csv`.

6. Add market benchmark and calibration.
   - Downloaded and ingested Football-Data's World Cup workbook into `data/raw/odds/WorldCup2026.xlsx`.
   - Removed bookmaker margin from average 1X2 odds to build market-implied probabilities.
   - Matched all 192 historical World Cup validation matches from 2014, 2018, and 2022.
   - Compared model-only, market-only, and leave-one-window-out model+market ensembles.
   - Added a validation gate so model+market is used only if it improves LOWO log loss over both model-only and market-only.

7. Improve top-scorer model.
   - Added squad-derived expected-minutes priors and national-team penalty-taker ranks.
   - Added optional current player availability overrides through `data/raw/world_cup_2026_player_availability.csv`.
   - Added injury/suspension status adjustments for top-scorer expectations and team expected goals.
   - Added team availability burden outputs and an optional historical availability-impact estimator.
   - Transfermarkt-adjusted scorer rankings now consume the availability-adjusted scorer baseline.

## Next

8. Generate final reports.
   - Group advancement probabilities.
   - Round-by-round probabilities.
   - Winner probabilities.
   - Top-scorer probabilities.
   - Match-by-match forecast tables.
