# Recent World Cup Holdout Backtest

Models are fit only on matches before each holdout block. The target blocks are the latest completed 2026 World Cup matches available in local data.

Historical model-market weight: model=0.36, market=0.64. No current 2026 market probabilities were matched.

## Summary

| holdout_size | model | n_matches | log_loss | brier_score | outcome_accuracy | average_observed_probability |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | ensemble | 5 | 1.1035 | 0.6998 | 0.4000 | 0.3733 |
| 5 | elo | 5 | 1.1074 | 0.7083 | 0.4000 | 0.3786 |
| 5 | poisson | 5 | 1.1332 | 0.6992 | 0.4000 | 0.3402 |
| 5 | form_adjusted_poisson | 5 | 1.1383 | 0.7085 | 0.4000 | 0.3430 |
| 10 | ensemble | 10 | 1.1212 | 0.7053 | 0.4000 | 0.3577 |
| 10 | form_adjusted_poisson | 10 | 1.1273 | 0.7118 | 0.4000 | 0.3422 |
| 10 | poisson | 10 | 1.1276 | 0.7085 | 0.4000 | 0.3408 |
| 10 | elo | 10 | 1.1295 | 0.7160 | 0.4000 | 0.3604 |
| 20 | ensemble | 12 | 1.0453 | 0.6490 | 0.5000 | 0.3866 |
| 20 | elo | 12 | 1.0469 | 0.6545 | 0.5000 | 0.3923 |
| 20 | poisson | 12 | 1.0823 | 0.6739 | 0.5000 | 0.3549 |
| 20 | form_adjusted_poisson | 12 | 1.0863 | 0.6799 | 0.4167 | 0.3546 |

## Last 20 Match Predictions

| match_number | date | home_team | away_team | observed_score | observed | ensemble_predicted_outcome | ensemble_observed_probability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-06-11 | Mexico | South Africa | 2-0 | home_win | home_win | 0.6661 |
| 2 | 2026-06-11 | South Korea | Czech Republic | 2-1 | home_win | home_win | 0.3965 |
| 3 | 2026-06-12 | Canada | Bosnia and Herzegovina | 1-1 | draw | home_win | 0.2244 |
| 4 | 2026-06-12 | United States | Paraguay | 4-1 | home_win | home_win | 0.4255 |
| 5 | 2026-06-13 | Haiti | Scotland | 0-1 | away_win | away_win | 0.5505 |
| 6 | 2026-06-13 | Australia | Turkey | 2-0 | home_win | away_win | 0.2874 |
| 7 | 2026-06-13 | Brazil | Morocco | 1-1 | draw | home_win | 0.2227 |
| 8 | 2026-06-13 | Qatar | Switzerland | 1-1 | draw | away_win | 0.2172 |
| 9 | 2026-06-14 | Ivory Coast | Ecuador | 1-0 | home_win | away_win | 0.2518 |
| 10 | 2026-06-14 | Germany | Curaçao | 7-1 | home_win | home_win | 0.7184 |
| 11 | 2026-06-14 | Netherlands | Japan | 2-2 | draw | away_win | 0.2253 |
| 12 | 2026-06-14 | Sweden | Tunisia | 5-1 | home_win | home_win | 0.4535 |
