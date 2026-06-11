# Score Strategy Backtest

This report compares two exact-score selection strategies on historical World Cup holdout windows with chronological roll-forward inside each tournament.

Strategies:

- `score_only`: choose the modal score from the form-adjusted Poisson score matrix.
- `1x2_compatible`: choose the most likely score whose outcome matches the 1X2 ensemble's most likely outcome.

The 1X2 ensemble weights are fit from prior World Cup validation windows only. The first window uses equal weights because no prior World Cup validation window exists.

## Interpretation

The 1X2-compatible strategy improves exact-score hit rate by 6.2 percentage points and improves outcome accuracy, but it has +0.005 higher goal MAE. Its conditional exact-score log loss is much worse because the gate assigns zero probability to scores from the other two 1X2 outcome classes whenever the aggregate outcome call is wrong. Therefore, use the 1X2-compatible rule only if the objective is a single contest-style score pick; keep the raw score distribution when calibrated probabilities or goal-error performance matter.

## All Windows

| window | strategy | n_matches | exact_score_log_loss | exact_score_accuracy | outcome_accuracy | goal_mae | goal_rmse | total_goal_mae | average_selected_score_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_windows | score_only | 192 | 2.9737 | 0.0469 | 0.3802 | 1.0026 | 1.4170 | 1.7135 | 0.1431 |
| all_windows | 1x2_compatible | 192 | 16.1968 | 0.1094 | 0.5677 | 1.0078 | 1.4280 | 1.7448 | 0.1323 |

## By World Cup Window

| window | strategy | n_matches | exact_score_log_loss | exact_score_accuracy | outcome_accuracy | goal_mae | goal_rmse | total_goal_mae | average_selected_score_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014_fifa_world_cup | score_only | 64 | 3.0008 | 0.0312 | 0.3438 | 0.9688 | 1.4087 | 1.6562 | 0.1323 |
| 2014_fifa_world_cup | 1x2_compatible | 64 | 14.7930 | 0.1562 | 0.6094 | 0.9688 | 1.4197 | 1.6875 | 0.1250 |
| 2018_fifa_world_cup | score_only | 64 | 2.8200 | 0.0781 | 0.4219 | 0.9609 | 1.3020 | 1.6094 | 0.1465 |
| 2018_fifa_world_cup | 1x2_compatible | 64 | 15.7656 | 0.1406 | 0.5781 | 0.9688 | 1.3405 | 1.6562 | 0.1378 |
| 2022_fifa_world_cup | score_only | 64 | 3.1002 | 0.0312 | 0.3750 | 1.0781 | 1.5309 | 1.8750 | 0.1504 |
| 2022_fifa_world_cup | 1x2_compatible | 64 | 18.0318 | 0.0312 | 0.5156 | 1.0859 | 1.5181 | 1.8906 | 0.1343 |

## First Predictions

| window | match_index | home_team | away_team | observed_score | score_only_score | compatible_score | compatible_outcome | ensemble_outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014_fifa_world_cup | 1 | Brazil | Croatia | 3-1 | 2-0 | 2-0 | home_win | home_win |
| 2014_fifa_world_cup | 2 | Chile | Australia | 3-1 | 2-1 | 2-1 | home_win | home_win |
| 2014_fifa_world_cup | 3 | Mexico | Cameroon | 1-0 | 1-1 | 1-0 | home_win | home_win |
| 2014_fifa_world_cup | 4 | Spain | Netherlands | 1-5 | 1-0 | 1-0 | home_win | home_win |
| 2014_fifa_world_cup | 5 | Colombia | Greece | 3-0 | 0-0 | 1-0 | home_win | home_win |
| 2014_fifa_world_cup | 6 | England | Italy | 1-2 | 1-0 | 1-0 | home_win | home_win |
| 2014_fifa_world_cup | 7 | Ivory Coast | Japan | 2-1 | 1-1 | 2-1 | home_win | home_win |
| 2014_fifa_world_cup | 8 | Uruguay | Costa Rica | 1-3 | 1-0 | 1-0 | home_win | home_win |
