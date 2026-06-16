# Baseline Model Summary

## Backtest Averages

| model | log_loss | brier_score |
| --- | --- | --- |
| elo | 0.9723 | 0.5716 |
| ensemble_calibrated_lowo | 0.9786 | 0.5762 |
| ensemble_equal_weight | 0.9880 | 0.5884 |
| form_adjusted_poisson | 1.0156 | 0.6090 |
| poisson | 1.0183 | 0.6110 |

## Most Decisive 2026 Group Forecasts

| match_number | group | home_team | away_team | predicted_score | predicted_score_outcome | predicted_score_probability | ensemble_home_win | ensemble_draw | ensemble_away_win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | E | Germany | Curaçao | 3-1 | home_win | 0.077 | 0.746 | 0.207 | 0.047 |
| 14 | H | Spain | Cape Verde | 2-0 | home_win | 0.120 | 0.743 | 0.216 | 0.041 |
| 38 | H | Spain | Saudi Arabia | 2-0 | home_win | 0.137 | 0.741 | 0.217 | 0.043 |
| 70 | J | Jordan | Argentina | 0-2 | away_win | 0.151 | 0.054 | 0.211 | 0.735 |
| 45 | L | England | Ghana | 2-0 | home_win | 0.158 | 0.734 | 0.215 | 0.051 |
| 8 | B | Qatar | Switzerland | 0-1 | away_win | 0.113 | 0.067 | 0.219 | 0.714 |
| 64 | G | New Zealand | Belgium | 0-2 | away_win | 0.099 | 0.086 | 0.212 | 0.703 |
| 42 | I | France | Iraq | 1-0 | home_win | 0.144 | 0.697 | 0.226 | 0.077 |
| 55 | E | Curaçao | Ivory Coast | 0-2 | away_win | 0.113 | 0.093 | 0.215 | 0.692 |
| 50 | C | Morocco | Haiti | 1-0 | home_win | 0.170 | 0.691 | 0.221 | 0.088 |

Notes:

- World Cup 2026 fixtures are treated as venue-neutral in this first baseline.
- Host-country advantage is not hard-coded; it should be retained only if it improves backtests.
- The primary ensemble uses weights learned from historical World Cup holdout predictions.

## Calibrated Ensemble Weights

| model | weight |
| --- | --- |
| elo | 0.850 |
| poisson | 0.000 |
| form_adjusted_poisson | 0.150 |

Weights are fit by minimizing multiclass log loss over the 2014, 2018, and 2022 World Cup validation predictions.

## Tuned Hyperparameters

| model | parameter | value |
| --- | --- | --- |
| elo | k_factor | 40.0 |
| elo | draw_probability | 0.22 |
| elo | home_advantage | 40.0 |
| elo | neutral_home_advantage | 0.0 |
| poisson | prior_strength | 3.0 |
| poisson | recency_half_life_days | 365 |
| poisson | use_tournament_importance | False |
| form_adjusted_poisson | form_window_matches | 10 |
| form_adjusted_poisson | form_prior_matches | 4.0 |
| form_adjusted_poisson | form_strength | 0.15 |

Hyperparameters are selected by minimizing validation log loss across the 2014, 2018, and 2022 World Cup holdout windows.

## Market Benchmark

| model | log_loss | brier_score |
| --- | --- | --- |
| market_only | 0.9613 | 0.5667 |
| model_market_lowo | 0.9616 | 0.5671 |
| model_only | 0.9707 | 0.5722 |

Gate decision: LOWO model+market does not improve over both model-only and market-only log loss.

## Availability Burden

| team | team_availability_burden | unavailable_players |
| --- | --- | --- |
| Morocco | 0.809 | 2 |
| Argentina | 0.080 | 1 |
| Netherlands | 0.040 | 1 |
| Algeria | 0.000 | 0 |
| Australia | 0.000 | 0 |
| Austria | 0.000 | 0 |
| Belgium | 0.000 | 0 |
| Bosnia and Herzegovina | 0.000 | 0 |
| Brazil | 0.000 | 0 |
| Canada | 0.000 | 0 |
| Cape Verde | 0.000 | 0 |
| Colombia | 0.000 | 0 |

| source | goal_penalty_per_burden | note |
| --- | --- | --- |
| default | 0.080 | No historical_team_availability.csv file was found; using conservative default. |

Availability burden is neutral unless current injury/suspension overrides are provided in `data/raw/world_cup_2026_player_availability.csv`.

## Transfermarkt Injury Burden

| team | transfermarkt_players_matched | players_with_recent_injury | players_with_open_injury | team_historical_injury_burden | team_current_open_injury_burden |
| --- | --- | --- | --- | --- | --- |
| Austria | 25 | 21 | 0 | 2.614 | 0.000 |
| Netherlands | 26 | 22 | 0 | 2.514 | 0.000 |
| Germany | 26 | 25 | 0 | 2.414 | 0.000 |
| Turkey | 26 | 22 | 0 | 2.193 | 0.000 |
| Australia | 26 | 14 | 0 | 2.080 | 0.000 |
| United States | 25 | 16 | 0 | 2.062 | 0.000 |
| Switzerland | 26 | 16 | 0 | 2.019 | 0.000 |
| Scotland | 23 | 16 | 0 | 1.930 | 0.000 |
| Spain | 25 | 16 | 0 | 1.824 | 0.000 |
| Uruguay | 24 | 18 | 0 | 1.790 | 0.000 |
| Brazil | 21 | 12 | 0 | 1.732 | 0.000 |
| Belgium | 26 | 22 | 0 | 1.712 | 0.000 |

Transfermarkt injury history is joined by player ID where possible and otherwise by normalized player, country, and club signals. Historical recurrence burden is diagnostic; verified current overrides should still be supplied through `data/raw/world_cup_2026_player_availability.csv` before the tournament.

## National-Team Top-Scorer Baseline

| team | scorer | weighted_goals | goal_share | expected_team_group_goals | expected_team_matches | expected_group_goals | expected_tournament_goals |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Norway | Erling Haaland | 21.718 | 0.414 | 5.005 | 4.877 | 2.070 | 3.365 |
| England | Harry Kane | 18.188 | 0.309 | 5.714 | 5.542 | 1.765 | 3.261 |
| Argentina | Lionel Messi | 9.785 | 0.242 | 4.587 | 5.941 | 1.111 | 2.200 |
| Portugal | Cristiano Ronaldo | 18.176 | 0.278 | 4.304 | 5.076 | 1.198 | 2.026 |
| Belgium | Romelu Lukaku | 12.172 | 0.201 | 5.380 | 5.163 | 1.079 | 1.857 |
| France | Kylian Mbappé | 14.445 | 0.261 | 4.165 | 4.947 | 1.089 | 1.796 |
| Netherlands | Memphis Depay | 12.339 | 0.180 | 5.737 | 4.965 | 1.030 | 1.705 |
| Argentina | Lautaro Martínez | 7.297 | 0.181 | 4.587 | 5.941 | 0.828 | 1.640 |
| Iran | Mehdi Taremi | 9.095 | 0.242 | 4.171 | 4.535 | 1.009 | 1.526 |
| Sweden | Viktor Gyökeres | 11.160 | 0.260 | 4.084 | 4.277 | 1.063 | 1.515 |
| Spain | Mikel Oyarzabal | 9.172 | 0.132 | 5.683 | 5.899 | 0.753 | 1.480 |
| Egypt | Mohamed Salah | 10.084 | 0.343 | 2.968 | 4.034 | 1.018 | 1.369 |
| Austria | Marko Arnautović | 10.867 | 0.226 | 4.002 | 4.530 | 0.906 | 1.368 |
| Morocco | Ayoub El Kaabi | 5.809 | 0.158 | 4.805 | 5.128 | 0.758 | 1.296 |
| Netherlands | Cody Gakpo | 9.250 | 0.135 | 5.737 | 4.965 | 0.772 | 1.278 |

This scorer table uses national-team goalscorer history, 2026 squad filtering, and official-bracket expected-match exposure.

## Official-Bracket Tournament Winner Probabilities

| team | round_of_32_probability | quarter_final_probability | semi_final_probability | final_probability | winner_probability |
| --- | --- | --- | --- | --- | --- |
| Argentina | 0.917 | 0.535 | 0.403 | 0.286 | 0.197 |
| Spain | 0.945 | 0.506 | 0.391 | 0.287 | 0.196 |
| England | 0.965 | 0.419 | 0.251 | 0.132 | 0.059 |
| France | 0.718 | 0.333 | 0.203 | 0.113 | 0.059 |
| Germany | 0.996 | 0.394 | 0.225 | 0.107 | 0.051 |
| Brazil | 0.738 | 0.293 | 0.173 | 0.083 | 0.044 |
| Portugal | 0.808 | 0.334 | 0.186 | 0.110 | 0.043 |
| Morocco | 0.905 | 0.327 | 0.181 | 0.089 | 0.041 |
| Japan | 0.919 | 0.311 | 0.164 | 0.074 | 0.037 |
| Belgium | 0.894 | 0.345 | 0.158 | 0.076 | 0.037 |
| Colombia | 0.694 | 0.261 | 0.136 | 0.076 | 0.036 |
| Netherlands | 0.887 | 0.295 | 0.142 | 0.061 | 0.031 |

These probabilities use group-stage simulation, FIFA Annex C third-place assignments, and the official 2026 knockout bracket.

## Availability-Adjusted Top-Scorer Baseline

| team | scorer | status | expected_minutes_share | penalty_taker_rank | expected_tournament_goals | expected_tournament_goals_availability_adjusted |
| --- | --- | --- | --- | --- | --- | --- |
| Norway | Erling Haaland | available | 0.950 | 1.0 | 3.365 | 3.477 |
| England | Harry Kane | available | 0.950 | 1.0 | 3.261 | 3.370 |
| Argentina | Lionel Messi | available | 0.950 | 1.0 | 2.200 | 2.273 |
| Portugal | Cristiano Ronaldo | available | 0.950 | 1.0 | 2.026 | 2.094 |
| Belgium | Romelu Lukaku | available | 0.950 | 1.0 | 1.857 | 1.919 |
| France | Kylian Mbappé | available | 0.950 | 1.0 | 1.796 | 1.856 |
| Netherlands | Memphis Depay | available | 0.950 | 1.0 | 1.705 | 1.762 |
| Iran | Mehdi Taremi | available | 0.950 | 1.0 | 1.526 | 1.577 |
| Argentina | Lautaro Martínez | available | 0.921 | NA | 1.640 | 1.576 |
| Sweden | Viktor Gyökeres | available | 0.922 | 1.0 | 1.515 | 1.544 |
| Spain | Mikel Oyarzabal | available | 0.922 | 1.0 | 1.480 | 1.508 |
| Egypt | Mohamed Salah | available | 0.950 | 1.0 | 1.369 | 1.414 |
| Austria | Marko Arnautović | available | 0.950 | 1.0 | 1.368 | 1.414 |
| Morocco | Ayoub El Kaabi | available | 0.950 | NA | 1.296 | 1.263 |
| Netherlands | Cody Gakpo | available | 0.894 | 2.0 | 1.278 | 1.247 |

This table applies expected-minutes, penalty-role, and injury/suspension status adjustments before club-form enrichment.

## Club-Form Adjusted Top 100

| team | scorer | club | club_form_match_quality | club_form_goals | expected_tournament_goals | expected_tournament_goals_club_adjusted |
| --- | --- | --- | --- | --- | --- | --- |
| Norway | Erling Haaland | Manchester City | player_and_club | 27.000 | 3.477 | 4.347 |
| England | Harry Kane | Bayern Munich | player_and_club | 36.000 | 3.370 | 4.212 |
| Portugal | Cristiano Ronaldo | Al-Nassr | player_and_club | 28.000 | 2.094 | 2.618 |
| France | Kylian Mbappé | Real Madrid | player_and_club | 25.000 | 1.856 | 2.320 |
| Belgium | Romelu Lukaku | Napoli | unmatched | NA | 1.919 | 1.919 |
| Argentina | Lionel Messi | Inter Miami CF | player_and_club | 12.000 | 2.273 | 1.905 |
| Netherlands | Memphis Depay | Corinthians | unmatched | NA | 1.762 | 1.762 |
| Argentina | Lautaro Martínez | Inter Milan | player_and_club | 17.000 | 1.576 | 1.639 |
| Iran | Mehdi Taremi | Olympiacos | unmatched | NA | 1.577 | 1.577 |
| Sweden | Viktor Gyökeres | Arsenal | unmatched | NA | 1.544 | 1.544 |
| Spain | Mikel Oyarzabal | Real Sociedad | player_and_club | 15.000 | 1.508 | 1.447 |
| Egypt | Mohamed Salah | Liverpool | unmatched | NA | 1.414 | 1.414 |
| Austria | Marko Arnautović | Red Star Belgrade | unmatched | NA | 1.414 | 1.414 |
| Morocco | Ayoub El Kaabi | Olympiacos | unmatched | NA | 1.263 | 1.263 |
| Netherlands | Cody Gakpo | Liverpool | unmatched | NA | 1.247 | 1.247 |

Club form is sourced from curated public top-scorer tables and currently covers only matched top-100 candidates.

## Transfermarkt-Adjusted Top 100

| team | scorer | transfermarkt_match_quality | club_goals_source | club_goals_model | club_weighted_goals_model | club_minutes_model | club_weighted_goals_per90 | transfermarkt_multiplier | latest_market_value_in_eur | expected_tournament_goals | expected_tournament_goals_transfermarkt_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Norway | Erling Haaland | player_country_club | appearances | 39.000 | 16.110 | 4235.000 | 0.342 | 1.431 | 200000000.000 | 3.477 | 4.976 |
| England | Harry Kane | player_country | appearances | 61.000 | 24.870 | 4140.000 | 0.541 | 1.450 | 65000000.000 | 3.370 | 4.886 |
| Argentina | Lionel Messi | player_country | game_events | 31.000 | 9.321 | 1725.000 | 0.486 | 1.382 | 35000000.000 | 2.273 | 3.142 |
| France | Kylian Mbappé | player_country_club | appearances | 43.000 | 16.140 | 3756.000 | 0.387 | 1.450 | 180000000.000 | 1.856 | 2.691 |
| Portugal | Cristiano Ronaldo | player_country_club | game_events | 28.000 | 9.773 | 2250.000 | 0.391 | 1.184 | 15000000.000 | 2.094 | 2.479 |
| Argentina | Lautaro Martínez | player_country | appearances | 22.000 | 10.350 | 2767.000 | 0.337 | 1.370 | 85000000.000 | 1.576 | 2.158 |
| Sweden | Viktor Gyökeres | player_country_club | appearances | 21.000 | 10.552 | 3427.000 | 0.277 | 1.268 | 70000000.000 | 1.544 | 1.957 |
| Belgium | Romelu Lukaku | player_country_club | appearances | 1.000 | 0.666 | 64.000 | 0.936 | 0.982 | 15000000.000 | 1.919 | 1.885 |
| Spain | Mikel Oyarzabal | player_country_club | appearances | 18.000 | 9.342 | 3195.000 | 0.263 | 1.193 | 25000000.000 | 1.508 | 1.799 |
| Iran | Mehdi Taremi | player_country | appearances | 15.000 | 4.635 | 1833.000 | 0.228 | 1.010 | 2500000.000 | 1.577 | 1.592 |
| Norway | Alexander Sørloth | player_country | appearances | 20.000 | 10.678 | 2816.000 | 0.341 | 1.308 | 20000000.000 | 1.139 | 1.490 |
| Germany | Kai Havertz | player_country_club | appearances | 6.000 | 3.909 | 976.000 | 0.360 | 1.363 | 55000000.000 | 1.063 | 1.449 |
| Egypt | Mohamed Salah | player_country_club | appearances | 12.000 | 5.431 | 3150.000 | 0.155 | 1.024 | 30000000.000 | 1.414 | 1.448 |
| Morocco | Ayoub El Kaabi | player_country | appearances | 20.000 | 6.711 | 2429.000 | 0.249 | 1.085 | 5000000.000 | 1.263 | 1.370 |
| Spain | Ferran Torres | player_country_club | appearances | 21.000 | 9.287 | 2692.000 | 0.310 | 1.276 | 30000000.000 | 1.042 | 1.329 |

### Transfermarkt Match Coverage

| match_quality | players |
| --- | --- |
| player_country_club | 64 |
| player_country | 31 |
| unmatched | 5 |

Transfermarkt features are preferred over sparse public top-scorer tables when the local Kaggle dump is available.
