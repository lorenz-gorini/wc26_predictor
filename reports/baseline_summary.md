# Baseline Model Summary

## Backtest Averages

| model | log_loss | brier_score |
| --- | --- | --- |
| elo | 0.9723 | 0.5716 |
| ensemble_calibrated_lowo | 0.9786 | 0.5762 |
| ensemble_equal_weight | 0.9880 | 0.5884 |
| form_adjusted_poisson | 1.0155 | 0.6090 |
| poisson | 1.0183 | 0.6110 |

## Most Decisive 2026 Group Forecasts

| match_number | group | home_team | away_team | predicted_score | predicted_score_outcome | predicted_score_probability | ensemble_home_win | ensemble_draw | ensemble_away_win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14 | H | Spain | Cape Verde | 2-0 | home_win | 0.120 | 0.743 | 0.216 | 0.041 |
| 38 | H | Spain | Saudi Arabia | 2-0 | home_win | 0.137 | 0.741 | 0.217 | 0.043 |
| 70 | J | Jordan | Argentina | 0-2 | away_win | 0.152 | 0.054 | 0.211 | 0.735 |
| 45 | L | England | Ghana | 1-0 | home_win | 0.158 | 0.729 | 0.217 | 0.054 |
| 8 | B | Qatar | Switzerland | 0-2 | away_win | 0.108 | 0.058 | 0.217 | 0.725 |
| 10 | E | Germany | Curaçao | 2-1 | home_win | 0.091 | 0.718 | 0.217 | 0.065 |
| 64 | G | New Zealand | Belgium | 0-2 | away_win | 0.099 | 0.086 | 0.212 | 0.703 |
| 42 | I | France | Iraq | 1-0 | home_win | 0.144 | 0.697 | 0.226 | 0.077 |
| 27 | B | Canada | Qatar | 1-0 | home_win | 0.175 | 0.695 | 0.227 | 0.078 |
| 50 | C | Morocco | Haiti | 1-0 | home_win | 0.165 | 0.687 | 0.220 | 0.093 |

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
| Norway | Erling Haaland | 23.321 | 0.415 | 5.010 | 5.007 | 2.080 | 3.471 |
| England | Harry Kane | 19.530 | 0.310 | 5.512 | 5.457 | 1.709 | 3.108 |
| Argentina | Lionel Messi | 10.507 | 0.243 | 4.632 | 6.052 | 1.127 | 2.274 |
| Portugal | Cristiano Ronaldo | 19.517 | 0.279 | 4.329 | 5.030 | 1.208 | 2.026 |
| Belgium | Romelu Lukaku | 13.071 | 0.201 | 5.386 | 5.136 | 1.084 | 1.856 |
| France | Kylian Mbappé | 15.511 | 0.262 | 4.168 | 5.025 | 1.094 | 1.832 |
| Argentina | Lautaro Martínez | 7.836 | 0.182 | 4.632 | 6.052 | 0.841 | 1.696 |
| Netherlands | Memphis Depay | 13.250 | 0.186 | 5.385 | 4.936 | 0.999 | 1.644 |
| Iran | Mehdi Taremi | 9.766 | 0.243 | 4.175 | 4.527 | 1.016 | 1.533 |
| Spain | Mikel Oyarzabal | 9.848 | 0.133 | 5.689 | 5.954 | 0.756 | 1.500 |
| Morocco | Ayoub El Kaabi | 6.237 | 0.163 | 5.030 | 5.272 | 0.820 | 1.442 |
| Egypt | Mohamed Salah | 10.828 | 0.345 | 2.969 | 4.092 | 1.025 | 1.399 |
| Austria | Marko Arnautović | 11.669 | 0.227 | 4.044 | 4.548 | 0.920 | 1.394 |
| Croatia | Andrej Kramarić | 9.784 | 0.186 | 4.494 | 4.529 | 0.835 | 1.260 |
| Netherlands | Cody Gakpo | 9.932 | 0.139 | 5.385 | 4.936 | 0.749 | 1.232 |

This scorer table uses national-team goalscorer history, 2026 squad filtering, and official-bracket expected-match exposure.

## Official-Bracket Tournament Winner Probabilities

| team | round_of_32_probability | quarter_final_probability | semi_final_probability | final_probability | winner_probability |
| --- | --- | --- | --- | --- | --- |
| Spain | 0.947 | 0.523 | 0.402 | 0.297 | 0.195 |
| Argentina | 0.926 | 0.572 | 0.428 | 0.294 | 0.192 |
| France | 0.706 | 0.352 | 0.224 | 0.129 | 0.074 |
| England | 0.953 | 0.405 | 0.232 | 0.123 | 0.063 |
| Portugal | 0.819 | 0.318 | 0.175 | 0.099 | 0.048 |
| Belgium | 0.884 | 0.341 | 0.159 | 0.088 | 0.041 |
| Germany | 0.779 | 0.302 | 0.168 | 0.086 | 0.040 |
| Brazil | 0.666 | 0.298 | 0.173 | 0.086 | 0.040 |
| Morocco | 0.945 | 0.364 | 0.191 | 0.084 | 0.038 |
| Japan | 0.941 | 0.356 | 0.197 | 0.085 | 0.037 |
| Norway | 0.840 | 0.313 | 0.151 | 0.065 | 0.035 |
| Netherlands | 0.844 | 0.292 | 0.157 | 0.072 | 0.035 |

These probabilities use group-stage simulation, FIFA Annex C third-place assignments, and the official 2026 knockout bracket.

## Availability-Adjusted Top-Scorer Baseline

| team | scorer | status | expected_minutes_share | penalty_taker_rank | expected_tournament_goals | expected_tournament_goals_availability_adjusted |
| --- | --- | --- | --- | --- | --- | --- |
| Norway | Erling Haaland | available | 0.950 | 1.0 | 3.471 | 3.588 |
| England | Harry Kane | available | 0.950 | 1.0 | 3.108 | 3.213 |
| Argentina | Lionel Messi | available | 0.950 | 1.0 | 2.274 | 2.351 |
| Portugal | Cristiano Ronaldo | available | 0.950 | 1.0 | 2.026 | 2.094 |
| Belgium | Romelu Lukaku | available | 0.950 | 1.0 | 1.856 | 1.918 |
| France | Kylian Mbappé | available | 0.950 | 1.0 | 1.832 | 1.893 |
| Netherlands | Memphis Depay | available | 0.950 | 1.0 | 1.644 | 1.699 |
| Argentina | Lautaro Martínez | available | 0.921 | NA | 1.696 | 1.629 |
| Iran | Mehdi Taremi | available | 0.950 | 1.0 | 1.533 | 1.584 |
| Spain | Mikel Oyarzabal | available | 0.922 | 1.0 | 1.500 | 1.528 |
| Egypt | Mohamed Salah | available | 0.950 | 1.0 | 1.399 | 1.445 |
| Austria | Marko Arnautović | available | 0.950 | 1.0 | 1.394 | 1.441 |
| Morocco | Ayoub El Kaabi | available | 0.950 | NA | 1.442 | 1.406 |
| Croatia | Andrej Kramarić | available | 0.894 | 2.0 | 1.260 | 1.229 |
| Belgium | Kevin De Bruyne | available | 0.922 | 2.0 | 1.220 | 1.208 |

This table applies expected-minutes, penalty-role, and injury/suspension status adjustments before club-form enrichment.

## Club-Form Adjusted Top 100

| team | scorer | club | club_form_match_quality | club_form_goals | expected_tournament_goals | expected_tournament_goals_club_adjusted |
| --- | --- | --- | --- | --- | --- | --- |
| Norway | Erling Haaland | Manchester City | player_and_club | 27.000 | 3.588 | 4.484 |
| England | Harry Kane | Bayern Munich | player_and_club | 36.000 | 3.213 | 4.016 |
| Portugal | Cristiano Ronaldo | Al-Nassr | player_and_club | 28.000 | 2.094 | 2.618 |
| France | Kylian Mbappé | Real Madrid | player_and_club | 25.000 | 1.893 | 2.367 |
| Argentina | Lionel Messi | Inter Miami CF | player_and_club | 12.000 | 2.351 | 1.967 |
| Belgium | Romelu Lukaku | Napoli | unmatched | NA | 1.918 | 1.918 |
| Netherlands | Memphis Depay | Corinthians | unmatched | NA | 1.699 | 1.699 |
| Argentina | Lautaro Martínez | Inter Milan | player_and_club | 17.000 | 1.629 | 1.695 |
| Iran | Mehdi Taremi | Olympiacos | unmatched | NA | 1.584 | 1.584 |
| Spain | Mikel Oyarzabal | Real Sociedad | player_and_club | 15.000 | 1.528 | 1.466 |
| Egypt | Mohamed Salah | Liverpool | unmatched | NA | 1.445 | 1.445 |
| Austria | Marko Arnautović | Red Star Belgrade | unmatched | NA | 1.441 | 1.441 |
| Morocco | Ayoub El Kaabi | Olympiacos | unmatched | NA | 1.406 | 1.406 |
| Belgium | Kevin De Bruyne | Napoli | unmatched | NA | 1.208 | 1.208 |
| Bosnia and Herzegovina | Edin Džeko | Schalke 04 | unmatched | NA | 1.205 | 1.205 |

Club form is sourced from curated public top-scorer tables and currently covers only matched top-100 candidates.

## Transfermarkt-Adjusted Top 100

| team | scorer | transfermarkt_match_quality | club_goals_source | club_goals_model | club_weighted_goals_model | club_minutes_model | club_weighted_goals_per90 | transfermarkt_multiplier | latest_market_value_in_eur | expected_tournament_goals | expected_tournament_goals_transfermarkt_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Norway | Erling Haaland | player_country_club | appearances | 39.000 | 16.110 | 4235.000 | 0.342 | 1.449 | 200000000.000 | 3.588 | 5.198 |
| England | Harry Kane | player_country | appearances | 61.000 | 24.870 | 4140.000 | 0.541 | 1.450 | 65000000.000 | 3.213 | 4.658 |
| Argentina | Lionel Messi | player_country | game_events | 31.000 | 9.321 | 1725.000 | 0.486 | 1.401 | 35000000.000 | 2.351 | 3.293 |
| France | Kylian Mbappé | player_country_club | appearances | 43.000 | 16.140 | 3756.000 | 0.387 | 1.450 | 180000000.000 | 1.893 | 2.746 |
| Portugal | Cristiano Ronaldo | player_country_club | game_events | 28.000 | 9.773 | 2250.000 | 0.391 | 1.207 | 15000000.000 | 2.094 | 2.527 |
| Argentina | Lautaro Martínez | player_country | appearances | 22.000 | 10.350 | 2767.000 | 0.337 | 1.390 | 85000000.000 | 1.629 | 2.264 |
| Belgium | Romelu Lukaku | player_country_club | appearances | 1.000 | 0.666 | 64.000 | 0.936 | 0.987 | 15000000.000 | 1.918 | 1.893 |
| Spain | Mikel Oyarzabal | player_country_club | appearances | 18.000 | 9.342 | 3195.000 | 0.263 | 1.217 | 25000000.000 | 1.528 | 1.860 |
| Iran | Mehdi Taremi | player_country | appearances | 15.000 | 4.635 | 1833.000 | 0.228 | 1.039 | 2500000.000 | 1.584 | 1.646 |
| Norway | Alexander Sørloth | player_country | appearances | 20.000 | 10.678 | 2816.000 | 0.341 | 1.330 | 20000000.000 | 1.175 | 1.563 |
| Morocco | Ayoub El Kaabi | player_country | appearances | 20.000 | 6.711 | 2429.000 | 0.249 | 1.112 | 5000000.000 | 1.406 | 1.563 |
| Egypt | Mohamed Salah | player_country_club | appearances | 12.000 | 5.431 | 3150.000 | 0.155 | 1.050 | 30000000.000 | 1.445 | 1.518 |
| Spain | Ferran Torres | player_country_club | appearances | 21.000 | 9.287 | 2692.000 | 0.310 | 1.298 | 30000000.000 | 1.056 | 1.370 |
| Croatia | Andrej Kramarić | player_country | appearances | 15.000 | 6.856 | 2422.000 | 0.255 | 1.098 | 3000000.000 | 1.229 | 1.349 |
| Sweden | Viktor Gyökeres | player_country_club | appearances | 21.000 | 10.552 | 3427.000 | 0.277 | 1.289 | 70000000.000 | 1.019 | 1.313 |

### Transfermarkt Match Coverage

| match_quality | players |
| --- | --- |
| player_country_club | 65 |
| player_country | 31 |
| unmatched | 4 |

Transfermarkt features are preferred over sparse public top-scorer tables when the local Kaggle dump is available.
