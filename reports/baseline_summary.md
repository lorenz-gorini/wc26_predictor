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

| match_number | group | home_team | away_team | ensemble_home_win | ensemble_draw | ensemble_away_win |
| --- | --- | --- | --- | --- | --- | --- |
| 14 | H | Spain | Cape Verde | 0.743 | 0.216 | 0.041 |
| 38 | H | Spain | Saudi Arabia | 0.741 | 0.217 | 0.043 |
| 70 | J | Jordan | Argentina | 0.054 | 0.211 | 0.735 |
| 45 | L | England | Ghana | 0.729 | 0.217 | 0.054 |
| 8 | B | Qatar | Switzerland | 0.058 | 0.217 | 0.725 |
| 10 | E | Germany | Curaçao | 0.718 | 0.217 | 0.065 |
| 64 | G | New Zealand | Belgium | 0.086 | 0.212 | 0.703 |
| 42 | I | France | Iraq | 0.697 | 0.226 | 0.077 |
| 27 | B | Canada | Qatar | 0.695 | 0.227 | 0.078 |
| 50 | C | Morocco | Haiti | 0.687 | 0.220 | 0.093 |

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
| Algeria | 0.000 | 0 |
| Argentina | 0.000 | 0 |
| Australia | 0.000 | 0 |
| Austria | 0.000 | 0 |
| Belgium | 0.000 | 0 |
| Bosnia and Herzegovina | 0.000 | 0 |
| Brazil | 0.000 | 0 |
| Canada | 0.000 | 0 |
| Cape Verde | 0.000 | 0 |
| Colombia | 0.000 | 0 |
| Croatia | 0.000 | 0 |
| Curaçao | 0.000 | 0 |

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
| Norway | Erling Haaland | 23.321 | 0.415 | 5.010 | 4.958 | 2.080 | 3.437 |
| England | Harry Kane | 19.530 | 0.310 | 5.512 | 5.473 | 1.709 | 3.117 |
| Argentina | Lionel Messi | 10.507 | 0.243 | 4.662 | 6.037 | 1.135 | 2.283 |
| Portugal | Cristiano Ronaldo | 19.517 | 0.279 | 4.329 | 5.028 | 1.208 | 2.025 |
| Belgium | Romelu Lukaku | 13.071 | 0.201 | 5.386 | 5.238 | 1.084 | 1.892 |
| France | Kylian Mbappé | 15.511 | 0.262 | 4.168 | 5.107 | 1.094 | 1.862 |
| Argentina | Lautaro Martínez | 7.836 | 0.182 | 4.662 | 6.037 | 0.846 | 1.703 |
| Netherlands | Memphis Depay | 13.250 | 0.186 | 5.402 | 4.888 | 1.002 | 1.633 |
| Morocco | Ayoub El Kaabi | 6.237 | 0.163 | 5.378 | 5.271 | 0.877 | 1.541 |
| Iran | Mehdi Taremi | 9.766 | 0.243 | 4.175 | 4.535 | 1.016 | 1.536 |
| Spain | Mikel Oyarzabal | 9.848 | 0.133 | 5.689 | 5.917 | 0.756 | 1.491 |
| Austria | Marko Arnautović | 11.669 | 0.227 | 4.044 | 4.610 | 0.920 | 1.413 |
| Egypt | Mohamed Salah | 10.828 | 0.345 | 2.969 | 4.095 | 1.025 | 1.400 |
| Morocco | Brahim Díaz | 5.290 | 0.138 | 5.378 | 5.271 | 0.744 | 1.307 |
| Croatia | Andrej Kramarić | 9.784 | 0.186 | 4.494 | 4.579 | 0.835 | 1.274 |

This scorer table uses national-team goalscorer history, 2026 squad filtering, and official-bracket expected-match exposure.

## Official-Bracket Tournament Winner Probabilities

| team | round_of_32_probability | quarter_final_probability | semi_final_probability | final_probability | winner_probability |
| --- | --- | --- | --- | --- | --- |
| Argentina | 0.920 | 0.567 | 0.433 | 0.297 | 0.189 |
| Spain | 0.939 | 0.527 | 0.388 | 0.271 | 0.174 |
| France | 0.699 | 0.370 | 0.254 | 0.151 | 0.089 |
| England | 0.944 | 0.406 | 0.243 | 0.134 | 0.072 |
| Portugal | 0.823 | 0.309 | 0.179 | 0.094 | 0.047 |
| Germany | 0.745 | 0.310 | 0.174 | 0.091 | 0.040 |
| Brazil | 0.679 | 0.318 | 0.179 | 0.092 | 0.040 |
| Belgium | 0.903 | 0.372 | 0.167 | 0.092 | 0.038 |
| Japan | 0.940 | 0.340 | 0.181 | 0.087 | 0.036 |
| Norway | 0.844 | 0.292 | 0.143 | 0.064 | 0.032 |
| Netherlands | 0.830 | 0.289 | 0.153 | 0.070 | 0.032 |
| Colombia | 0.690 | 0.224 | 0.126 | 0.064 | 0.030 |

These probabilities use group-stage simulation, FIFA Annex C third-place assignments, and the official 2026 knockout bracket.

## Availability-Adjusted Top-Scorer Baseline

| team | scorer | status | expected_minutes_share | penalty_taker_rank | expected_tournament_goals | expected_tournament_goals_availability_adjusted |
| --- | --- | --- | --- | --- | --- | --- |
| Norway | Erling Haaland | available | 0.950 | 1.0 | 3.437 | 3.552 |
| England | Harry Kane | available | 0.950 | 1.0 | 3.117 | 3.222 |
| Argentina | Lionel Messi | available | 0.950 | 1.0 | 2.283 | 2.360 |
| Portugal | Cristiano Ronaldo | available | 0.950 | 1.0 | 2.025 | 2.093 |
| Belgium | Romelu Lukaku | available | 0.950 | 1.0 | 1.892 | 1.956 |
| France | Kylian Mbappé | available | 0.950 | 1.0 | 1.862 | 1.924 |
| Netherlands | Memphis Depay | available | 0.950 | 1.0 | 1.633 | 1.688 |
| Argentina | Lautaro Martínez | available | 0.921 | NA | 1.703 | 1.635 |
| Iran | Mehdi Taremi | available | 0.950 | 1.0 | 1.536 | 1.587 |
| Spain | Mikel Oyarzabal | available | 0.922 | 1.0 | 1.491 | 1.519 |
| Morocco | Ayoub El Kaabi | available | 0.950 | NA | 1.541 | 1.503 |
| Austria | Marko Arnautović | available | 0.950 | 1.0 | 1.413 | 1.461 |
| Egypt | Mohamed Salah | available | 0.950 | 1.0 | 1.400 | 1.447 |
| Croatia | Andrej Kramarić | available | 0.894 | 2.0 | 1.274 | 1.243 |
| Belgium | Kevin De Bruyne | available | 0.922 | 2.0 | 1.245 | 1.232 |

This table applies expected-minutes, penalty-role, and injury/suspension status adjustments before club-form enrichment.

## Club-Form Adjusted Top 100

| team | scorer | club | club_form_match_quality | club_form_goals | expected_tournament_goals | expected_tournament_goals_club_adjusted |
| --- | --- | --- | --- | --- | --- | --- |
| Norway | Erling Haaland | Manchester City | player_and_club | 27.000 | 3.552 | 4.441 |
| England | Harry Kane | Bayern Munich | player_and_club | 36.000 | 3.222 | 4.027 |
| Portugal | Cristiano Ronaldo | Al-Nassr | player_and_club | 28.000 | 2.093 | 2.617 |
| France | Kylian Mbappé | Real Madrid | player_and_club | 25.000 | 1.924 | 2.405 |
| Argentina | Lionel Messi | Inter Miami CF | player_and_club | 12.000 | 2.360 | 1.975 |
| Belgium | Romelu Lukaku | Napoli | unmatched | NA | 1.956 | 1.956 |
| Argentina | Lautaro Martínez | Inter Milan | player_and_club | 17.000 | 1.635 | 1.702 |
| Netherlands | Memphis Depay | Corinthians | unmatched | NA | 1.688 | 1.688 |
| Iran | Mehdi Taremi | Olympiacos | unmatched | NA | 1.587 | 1.587 |
| Morocco | Ayoub El Kaabi | Olympiacos | unmatched | NA | 1.503 | 1.503 |
| Austria | Marko Arnautović | Red Star Belgrade | unmatched | NA | 1.461 | 1.461 |
| Spain | Mikel Oyarzabal | Real Sociedad | player_and_club | 15.000 | 1.519 | 1.457 |
| Egypt | Mohamed Salah | Liverpool | unmatched | NA | 1.447 | 1.447 |
| Belgium | Kevin De Bruyne | Napoli | unmatched | NA | 1.232 | 1.232 |
| Morocco | Brahim Díaz | Real Madrid | unmatched | NA | 1.215 | 1.215 |

Club form is sourced from curated public top-scorer tables and currently covers only matched top-100 candidates.

## Transfermarkt-Adjusted Top 100

| team | scorer | transfermarkt_match_quality | club_goals_source | club_goals_model | club_weighted_goals_model | club_minutes_model | club_weighted_goals_per90 | transfermarkt_multiplier | latest_market_value_in_eur | expected_tournament_goals | expected_tournament_goals_transfermarkt_adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Norway | Erling Haaland | player_country_club | appearances | 39.000 | 16.110 | 4235.000 | 0.342 | 1.449 | 200000000.000 | 3.552 | 5.147 |
| England | Harry Kane | player_country | appearances | 61.000 | 24.870 | 4140.000 | 0.541 | 1.450 | 65000000.000 | 3.222 | 4.672 |
| Argentina | Lionel Messi | player_country | game_events | 31.000 | 9.321 | 1725.000 | 0.486 | 1.401 | 35000000.000 | 2.360 | 3.306 |
| France | Kylian Mbappé | player_country_club | appearances | 43.000 | 16.140 | 3756.000 | 0.387 | 1.450 | 180000000.000 | 1.924 | 2.790 |
| Portugal | Cristiano Ronaldo | player_country_club | game_events | 28.000 | 9.773 | 2250.000 | 0.391 | 1.207 | 15000000.000 | 2.093 | 2.526 |
| Argentina | Lautaro Martínez | player_country | appearances | 22.000 | 10.350 | 2767.000 | 0.337 | 1.390 | 85000000.000 | 1.635 | 2.273 |
| Belgium | Romelu Lukaku | player_country_club | appearances | 1.000 | 0.666 | 64.000 | 0.936 | 0.987 | 15000000.000 | 1.956 | 1.931 |
| Spain | Mikel Oyarzabal | player_country_club | appearances | 18.000 | 9.342 | 3195.000 | 0.263 | 1.217 | 25000000.000 | 1.519 | 1.848 |
| Morocco | Ayoub El Kaabi | player_country | appearances | 20.000 | 6.711 | 2429.000 | 0.249 | 1.112 | 5000000.000 | 1.503 | 1.671 |
| Iran | Mehdi Taremi | player_country | appearances | 15.000 | 4.635 | 1833.000 | 0.228 | 1.039 | 2500000.000 | 1.587 | 1.649 |
| Norway | Alexander Sørloth | player_country | appearances | 20.000 | 10.678 | 2816.000 | 0.341 | 1.330 | 20000000.000 | 1.164 | 1.548 |
| Egypt | Mohamed Salah | player_country_club | appearances | 12.000 | 5.431 | 3150.000 | 0.155 | 1.050 | 30000000.000 | 1.447 | 1.520 |
| Croatia | Andrej Kramarić | player_country | appearances | 15.000 | 6.856 | 2422.000 | 0.255 | 1.098 | 3000000.000 | 1.243 | 1.364 |
| Spain | Ferran Torres | player_country_club | appearances | 21.000 | 9.287 | 2692.000 | 0.310 | 1.298 | 30000000.000 | 1.049 | 1.362 |
| Sweden | Viktor Gyökeres | player_country_club | appearances | 21.000 | 10.552 | 3427.000 | 0.277 | 1.289 | 70000000.000 | 1.026 | 1.323 |

### Transfermarkt Match Coverage

| match_quality | players |
| --- | --- |
| player_country_club | 65 |
| player_country | 31 |
| unmatched | 4 |

Transfermarkt features are preferred over sparse public top-scorer tables when the local Kaggle dump is available.
