# World Cup 2026 Final Forecast Report

This report summarizes the current model outputs generated from the processed pipeline artifacts. Group advancement means probability of reaching the round of 32, not exact group-position probabilities.

Top-scorer probabilities use 100,000 independent Poisson simulations over the Transfermarkt-adjusted top-100 scorer list, with tied top-scorer outcomes split evenly.

Market benchmark: not using model+market ensemble. Reason: LOWO model+market does not improve over both model-only and market-only log loss. Log loss model=0.971, market=0.961, model+market=0.962.
Availability: current nonzero burdens are Morocco (0.809, 2 players); Argentina (0.080, 1 player); Netherlands (0.040, 1 player).

## Winner probabilities
| team | group | winner_probability | final_probability | semi_final_probability | quarter_final_probability | round_of_16_probability | round_of_32_probability | expected_team_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Argentina | J | 18.5% | 29.3% | 42.9% | 56.9% | 68.9% | 92.0% | 6.036 |
| Spain | H | 18.2% | 28.5% | 40.1% | 53.1% | 68.5% | 94.9% | 5.967 |
| France | I | 7.7% | 13.4% | 23.9% | 36.2% | 52.6% | 69.8% | 5.064 |
| England | L | 6.8% | 13.3% | 24.7% | 41.8% | 63.4% | 94.7% | 5.493 |
| Belgium | G | 4.7% | 9.4% | 17.2% | 36.5% | 59.8% | 88.4% | 5.191 |
| Portugal | K | 4.6% | 9.7% | 17.6% | 32.6% | 53.6% | 81.7% | 5.031 |
| Norway | I | 4.5% | 7.0% | 15.0% | 30.2% | 55.1% | 84.0% | 4.993 |
| Germany | E | 3.9% | 8.7% | 18.1% | 32.4% | 51.3% | 77.2% | 4.971 |
| Morocco | C | 3.7% | 8.6% | 20.2% | 36.3% | 58.8% | 94.8% | 5.303 |
| Brazil | C | 3.7% | 8.8% | 17.4% | 30.4% | 43.4% | 66.2% | 4.748 |
| Japan | F | 3.6% | 8.6% | 19.0% | 33.4% | 52.9% | 94.1% | 5.184 |
| Colombia | K | 3.1% | 7.0% | 13.0% | 24.9% | 44.5% | 70.2% | 4.656 |
| Netherlands | F | 2.7% | 6.4% | 14.0% | 29.1% | 47.9% | 83.8% | 4.888 |
| Switzerland | B | 2.1% | 5.2% | 10.5% | 27.1% | 55.1% | 86.0% | 4.892 |
| Ecuador | E | 1.9% | 3.7% | 9.5% | 19.8% | 37.9% | 65.2% | 4.419 |

## Group advancement probabilities
| group | team | advance_probability | expected_team_matches |
| --- | --- | --- | --- |
| A | Mexico | 72.3% | 4.599 |
| A | Czech Republic | 67.9% | 4.149 |
| A | South Korea | 66.5% | 4.154 |
| A | South Africa | 62.4% | 3.840 |
| B | Switzerland | 86.0% | 4.892 |
| B | Canada | 83.9% | 4.540 |
| B | Bosnia and Herzegovina | 60.0% | 3.791 |
| B | Qatar | 38.2% | 3.436 |
| C | Morocco | 94.8% | 5.303 |
| C | Brazil | 66.2% | 4.748 |
| C | Haiti | 60.3% | 3.764 |
| C | Scotland | 43.3% | 3.681 |
| D | Australia | 80.7% | 4.431 |
| D | Turkey | 79.3% | 4.705 |
| D | United States | 56.2% | 3.958 |
| D | Paraguay | 54.6% | 3.957 |
| E | Ivory Coast | 82.8% | 4.402 |
| E | Germany | 77.2% | 4.971 |
| E | Ecuador | 65.2% | 4.419 |
| E | Curaçao | 45.0% | 3.532 |
| F | Japan | 94.1% | 5.184 |
| F | Netherlands | 83.8% | 4.888 |
| F | Tunisia | 56.9% | 3.744 |
| F | Sweden | 29.7% | 3.430 |
| G | Belgium | 88.4% | 5.191 |
| G | Iran | 79.8% | 4.498 |
| G | Egypt | 68.3% | 4.094 |
| G | New Zealand | 29.9% | 3.384 |
| H | Spain | 94.9% | 5.967 |
| H | Uruguay | 61.7% | 4.106 |
| H | Cape Verde | 55.5% | 3.685 |
| H | Saudi Arabia | 46.1% | 3.560 |
| I | Norway | 84.0% | 4.993 |
| I | Senegal | 79.0% | 4.425 |
| I | France | 69.8% | 5.064 |
| I | Iraq | 35.2% | 3.464 |
| J | Argentina | 92.0% | 6.036 |
| J | Algeria | 84.8% | 4.547 |
| J | Austria | 78.3% | 4.577 |
| J | Jordan | 16.7% | 3.237 |
| K | Portugal | 81.7% | 5.031 |
| K | Colombia | 70.2% | 4.656 |
| K | DR Congo | 59.4% | 3.854 |
| K | Uzbekistan | 56.3% | 3.864 |
| L | England | 94.7% | 5.493 |
| L | Croatia | 76.2% | 4.538 |
| L | Panama | 53.4% | 3.773 |
| L | Ghana | 36.4% | 3.445 |

## Round-by-round probabilities
| team | group | round_of_32_probability | round_of_16_probability | quarter_final_probability | semi_final_probability | final_probability | third_place_match_probability | winner_probability | expected_knockout_matches | expected_team_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Argentina | J | 92.0% | 68.9% | 56.9% | 42.9% | 29.3% | 13.6% | 18.5% | 3.036 | 6.036 |
| Spain | H | 94.9% | 68.5% | 53.1% | 40.1% | 28.5% | 11.6% | 18.2% | 2.967 | 5.967 |
| France | I | 69.8% | 52.6% | 36.2% | 23.9% | 13.4% | 10.5% | 7.7% | 2.064 | 5.064 |
| England | L | 94.7% | 63.4% | 41.8% | 24.7% | 13.3% | 11.4% | 6.8% | 2.493 | 5.493 |
| Belgium | G | 88.4% | 59.8% | 36.5% | 17.2% | 9.4% | 7.8% | 4.7% | 2.191 | 5.191 |
| Portugal | K | 81.7% | 53.6% | 32.6% | 17.6% | 9.7% | 7.9% | 4.6% | 2.031 | 5.031 |
| Norway | I | 84.0% | 55.1% | 30.2% | 15.0% | 7.0% | 8.0% | 4.5% | 1.993 | 4.993 |
| Germany | E | 77.2% | 51.3% | 32.4% | 18.1% | 8.7% | 9.4% | 3.9% | 1.971 | 4.971 |
| Brazil | C | 66.2% | 43.4% | 30.4% | 17.4% | 8.8% | 8.6% | 3.7% | 1.748 | 4.748 |
| Morocco | C | 94.8% | 58.8% | 36.3% | 20.2% | 8.6% | 11.6% | 3.7% | 2.303 | 5.303 |
| Japan | F | 94.1% | 52.9% | 33.4% | 19.0% | 8.6% | 10.4% | 3.6% | 2.184 | 5.184 |
| Colombia | K | 70.2% | 44.5% | 24.9% | 13.0% | 7.0% | 6.0% | 3.1% | 1.656 | 4.656 |
| Netherlands | F | 83.8% | 47.9% | 29.1% | 14.0% | 6.4% | 7.6% | 2.7% | 1.888 | 4.888 |
| Switzerland | B | 86.0% | 55.1% | 27.1% | 10.5% | 5.2% | 5.3% | 2.1% | 1.892 | 4.892 |
| Ecuador | E | 65.2% | 37.9% | 19.8% | 9.5% | 3.7% | 5.8% | 1.9% | 1.419 | 4.419 |

## Top-scorer probabilities
| rank | team | scorer | club | expected_tournament_goals | top_scorer_probability | transfermarkt_match_quality | status | availability_multiplier | transfermarkt_multiplier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Norway | Erling Haaland | Manchester City | 5.184 | 39.0% | player_country_club | available | 1.000 | 1.449 |
| 2 | England | Harry Kane | Bayern Munich | 4.689 | 29.0% | player_country | available | 1.000 | 1.450 |
| 3 | Argentina | Lionel Messi | Inter Miami CF | 3.284 | 9.3% | player_country | available | 1.000 | 1.401 |
| 4 | France | Kylian Mbappé | Real Madrid | 2.767 | 5.3% | player_country_club | available | 1.000 | 1.450 |
| 5 | Portugal | Cristiano Ronaldo | Al-Nassr | 2.528 | 3.8% | player_country_club | available | 1.000 | 1.207 |
| 6 | Argentina | Lautaro Martínez | Inter Milan | 2.258 | 2.5% | player_country | available | 1.000 | 1.390 |
| 7 | Belgium | Romelu Lukaku | Napoli | 1.913 | 1.5% | player_country_club | available | 1.000 | 0.987 |
| 8 | Spain | Mikel Oyarzabal | Real Sociedad | 1.864 | 1.3% | player_country_club | available | 1.000 | 1.217 |
| 9 | Iran | Mehdi Taremi | Olympiacos | 1.635 | 0.8% | player_country | available | 1.000 | 1.039 |
| 10 | Morocco | Ayoub El Kaabi | Olympiacos | 1.572 | 0.7% | player_country | available | 1.000 | 1.112 |
| 11 | Norway | Alexander Sørloth | Atlético Madrid | 1.559 | 0.7% | player_country | available | 1.000 | 1.330 |
| 12 | Egypt | Mohamed Salah | Liverpool | 1.519 | 0.6% | player_country_club | available | 1.000 | 1.050 |
| 13 | Croatia | Andrej Kramarić | TSG Hoffenheim | 1.352 | 0.4% | player_country | available | 1.000 | 1.098 |
| 14 | Spain | Ferran Torres | Barcelona | 1.373 | 0.4% | player_country_club | available | 1.000 | 1.298 |
| 15 | Sweden | Viktor Gyökeres | Arsenal | 1.322 | 0.3% | player_country_club | available | 1.000 | 1.289 |
| 16 | Turkey | Kerem Aktürkoğlu | Fenerbahçe | 1.241 | 0.3% | player_country_club | available | 1.000 | 1.149 |
| 17 | Colombia | Luis Díaz | Bayern Munich | 1.244 | 0.3% | player_country | available | 1.000 | 1.209 |
| 18 | Japan | Ayase Ueda | Feyenoord | 1.249 | 0.3% | player_country_club | available | 1.000 | 1.122 |
| 19 | Haiti | Duckens Nazon | Esteghlal | 1.202 | 0.2% | player_country | available | 1.000 | 1.000 |
| 20 | Austria | Marko Arnautović | Red Star Belgrade | 1.178 | 0.2% | player_country | available | 1.000 | 0.812 |

## Most decisive group matches
| match_number | date | time_local | group | home_team | away_team | ensemble_home_win | ensemble_draw | ensemble_away_win | form_poisson_home_expected_goals | form_poisson_away_expected_goals | most_likely_outcome | most_likely_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14 | 2026-06-15 | 12:00 p.m. | H | Spain | Cape Verde | 74.3% | 21.6% | 4.1% | 2.100 | 0.813 | home_win | 74.3% |
| 38 | 2026-06-21 | 12:00 p.m. | H | Spain | Saudi Arabia | 74.1% | 21.7% | 4.3% | 2.014 | 0.681 | home_win | 74.1% |
| 70 | 2026-06-27 | 9:00 p.m. | J | Jordan | Argentina | 5.4% | 21.1% | 73.5% | 0.560 | 2.301 | away_win | 73.5% |
| 45 | 2026-06-23 | 4:00 p.m. | L | England | Ghana | 72.9% | 21.7% | 5.4% | 1.946 | 0.564 | home_win | 72.9% |
| 8 | 2026-06-13 | 12:00 p.m. | B | Qatar | Switzerland | 5.8% | 21.7% | 72.5% | 0.914 | 2.079 | away_win | 72.5% |
| 10 | 2026-06-14 | 12:00 p.m. | E | Germany | Curaçao | 71.8% | 21.7% | 6.5% | 2.301 | 1.414 | home_win | 71.8% |
| 64 | 2026-06-26 | 8:00 p.m. | G | New Zealand | Belgium | 8.6% | 21.2% | 70.3% | 0.951 | 2.528 | away_win | 70.3% |
| 42 | 2026-06-22 | 5:00 p.m. | I | France | Iraq | 69.7% | 22.6% | 7.7% | 1.513 | 0.837 | home_win | 69.7% |
| 27 | 2026-06-18 | 3:00 p.m. | B | Canada | Qatar | 69.5% | 22.7% | 7.8% | 1.468 | 0.661 | home_win | 69.5% |
| 50 | 2026-06-24 | 6:00 p.m. | C | Morocco | Haiti | 68.7% | 22.0% | 9.3% | 1.777 | 0.598 | home_win | 68.7% |
| 34 | 2026-06-20 | 7:00 p.m. | E | Ecuador | Curaçao | 68.6% | 23.3% | 8.1% | 1.103 | 0.857 | home_win | 68.6% |
| 68 | 2026-06-27 | 5:00 p.m. | L | Croatia | Ghana | 67.9% | 22.1% | 10.0% | 1.805 | 0.985 | home_win | 67.9% |
| 29 | 2026-06-19 | 8:30 p.m. | C | Brazil | Haiti | 67.7% | 22.2% | 10.0% | 1.677 | 1.501 | home_win | 67.7% |
| 18 | 2026-06-16 | 6:00 p.m. | I | Iraq | Norway | 10.9% | 22.1% | 67.0% | 0.786 | 1.777 | away_win | 67.0% |
| 67 | 2026-06-27 | 5:00 p.m. | L | Panama | England | 11.7% | 21.7% | 66.6% | 0.743 | 2.004 | away_win | 66.6% |

## Most likely knockout pairings
| match_number | round | first_team | second_team | pairing_probability | first_advancement_probability | second_advancement_probability | simulation_count | most_likely_advancing_team | most_likely_advancement_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 73 | round_of_32 | Czech Republic | Canada | 8.7% | 41.1% | 58.9% | 87 | Canada | 58.9% |
| 74 | round_of_32 | Ivory Coast | Paraguay | 5.7% | 50.5% | 49.5% | 57 | Ivory Coast | 50.5% |
| 75 | round_of_32 | Japan | Brazil | 17.9% | 44.8% | 55.2% | 179 | Brazil | 55.2% |
| 76 | round_of_32 | Morocco | Netherlands | 25.5% | 51.4% | 48.6% | 255 | Morocco | 51.4% |
| 77 | round_of_32 | Senegal | Tunisia | 7.0% | 72.7% | 27.3% | 70 | Senegal | 72.7% |
| 78 | round_of_32 | Ivory Coast | Norway | 10.1% | 31.6% | 68.4% | 101 | Norway | 68.4% |
| 79 | round_of_32 | Mexico | Ecuador | 3.2% | 45.7% | 54.3% | 32 | Ecuador | 54.3% |
| 80 | round_of_32 | England | DR Congo | 11.9% | 85.3% | 14.7% | 119 | England | 85.3% |
| 81 | round_of_32 | Australia | Bosnia and Herzegovina | 8.7% | 76.9% | 23.1% | 87 | Australia | 76.9% |
| 82 | round_of_32 | Belgium | Czech Republic | 9.1% | 73.3% | 26.7% | 91 | Belgium | 73.3% |
| 83 | round_of_32 | Colombia | Croatia | 11.5% | 63.1% | 36.9% | 115 | Colombia | 63.1% |
| 84 | round_of_32 | Spain | Algeria | 22.3% | 80.6% | 19.4% | 223 | Spain | 80.6% |
| 85 | round_of_32 | Switzerland | Austria | 7.0% | 52.9% | 47.1% | 70 | Switzerland | 52.9% |
| 86 | round_of_32 | Argentina | Cape Verde | 15.1% | 95.3% | 4.7% | 151 | Argentina | 95.3% |
| 87 | round_of_32 | Portugal | Panama | 8.5% | 83.7% | 16.3% | 85 | Portugal | 83.7% |
| 88 | round_of_32 | Australia | Iran | 10.0% | 45.4% | 54.6% | 100 | Iran | 54.6% |
| 89 | round_of_16 | Mexico | Japan | 7.3% | 41.3% | 58.7% | 73 | Japan | 58.7% |
| 90 | round_of_16 | Germany | Senegal | 4.9% | 68.3% | 31.7% | 49 | Germany | 68.3% |
| 91 | round_of_16 | Morocco | Norway | 8.7% | 52.8% | 47.2% | 87 | Morocco | 52.8% |
| 92 | round_of_16 | Mexico | England | 8.2% | 35.8% | 64.2% | 82 | England | 64.2% |
| 93 | round_of_16 | Croatia | Spain | 9.6% | 20.5% | 79.5% | 96 | Spain | 79.5% |
| 94 | round_of_16 | Australia | Belgium | 7.3% | 30.2% | 69.8% | 73 | Belgium | 69.8% |
| 95 | round_of_16 | Argentina | Iran | 8.7% | 82.8% | 17.2% | 87 | Argentina | 82.8% |
| 96 | round_of_16 | Switzerland | Portugal | 7.9% | 39.7% | 60.3% | 79 | Portugal | 60.3% |
| 97 | quarter_final | Japan | Norway | 4.5% | 52.5% | 47.5% | 45 | Japan | 52.5% |
| 98 | quarter_final | Spain | Belgium | 10.3% | 71.8% | 28.2% | 103 | Spain | 71.8% |
| 99 | quarter_final | Morocco | England | 6.0% | 44.3% | 55.7% | 60 | England | 55.7% |
| 100 | quarter_final | Argentina | Portugal | 7.0% | 68.8% | 31.2% | 70 | Argentina | 68.8% |
| 101 | semi_final | Japan | Spain | 5.0% | 29.2% | 70.8% | 50 | Spain | 70.8% |
| 102 | semi_final | England | Argentina | 5.0% | 33.8% | 66.2% | 50 | Argentina | 66.2% |
| 103 | final | Spain | Argentina | 5.4% | 50.4% | 49.6% | 54 | Spain | 50.4% |
| 104 | third_place | Japan | Morocco | 1.0% | 49.8% | 50.2% | 10 | Morocco | 50.2% |

## Generated files
- `reports/final_group_advancement_probabilities.csv`
- `reports/final_round_by_round_probabilities.csv`
- `reports/final_winner_probabilities.csv`
- `reports/final_top_scorer_probabilities.csv`
- `reports/final_group_match_forecasts.csv`
- `reports/final_knockout_match_forecasts.csv`
- `reports/final_full_match_forecasts.csv`
