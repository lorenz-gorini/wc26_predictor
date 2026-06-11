# World Cup 2026 Final Forecast Report

This report summarizes the current model outputs generated from the processed pipeline artifacts. Group advancement means probability of reaching the round of 32, not exact group-position probabilities.

Top-scorer probabilities use 100,000 independent Poisson simulations over the Transfermarkt-adjusted top-100 scorer list, with tied top-scorer outcomes split evenly.

Market benchmark: not using model+market ensemble. Reason: LOWO model+market does not improve over both model-only and market-only log loss. Log loss model=0.971, market=0.961, model+market=0.962.
Availability: current nonzero burdens are Morocco (0.809, 2 players); Argentina (0.080, 1 player); Netherlands (0.040, 1 player).

## Winner probabilities
| team | group | winner_probability | final_probability | semi_final_probability | quarter_final_probability | round_of_16_probability | round_of_32_probability | expected_team_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spain | H | 19.5% | 29.7% | 40.2% | 52.3% | 68.0% | 94.7% | 5.954 |
| Argentina | J | 19.1% | 29.4% | 42.8% | 57.1% | 69.8% | 92.6% | 6.052 |
| France | I | 7.4% | 12.8% | 22.4% | 35.2% | 51.9% | 70.5% | 5.025 |
| England | L | 6.3% | 12.3% | 23.2% | 40.5% | 63.7% | 95.2% | 5.457 |
| Portugal | K | 4.8% | 9.9% | 17.5% | 31.8% | 54.3% | 81.8% | 5.030 |
| Belgium | G | 4.1% | 8.8% | 15.8% | 34.1% | 59.4% | 88.4% | 5.136 |
| Germany | E | 4.0% | 8.6% | 16.8% | 30.2% | 51.6% | 77.9% | 4.934 |
| Brazil | C | 4.0% | 8.6% | 17.3% | 29.8% | 42.4% | 66.6% | 4.734 |
| Morocco | C | 3.8% | 8.3% | 19.1% | 36.4% | 58.2% | 94.5% | 5.272 |
| Japan | F | 3.8% | 8.5% | 19.7% | 35.6% | 54.2% | 94.1% | 5.234 |
| Norway | I | 3.5% | 6.5% | 15.1% | 31.3% | 55.2% | 84.0% | 5.007 |
| Netherlands | F | 3.5% | 7.2% | 15.7% | 29.2% | 48.6% | 84.4% | 4.936 |
| Colombia | K | 3.2% | 7.2% | 13.5% | 24.8% | 45.1% | 69.1% | 4.660 |
| Switzerland | B | 1.9% | 5.1% | 11.7% | 28.1% | 55.8% | 86.6% | 4.938 |
| Ecuador | E | 1.8% | 4.2% | 9.7% | 19.5% | 38.3% | 65.8% | 4.428 |

## Group advancement probabilities
| group | team | advance_probability | expected_team_matches |
| --- | --- | --- | --- |
| A | Mexico | 70.8% | 4.530 |
| A | Czech Republic | 67.8% | 4.152 |
| A | South Korea | 66.9% | 4.157 |
| A | South Africa | 63.4% | 3.865 |
| B | Switzerland | 86.6% | 4.938 |
| B | Canada | 84.0% | 4.590 |
| B | Bosnia and Herzegovina | 58.2% | 3.752 |
| B | Qatar | 38.0% | 3.432 |
| C | Morocco | 94.5% | 5.272 |
| C | Brazil | 66.6% | 4.734 |
| C | Haiti | 60.2% | 3.772 |
| C | Scotland | 44.9% | 3.716 |
| D | Australia | 80.2% | 4.453 |
| D | Turkey | 78.2% | 4.667 |
| D | United States | 56.8% | 3.960 |
| D | Paraguay | 53.9% | 3.970 |
| E | Ivory Coast | 81.3% | 4.419 |
| E | Germany | 77.9% | 4.934 |
| E | Ecuador | 65.8% | 4.428 |
| E | Curaçao | 46.2% | 3.550 |
| F | Japan | 94.1% | 5.234 |
| F | Netherlands | 84.4% | 4.936 |
| F | Tunisia | 58.0% | 3.759 |
| F | Sweden | 28.7% | 3.408 |
| G | Belgium | 88.4% | 5.136 |
| G | Iran | 81.3% | 4.527 |
| G | Egypt | 67.7% | 4.092 |
| G | New Zealand | 29.6% | 3.382 |
| H | Spain | 94.7% | 5.954 |
| H | Uruguay | 62.3% | 4.117 |
| H | Cape Verde | 54.4% | 3.652 |
| H | Saudi Arabia | 46.9% | 3.560 |
| I | Norway | 84.0% | 5.007 |
| I | Senegal | 79.6% | 4.486 |
| I | France | 70.5% | 5.025 |
| I | Iraq | 34.5% | 3.457 |
| J | Argentina | 92.6% | 6.052 |
| J | Algeria | 83.8% | 4.551 |
| J | Austria | 77.5% | 4.548 |
| J | Jordan | 17.3% | 3.245 |
| K | Portugal | 81.8% | 5.030 |
| K | Colombia | 69.1% | 4.660 |
| K | DR Congo | 60.6% | 3.870 |
| K | Uzbekistan | 54.5% | 3.833 |
| L | England | 95.2% | 5.457 |
| L | Croatia | 76.8% | 4.529 |
| L | Panama | 52.5% | 3.759 |
| L | Ghana | 36.9% | 3.443 |

## Round-by-round probabilities
| team | group | round_of_32_probability | round_of_16_probability | quarter_final_probability | semi_final_probability | final_probability | third_place_match_probability | winner_probability | expected_knockout_matches | expected_team_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spain | H | 94.7% | 68.0% | 52.3% | 40.2% | 29.7% | 10.5% | 19.5% | 2.954 | 5.954 |
| Argentina | J | 92.6% | 69.8% | 57.1% | 42.8% | 29.4% | 13.4% | 19.1% | 3.052 | 6.052 |
| France | I | 70.5% | 51.9% | 35.2% | 22.4% | 12.8% | 9.6% | 7.4% | 2.025 | 5.025 |
| England | L | 95.2% | 63.7% | 40.5% | 23.2% | 12.3% | 10.8% | 6.3% | 2.458 | 5.457 |
| Portugal | K | 81.8% | 54.3% | 31.8% | 17.5% | 9.9% | 7.6% | 4.8% | 2.030 | 5.030 |
| Belgium | G | 88.4% | 59.4% | 34.1% | 15.8% | 8.8% | 7.0% | 4.1% | 2.136 | 5.136 |
| Germany | E | 77.9% | 51.6% | 30.2% | 16.8% | 8.6% | 8.2% | 4.0% | 1.933 | 4.934 |
| Brazil | C | 66.6% | 42.4% | 29.8% | 17.3% | 8.6% | 8.6% | 4.0% | 1.734 | 4.734 |
| Morocco | C | 94.5% | 58.2% | 36.4% | 19.1% | 8.3% | 10.8% | 3.8% | 2.272 | 5.272 |
| Japan | F | 94.1% | 54.2% | 35.6% | 19.7% | 8.5% | 11.2% | 3.8% | 2.234 | 5.234 |
| Norway | I | 84.0% | 55.2% | 31.3% | 15.1% | 6.5% | 8.6% | 3.5% | 2.007 | 5.007 |
| Netherlands | F | 84.4% | 48.6% | 29.2% | 15.7% | 7.2% | 8.5% | 3.5% | 1.935 | 4.936 |
| Colombia | K | 69.1% | 45.1% | 24.8% | 13.5% | 7.2% | 6.2% | 3.2% | 1.660 | 4.660 |
| Switzerland | B | 86.6% | 55.8% | 28.1% | 11.7% | 5.1% | 6.6% | 1.9% | 1.938 | 4.938 |
| Ecuador | E | 65.8% | 38.3% | 19.5% | 9.7% | 4.2% | 5.5% | 1.8% | 1.429 | 4.428 |

## Top-scorer probabilities
| rank | team | scorer | club | expected_tournament_goals | top_scorer_probability | transfermarkt_match_quality | status | availability_multiplier | transfermarkt_multiplier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Norway | Erling Haaland | Manchester City | 5.198 | 39.3% | player_country_club | available | 1.000 | 1.449 |
| 2 | England | Harry Kane | Bayern Munich | 4.658 | 28.6% | player_country | available | 1.000 | 1.450 |
| 3 | Argentina | Lionel Messi | Inter Miami CF | 3.293 | 9.4% | player_country | available | 1.000 | 1.401 |
| 4 | France | Kylian Mbappé | Real Madrid | 2.746 | 5.1% | player_country_club | available | 1.000 | 1.450 |
| 5 | Portugal | Cristiano Ronaldo | Al-Nassr | 2.527 | 3.8% | player_country_club | available | 1.000 | 1.207 |
| 6 | Argentina | Lautaro Martínez | Inter Milan | 2.264 | 2.6% | player_country | available | 1.000 | 1.390 |
| 7 | Belgium | Romelu Lukaku | Napoli | 1.893 | 1.3% | player_country_club | available | 1.000 | 0.987 |
| 8 | Spain | Mikel Oyarzabal | Real Sociedad | 1.860 | 1.3% | player_country_club | available | 1.000 | 1.217 |
| 9 | Iran | Mehdi Taremi | Olympiacos | 1.646 | 0.8% | player_country | available | 1.000 | 1.039 |
| 10 | Norway | Alexander Sørloth | Atlético Madrid | 1.563 | 0.7% | player_country | available | 1.000 | 1.330 |
| 11 | Morocco | Ayoub El Kaabi | Olympiacos | 1.563 | 0.7% | player_country | available | 1.000 | 1.112 |
| 12 | Egypt | Mohamed Salah | Liverpool | 1.518 | 0.6% | player_country_club | available | 1.000 | 1.050 |
| 13 | Spain | Ferran Torres | Barcelona | 1.370 | 0.4% | player_country_club | available | 1.000 | 1.298 |
| 14 | Croatia | Andrej Kramarić | TSG Hoffenheim | 1.349 | 0.4% | player_country | available | 1.000 | 1.098 |
| 15 | Sweden | Viktor Gyökeres | Arsenal | 1.313 | 0.3% | player_country_club | available | 1.000 | 1.289 |
| 16 | Japan | Ayase Ueda | Feyenoord | 1.261 | 0.3% | player_country_club | available | 1.000 | 1.122 |
| 17 | Turkey | Kerem Aktürkoğlu | Fenerbahçe | 1.231 | 0.3% | player_country_club | available | 1.000 | 1.149 |
| 18 | Switzerland | Breel Embolo | Rennes | 1.192 | 0.3% | player_country | available | 1.000 | 1.038 |
| 19 | Colombia | Luis Díaz | Bayern Munich | 1.245 | 0.3% | player_country | available | 1.000 | 1.209 |
| 20 | Netherlands | Cody Gakpo | Liverpool | 1.199 | 0.2% | player_country_club | available | 1.000 | 0.998 |

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
| 73 | round_of_32 | Mexico | Canada | 9.6% | 57.9% | 42.1% | 191 | Mexico | 57.9% |
| 74 | round_of_32 | Ivory Coast | Brazil | 5.1% | 26.4% | 73.6% | 103 | Brazil | 73.6% |
| 75 | round_of_32 | Japan | Brazil | 17.6% | 44.8% | 55.2% | 352 | Brazil | 55.2% |
| 76 | round_of_32 | Morocco | Netherlands | 24.6% | 51.4% | 48.6% | 491 | Morocco | 51.4% |
| 77 | round_of_32 | Norway | Tunisia | 7.5% | 83.4% | 16.6% | 150 | Norway | 83.4% |
| 78 | round_of_32 | Ivory Coast | Norway | 9.3% | 31.6% | 68.4% | 186 | Norway | 68.4% |
| 79 | round_of_32 | Czech Republic | Ecuador | 3.6% | 30.3% | 69.7% | 73 | Ecuador | 69.7% |
| 80 | round_of_32 | England | DR Congo | 11.1% | 85.3% | 14.7% | 222 | England | 85.3% |
| 81 | round_of_32 | Australia | Bosnia and Herzegovina | 8.1% | 76.9% | 23.1% | 162 | Australia | 76.9% |
| 82 | round_of_32 | Belgium | South Korea | 8.9% | 72.3% | 27.7% | 178 | Belgium | 72.3% |
| 83 | round_of_32 | Colombia | Croatia | 10.6% | 63.1% | 36.9% | 212 | Colombia | 63.1% |
| 84 | round_of_32 | Spain | Algeria | 22.5% | 80.6% | 19.4% | 450 | Spain | 80.6% |
| 85 | round_of_32 | Switzerland | Egypt | 7.6% | 71.9% | 28.1% | 152 | Switzerland | 71.9% |
| 86 | round_of_32 | Argentina | Cape Verde | 14.3% | 95.3% | 4.7% | 287 | Argentina | 95.3% |
| 87 | round_of_32 | Portugal | Panama | 8.4% | 83.7% | 16.3% | 168 | Portugal | 83.7% |
| 88 | round_of_32 | Australia | Iran | 9.7% | 45.4% | 54.6% | 193 | Iran | 54.6% |
| 89 | round_of_16 | Switzerland | Japan | 7.1% | 42.8% | 57.2% | 142 | Japan | 57.2% |
| 90 | round_of_16 | Germany | Norway | 5.1% | 53.6% | 46.4% | 102 | Germany | 53.6% |
| 91 | round_of_16 | Morocco | France | 8.3% | 39.5% | 60.5% | 167 | France | 60.5% |
| 92 | round_of_16 | Mexico | England | 8.1% | 35.8% | 64.2% | 162 | England | 64.2% |
| 93 | round_of_16 | Croatia | Spain | 9.2% | 20.5% | 79.5% | 185 | Spain | 79.5% |
| 94 | round_of_16 | Australia | Belgium | 7.5% | 30.2% | 69.8% | 151 | Belgium | 69.8% |
| 95 | round_of_16 | Argentina | Iran | 7.7% | 82.8% | 17.2% | 154 | Argentina | 82.8% |
| 96 | round_of_16 | Switzerland | Portugal | 7.9% | 39.7% | 60.3% | 158 | Portugal | 60.3% |
| 97 | quarter_final | Japan | Norway | 4.2% | 52.5% | 47.5% | 85 | Japan | 52.5% |
| 98 | quarter_final | Spain | Belgium | 10.0% | 71.8% | 28.2% | 200 | Spain | 71.8% |
| 99 | quarter_final | Morocco | England | 6.4% | 44.3% | 55.7% | 128 | England | 55.7% |
| 100 | quarter_final | Argentina | Portugal | 6.3% | 68.8% | 31.2% | 126 | Argentina | 68.8% |
| 101 | semi_final | Japan | Spain | 4.9% | 29.2% | 70.8% | 98 | Spain | 70.8% |
| 102 | semi_final | England | Argentina | 4.8% | 33.8% | 66.2% | 95 | Argentina | 66.2% |
| 103 | final | Spain | Argentina | 5.3% | 50.4% | 49.6% | 106 | Spain | 50.4% |
| 104 | third_place | Spain | Argentina | 0.9% | 50.4% | 49.6% | 19 | Spain | 50.4% |

## Generated files
- `reports/final_group_advancement_probabilities.csv`
- `reports/final_round_by_round_probabilities.csv`
- `reports/final_winner_probabilities.csv`
- `reports/final_top_scorer_probabilities.csv`
- `reports/final_group_match_forecasts.csv`
- `reports/final_knockout_match_forecasts.csv`
- `reports/final_full_match_forecasts.csv`

## Related validation review
- `reports/model_validation_review.md`
