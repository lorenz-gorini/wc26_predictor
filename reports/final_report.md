# World Cup 2026 Final Forecast Report

This report summarizes the current model outputs generated from the processed pipeline artifacts. Group advancement means probability of reaching the round of 32, not exact group-position probabilities.

Group match score predictions use the form-adjusted Poisson exact-score distribution. Aggregate 1X2 probabilities remain shown as secondary outcome diagnostics.

Top-scorer probabilities use 100,000 independent Poisson simulations over the Transfermarkt-adjusted top-100 scorer list, with tied top-scorer outcomes split evenly.

Market benchmark: not using model+market ensemble. Reason: LOWO model+market does not improve over both model-only and market-only log loss. Log loss model=0.971, market=0.961, model+market=0.962.
Availability: current nonzero burdens are Morocco (0.809, 2 players); Argentina (0.080, 1 player); Netherlands (0.040, 1 player).

## Winner probabilities
| team | group | winner_probability | final_probability | semi_final_probability | quarter_final_probability | round_of_16_probability | round_of_32_probability | expected_team_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Argentina | J | 19.7% | 28.6% | 40.3% | 53.5% | 68.2% | 91.8% | 5.941 |
| Spain | H | 19.6% | 28.7% | 39.1% | 50.6% | 66.6% | 94.5% | 5.899 |
| England | L | 5.9% | 13.2% | 25.1% | 41.9% | 65.7% | 96.5% | 5.542 |
| France | I | 5.9% | 11.3% | 20.3% | 33.2% | 49.1% | 71.8% | 4.947 |
| Germany | E | 5.1% | 10.7% | 22.4% | 39.4% | 66.5% | 99.6% | 5.503 |
| Brazil | C | 4.5% | 8.2% | 17.3% | 29.3% | 45.1% | 73.8% | 4.830 |
| Portugal | K | 4.3% | 11.0% | 18.6% | 33.4% | 56.2% | 80.8% | 5.076 |
| Morocco | C | 4.0% | 8.9% | 18.1% | 32.7% | 53.3% | 90.5% | 5.128 |
| Japan | F | 3.8% | 7.4% | 16.4% | 31.1% | 50.2% | 92.0% | 5.060 |
| Belgium | G | 3.7% | 7.6% | 15.8% | 34.5% | 60.8% | 89.5% | 5.163 |
| Colombia | K | 3.6% | 7.6% | 13.6% | 26.1% | 46.5% | 69.3% | 4.691 |
| Netherlands | F | 3.1% | 6.2% | 14.2% | 29.5% | 49.7% | 88.7% | 4.965 |
| Norway | I | 3.0% | 6.3% | 14.0% | 27.4% | 50.3% | 82.0% | 4.877 |
| Mexico | A | 2.1% | 5.2% | 13.9% | 29.3% | 60.9% | 96.4% | 5.144 |
| Algeria | J | 1.9% | 4.7% | 10.7% | 24.2% | 45.5% | 86.7% | 4.777 |

## Group advancement probabilities
| group | team | advance_probability | expected_team_matches |
| --- | --- | --- | --- |
| A | Mexico | 96.4% | 5.144 |
| A | South Korea | 93.2% | 4.759 |
| A | Czech Republic | 49.1% | 3.825 |
| A | South Africa | 32.6% | 3.428 |
| B | Canada | 76.7% | 4.429 |
| B | Switzerland | 76.2% | 4.609 |
| B | Bosnia and Herzegovina | 61.2% | 3.776 |
| B | Qatar | 50.2% | 3.579 |
| C | Morocco | 90.5% | 5.128 |
| C | Scotland | 78.8% | 4.259 |
| C | Brazil | 73.8% | 4.830 |
| C | Haiti | 27.5% | 3.351 |
| D | Australia | 98.4% | 4.929 |
| D | United States | 94.8% | 4.735 |
| D | Turkey | 58.2% | 4.088 |
| D | Paraguay | 18.1% | 3.271 |
| E | Germany | 99.6% | 5.503 |
| E | Ivory Coast | 98.2% | 4.752 |
| E | Ecuador | 50.3% | 4.034 |
| E | Curaçao | 11.1% | 3.131 |
| F | Japan | 92.0% | 5.060 |
| F | Netherlands | 88.7% | 4.965 |
| F | Sweden | 87.4% | 4.277 |
| F | Tunisia | 11.4% | 3.143 |
| G | Belgium | 89.5% | 5.163 |
| G | Iran | 80.8% | 4.535 |
| G | Egypt | 64.5% | 4.034 |
| G | New Zealand | 29.2% | 3.378 |
| H | Spain | 94.5% | 5.899 |
| H | Uruguay | 61.7% | 4.053 |
| H | Cape Verde | 54.0% | 3.635 |
| H | Saudi Arabia | 47.8% | 3.552 |
| I | Norway | 82.0% | 4.877 |
| I | Senegal | 80.4% | 4.441 |
| I | France | 71.8% | 4.947 |
| I | Iraq | 33.1% | 3.422 |
| J | Argentina | 91.8% | 5.941 |
| J | Algeria | 86.7% | 4.777 |
| J | Austria | 76.8% | 4.530 |
| J | Jordan | 15.3% | 3.208 |
| K | Portugal | 80.8% | 5.076 |
| K | Colombia | 69.3% | 4.691 |
| K | DR Congo | 61.4% | 3.874 |
| K | Uzbekistan | 52.6% | 3.795 |
| L | England | 96.5% | 5.542 |
| L | Croatia | 77.8% | 4.491 |
| L | Panama | 52.6% | 3.723 |
| L | Ghana | 34.8% | 3.412 |

## Round-by-round probabilities
| team | group | round_of_32_probability | round_of_16_probability | quarter_final_probability | semi_final_probability | final_probability | third_place_match_probability | winner_probability | expected_knockout_matches | expected_team_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Argentina | J | 91.8% | 68.2% | 53.5% | 40.3% | 28.6% | 11.7% | 19.7% | 2.941 | 5.941 |
| Spain | H | 94.5% | 66.6% | 50.6% | 39.1% | 28.7% | 10.4% | 19.6% | 2.899 | 5.899 |
| England | L | 96.5% | 65.7% | 41.9% | 25.1% | 13.2% | 11.9% | 5.9% | 2.542 | 5.542 |
| France | I | 71.8% | 49.1% | 33.2% | 20.3% | 11.3% | 9.0% | 5.9% | 1.948 | 4.947 |
| Germany | E | 99.6% | 66.5% | 39.4% | 22.4% | 10.7% | 11.8% | 5.1% | 2.503 | 5.503 |
| Brazil | C | 73.8% | 45.1% | 29.3% | 17.3% | 8.2% | 9.1% | 4.5% | 1.830 | 4.830 |
| Portugal | K | 80.8% | 56.2% | 33.4% | 18.6% | 11.0% | 7.6% | 4.3% | 2.076 | 5.076 |
| Morocco | C | 90.5% | 53.3% | 32.7% | 18.1% | 8.9% | 9.2% | 4.0% | 2.128 | 5.128 |
| Japan | F | 92.0% | 50.2% | 31.1% | 16.4% | 7.4% | 8.9% | 3.8% | 2.060 | 5.060 |
| Belgium | G | 89.5% | 60.8% | 34.5% | 15.8% | 7.6% | 8.2% | 3.7% | 2.163 | 5.163 |
| Colombia | K | 69.3% | 46.5% | 26.1% | 13.6% | 7.6% | 5.9% | 3.6% | 1.691 | 4.691 |
| Netherlands | F | 88.7% | 49.7% | 29.5% | 14.2% | 6.2% | 8.1% | 3.1% | 1.964 | 4.965 |
| Norway | I | 82.0% | 50.3% | 27.4% | 14.0% | 6.3% | 7.7% | 3.0% | 1.877 | 4.877 |
| Mexico | A | 96.4% | 60.9% | 29.3% | 13.9% | 5.2% | 8.7% | 2.1% | 2.144 | 5.144 |
| Algeria | J | 86.7% | 45.5% | 24.2% | 10.7% | 4.7% | 5.9% | 1.9% | 1.776 | 4.777 |

## Top-scorer probabilities
| rank | team | scorer | club | expected_tournament_goals | top_scorer_probability | transfermarkt_match_quality | status | availability_multiplier | transfermarkt_multiplier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Norway | Erling Haaland | Manchester City | 4.976 | 35.4% | player_country_club | available | 1.000 | 1.431 |
| 2 | England | Harry Kane | Bayern Munich | 4.886 | 33.4% | player_country | available | 1.000 | 1.450 |
| 3 | Argentina | Lionel Messi | Inter Miami CF | 3.142 | 8.3% | player_country | available | 1.000 | 1.382 |
| 4 | France | Kylian Mbappé | Real Madrid | 2.691 | 4.9% | player_country_club | available | 1.000 | 1.450 |
| 5 | Portugal | Cristiano Ronaldo | Al-Nassr | 2.479 | 3.7% | player_country_club | available | 1.000 | 1.184 |
| 6 | Argentina | Lautaro Martínez | Inter Milan | 2.158 | 2.2% | player_country | available | 1.000 | 1.370 |
| 7 | Sweden | Viktor Gyökeres | Arsenal | 1.957 | 1.6% | player_country_club | available | 1.000 | 1.268 |
| 8 | Belgium | Romelu Lukaku | Napoli | 1.885 | 1.4% | player_country_club | available | 1.000 | 0.982 |
| 9 | Spain | Mikel Oyarzabal | Real Sociedad | 1.799 | 1.2% | player_country_club | available | 1.000 | 1.193 |
| 10 | Iran | Mehdi Taremi | Olympiacos | 1.592 | 0.7% | player_country | available | 1.000 | 1.010 |
| 11 | Norway | Alexander Sørloth | Atlético Madrid | 1.490 | 0.6% | player_country | available | 1.000 | 1.308 |
| 12 | Egypt | Mohamed Salah | Liverpool | 1.448 | 0.5% | player_country_club | available | 1.000 | 1.024 |
| 13 | Germany | Kai Havertz | Arsenal | 1.449 | 0.5% | player_country_club | available | 1.000 | 1.363 |
| 14 | Morocco | Ayoub El Kaabi | Olympiacos | 1.370 | 0.4% | player_country | available | 1.000 | 1.085 |
| 15 | Spain | Ferran Torres | Barcelona | 1.329 | 0.4% | player_country_club | available | 1.000 | 1.276 |
| 16 | Croatia | Andrej Kramarić | TSG Hoffenheim | 1.286 | 0.3% | player_country | available | 1.000 | 1.070 |
| 17 | Netherlands | Memphis Depay | Corinthians | 1.234 | 0.3% | player_country_club | available | 1.000 | 0.700 |
| 18 | South Korea | Son Heung-min | Los Angeles FC | 1.209 | 0.3% | unmatched | available | 1.000 | 1.000 |
| 19 | Colombia | Luis Díaz | Bayern Munich | 1.217 | 0.3% | player_country | available | 1.000 | 1.186 |
| 20 | Netherlands | Cody Gakpo | Liverpool | 1.210 | 0.3% | player_country_club | available | 1.000 | 0.971 |

## Most decisive group matches
| match_number | date | time_local | group | home_team | away_team | predicted_score | predicted_score_outcome | predicted_score_probability | top_scorelines | ensemble_home_win | ensemble_draw | ensemble_away_win | form_poisson_home_expected_goals | form_poisson_away_expected_goals | most_likely_outcome | most_likely_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 2026-06-14 | 12:00 p.m. | E | Germany | Curaçao | 3-1 | home_win | 7.7% | 3-1 (0.077); 2-1 (0.071); 4-1 (0.063); 3-0 (0.057); 3-2 (0.053) | 74.6% | 20.7% | 4.7% | 3.281 | 1.364 | home_win | 74.6% |
| 14 | 2026-06-15 | 12:00 p.m. | H | Spain | Cape Verde | 2-0 | home_win | 12.0% | 2-0 (0.120); 1-0 (0.114); 2-1 (0.097); 1-1 (0.093); 3-0 (0.084) | 74.3% | 21.6% | 4.1% | 2.097 | 0.813 | home_win | 74.3% |
| 38 | 2026-06-21 | 12:00 p.m. | H | Spain | Saudi Arabia | 2-0 | home_win | 13.7% | 2-0 (0.137); 1-0 (0.136); 2-1 (0.093); 1-1 (0.093); 3-0 (0.092) | 74.1% | 21.7% | 4.3% | 2.012 | 0.681 | home_win | 74.1% |
| 70 | 2026-06-27 | 9:00 p.m. | J | Jordan | Argentina | 0-2 | away_win | 15.1% | 0-2 (0.151); 0-1 (0.132); 0-3 (0.116); 1-2 (0.085); 1-1 (0.074) | 5.4% | 21.1% | 73.5% | 0.560 | 2.297 | away_win | 73.5% |
| 45 | 2026-06-23 | 4:00 p.m. | L | England | Ghana | 2-0 | home_win | 15.8% | 2-0 (0.158); 1-0 (0.157); 3-0 (0.106); 2-1 (0.085); 1-1 (0.084) | 73.4% | 21.5% | 5.1% | 2.017 | 0.537 | home_win | 73.4% |
| 8 | 2026-06-13 | 12:00 p.m. | B | Qatar | Switzerland | 0-1 | away_win | 11.3% | 0-1 (0.113); 0-2 (0.110); 1-1 (0.101); 1-2 (0.099); 0-3 (0.072) | 6.7% | 21.9% | 71.4% | 0.898 | 1.953 | away_win | 71.4% |
| 64 | 2026-06-26 | 8:00 p.m. | G | New Zealand | Belgium | 0-2 | away_win | 9.9% | 0-2 (0.099); 1-2 (0.094); 0-3 (0.083); 1-3 (0.079); 0-1 (0.078) | 8.6% | 21.2% | 70.3% | 0.950 | 2.524 | away_win | 70.3% |
| 42 | 2026-06-22 | 5:00 p.m. | I | France | Iraq | 1-0 | home_win | 14.4% | 1-0 (0.144); 1-1 (0.121); 2-0 (0.109); 0-0 (0.096); 2-1 (0.091) | 69.7% | 22.6% | 7.7% | 1.512 | 0.836 | home_win | 69.7% |
| 55 | 2026-06-25 | 4:00 p.m. | E | Curaçao | Ivory Coast | 0-2 | away_win | 11.3% | 0-2 (0.113); 0-1 (0.101); 1-2 (0.097); 1-1 (0.088); 0-3 (0.083) | 9.3% | 21.5% | 69.2% | 0.866 | 2.221 | away_win | 69.2% |
| 50 | 2026-06-24 | 6:00 p.m. | C | Morocco | Haiti | 1-0 | home_win | 17.0% | 1-0 (0.170); 2-0 (0.147); 1-1 (0.100); 0-0 (0.099); 2-1 (0.087) | 69.1% | 22.1% | 8.8% | 1.728 | 0.589 | home_win | 69.1% |
| 34 | 2026-06-20 | 7:00 p.m. | E | Ecuador | Curaçao | 1-0 | home_win | 14.7% | 1-0 (0.147); 1-1 (0.128); 0-0 (0.109); 2-0 (0.099); 0-1 (0.095) | 69.1% | 22.9% | 8.0% | 1.347 | 0.866 | home_win | 69.1% |
| 27 | 2026-06-18 | 3:00 p.m. | B | Canada | Qatar | 1-0 | home_win | 18.0% | 1-0 (0.180); 0-0 (0.130); 2-0 (0.125); 1-1 (0.118); 0-1 (0.085) | 68.2% | 22.9% | 8.9% | 1.390 | 0.652 | home_win | 68.2% |
| 29 | 2026-06-19 | 8:30 p.m. | C | Brazil | Haiti | 1-1 | draw | 11.0% | 1-1 (0.110); 2-1 (0.089); 1-2 (0.078); 1-0 (0.077); 0-1 (0.068) | 67.9% | 22.3% | 9.8% | 1.621 | 1.427 | home_win | 67.9% |
| 68 | 2026-06-27 | 5:00 p.m. | L | Croatia | Ghana | 1-0 | home_win | 11.1% | 1-0 (0.111); 1-1 (0.109); 2-0 (0.100); 2-1 (0.099); 0-0 (0.062) | 67.9% | 22.2% | 10.0% | 1.803 | 0.985 | home_win | 67.9% |
| 36 | 2026-06-20 | 10:00 p.m. | F | Tunisia | Japan | 0-1 | away_win | 14.4% | 0-1 (0.144); 0-2 (0.130); 1-1 (0.104); 1-2 (0.094); 0-0 (0.080) | 10.2% | 22.0% | 67.8% | 0.722 | 1.808 | away_win | 67.8% |

## Most likely knockout pairings
| match_number | round | first_team | second_team | pairing_probability | first_advancement_probability | second_advancement_probability | simulation_count | most_likely_advancing_team | most_likely_advancement_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 73 | round_of_32 | South Korea | Canada | 10.2% | 46.4% | 53.6% | 205 | Canada | 53.6% |
| 74 | round_of_32 | Germany | Turkey | 10.5% | 67.5% | 32.5% | 211 | Germany | 67.5% |
| 75 | round_of_32 | Japan | Brazil | 14.2% | 44.9% | 55.1% | 284 | Brazil | 55.1% |
| 76 | round_of_32 | Morocco | Netherlands | 18.4% | 52.0% | 48.0% | 369 | Morocco | 52.0% |
| 77 | round_of_32 | Norway | Sweden | 13.4% | 75.1% | 24.9% | 267 | Norway | 75.1% |
| 78 | round_of_32 | Ivory Coast | Senegal | 13.7% | 50.0% | 50.0% | 274 | Ivory Coast | 50.0% |
| 79 | round_of_32 | Mexico | Ecuador | 8.2% | 50.7% | 49.3% | 163 | Mexico | 50.7% |
| 80 | round_of_32 | England | DR Congo | 12.2% | 85.9% | 14.1% | 243 | England | 85.9% |
| 81 | round_of_32 | Australia | Bosnia and Herzegovina | 9.4% | 79.8% | 20.2% | 188 | Australia | 79.8% |
| 82 | round_of_32 | Belgium | Czech Republic | 10.8% | 75.6% | 24.4% | 215 | Belgium | 75.6% |
| 83 | round_of_32 | Colombia | Croatia | 9.8% | 63.1% | 36.9% | 197 | Colombia | 63.1% |
| 84 | round_of_32 | Spain | Argentina | 23.4% | 50.4% | 49.6% | 469 | Spain | 50.4% |
| 85 | round_of_32 | Canada | Egypt | 5.8% | 62.2% | 37.8% | 115 | Canada | 62.2% |
| 86 | round_of_32 | Argentina | Uruguay | 13.5% | 82.4% | 17.6% | 269 | Argentina | 82.4% |
| 87 | round_of_32 | Portugal | Panama | 8.4% | 84.2% | 15.8% | 168 | Portugal | 84.2% |
| 88 | round_of_32 | United States | Iran | 11.8% | 47.0% | 53.0% | 235 | Iran | 53.0% |
| 89 | round_of_16 | South Korea | Japan | 5.1% | 29.4% | 70.6% | 103 | Japan | 70.6% |
| 90 | round_of_16 | Germany | Norway | 9.3% | 55.1% | 44.9% | 187 | Germany | 55.1% |
| 91 | round_of_16 | Morocco | Germany | 7.4% | 47.9% | 52.1% | 149 | Germany | 52.1% |
| 92 | round_of_16 | Mexico | England | 14.3% | 35.6% | 64.4% | 286 | England | 64.4% |
| 93 | round_of_16 | Croatia | Spain | 9.7% | 20.5% | 79.5% | 193 | Spain | 79.5% |
| 94 | round_of_16 | Australia | Belgium | 10.7% | 35.4% | 64.6% | 213 | Belgium | 64.6% |
| 95 | round_of_16 | Argentina | Australia | 7.2% | 82.3% | 17.7% | 145 | Argentina | 82.3% |
| 96 | round_of_16 | Switzerland | Portugal | 5.3% | 36.5% | 63.5% | 107 | Portugal | 63.5% |
| 97 | quarter_final | Brazil | Germany | 3.5% | 51.2% | 48.8% | 69 | Brazil | 51.2% |
| 98 | quarter_final | Spain | Belgium | 8.5% | 71.8% | 28.2% | 170 | Spain | 71.8% |
| 99 | quarter_final | Morocco | England | 5.0% | 43.3% | 56.7% | 100 | England | 56.7% |
| 100 | quarter_final | Argentina | Portugal | 5.9% | 67.9% | 32.1% | 117 | Argentina | 67.9% |
| 101 | semi_final | Germany | Spain | 3.6% | 31.1% | 68.9% | 73 | Spain | 68.9% |
| 102 | semi_final | England | Argentina | 4.4% | 35.0% | 65.0% | 88 | Argentina | 65.0% |
| 103 | final | Spain | Argentina | 5.3% | 50.4% | 49.6% | 107 | Spain | 50.4% |
| 104 | third_place | Spain | Argentina | 0.9% | 50.4% | 49.6% | 17 | Spain | 50.4% |

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
