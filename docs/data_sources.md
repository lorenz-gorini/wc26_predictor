# Data Sources

This page records the current public sources used by the project and where to
download them manually if the script cannot access the network.

## Historical International Results

Source:

```text
https://raw.githubusercontent.com/martj42/international_results/master/results.csv
```

Repository:

```text
https://github.com/martj42/international_results
```

Use:

- model training;
- Elo updates;
- Poisson attack/defense estimation;
- historical backtesting.

Important detail: the upstream file can include scheduled fixtures with missing
scores. The raw file is saved unchanged to `data/raw/international_results.csv`.
The modeling file `data/processed/international_results.csv` keeps completed
matches only.

## World Cup 2026 Group Fixtures

Source pages:

```text
https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_A
https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_B
https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_C
https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_D
https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_E
https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_F
https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_G
https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_H
https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_I
https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_J
https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_K
https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_L
```

Use:

- World Cup 2026 group-stage prediction;
- group simulation;
- later full tournament simulation.

These pages cite FIFA fixture reports, but they are still a secondary mirror.
Before final forecasts, audit `data/raw/world_cup_2026_fixtures.csv` against
FIFA's official fixture page:

```text
https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/scores-fixtures
```

## Download Command

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/download_data.py
```

Generated files:

```text
data/raw/international_results.csv
data/processed/international_results.csv
data/raw/world_cup_2026_fixtures.csv
data/raw/world_cup_2026_squads.csv
data/raw/club_top_scorers.csv
data/raw/download_metadata.json
```

## Historical Goalscorers

Source:

```text
https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv
```

Use:

- national-team top-scorer baseline;
- penalty/own-goal filtering;
- later player-level form features.

Some historical rows identify the scoring team but not the individual scorer.
Those rows are kept in raw data and dropped only for the player scorer model.

## World Cup 2026 Squads

Source:

```text
https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads
```

Use:

- filter scorer candidates to listed squad players;
- player position, caps, goals, and club fields;
- later club-form enrichment and expected-minutes modeling.

This is a secondary public source. Final forecasts should be audited against
FIFA's official squad lists.

## Club Top-Scorer Tables

The current club-form enrichment uses a curated set of Wikipedia competition
pages for 2025-26 and 2026 club scoring leaders. This is a sparse fallback
because FBref was blocked by Cloudflare in the local environment.

Use:

- club-form enrichment for the current top 100 scorer candidates;
- player+club identity matching;
- conservative club-form adjustment to expected tournament goals.

The resulting file is:

```text
data/raw/club_top_scorers.csv
```

The adjusted scorer file is:

```text
data/processed/world_cup_2026_top_scorer_club_adjusted_top100.csv
```

## Transfermarkt Kaggle Dataset

Source:

```text
https://www.kaggle.com/datasets/davidcariboo/player-scores
```

Local directory:

```text
data/raw/transfermarkt_kaggle/
```

The local files include:

```text
competitions.csv
games.csv
clubs.csv
players.csv
player_valuations.csv
appearances.csv
game_events.csv
game_lineups.csv
club_games.csv
transfers.csv
countries.csv
national_teams.csv
```

This is now the preferred source for top-scorer club-form enrichment. The current
pipeline uses:

- `players.csv` for identity, position, country, club, market value, caps, and
  international goals;
- `appearances.csv` for 2025-26 club goals, assists, and minutes;
- `game_events.csv` for goal-event fallback when appearance rows are incomplete;
- `game_lineups.csv` for starts;
- `player_valuations.csv` for latest market value;
- `competitions.csv` to exclude national-team competitions from club-form
  aggregation.

The resulting file is:

```text
data/processed/world_cup_2026_top_scorer_transfermarkt_adjusted_top100.csv
```

If a player has no recent appearance rows in the Kaggle dump, the model first
tries to recover club-form signal from goal events and starting-lineup records.
Only players with no usable appearance, event, or lineup coverage are treated as
neutral.

## Official 2026 Knockout Bracket

The official tournament simulator uses a packaged copy of the 495-row
third-place assignment table from the 2026 knockout-stage page, which mirrors
FIFA Annex C:

```text
src/wc26_predictor/simulation/data/third_place_assignments_2026.csv
```

The simulator also encodes the published match-number tree for matches 73-104,
including the final and third-place match.

## Bookmaker Odds

Source:

```text
https://www.football-data.co.uk/WorldCup2026.xlsx
```

Local file:

```text
data/raw/odds/WorldCup2026.xlsx
```

The workbook currently provides historical World Cup finals odds for 2014, 2018,
and 2022, plus 2026 qualifier odds. The validation workflow uses only the
historical finals sheets because those correspond to the model holdout windows.

The processed market files are:

```text
data/processed/world_cup_market_validation_matches.csv
data/processed/world_cup_market_validation_metrics.csv
data/processed/world_cup_market_ensemble_weights_lowo.csv
data/processed/world_cup_market_gate_decision.csv
```

## Player Availability and Injuries

Optional current overrides:

```text
data/raw/world_cup_2026_player_availability.csv
```

Required columns:

```text
team,player,status
```

Optional columns:

```text
expected_minutes_share,penalty_taker_rank,reason,return_date,source_url
```

Status must be one of `available`, `doubtful`, `injured`, `suspended`, or `out`.
Rows are matched against the squad file; unknown players raise an error.

Optional historical calibration file:

```text
data/raw/historical_team_availability.csv
```

Required columns:

```text
date,team,team_availability_burden
```

If present, the project estimates the expected-goal penalty associated with team
availability burden from historical matches. If absent, the pipeline records
that no historical injury-impact calibration was available and uses a
conservative default coefficient.

Transfermarkt-derived injury history:

```text
data/raw/transfermarkt_injuries/player_injuries.csv
```

The pipeline joins this file to `data/raw/transfermarkt_kaggle/players.csv` and
the World Cup squad file through Transfermarkt `player_id` when available, with
normalized player/country/club matching as a fallback. It writes:

```text
data/processed/world_cup_2026_transfermarkt_squad_injury_features.csv
data/processed/world_cup_2026_transfermarkt_team_injury_burden.csv
```

Open-ended injury rows in the historical source are treated as open at the
source snapshot, not automatically as verified current World Cup injuries. They
affect historical recurrence burden; hard current absences should still be
provided through `data/raw/world_cup_2026_player_availability.csv` or a fresh
API-backed feed.

Local API keys should be stored in the Git-ignored project `.env` file:

```text
SOCCERDATA_API_KEY=your_key_here
```

SoccerDataAPI can provide fresh sidelined-player information through match
payloads. The guarded fetch script first downloads the upcoming preview index
and only fetches match details up to an explicit request cap:

```bash
PYTHONPATH=src /Users/lorenzogorini/anaconda3/envs/general/bin/python scripts/fetch_soccerdata_injuries.py --max-match-requests 0
```

Outputs:

```text
data/raw/soccerdata_upcoming_match_previews.csv
data/raw/soccerdata_filtered_match_previews.csv
data/raw/soccerdata_current_sidelined_players.csv
```

The default league filter is `World Cup`. If those previews are absent, do not
spend match-detail requests; use the Transfermarkt recurrence features plus
manual verified overrides instead.

## Future Sources

Useful next additions:

- World Cup 2026 finals bookmaker odds when available;
- verified current injury/suspension feeds;
- FIFA ranking snapshots;
- squad/player snapshots;
- injury and suspension sources;
- venue metadata such as altitude, indoor roof, and expected climate.
