# Top-Scorer Model

The current top-scorer model is deliberately separate from the match-outcome
model. It is a national-team baseline, not yet a full player model.

## Current Method

1. Load `data/raw/goalscorers.csv`.
2. Drop own goals and rows with unknown individual scorer names.
3. Keep players from World Cup 2026 teams.
4. Apply time decay to historical national-team goals.
5. Estimate each player's recent weighted share of his team's goals.
6. Estimate team tournament exposure with group-stage Monte Carlo simulation and
   the official 2026 knockout bracket.
7. Multiply each player's goal share by both expected group-stage goals and
   expected tournament goals.

Output:

```text
data/processed/world_cup_2026_top_scorer_baseline.csv
```

## Limitations

This model still does not know:

- confirmed starting XIs;
- live tactical role changes;
- verified current injuries unless provided through the availability override file;
- official penalty-taking hierarchy today beyond historical national-team penalty evidence.

Because of this, the current output is best interpreted as a baseline ranking,
not a final Golden Boot forecast.

## Next Layer

The next useful player data layer should add:

- final squad lists;
- club minutes and goals in 2025-26;
- club expected goals if available;
- player age and position;
- penalty-taker status;
- team advancement probabilities from the official bracket simulator.

## Club-Form Enrichment

The project now includes two club-form enrichment layers for the top 100 current
scorer candidates.

### Transfermarkt Kaggle

Preferred output:

```text
data/processed/world_cup_2026_top_scorer_transfermarkt_adjusted_top100.csv
```

This uses the local Kaggle Transfermarkt dump in:

```text
data/raw/transfermarkt_kaggle/
```

Features:

- Transfermarkt player identity match quality;
- current club and country;
- 2025-26 club goals, assists, minutes, and starts;
- goals per 90, weighted goals per 90, and assists per 90;
- latest and profile market values;
- international caps and goals;
- conservative adjusted expected tournament goals.

### Availability Layer

The scorer workflow now builds:

```text
data/processed/world_cup_2026_player_availability_features.csv
data/processed/world_cup_2026_team_availability_burden.csv
data/processed/world_cup_2026_availability_impact.csv
data/processed/world_cup_2026_top_scorer_availability_adjusted.csv
```

By default, every listed squad player is treated as available. The model derives:

- expected-minutes share from squad caps, national-team goals, and position;
- penalty-taker rank from historical national-team penalty goals;
- a player availability multiplier from status;
- a team availability burden from unavailable expected-minutes share.

Manual injury or suspension overrides can be added in:

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

Accepted status values are `available`, `doubtful`, `injured`, `suspended`, and
`out`. Unknown players raise an error rather than being silently ignored.

The match-forecast layer can also use team availability burden to adjust expected
goals. If `data/raw/historical_team_availability.csv` is present, the project
estimates the goal penalty from historical team-match availability burden.
Otherwise it records that no historical calibration file was available and uses
a conservative default coefficient.

The pipeline reads the large appearances, events, and lineups files in chunks and
only enriches the current top 100 scorer candidates. It uses the 2025-26 season
through the forecast date, not only the last few weeks. This avoids incorrectly
penalizing players whose `appearances.csv` rows are sparse or delayed in the
Kaggle dump. Within that season, goals are exponentially time-decayed with a
120-day half-life, so a goal scored close to the forecast date receives more
weight than a goal scored early in the season.

When `appearances.csv` has no recent rows for a matched player, the model falls
back to Transfermarkt goal events and starting-lineup records. Event goals are
used as the goal count, and starts are converted to an approximate minutes
exposure using 75 minutes per start. This is why players such as Messi or Ronaldo
can still receive club-form information even if their appearances table coverage
is incomplete.

Club goals are adjusted before entering the model:

- Champions League goals receive a 1.15 competition weight;
- Europa League and Conference League goals receive 1.05;
- other international cups receive 1.10;
- first-tier goals in England, Spain, Germany, Italy, and France receive 1.00;
- first-tier goals in Portugal, Netherlands, and Belgium receive 0.90;
- first-tier goals in Saudi Arabia, United States, Mexico, Japan, and Turkey
  receive 0.75;
- domestic cup goals receive 0.85;
- other club competitions receive 0.80.

The model also normalizes for competition scoring environment by estimating each
competition's goals per 90 from the Kaggle appearances file and applying a
bounded correction factor. High-scoring leagues are deflated and low-scoring
leagues are inflated, with the factor clipped between 0.80 and 1.20.

The club-form signal is deliberately an adjustment to the national-team baseline,
not a replacement for it. The base expectation comes from national-team scoring
share times team tournament exposure. Transfermarkt features then create a
multiplier from weighted club goals per 90, minutes coverage, and market value,
with missing recent club coverage treated as neutral.

### Public Top-Scorer Tables

Fallback output:

```text
data/raw/club_top_scorers.csv
data/processed/world_cup_2026_top_scorer_club_adjusted_top100.csv
```

Because FBref blocks automated access in this environment, this fallback layer
uses curated public top-scorer tables from Wikipedia competition pages.
The enrichment:

- uses the 2026 squad club field;
- matches players by normalized player name and club name;
- marks match quality as `player_and_club`, `player_only`, or `unmatched`;
- applies a conservative multiplier to expected tournament goals.

This fallback is not a complete club-form model. Coverage is sparse because public
top-scorer tables generally include only the leading scorers in each competition.
The correct interpretation is: when a high-confidence club-form match exists,
use it as an extra signal; otherwise keep the national-team/tournament-exposure
estimate unchanged.
