#!/usr/bin/env python
"""Fetch Sportsgambler current football injuries and map them to WC squads."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc26_predictor.data.sportsgambler import (
    SportsgamblerConfig,
    fetch_sportsgambler_football_injuries,
    match_sportsgambler_injuries_to_squads,
)
from wc26_predictor.data.squads import load_squads_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root directory.",
    )
    parser.add_argument(
        "--include-club-suspensions",
        action="store_true",
        help="Include club red-card suspensions in the model-ready override file.",
    )
    args = parser.parse_args()

    raw_dir = args.project_root / "data" / "raw"
    squads = load_squads_csv(raw_dir / "world_cup_2026_squads.csv")
    config = SportsgamblerConfig(include_club_suspensions=args.include_club_suspensions)
    injuries = fetch_sportsgambler_football_injuries(config)
    matched, overrides = match_sportsgambler_injuries_to_squads(
        injuries=injuries,
        squads=squads,
        config=config,
    )

    injuries_path = raw_dir / "sportsgambler_all_football_injuries.csv"
    matched_path = raw_dir / "sportsgambler_squad_injury_matches.csv"
    overrides_path = raw_dir / "sportsgambler_player_availability_overrides.csv"
    injuries.to_csv(injuries_path, index=False)
    matched.to_csv(matched_path, index=False)
    overrides.to_csv(overrides_path, index=False)

    print(f"Sportsgambler all injuries -> {injuries_path} ({len(injuries)} rows)")
    print(f"Sportsgambler squad injury matches -> {matched_path} ({len(matched)} rows)")
    print(f"Sportsgambler availability overrides -> {overrides_path} ({len(overrides)} rows)")


if __name__ == "__main__":
    main()
