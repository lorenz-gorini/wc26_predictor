#!/usr/bin/env python
"""Fetch cached SoccerDataAPI injury/sidelined data with strict request limits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wc26_predictor.data.soccerdata_api import (
    SoccerDataAPIClient,
    extract_sidelined_availability,
    load_soccerdata_api_key,
    resolve_sidelined_to_squad_availability,
    summarize_upcoming_match_previews,
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
        "--refresh-upcoming",
        action="store_true",
        help="Refresh the upcoming match-preview index with one API request.",
    )
    parser.add_argument(
        "--league-name-contains",
        default="World Cup",
        help="Case-insensitive league filter before fetching match details.",
    )
    parser.add_argument(
        "--max-match-requests",
        type=int,
        default=0,
        help="Maximum uncached match-detail requests to spend.",
    )
    parser.add_argument(
        "--force-match-refresh",
        action="store_true",
        help="Ignore cached match-detail JSON files.",
    )
    args = parser.parse_args()
    if args.max_match_requests < 0:
        raise ValueError("max-match-requests cannot be negative.")

    cache_dir = args.project_root / "data" / "raw" / "soccerdata_api"
    output_dir = args.project_root / "data" / "raw"
    key = load_soccerdata_api_key(args.project_root / ".env")
    client = SoccerDataAPIClient(api_key=key, cache_dir=cache_dir)

    upcoming_payload, used_request, upcoming_cache = client.get(
        "match-previews-upcoming",
        force=args.refresh_upcoming,
    )
    requests_used = int(used_request)
    previews = summarize_upcoming_match_previews(upcoming_payload)
    previews_path = output_dir / "soccerdata_upcoming_match_previews.csv"
    previews.to_csv(previews_path, index=False)

    filtered = previews
    if args.league_name_contains and not previews.empty:
        pattern = args.league_name_contains.casefold()
        filtered = previews[
            previews["league_name"].astype("string").str.casefold().str.contains(pattern, na=False)
        ].copy()
    filtered_path = output_dir / "soccerdata_filtered_match_previews.csv"
    filtered.to_csv(filtered_path, index=False)

    availability_frames = []
    uncached_match_requests = 0
    for match_id in filtered["match_id"].dropna().astype(int).tolist():
        if uncached_match_requests >= args.max_match_requests:
            break
        payload, match_used_request, _ = client.get(
            "match",
            {"match_id": match_id},
            force=args.force_match_refresh,
        )
        if match_used_request:
            uncached_match_requests += 1
            requests_used += 1
        availability = extract_sidelined_availability(payload)
        if not availability.empty:
            availability_frames.append(availability)

    current_availability = (
        pd.concat(availability_frames, ignore_index=True)
        if availability_frames
        else pd.DataFrame(columns=["match_id", "team", "player", "status", "reason", "source_url"])
    )
    current_path = output_dir / "soccerdata_current_sidelined_players.csv"
    current_availability.to_csv(current_path, index=False)
    squad_overrides_path = output_dir / "soccerdata_player_availability_overrides.csv"
    unmatched_path = output_dir / "soccerdata_unmatched_sidelined_players.csv"
    squads_path = output_dir / "world_cup_2026_squads.csv"
    if squads_path.exists():
        squads = load_squads_csv(squads_path)
        matched_overrides, unmatched = resolve_sidelined_to_squad_availability(
            current_availability,
            squads,
        )
        matched_overrides.to_csv(squad_overrides_path, index=False)
        unmatched.to_csv(unmatched_path, index=False)

    print(f"Upcoming preview cache -> {upcoming_cache}")
    print(f"Upcoming preview rows -> {len(previews)}")
    print(f"Filtered preview rows -> {len(filtered)}")
    print(f"Current sidelined rows -> {len(current_availability)}")
    print(f"API requests used in this run -> {requests_used}")
    print(f"Upcoming previews CSV -> {previews_path}")
    print(f"Filtered previews CSV -> {filtered_path}")
    print(f"Current sidelined CSV -> {current_path}")
    if squads_path.exists():
        print(f"Matched squad availability overrides -> {squad_overrides_path}")
        print(f"Unmatched sidelined players -> {unmatched_path}")


if __name__ == "__main__":
    main()
