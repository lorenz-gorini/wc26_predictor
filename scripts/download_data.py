#!/usr/bin/env python
"""Download public raw data for the World Cup 2026 predictor."""

from __future__ import annotations

import argparse
from pathlib import Path

from wc26_predictor.data.download_public import (
    download_club_top_scorers,
    download_goalscorers,
    download_international_results,
    download_world_cup_2026_group_fixtures,
    download_world_cup_2026_squads,
    write_download_metadata,
)
from wc26_predictor.data.ingest_results import load_completed_results_csv, save_processed_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root directory.",
    )
    parser.add_argument(
        "--skip-fixtures",
        action="store_true",
        help="Only download historical results.",
    )
    args = parser.parse_args()

    raw_dir = args.project_root / "data" / "raw"
    processed_dir = args.project_root / "data" / "processed"

    records = [
        download_international_results(raw_dir / "international_results.csv"),
        download_goalscorers(raw_dir / "goalscorers.csv"),
    ]
    save_processed_results(
        load_completed_results_csv(raw_dir / "international_results.csv"),
        processed_dir / "international_results.csv",
    )

    if not args.skip_fixtures:
        records.append(
            download_world_cup_2026_group_fixtures(raw_dir / "world_cup_2026_fixtures.csv")
        )
        records.append(download_world_cup_2026_squads(raw_dir / "world_cup_2026_squads.csv"))
        records.append(download_club_top_scorers(raw_dir / "club_top_scorers.csv"))

    write_download_metadata(records, raw_dir / "download_metadata.json")
    for record in records:
        print(f"{record.name}: {record.rows} rows -> {record.destination}")


if __name__ == "__main__":
    main()
