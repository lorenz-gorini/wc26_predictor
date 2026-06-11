"""Download public football data used by the project.

The functions in this module are intentionally source-specific. Generic scraping
usually hides assumptions; explicit source adapters make failures easier to audit.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from string import ascii_uppercase
from urllib.parse import urljoin
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

from wc26_predictor.data.club_form import parse_wikipedia_top_scorers

USER_AGENT = "wc26-predictor/0.1 (+https://github.com/martj42/international_results)"
INTERNATIONAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
GOALSCORERS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
)
WORLD_CUP_2026_SQUADS_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
WORLD_CUP_GROUP_PAGE_TEMPLATE = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_{group}"
CLUB_TOP_SCORER_SOURCES = [
    ("Premier League", "https://en.wikipedia.org/wiki/2025%E2%80%9326_Premier_League"),
    ("La Liga", "https://en.wikipedia.org/wiki/2025%E2%80%9326_La_Liga"),
    ("Bundesliga", "https://en.wikipedia.org/wiki/2025%E2%80%9326_Bundesliga"),
    ("Serie A", "https://en.wikipedia.org/wiki/2025%E2%80%9326_Serie_A"),
    ("Ligue 1", "https://en.wikipedia.org/wiki/2025%E2%80%9326_Ligue_1"),
    ("Saudi Pro League", "https://en.wikipedia.org/wiki/2025%E2%80%9326_Saudi_Pro_League"),
    ("Major League Soccer", "https://en.wikipedia.org/wiki/2026_Major_League_Soccer_season"),
    ("Super Lig", "https://en.wikipedia.org/wiki/2025%E2%80%9326_S%C3%BCper_Lig"),
    (
        "UEFA Champions League",
        "https://en.wikipedia.org/wiki/2025%E2%80%9326_UEFA_Champions_League",
    ),
    ("UEFA Europa League", "https://en.wikipedia.org/wiki/2025%E2%80%9326_UEFA_Europa_League"),
]

FIXTURE_COLUMNS = [
    "date",
    "time_local",
    "utc_offset",
    "group",
    "match_number",
    "home_team",
    "away_team",
    "stadium",
    "city",
    "source_url",
]

VENUE_CITY_UTC_OFFSETS = {
    "Arlington": "UTC-5",
    "Atlanta": "UTC-4",
    "East Rutherford": "UTC-4",
    "Foxborough": "UTC-4",
    "Guadalajara": "UTC-6",
    "Houston": "UTC-5",
    "Inglewood": "UTC-7",
    "Kansas City": "UTC-5",
    "Mexico City": "UTC-6",
    "Miami Gardens": "UTC-4",
    "Monterrey": "UTC-6",
    "Philadelphia": "UTC-4",
    "Santa Clara": "UTC-7",
    "Seattle": "UTC-7",
    "Toronto": "UTC-4",
    "Vancouver": "UTC-7",
    "Zapopan": "UTC-6",
}


@dataclass(frozen=True, slots=True)
class DownloadRecord:
    """Metadata for a downloaded public data source."""

    name: str
    source_url: str
    destination: str
    rows: int
    downloaded_at_utc: str


def download_international_results(destination: str | Path) -> DownloadRecord:
    """Download the public international results CSV.

    The upstream file can include scheduled fixtures with missing scores. This is
    acceptable in raw data; completed-match validation happens in processing.
    """

    destination_path = Path(destination)
    response = _get(INTERNATIONAL_RESULTS_URL)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(response.content)

    raw_rows = len(pd.read_csv(destination_path))
    return DownloadRecord(
        name="martj42 international_results results.csv",
        source_url=INTERNATIONAL_RESULTS_URL,
        destination=str(destination_path),
        rows=raw_rows,
        downloaded_at_utc=_utc_now(),
    )


def download_goalscorers(destination: str | Path) -> DownloadRecord:
    """Download the public goalscorers CSV from martj42/international_results."""

    destination_path = Path(destination)
    response = _get(GOALSCORERS_URL)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(response.content)

    raw_rows = len(pd.read_csv(destination_path))
    return DownloadRecord(
        name="martj42 international_results goalscorers.csv",
        source_url=GOALSCORERS_URL,
        destination=str(destination_path),
        rows=raw_rows,
        downloaded_at_utc=_utc_now(),
    )


def download_world_cup_2026_squads(destination: str | Path) -> DownloadRecord:
    """Scrape World Cup 2026 squad tables from Wikipedia."""

    destination_path = Path(destination)
    response = _get(WORLD_CUP_2026_SQUADS_URL)
    squads = _parse_world_cup_squads(response.text, source_url=WORLD_CUP_2026_SQUADS_URL)
    _validate_squads(squads)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    squads.to_csv(destination_path, index=False)
    return DownloadRecord(
        name="Wikipedia FIFA World Cup 2026 squads",
        source_url=WORLD_CUP_2026_SQUADS_URL,
        destination=str(destination_path),
        rows=len(squads),
        downloaded_at_utc=_utc_now(),
    )


def download_club_top_scorers(destination: str | Path) -> DownloadRecord:
    """Scrape a curated set of public club top-scorer tables."""

    destination_path = Path(destination)
    frames = []
    source_urls = []
    for competition, url in CLUB_TOP_SCORER_SOURCES:
        response = _get(url)
        frames.append(parse_wikipedia_top_scorers(response.text, competition, url))
        source_urls.append(url)

    club_form = pd.concat(frames, ignore_index=True)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    club_form.to_csv(destination_path, index=False)
    return DownloadRecord(
        name="Wikipedia club top-scorer tables",
        source_url=", ".join(source_urls),
        destination=str(destination_path),
        rows=len(club_form),
        downloaded_at_utc=_utc_now(),
    )


def download_world_cup_2026_group_fixtures(destination: str | Path) -> DownloadRecord:
    """Scrape World Cup 2026 group-stage fixtures from Wikipedia group pages.

    Wikipedia is used here as a machine-readable mirror of FIFA-cited fixture
    tables. Treat this file as raw data: audit before publishing final forecasts.
    """

    destination_path = Path(destination)
    frames = []
    source_urls = []
    for group in ascii_uppercase[:12]:
        url = WORLD_CUP_GROUP_PAGE_TEMPLATE.format(group=group)
        source_urls.append(url)
        response = _get(url)
        frames.append(_parse_group_page(response.text, source_url=url, group=group))

    fixtures = pd.concat(frames, ignore_index=True)
    _validate_group_fixtures(fixtures)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(destination_path, index=False)
    return DownloadRecord(
        name="Wikipedia FIFA World Cup 2026 group fixture pages",
        source_url=", ".join(source_urls),
        destination=str(destination_path),
        rows=len(fixtures),
        downloaded_at_utc=_utc_now(),
    )


def write_download_metadata(records: list[DownloadRecord], destination: str | Path) -> None:
    """Write source metadata for reproducibility."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        json.dumps([asdict(record) for record in records], indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_group_page(html: str, source_url: str, group: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    boxes = soup.select("div.footballbox")
    if len(boxes) != 6:
        raise ValueError(f"Expected 6 fixture boxes for group {group}, found {len(boxes)}.")

    rows = [_parse_footballbox(box, source_url=source_url, group=group) for box in boxes]
    return pd.DataFrame(rows, columns=FIXTURE_COLUMNS)


def _parse_world_cup_squads(html: str, source_url: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for table in soup.select("table.wikitable"):
        team_heading = table.find_previous("div", class_="mw-heading3")
        if team_heading is None:
            continue
        team = team_heading.get_text(" ", strip=True).replace("[ edit ]", "").strip()
        parsed = pd.read_html(StringIO(str(table)))[0]
        expected_columns = {"No.", "Pos.", "Player", "Caps", "Goals", "Club"}
        if not expected_columns.issubset(parsed.columns):
            continue
        for player in parsed.itertuples(index=False):
            rows.append(
                {
                    "team": team,
                    "number": int(getattr(player, "_0")),
                    "position": str(getattr(player, "_1")),
                    "player": _clean_player_name(str(getattr(player, "Player"))),
                    "caps": int(getattr(player, "Caps")),
                    "goals": int(getattr(player, "Goals")),
                    "club": str(getattr(player, "Club")).strip(),
                    "source_url": source_url,
                }
            )
    return pd.DataFrame(rows)


def _clean_player_name(player: str) -> str:
    return re.sub(r"\s+\((captain|vice-captain)\)", "", player, flags=re.IGNORECASE).strip()


def _parse_footballbox(box: Tag, source_url: str, group: str) -> dict[str, object]:
    date_text = _required_text(box, ".fdate")
    iso_dates = re.findall(r"\d{4}-\d{2}-\d{2}", date_text)
    if len(iso_dates) != 1:
        raise ValueError(f"Could not extract unique ISO date from fixture text: {date_text!r}")

    score_text = _required_text(box, ".fscore")
    match_numbers = re.findall(r"\d+", score_text)
    if len(match_numbers) != 1:
        raise ValueError(f"Could not extract unique match number from score text: {score_text!r}")

    stadium, city = _parse_venue(_required_text(box, ".fright [itemprop='name address']"))
    time_text = _required_text(box, ".ftime")
    utc_offset = _parse_utc_offset(time_text, city)

    return {
        "date": iso_dates[0],
        "time_local": time_text.replace(utc_offset, "").replace(utc_offset.replace("-", "−"), "").strip(),
        "utc_offset": utc_offset,
        "group": group,
        "match_number": int(match_numbers[0]),
        "home_team": _required_text(box, ".fhome [itemprop='name']"),
        "away_team": _required_text(box, ".faway [itemprop='name']"),
        "stadium": stadium,
        "city": city,
        "source_url": source_url,
    }


def _parse_venue(venue_text: str) -> tuple[str, str]:
    parts = [part.strip() for part in venue_text.split(",")]
    if len(parts) < 2:
        raise ValueError(f"Could not parse stadium and city from venue text: {venue_text!r}")
    return parts[0], ", ".join(parts[1:])


def _parse_utc_offset(time_text: str, city: str) -> str:
    utc_offsets = re.findall(r"UTC[+-−]\d+", time_text)
    if len(utc_offsets) == 1:
        return utc_offsets[0].replace("−", "-")
    if len(utc_offsets) > 1:
        raise ValueError(f"Could not extract unique UTC offset from time text: {time_text!r}")
    if city not in VENUE_CITY_UTC_OFFSETS:
        raise ValueError(f"No UTC offset found and city {city!r} is not in the venue offset map.")
    return VENUE_CITY_UTC_OFFSETS[city]


def _validate_group_fixtures(fixtures: pd.DataFrame) -> None:
    missing = set(FIXTURE_COLUMNS).difference(fixtures.columns)
    if missing:
        raise ValueError(f"Missing fixture columns: {sorted(missing)}")

    if len(fixtures) != 72:
        raise ValueError(f"Expected 72 group-stage fixtures, found {len(fixtures)}.")
    if fixtures["match_number"].nunique() != 72:
        raise ValueError("Fixture match numbers must be unique.")
    if fixtures[["date", "home_team", "away_team", "stadium", "city"]].isna().any().any():
        raise ValueError("Fixture table contains missing values.")
    if (fixtures["home_team"] == fixtures["away_team"]).any():
        raise ValueError("Fixture table contains a match with identical teams.")


def _validate_squads(squads: pd.DataFrame) -> None:
    required_columns = {"team", "number", "position", "player", "caps", "goals", "club", "source_url"}
    missing = required_columns.difference(squads.columns)
    if missing:
        raise ValueError(f"Missing squad columns: {sorted(missing)}")
    if squads.empty:
        raise ValueError("Squad table is empty.")
    if squads[["team", "position", "player", "club"]].isna().any().any():
        raise ValueError("Squad table contains missing string values.")


def _required_text(node: Tag, selector: str) -> str:
    match = node.select_one(selector)
    if match is None:
        raise ValueError(f"Could not find required selector {selector!r}.")
    text = " ".join(match.get_text(" ", strip=True).split())
    if not text:
        raise ValueError(f"Selector {selector!r} is empty.")
    return text


def _get(url: str) -> requests.Response:
    response = requests.get(
        urljoin(url, ""),
        timeout=30,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response


def _utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
