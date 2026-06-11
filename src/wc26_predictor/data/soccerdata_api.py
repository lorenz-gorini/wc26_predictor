"""SoccerDataAPI client and injury extraction helpers."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from wc26_predictor.data.club_form import normalize_name

SOCCERDATA_BASE_URL = "https://api.soccerdataapi.com"


@dataclass(frozen=True, slots=True)
class SoccerDataAPIClient:
    """Tiny cached SoccerDataAPI client.

    SoccerDataAPI plans can have low daily request limits. This client always
    writes successful responses to disk and reuses them unless `force=True`.
    """

    api_key: str
    cache_dir: Path
    base_url: str = SOCCERDATA_BASE_URL
    timeout_seconds: int = 30

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> tuple[dict[str, Any] | list[Any], bool, Path]:
        """Return JSON response, whether a network request was used, and cache path."""

        query = dict(params or {})
        query["auth_token"] = self.api_key
        cache_path = self.cache_dir / f"{_cache_key(endpoint, params or {})}.json"
        if cache_path.exists() and not force:
            return json.loads(cache_path.read_text(encoding="utf-8")), False, cache_path

        url = f"{self.base_url.rstrip('/')}/{endpoint.strip('/')}/"
        try:
            response = requests.get(
                url,
                headers={"Accept-Encoding": "gzip", "Content-Type": "application/json"},
                params=query,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException:
            raise ValueError(
                f"SoccerDataAPI request failed for endpoint {endpoint.strip('/')!r}."
            ) from None
        payload = response.json()
        if isinstance(payload, dict) and "detail" in payload:
            raise ValueError(f"SoccerDataAPI returned an error: {payload['detail']}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return payload, True, cache_path


def load_soccerdata_api_key(env_path: str | Path = ".env") -> str:
    """Load SoccerDataAPI key from environment or a local `.env` file."""

    env_value = os.environ.get("SOCCERDATA_API_KEY", "").strip()
    if env_value:
        return env_value

    path = Path(env_path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == "SOCCERDATA_API_KEY":
                cleaned = value.strip().strip('"').strip("'")
                if cleaned:
                    return cleaned
    raise ValueError("SOCCERDATA_API_KEY was not found in the environment or .env file.")


def summarize_upcoming_match_previews(payload: dict[str, Any] | list[Any]) -> pd.DataFrame:
    """Flatten SoccerDataAPI upcoming match-preview groups to one row per match."""

    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        groups = payload["results"]
    elif isinstance(payload, list):
        groups = payload
    else:
        raise ValueError("Expected upcoming match previews response to be a list or results dict.")

    rows: list[dict[str, Any]] = []
    for league_group in groups:
        previews = league_group.get("match_previews", [])
        if not isinstance(previews, list):
            continue
        league_id = league_group.get("league_id")
        league_name = league_group.get("league_name")
        country = league_group.get("country", {})
        country_name = country.get("name") if isinstance(country, dict) else None
        for preview in previews:
            teams = preview.get("teams", {})
            home = teams.get("home", {}) if isinstance(teams, dict) else {}
            away = teams.get("away", {}) if isinstance(teams, dict) else {}
            rows.append(
                {
                    "league_id": league_id,
                    "league_name": league_name,
                    "country": country_name,
                    "match_id": preview.get("id"),
                    "date": preview.get("date"),
                    "time": preview.get("time"),
                    "home_team_id": home.get("id"),
                    "home_team": home.get("name"),
                    "away_team_id": away.get("id"),
                    "away_team": away.get("name"),
                    "word_count": preview.get("word_count"),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["match_id"] = pd.to_numeric(frame["match_id"], errors="raise").astype(int)
    return frame.sort_values(["date", "league_name", "match_id"]).reset_index(drop=True)


def extract_sidelined_availability(match_payload: dict[str, Any]) -> pd.DataFrame:
    """Extract sidelined players from a SoccerDataAPI match response."""

    if not isinstance(match_payload, dict):
        raise ValueError("Expected match payload to be a dictionary.")

    teams = match_payload.get("teams", {})
    if not isinstance(teams, dict):
        teams = {}
    lineups = match_payload.get("lineups", {})
    if not isinstance(lineups, dict):
        lineups = {}
    sidelined = lineups.get("sidelined", {})
    if not isinstance(sidelined, dict):
        sidelined = {}

    rows = []
    for side in ["home", "away"]:
        side_team = teams.get(side, {})
        team_name = side_team.get("name") if isinstance(side_team, dict) else None
        for item in sidelined.get(side, []) or []:
            player = item.get("player", {})
            if not isinstance(player, dict):
                continue
            rows.append(
                {
                    "match_id": match_payload.get("id"),
                    "team": team_name,
                    "player": player.get("name"),
                    "status": _normalize_soccerdata_status(item.get("status")),
                    "reason": item.get("desc", ""),
                    "source_url": "https://soccerdataapi.com/docs/",
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=["match_id", "team", "player", "status", "reason", "source_url"]
        )
    for column in ["team", "player", "status"]:
        if frame[column].isna().any() or (frame[column].astype("string").str.strip() == "").any():
            raise ValueError(f"SoccerDataAPI sidelined rows contain missing {column!r}.")
    return frame.sort_values(["team", "player"]).reset_index(drop=True)


def resolve_sidelined_to_squad_availability(
    sidelined: pd.DataFrame,
    squads: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve SoccerDataAPI sidelined rows to canonical World Cup squad names."""

    if sidelined.empty:
        return (
            pd.DataFrame(
                columns=[
                    "team",
                    "player",
                    "status",
                    "reason",
                    "source_url",
                    "source_player",
                    "match_id",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "match_id",
                    "team",
                    "player",
                    "status",
                    "reason",
                    "source_url",
                ]
            ),
        )

    required_sidelined = {"match_id", "team", "player", "status", "reason", "source_url"}
    missing_sidelined = required_sidelined.difference(sidelined.columns)
    if missing_sidelined:
        raise ValueError(f"Missing sidelined columns: {sorted(missing_sidelined)}")

    required_squads = {"team", "player"}
    missing_squads = required_squads.difference(squads.columns)
    if missing_squads:
        raise ValueError(f"Missing squad columns: {sorted(missing_squads)}")

    squad_lookup = squads.loc[:, ["team", "player"]].copy()
    squad_lookup["team_key"] = squad_lookup["team"].map(normalize_name)
    squad_lookup["player_key"] = squad_lookup["player"].map(normalize_name)
    squad_lookup["initial_last_key"] = squad_lookup["player"].map(_initial_last_key)

    matched_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    for row in sidelined.itertuples(index=False):
        team_key = normalize_name(row.team)
        player_key = normalize_name(row.player)
        candidates = squad_lookup[squad_lookup["team_key"] == team_key].copy()
        exact = candidates[candidates["player_key"] == player_key]
        if len(exact) == 1:
            match = exact.iloc[0]
        else:
            initial_last = _initial_last_key(row.player)
            abbreviated = candidates[
                (candidates["initial_last_key"] == initial_last) & (initial_last != "")
            ]
            match = abbreviated.iloc[0] if len(abbreviated) == 1 else None

        if match is None:
            unmatched_rows.append(row._asdict())
            continue
        matched_rows.append(
            {
                "team": match["team"],
                "player": match["player"],
                "status": row.status,
                "reason": row.reason,
                "source_url": row.source_url,
                "source_player": row.player,
                "match_id": row.match_id,
            }
        )

    matched = pd.DataFrame(matched_rows)
    unmatched = pd.DataFrame(unmatched_rows)
    if matched.empty:
        matched = pd.DataFrame(
            columns=[
                "team",
                "player",
                "status",
                "reason",
                "source_url",
                "source_player",
                "match_id",
            ]
        )
    if unmatched.empty:
        unmatched = pd.DataFrame(
            columns=["match_id", "team", "player", "status", "reason", "source_url"]
        )
    return matched.sort_values(["team", "player"]).reset_index(drop=True), unmatched.reset_index(
        drop=True
    )


def _normalize_soccerdata_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status in {"out", "injured", "suspended", "doubtful"}:
        return status
    if status in {"questionable", "uncertain"}:
        return "doubtful"
    return "out"


def _initial_last_key(player_name: Any) -> str:
    parts = normalize_name(player_name).split()
    if len(parts) < 2:
        return ""
    return f"{parts[0][0]} {parts[-1]}"


def _cache_key(endpoint: str, params: dict[str, Any]) -> str:
    public_params = {key: params[key] for key in sorted(params)}
    raw = json.dumps(
        {"endpoint": endpoint.strip("/"), "params": public_params},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    endpoint_label = endpoint.strip("/").replace("/", "_") or "root"
    return f"{endpoint_label}_{digest}"
