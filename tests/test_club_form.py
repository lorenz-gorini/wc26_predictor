from __future__ import annotations

import pandas as pd

from wc26_predictor.data.club_form import enrich_top_scorers_with_club_form, parse_wikipedia_top_scorers


def test_parse_wikipedia_top_scorers_extracts_goals_and_minutes() -> None:
    html = """
    <table>
      <tr><th>Rank</th><th>Player</th><th>Club</th><th>Goals</th><th>Minutes played</th></tr>
      <tr><td>1</td><td>Example Player</td><td>Example Club</td><td>12</td><td>900</td></tr>
    </table>
    """

    parsed = parse_wikipedia_top_scorers(html, "Example League", "https://example.com")

    assert parsed.loc[0, "player"] == "Example Player"
    assert parsed.loc[0, "club"] == "Example Club"
    assert parsed.loc[0, "goals"] == 12
    assert parsed.loc[0, "minutes"] == 900


def test_enrich_top_scorers_with_club_form_prefers_player_and_club_match() -> None:
    top_scorers = pd.DataFrame(
        {
            "team": ["Norway"],
            "scorer": ["Erling Haaland"],
            "expected_tournament_goals": [3.0],
        }
    )
    squads = pd.DataFrame(
        {
            "team": ["Norway"],
            "player": ["Erling Haaland"],
            "position": ["FW"],
            "caps": [50],
            "goals": [45],
            "club": ["Manchester City"],
        }
    )
    club_form = pd.DataFrame(
        {
            "competition": ["Premier League"],
            "source_url": ["https://example.com"],
            "player": ["Erling Haaland"],
            "club": ["Manchester City"],
            "goals": [26],
            "minutes": [2400],
        }
    )

    enriched = enrich_top_scorers_with_club_form(top_scorers, squads, club_form)

    assert enriched.loc[0, "club_form_match_quality"] == "player_and_club"
    assert enriched.loc[0, "club_form_goals"] == 26
    assert enriched.loc[0, "club_form_goals_per90"] > 0

