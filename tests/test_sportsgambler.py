from __future__ import annotations

import pandas as pd

from wc26_predictor.data.sportsgambler import (
    SportsgamblerConfig,
    match_sportsgambler_injuries_to_squads,
    parse_sportsgambler_football_injuries,
)

HTML = """
<article>
  <h3>Aston Villa</h3>
  <div class="inj-container inj-titles"></div>
  <div class="inj-row">
    <div class="inj-container">
      <span class="inj-type injury-plus"></span>
      <span class="inj-player">Emiliano Martinez</span>
      <span class="inj-position h-sm">G</span>
      <span class="inj-game h-sm">32</span>
      <span class="inj-goals h-sm">-</span>
      <span class="inj-assist h-sm">1</span>
      <span class="inj-info">Wrist</span>
      <span class="inj-return h-sm">-</span>
    </div>
  </div>
  <h3>Man City</h3>
  <div class="inj-row">
    <div class="inj-container">
      <span class="inj-type redcard"></span>
      <span class="inj-player">Mateo Kovacic</span>
      <span class="inj-position h-sm">M</span>
      <span class="inj-game h-sm">6</span>
      <span class="inj-goals h-sm">-</span>
      <span class="inj-assist h-sm">1</span>
      <span class="inj-info">Red card (direct)</span>
      <span class="inj-return h-sm">-</span>
    </div>
  </div>
</article>
"""


def test_parse_sportsgambler_football_injuries() -> None:
    injuries = parse_sportsgambler_football_injuries(HTML, "https://example.test")

    assert len(injuries) == 2
    assert set(injuries["player"]) == {"Emiliano Martinez", "Mateo Kovacic"}
    emiliano_type = injuries.loc[
        injuries["player"] == "Emiliano Martinez",
        "type_class",
    ].iloc[0]
    assert "injury-plus" in emiliano_type


def test_match_sportsgambler_injuries_excludes_club_suspensions_by_default() -> None:
    injuries = parse_sportsgambler_football_injuries(HTML, "https://example.test")
    squads = pd.DataFrame(
        {
            "team": ["Argentina", "Croatia", "Uruguay"],
            "player": ["Emiliano Martínez", "Mateo Kovačić", "Emiliano Martínez"],
            "club": ["Aston Villa", "Manchester City", "Palmeiras"],
            "position": ["GK", "MF", "MF"],
        }
    )

    matched, overrides = match_sportsgambler_injuries_to_squads(
        injuries,
        squads,
        SportsgamblerConfig(),
    )

    assert len(matched) == 3
    assert not bool(matched.loc[matched["team"] == "Uruguay", "club_match"].iloc[0])
    assert overrides["player"].tolist() == ["Emiliano Martínez"]
    assert overrides.loc[0, "status"] == "injured"
