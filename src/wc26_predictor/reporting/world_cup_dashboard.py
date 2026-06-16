"""Generate a multi-page static World Cup dashboard."""

# ruff: noqa: E501

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd

from wc26_predictor.reporting.prediction_history import PLAYED_CHECKS_FILE


@dataclass(frozen=True, slots=True)
class WorldCupDashboardPaths:
    """Generated multi-page dashboard paths."""

    index: Path
    future_matches: Path
    group_stage: Path
    next_phases: Path


def generate_world_cup_dashboard(
    project_root: str | Path,
    index_path: str | Path | None = None,
    dashboard_dir: str | Path | None = None,
    as_of: date | None = None,
) -> WorldCupDashboardPaths:
    """Generate split dashboard pages from processed and report artifacts."""

    root = Path(project_root)
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    pages_dir = Path(dashboard_dir) if dashboard_dir is not None else reports_dir / "dashboard"
    pages_dir.mkdir(parents=True, exist_ok=True)
    root_index = (
        Path(index_path)
        if index_path is not None
        else reports_dir / "model_performance_dashboard.html"
    )
    dashboard_date = as_of or date.today()

    match_details = _read_csv(processed_dir / "world_cup_2026_upcoming_match_details.csv")
    drivers = _read_optional_csv(processed_dir / "world_cup_2026_match_prediction_drivers.csv")
    impacts = _read_optional_csv(processed_dir / "world_cup_2026_match_final_stage_impacts.csv")
    played_checks = _read_optional_csv(processed_dir / PLAYED_CHECKS_FILE)
    group_advancement = _read_optional_csv(
        reports_dir / "final_group_advancement_probabilities.csv"
    )
    round_by_round = _read_optional_csv(reports_dir / "final_round_by_round_probabilities.csv")
    winners = _read_optional_csv(reports_dir / "final_winner_probabilities.csv")
    knockout = _read_optional_csv(reports_dir / "final_knockout_match_forecasts.csv")

    future_path = pages_dir / "future_matches.html"
    group_path = pages_dir / "group_stage.html"
    phases_path = pages_dir / "next_phases.html"
    performance_path = pages_dir / "model_performance.html"

    root_index.write_text(
        _index_html(
            match_details=match_details,
            played_checks=played_checks,
            as_of=dashboard_date,
        ),
        encoding="utf-8",
    )
    future_path.write_text(
        _page(
            title="Future Match Predictions",
            active="future",
            body=_future_matches_section(match_details, drivers, impacts, dashboard_date),
        ),
        encoding="utf-8",
    )
    group_path.write_text(
        _page(
            title="Group-Stage Pools",
            active="groups",
            body=_group_stage_section(
                match_details=match_details,
                group_advancement=group_advancement,
                played_checks=played_checks,
                as_of=dashboard_date,
            ),
        ),
        encoding="utf-8",
    )
    phases_path.write_text(
        _page(
            title="Next-Phase Predictions",
            active="phases",
            body=_next_phases_section(winners, round_by_round, knockout),
        ),
        encoding="utf-8",
    )
    if not performance_path.exists():
        performance_path.write_text(
            _page(
                title="Model Performance",
                active="performance",
                body=(
                    '<section class="panel"><h2>Model performance diagnostics</h2>'
                    "<p>Run <code>scripts/generate_validation_dashboard.py</code> to populate "
                    "the performance diagnostics page.</p></section>"
                ),
            ),
            encoding="utf-8",
        )
    return WorldCupDashboardPaths(
        index=root_index,
        future_matches=future_path,
        group_stage=group_path,
        next_phases=phases_path,
    )


def _index_html(
    match_details: pd.DataFrame,
    played_checks: pd.DataFrame | None,
    as_of: date,
) -> str:
    completed = _completed_matches(match_details)
    future = _future_matches(match_details, as_of)
    pending_past = _pending_past_matches(match_details, as_of)
    played_count = len(completed)
    checked_count = 0 if played_checks is None else len(played_checks)
    body = (
        '<section class="hero">'
        "<p>World Cup 2026 predictor</p>"
        "<h1>Forecast dashboard</h1>"
        f"<span>Updated view as of {escape(as_of.isoformat())}</span>"
        "</section>"
        '<section class="kpi-grid">'
        f"{_kpi('Upcoming fixtures', str(len(future)), 'future predictions shown separately')}"
        f"{_kpi('Played fixtures', str(played_count), f'{checked_count} with archived prediction checks')}"
        f"{_kpi('Awaiting results', str(len(pending_past)), 'past fixtures not yet in downloaded results')}"
        "</section>"
        '<section class="route-grid">'
        f"{_route_card('Future match predictions', 'dashboard/future_matches.html', 'Next unplayed fixtures, score distributions, and model drivers.')}"
        f"{_route_card('Group-stage pools', 'dashboard/group_stage.html', 'Pool standings, played matches, and pre-match forecast checks.')}"
        f"{_route_card('Next-phase predictions', 'dashboard/next_phases.html', 'Winner probabilities, round-by-round probabilities, and likely knockout pairings.')}"
        f"{_route_card('Model performance', 'dashboard/model_performance.html', 'Historical validation, uncertainty diagnostics, and benchmarks.')}"
        "</section>"
    )
    return _page("World Cup 2026 Dashboard", active="home", body=body, root=True)


def _future_matches_section(
    match_details: pd.DataFrame,
    drivers: pd.DataFrame | None,
    impacts: pd.DataFrame | None,
    as_of: date,
    limit: int = 20,
) -> str:
    future = _future_matches(match_details, as_of).head(limit)
    if future.empty:
        return '<section class="panel"><h2>Future match predictions</h2><p>No future unplayed fixtures are available.</p></section>'

    driver_lookup = _frame_lookup(drivers, "match_number")
    impact_lookup = _series_lookup(impacts, "match_number")
    cards = [
        _match_card(
            row, driver_lookup.get(int(row.match_number)), impact_lookup.get(int(row.match_number))
        )
        for row in future.itertuples(index=False)
    ]
    return (
        '<section class="panel wide">'
        "<h2>Future match predictions</h2>"
        f"<p>Showing the next {len(future)} unplayed fixtures from {escape(as_of.isoformat())} onward.</p>"
        '<div class="forecast-list">' + "".join(cards) + "</div></section>"
    )


def _group_stage_section(
    match_details: pd.DataFrame,
    group_advancement: pd.DataFrame | None,
    played_checks: pd.DataFrame | None,
    as_of: date,
) -> str:
    standings = _group_standings(match_details, group_advancement)
    checks = played_checks if played_checks is not None else pd.DataFrame()
    completed = _completed_matches(match_details)
    pending = _pending_past_matches(match_details, as_of)
    cards = []
    for group_name, group_frame in standings.groupby("group", sort=True):
        cards.append(
            _group_card(
                group=str(group_name),
                standings=group_frame,
                completed=completed,
                checks=checks,
            )
        )
    return (
        '<section class="panel wide">'
        "<h2>Group-stage pools</h2>"
        "<p>Open a team row to inspect played matches and the archived prediction used for evaluation.</p>"
        '<div class="group-grid">'
        + "".join(cards)
        + "</div></section>"
        + _played_checks_panel(checks)
        + _pending_results_panel(pending)
    )


def _next_phases_section(
    winners: pd.DataFrame | None,
    round_by_round: pd.DataFrame | None,
    knockout: pd.DataFrame | None,
) -> str:
    return (
        '<section class="grid two">'
        + _table_panel(
            "Winner probabilities",
            _format_probability_table(
                winners,
                [
                    "team",
                    "group",
                    "winner_probability",
                    "final_probability",
                    "semi_final_probability",
                ],
                limit=16,
            ),
        )
        + _table_panel(
            "Round-by-round probabilities",
            _format_probability_table(
                round_by_round,
                [
                    "team",
                    "group",
                    "round_of_32_probability",
                    "round_of_16_probability",
                    "quarter_final_probability",
                    "final_probability",
                ],
                limit=24,
            ),
        )
        + "</section>"
        + '<section class="panel wide">'
        "<h2>Most likely knockout pairings</h2>" + _knockout_pairing_table(knockout) + "</section>"
    )


def _group_card(
    group: str,
    standings: pd.DataFrame,
    completed: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    rows = []
    for row in standings.itertuples(index=False):
        team_matches = completed[
            (completed["home_team"] == row.team) | (completed["away_team"] == row.team)
        ].copy()
        team_checks = (
            checks[(checks["home_team"] == row.team) | (checks["away_team"] == row.team)].copy()
            if not checks.empty
            else pd.DataFrame()
        )
        rows.append(
            '<details class="team-row">'
            "<summary>"
            f"<strong>{escape(str(row.team))}</strong>"
            f"<span>{int(row.played)}P</span>"
            f"<span>{int(row.points)} pts</span>"
            f"<span>GD {int(row.goal_difference):+d}</span>"
            f"<span>Adv {_fmt_pct(row.advance_probability)}</span>"
            "</summary>"
            f"{_team_match_list(team_matches, team_checks)}"
            "</details>"
        )
    return (
        '<article class="group-card">'
        f"<h3>Group {escape(group)}</h3>"
        '<div class="team-header"><span>Team</span><span>P</span><span>Pts</span><span>GD</span><span>Adv</span></div>'
        + "".join(rows)
        + "</article>"
    )


def _team_match_list(matches: pd.DataFrame, checks: pd.DataFrame) -> str:
    if matches.empty:
        return '<p class="muted inset">No downloaded played matches for this team yet.</p>'
    rows = []
    check_lookup = _series_lookup(checks, "match_number")
    for match in matches.sort_values(["date", "match_number"]).itertuples(index=False):
        check = check_lookup.get(int(match.match_number))
        prediction = (
            f"Predicted {check.predicted_score}; outcome {'hit' if bool(check.outcome_hit) else 'miss'}"
            if check is not None
            else "No archived pre-match prediction"
        )
        rows.append(
            "<li>"
            f"<strong>{escape(str(match.home_team))} {int(match.completed_home_score)}-"
            f"{int(match.completed_away_score)} {escape(str(match.away_team))}</strong>"
            f"<span>{escape(prediction)}</span>"
            "</li>"
        )
    return '<ul class="match-list">' + "".join(rows) + "</ul>"


def _played_checks_panel(checks: pd.DataFrame) -> str:
    if checks.empty:
        return (
            '<section class="panel wide"><h2>Past match prediction checks</h2>'
            "<p>No completed World Cup matches with archived pre-match predictions are available yet. "
            "After running the update workflow once before new results arrive, later completed matches "
            "will appear here with exact-score and outcome diagnostics.</p></section>"
        )
    summary = (
        '<section class="kpi-grid">'
        f"{_kpi('Exact score hits', f'{checks["exact_score_hit"].mean():.1%}', 'archived prediction vs result')}"
        f"{_kpi('Outcome hits', f'{checks["outcome_hit"].mean():.1%}', 'home/draw/away direction')}"
        f"{_kpi('Goal MAE', f'{checks["goal_abs_error"].mean():.2f}', 'average team-goal absolute error')}"
        "</section>"
    )
    return (
        '<section class="panel wide"><h2>Past match prediction checks</h2>'
        + summary
        + _checks_table(checks)
        + "</section>"
    )


def _pending_results_panel(pending: pd.DataFrame) -> str:
    if pending.empty:
        return ""
    rows = []
    for row in pending.sort_values(["date", "match_number"]).itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{int(row.match_number)}</td><td>{escape(str(row.group))}</td>"
            f"<td>{escape(str(row.date))}</td><td>{escape(str(row.home_team))}</td>"
            f"<td>{escape(str(row.away_team))}</td><td>{escape(str(row.predicted_score))}</td>"
            "</tr>"
        )
    return (
        '<section class="panel wide"><h2>Past fixtures awaiting results</h2>'
        "<p>These fixtures are before the dashboard date but are not present in the downloaded completed-results file yet.</p>"
        '<div class="table-wrap"><table><thead><tr><th>#</th><th>Group</th><th>Date</th><th>Home</th><th>Away</th><th>Last prediction</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def _match_card(row: object, drivers: pd.DataFrame | None, impact: pd.Series | None) -> str:
    return (
        '<details class="match-card">'
        "<summary>"
        '<span class="match-main">'
        f"<strong>{escape(str(row.home_team))} vs {escape(str(row.away_team))}</strong>"
        f"<small>Match {int(row.match_number)} · Group {escape(str(row.group))} · "
        f"{escape(str(row.date))} {escape(str(row.time_local))}</small>"
        "</span>"
        f'<span class="score-pill">{escape(str(row.predicted_score))}'
        f"<small>{float(row.predicted_score_probability):.1%}</small></span>"
        "</summary>"
        '<div class="match-detail">'
        f"{_match_metrics(row, impact)}"
        f"{_driver_bars(drivers)}"
        "</div>"
        "</details>"
    )


def _match_metrics(row: object, impact: pd.Series | None) -> str:
    impact_text = "n/a" if impact is None else f"{float(impact.total_final_stage_impact):.3f}"
    items = [
        (
            "Expected goals",
            f"{float(row.home_expected_goals):.2f} - {float(row.away_expected_goals):.2f}",
        ),
        ("Home win", f"{float(row.home_win_probability):.1%}"),
        ("Draw", f"{float(row.draw_probability):.1%}"),
        ("Away win", f"{float(row.away_win_probability):.1%}"),
        ("Top scorelines", str(row.top_scorelines)),
        ("Final-stage impact", impact_text),
        ("Venue", f"{row.stadium}, {row.city}"),
    ]
    return (
        '<div class="match-metrics">'
        + "".join(
            '<div class="metric-cell">'
            f"<span>{escape(label)}</span><strong>{escape(value)}</strong>"
            "</div>"
            for label, value in items
        )
        + "</div>"
    )


def _driver_bars(drivers: pd.DataFrame | None) -> str:
    if drivers is None or drivers.empty:
        return '<p class="muted">No driver decomposition available.</p>'
    display = drivers[drivers["driver"] != "base_goal_environment"].copy()
    display = display.sort_values("abs_contribution_log_expected_goals", ascending=False).head(8)
    max_abs = max(float(display["contribution_log_expected_goals"].abs().max()), 1e-9)
    rows = ["<h3>Main score drivers</h3>"]
    for row in display.itertuples(index=False):
        contribution = float(row.contribution_log_expected_goals)
        width = abs(contribution) / max_abs * 100.0
        side_class = "positive" if contribution >= 0 else "negative"
        rows.append(
            '<div class="driver-row">'
            '<div class="driver-label">'
            f"<strong>{escape(str(row.team))}</strong>"
            f"<span>{escape(str(row.driver).replace('_', ' '))}</span>"
            "</div>"
            '<div class="driver-track">'
            f'<span class="driver-fill {side_class}" style="width: {width:.1f}%"></span>'
            "</div>"
            f"<small>{contribution:+.3f} · x{float(row.multiplier):.2f}</small>"
            "</div>"
        )
    return '<div class="driver-list">' + "".join(rows) + "</div>"


def _group_standings(
    match_details: pd.DataFrame,
    group_advancement: pd.DataFrame | None,
) -> pd.DataFrame:
    teams = []
    for row in match_details.itertuples(index=False):
        teams.append({"group": row.group, "team": row.home_team})
        teams.append({"group": row.group, "team": row.away_team})
    standings = pd.DataFrame(teams).drop_duplicates().sort_values(["group", "team"])
    for column in ["played", "wins", "draws", "losses", "goals_for", "goals_against", "points"]:
        standings[column] = 0
    completed = _completed_matches(match_details)
    for match in completed.itertuples(index=False):
        _apply_result(
            standings,
            match.home_team,
            int(match.completed_home_score),
            int(match.completed_away_score),
        )
        _apply_result(
            standings,
            match.away_team,
            int(match.completed_away_score),
            int(match.completed_home_score),
        )
    standings["goal_difference"] = standings["goals_for"] - standings["goals_against"]
    if group_advancement is not None and not group_advancement.empty:
        standings = standings.merge(
            group_advancement.loc[:, ["team", "advance_probability"]],
            on="team",
            how="left",
        )
    else:
        standings["advance_probability"] = 0.0
    standings["advance_probability"] = standings["advance_probability"].fillna(0.0)
    return standings.sort_values(
        ["group", "points", "goal_difference", "goals_for", "advance_probability", "team"],
        ascending=[True, False, False, False, False, True],
        ignore_index=True,
    )


def _apply_result(standings: pd.DataFrame, team: str, goals_for: int, goals_against: int) -> None:
    selector = standings["team"] == team
    standings.loc[selector, "played"] += 1
    standings.loc[selector, "goals_for"] += goals_for
    standings.loc[selector, "goals_against"] += goals_against
    if goals_for > goals_against:
        standings.loc[selector, "wins"] += 1
        standings.loc[selector, "points"] += 3
    elif goals_for == goals_against:
        standings.loc[selector, "draws"] += 1
        standings.loc[selector, "points"] += 1
    else:
        standings.loc[selector, "losses"] += 1


def _checks_table(checks: pd.DataFrame) -> str:
    rows = []
    for row in checks.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{int(row.match_number)}</td><td>{escape(str(row.group))}</td>"
            f"<td>{escape(str(row.home_team))}</td><td>{escape(str(row.away_team))}</td>"
            f"<td>{escape(str(row.observed_score))}</td><td>{escape(str(row.predicted_score))}</td>"
            f"<td>{'yes' if bool(row.exact_score_hit) else 'no'}</td>"
            f"<td>{'yes' if bool(row.outcome_hit) else 'no'}</td>"
            f"<td>{float(row.goal_abs_error):.2f}</td>"
            f"<td>{float(row.outcome_log_loss):.3f}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>#</th><th>Group</th><th>Home</th>'
        "<th>Away</th><th>Observed</th><th>Predicted</th><th>Exact</th><th>Outcome</th>"
        f"<th>Goal MAE</th><th>Outcome LL</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _knockout_pairing_table(knockout: pd.DataFrame | None) -> str:
    if knockout is None or knockout.empty:
        return "<p>No knockout pairing forecasts available.</p>"
    top = knockout.sort_values(["match_number", "pairing_probability"], ascending=[True, False])
    top = top.groupby("match_number", as_index=False).head(1)
    rows = []
    for row in top.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{int(row.match_number)}</td><td>{escape(str(row.round))}</td>"
            f"<td>{escape(str(row.first_team))}</td><td>{escape(str(row.second_team))}</td>"
            f"<td>{float(row.pairing_probability):.1%}</td>"
            f"<td>{float(row.first_advancement_probability):.1%}</td>"
            f"<td>{float(row.second_advancement_probability):.1%}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>#</th><th>Round</th><th>First</th>'
        "<th>Second</th><th>Pairing</th><th>First advances</th><th>Second advances</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _format_probability_table(
    frame: pd.DataFrame | None,
    columns: list[str],
    limit: int,
) -> str:
    if frame is None or frame.empty:
        return "<p>No data available.</p>"
    display = (
        frame.loc[:, [column for column in columns if column in frame.columns]].head(limit).copy()
    )
    for column in display.columns:
        if column.endswith("_probability"):
            display[column] = display[column].map(_fmt_pct)
    rows = []
    for row in display.itertuples(index=False):
        rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>")
    headers = "".join(f"<th>{escape(column)}</th>" for column in display.columns)
    return f'<div class="table-wrap"><table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _table_panel(title: str, table_html: str) -> str:
    return f'<section class="panel"><h2>{escape(title)}</h2>{table_html}</section>'


def _kpi(title: str, value: str, detail: str) -> str:
    return (
        '<article class="kpi">'
        f"<span>{escape(title)}</span><strong>{escape(value)}</strong><small>{escape(detail)}</small>"
        "</article>"
    )


def _route_card(title: str, href: str, detail: str) -> str:
    return (
        f'<a class="route-card" href="{escape(href)}">'
        f"<strong>{escape(title)}</strong><span>{escape(detail)}</span>"
        "</a>"
    )


def _page(title: str, active: str, body: str, root: bool = False) -> str:
    prefix = "dashboard/" if root else ""
    home_href = (
        "model_performance_dashboard.html" if root else "../model_performance_dashboard.html"
    )
    nav = [
        ("home", home_href, "Home"),
        ("future", f"{prefix}future_matches.html", "Future matches"),
        ("groups", f"{prefix}group_stage.html", "Group stage"),
        ("phases", f"{prefix}next_phases.html", "Next phases"),
        ("performance", f"{prefix}model_performance.html", "Model performance"),
    ]
    links = "".join(
        f'<a class="{"active" if key == active else ""}" href="{escape(href)}">{escape(label)}</a>'
        for key, href, label in nav
    )
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(title)}</title>",
            f"<style>{_css()}</style>",
            "</head>",
            "<body>",
            '<main class="page">',
            f"<nav>{links}</nav>",
            body,
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _future_matches(match_details: pd.DataFrame, as_of: date) -> pd.DataFrame:
    frame = match_details.copy()
    frame["date_sort"] = pd.to_datetime(frame["date"], errors="raise").dt.date
    mask = (~frame["is_completed"].map(_parse_boolish)) & (frame["date_sort"] >= as_of)
    return frame[mask].sort_values(["date_sort", "match_number"], ignore_index=True)


def _pending_past_matches(match_details: pd.DataFrame, as_of: date) -> pd.DataFrame:
    frame = match_details.copy()
    frame["date_sort"] = pd.to_datetime(frame["date"], errors="raise").dt.date
    mask = (~frame["is_completed"].map(_parse_boolish)) & (frame["date_sort"] < as_of)
    return frame[mask].sort_values(["date_sort", "match_number"], ignore_index=True)


def _completed_matches(match_details: pd.DataFrame) -> pd.DataFrame:
    return match_details[match_details["is_completed"].map(_parse_boolish)].copy()


def _frame_lookup(frame: pd.DataFrame | None, key: str) -> dict[int, pd.DataFrame]:
    if frame is None or frame.empty:
        return {}
    return {int(value): group.copy() for value, group in frame.groupby(key, sort=False)}


def _series_lookup(frame: pd.DataFrame | None, key: str) -> dict[int, pd.Series]:
    if frame is None or frame.empty:
        return {}
    return {
        int(getattr(row, key)): pd.Series(row._asdict()) for row in frame.itertuples(index=False)
    }


def _fmt_pct(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def _parse_boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"Required dashboard input does not exist: {path}")
    return pd.read_csv(path)


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f4f1ea;
  --panel: #fffdf8;
  --ink: #17201f;
  --muted: #66706b;
  --line: #ded7ca;
  --blue: #1f5fbf;
  --green: #2f6f73;
  --red: #b75d3b;
  --shadow: 0 14px 38px rgba(45, 38, 24, 0.10);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.page { width: min(1440px, 100%); margin: 0 auto; padding: 26px; }
nav {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 22px;
}
nav a, .route-card {
  color: var(--ink);
  text-decoration: none;
}
nav a {
  padding: 9px 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}
nav a.active { color: #fff; background: var(--blue); border-color: var(--blue); }
.hero { padding: 42px 0 30px; border-bottom: 1px solid var(--line); }
.hero p { margin: 0 0 8px; color: var(--blue); font-weight: 800; text-transform: uppercase; }
.hero h1 { margin: 0; font-size: clamp(2.4rem, 6vw, 5.8rem); line-height: 0.95; }
.hero span { display: block; margin-top: 14px; color: var(--muted); }
.kpi-grid, .route-grid, .grid, .forecast-list, .group-grid {
  display: grid;
  gap: 16px;
  margin-top: 18px;
}
.kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.route-grid, .grid.two, .forecast-list { grid-template-columns: repeat(2, minmax(0, 1fr)); align-items: start; }
.group-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); align-items: start; }
.panel, .kpi, .route-card, .match-card, .group-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}
.panel, .kpi, .route-card, .group-card { padding: 20px; }
.panel.wide { margin-top: 18px; }
.route-card strong, .route-card span, .kpi span, .kpi strong, .kpi small { display: block; }
.route-card strong, .kpi strong { font-size: 1.35rem; margin: 8px 0; }
.route-card span, .kpi span, .kpi small, .muted, .panel p { color: var(--muted); }
h2, h3 { margin: 0 0 10px; }
.match-card { overflow: hidden; }
.match-card summary {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 14px;
  align-items: center;
  padding: 16px;
  cursor: pointer;
  list-style: none;
}
.match-card summary::-webkit-details-marker, .team-row summary::-webkit-details-marker { display: none; }
.match-main strong, .match-main small, .score-pill, .score-pill small { display: block; }
.match-main small, .score-pill small { margin-top: 4px; color: var(--muted); font-size: 0.78rem; }
.score-pill {
  min-width: 78px;
  padding: 9px;
  border: 1px solid var(--line);
  border-radius: 8px;
  text-align: center;
  font-weight: 800;
  background: #f7f2e8;
}
.match-detail { padding: 0 16px 16px; }
.match-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin-bottom: 14px;
}
.metric-cell { padding: 10px; border: 1px solid var(--line); border-radius: 8px; }
.metric-cell span { display: block; color: var(--muted); font-size: 0.74rem; font-weight: 700; }
.metric-cell strong { display: block; margin-top: 4px; overflow-wrap: anywhere; font-size: 0.88rem; }
.driver-row {
  display: grid;
  grid-template-columns: minmax(150px, 0.9fr) minmax(120px, 1fr) auto;
  gap: 10px;
  align-items: center;
  padding: 8px 0;
  border-top: 1px solid var(--line);
}
.driver-label strong, .driver-label span { display: block; }
.driver-label span, .driver-row small { color: var(--muted); font-size: 0.76rem; }
.driver-track { height: 9px; overflow: hidden; border-radius: 999px; background: #ebe4d8; }
.driver-fill { display: block; height: 100%; border-radius: 999px; }
.driver-fill.positive { background: var(--green); }
.driver-fill.negative { background: var(--red); }
.team-header, .team-row summary {
  display: grid;
  grid-template-columns: minmax(150px, 1fr) 42px 52px 52px 64px;
  gap: 8px;
  align-items: center;
}
.team-header { color: var(--muted); font-size: 0.76rem; font-weight: 800; padding-bottom: 8px; }
.team-row { border-top: 1px solid var(--line); }
.team-row summary { padding: 10px 0; cursor: pointer; }
.inset { margin: 0 0 10px 0; padding: 0 0 8px; }
.match-list { margin: 0 0 12px; padding-left: 18px; color: var(--muted); }
.match-list strong, .match-list span { display: block; color: var(--ink); }
.match-list span { color: var(--muted); font-size: 0.86rem; }
.table-wrap { max-height: 520px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
th { position: sticky; top: 0; background: #f7f2e8; z-index: 1; }
code { background: #f7f2e8; padding: 2px 5px; border-radius: 5px; }
@media (max-width: 1000px) {
  .page { padding: 18px; }
  .kpi-grid, .route-grid, .grid.two, .forecast-list, .group-grid, .match-metrics {
    grid-template-columns: 1fr;
  }
  .driver-row { grid-template-columns: 1fr; }
}
"""
