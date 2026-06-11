"""Final forecast report generation for the World Cup 2026 project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FinalReportConfig:
    """Configuration for deterministic final report generation."""

    top_scorer_simulations: int = 100_000
    random_seed: int = 2026


@dataclass(frozen=True, slots=True)
class FinalReportPaths:
    """Paths to the generated report artifacts."""

    group_advancement: Path
    round_by_round: Path
    winners: Path
    top_scorers: Path
    group_matches: Path
    knockout_matches: Path
    full_matches: Path
    markdown_report: Path


def generate_final_reports(
    processed_dir: Path,
    reports_dir: Path,
    config: FinalReportConfig | None = None,
) -> FinalReportPaths:
    """Generate final CSV tables and a compact Markdown report.

    The function expects the baseline workflow to have already written its processed
    outputs. Missing required inputs or columns raise ``ValueError`` instead of producing
    partial reports from stale or malformed data.
    """

    report_config = config or FinalReportConfig()
    processed_dir = Path(processed_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    tournament = _read_csv(
        processed_dir / "world_cup_2026_official_tournament_probabilities.csv"
    )
    group_forecasts = _read_csv(processed_dir / "world_cup_2026_baseline_forecasts.csv")
    knockout_forecasts = _read_csv(
        processed_dir / "world_cup_2026_knockout_match_forecasts.csv"
    )
    full_forecasts = _read_csv(processed_dir / "world_cup_2026_full_match_forecasts.csv")
    top_scorers = _read_csv(
        processed_dir / "world_cup_2026_top_scorer_transfermarkt_adjusted_top100.csv"
    )

    group_map = _group_membership_from_forecasts(group_forecasts)
    tournament_with_groups = tournament.merge(group_map, on="team", how="left")
    if tournament_with_groups["group"].isna().any():
        missing = sorted(tournament_with_groups.loc[tournament_with_groups["group"].isna(), "team"])
        raise ValueError(f"Missing group labels for teams: {missing}")

    group_advancement = _build_group_advancement(tournament_with_groups)
    round_by_round = _build_round_by_round(tournament_with_groups)
    winners = _build_winner_probabilities(tournament_with_groups)
    top_scorer_probabilities = estimate_top_scorer_probabilities(
        top_scorers,
        n_simulations=report_config.top_scorer_simulations,
        seed=report_config.random_seed,
    )
    group_matches = _build_group_match_forecasts(group_forecasts)
    knockout_matches = _build_knockout_match_forecasts(knockout_forecasts)
    full_matches = _build_full_match_forecasts(full_forecasts)

    paths = FinalReportPaths(
        group_advancement=reports_dir / "final_group_advancement_probabilities.csv",
        round_by_round=reports_dir / "final_round_by_round_probabilities.csv",
        winners=reports_dir / "final_winner_probabilities.csv",
        top_scorers=reports_dir / "final_top_scorer_probabilities.csv",
        group_matches=reports_dir / "final_group_match_forecasts.csv",
        knockout_matches=reports_dir / "final_knockout_match_forecasts.csv",
        full_matches=reports_dir / "final_full_match_forecasts.csv",
        markdown_report=reports_dir / "final_report.md",
    )

    group_advancement.to_csv(paths.group_advancement, index=False)
    round_by_round.to_csv(paths.round_by_round, index=False)
    winners.to_csv(paths.winners, index=False)
    top_scorer_probabilities.to_csv(paths.top_scorers, index=False)
    group_matches.to_csv(paths.group_matches, index=False)
    knockout_matches.to_csv(paths.knockout_matches, index=False)
    full_matches.to_csv(paths.full_matches, index=False)
    paths.markdown_report.write_text(
        _build_markdown_report(
            group_advancement=group_advancement,
            round_by_round=round_by_round,
            winners=winners,
            top_scorers=top_scorer_probabilities,
            group_matches=group_matches,
            knockout_matches=knockout_matches,
            processed_dir=processed_dir,
            config=report_config,
        ),
        encoding="utf-8",
    )
    return paths


def estimate_top_scorer_probabilities(
    top_scorers: pd.DataFrame,
    n_simulations: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate tournament top-scorer probabilities from expected goals.

    The approximation treats each top-100 player's tournament goals as an independent
    Poisson random variable and splits tied top-scorer outcomes evenly.
    """

    required = {
        "team",
        "scorer",
        "club",
        "expected_tournament_goals_transfermarkt_adjusted",
        "expected_tournament_goals",
        "transfermarkt_match_quality",
    }
    _require_columns(top_scorers, required, "top scorer table")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    scorers = top_scorers.copy()
    scorers["expected_tournament_goals"] = scorers[
        "expected_tournament_goals_transfermarkt_adjusted"
    ].astype(float)
    lambdas = scorers["expected_tournament_goals"].clip(lower=0.0).to_numpy(dtype=float)
    if not np.isfinite(lambdas).all():
        raise ValueError("Expected top-scorer goals contain non-finite values.")

    rng = np.random.default_rng(seed)
    top_counts = np.zeros(len(scorers), dtype=float)
    batch_size = min(10_000, n_simulations)
    completed = 0
    while completed < n_simulations:
        current_batch = min(batch_size, n_simulations - completed)
        samples = rng.poisson(lambdas, size=(current_batch, len(lambdas)))
        maxima = samples.max(axis=1)
        tied_winners = samples == maxima[:, None]
        top_counts += (tied_winners / tied_winners.sum(axis=1)[:, None]).sum(axis=0)
        completed += current_batch

    scorers["top_scorer_probability"] = top_counts / n_simulations
    output_columns = [
        "team",
        "scorer",
        "club",
        "expected_tournament_goals",
        "top_scorer_probability",
        "transfermarkt_match_quality",
    ]
    optional_columns = ["status", "availability_multiplier", "transfermarkt_multiplier"]
    output_columns.extend(column for column in optional_columns if column in scorers.columns)
    output = scorers.loc[:, output_columns].sort_values(
        ["top_scorer_probability", "expected_tournament_goals"],
        ascending=[False, False],
        ignore_index=True,
    )
    output.insert(0, "rank", np.arange(1, len(output) + 1))
    return output


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"Required processed file does not exist: {path}")
    return pd.read_csv(path)


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {label}: {sorted(missing)}")


def _group_membership_from_forecasts(group_forecasts: pd.DataFrame) -> pd.DataFrame:
    _require_columns(group_forecasts, {"group", "home_team", "away_team"}, "group forecasts")
    home = group_forecasts.loc[:, ["group", "home_team"]].rename(columns={"home_team": "team"})
    away = group_forecasts.loc[:, ["group", "away_team"]].rename(columns={"away_team": "team"})
    group_map = pd.concat([home, away], ignore_index=True).drop_duplicates()
    duplicated = group_map[group_map.duplicated("team", keep=False)]
    if not duplicated.empty:
        teams = sorted(duplicated["team"].unique())
        raise ValueError(f"Teams assigned to multiple groups: {teams}")
    return group_map.sort_values(["group", "team"], ignore_index=True)


def _build_group_advancement(tournament: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "group",
        "team",
        "round_of_32_probability",
        "expected_team_matches",
    ]
    _require_columns(tournament, set(columns), "tournament probabilities")
    output = tournament.loc[:, list(columns)].rename(
        columns={"round_of_32_probability": "advance_probability"}
    )
    return output.sort_values(
        ["group", "advance_probability", "expected_team_matches"],
        ascending=[True, False, False],
        ignore_index=True,
    )


def _build_round_by_round(tournament: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "team",
        "group",
        "round_of_32_probability",
        "round_of_16_probability",
        "quarter_final_probability",
        "semi_final_probability",
        "final_probability",
        "third_place_match_probability",
        "winner_probability",
        "expected_knockout_matches",
        "expected_team_matches",
    ]
    _require_columns(tournament, set(columns), "tournament probabilities")
    return tournament.loc[:, columns].sort_values(
        ["winner_probability", "final_probability"],
        ascending=[False, False],
        ignore_index=True,
    )


def _build_winner_probabilities(tournament: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "team",
        "group",
        "winner_probability",
        "final_probability",
        "semi_final_probability",
        "quarter_final_probability",
        "round_of_16_probability",
        "round_of_32_probability",
        "expected_team_matches",
    ]
    _require_columns(tournament, set(columns), "tournament probabilities")
    return tournament.loc[:, columns].sort_values(
        "winner_probability",
        ascending=False,
        ignore_index=True,
    )


def _build_group_match_forecasts(group_forecasts: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "match_number",
        "date",
        "time_local",
        "group",
        "home_team",
        "away_team",
        "ensemble_home_win",
        "ensemble_draw",
        "ensemble_away_win",
        "form_poisson_home_expected_goals",
        "form_poisson_away_expected_goals",
    ]
    _require_columns(group_forecasts, set(columns), "group forecasts")
    output = group_forecasts.loc[:, columns].copy()
    probabilities = output.loc[
        :,
        ["ensemble_home_win", "ensemble_draw", "ensemble_away_win"],
    ].to_numpy(dtype=float)
    outcomes = np.array(["home_win", "draw", "away_win"])
    output["most_likely_outcome"] = outcomes[probabilities.argmax(axis=1)]
    output["most_likely_probability"] = probabilities.max(axis=1)
    return output.sort_values("match_number", ignore_index=True)


def _build_knockout_match_forecasts(knockout_forecasts: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "match_number",
        "round",
        "first_team",
        "second_team",
        "pairing_probability",
        "first_advancement_probability",
        "second_advancement_probability",
        "simulation_count",
    ]
    _require_columns(knockout_forecasts, set(columns), "knockout forecasts")
    output = knockout_forecasts.loc[:, columns].copy()
    output["most_likely_advancing_team"] = np.where(
        output["first_advancement_probability"]
        >= output["second_advancement_probability"],
        output["first_team"],
        output["second_team"],
    )
    output["most_likely_advancement_probability"] = output[
        ["first_advancement_probability", "second_advancement_probability"]
    ].max(axis=1)
    return output.sort_values(
        ["match_number", "pairing_probability"],
        ascending=[True, False],
        ignore_index=True,
    )


def _build_full_match_forecasts(full_forecasts: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "match_number",
        "round",
        "group",
        "first_team",
        "second_team",
        "pairing_probability",
        "first_win_probability",
        "draw_probability",
        "second_win_probability",
        "first_advancement_probability",
        "second_advancement_probability",
        "simulation_count",
    ]
    _require_columns(full_forecasts, set(columns), "full match forecasts")
    return full_forecasts.loc[:, columns].sort_values(
        ["match_number", "pairing_probability"],
        ascending=[True, False],
        ignore_index=True,
    )


def _build_markdown_report(
    group_advancement: pd.DataFrame,
    round_by_round: pd.DataFrame,
    winners: pd.DataFrame,
    top_scorers: pd.DataFrame,
    group_matches: pd.DataFrame,
    knockout_matches: pd.DataFrame,
    processed_dir: Path,
    config: FinalReportConfig,
) -> str:
    market_note = _market_gate_note(processed_dir)
    availability_note = _availability_note(processed_dir)
    top_knockout_pairings = knockout_matches.groupby("match_number", as_index=False).head(1)
    decisive_group_matches = group_matches.sort_values(
        "most_likely_probability",
        ascending=False,
        ignore_index=True,
    )

    return "\n".join(
        [
            "# World Cup 2026 Final Forecast Report",
            "",
            "This report summarizes the current model outputs generated from the processed "
            "pipeline artifacts. Group advancement means probability of reaching the "
            "round of 32, not exact group-position probabilities.",
            "",
            f"Top-scorer probabilities use {config.top_scorer_simulations:,} independent "
            "Poisson simulations over the Transfermarkt-adjusted top-100 scorer list, "
            "with tied top-scorer outcomes split evenly.",
            "",
            market_note,
            availability_note,
            "",
            "## Winner probabilities",
            _markdown_table(_format_probability_columns(winners.head(15))),
            "",
            "## Group advancement probabilities",
            _markdown_table(_format_probability_columns(group_advancement)),
            "",
            "## Round-by-round probabilities",
            _markdown_table(_format_probability_columns(round_by_round.head(15))),
            "",
            "## Top-scorer probabilities",
            _markdown_table(_format_probability_columns(top_scorers.head(20))),
            "",
            "## Most decisive group matches",
            _markdown_table(_format_probability_columns(decisive_group_matches.head(15))),
            "",
            "## Most likely knockout pairings",
            _markdown_table(_format_probability_columns(top_knockout_pairings)),
            "",
            "## Generated files",
            "- `reports/final_group_advancement_probabilities.csv`",
            "- `reports/final_round_by_round_probabilities.csv`",
            "- `reports/final_winner_probabilities.csv`",
            "- `reports/final_top_scorer_probabilities.csv`",
            "- `reports/final_group_match_forecasts.csv`",
            "- `reports/final_knockout_match_forecasts.csv`",
            "- `reports/final_full_match_forecasts.csv`",
            "",
            "## Related validation review",
            "- `reports/model_validation_review.md`",
            "",
        ]
    )


def _market_gate_note(processed_dir: Path) -> str:
    path = processed_dir / "world_cup_market_gate_decision.csv"
    if not path.exists():
        return "Market benchmark: no market validation gate file was found."
    decision = pd.read_csv(path)
    _require_columns(
        decision,
        {
            "use_model_market",
            "reason",
            "model_log_loss",
            "market_log_loss",
            "model_market_log_loss",
        },
        "market gate decision",
    )
    row = decision.iloc[0]
    use_model_market = bool(row["use_model_market"])
    reason = str(row["reason"]).rstrip(".")
    return (
        "Market benchmark: "
        f"{'using' if use_model_market else 'not using'} model+market ensemble. "
        f"Reason: {reason}. "
        f"Log loss model={row['model_log_loss']:.3f}, "
        f"market={row['market_log_loss']:.3f}, "
        f"model+market={row['model_market_log_loss']:.3f}."
    )


def _availability_note(processed_dir: Path) -> str:
    path = processed_dir / "world_cup_2026_team_availability_burden.csv"
    if not path.exists():
        return "Availability: no team availability burden file was found."
    burden = pd.read_csv(path)
    _require_columns(
        burden,
        {"team", "team_availability_burden", "unavailable_players"},
        "team availability burden",
    )
    nonzero = burden[burden["team_availability_burden"] > 0].sort_values(
        "team_availability_burden",
        ascending=False,
    )
    if nonzero.empty:
        return "Availability: no nonzero current injury or suspension burden in the model."
    items = [
        (
            f"{row.team} ({row.team_availability_burden:.3f}, "
            f"{row.unavailable_players} {_player_label(int(row.unavailable_players))})"
        )
        for row in nonzero.itertuples(index=False)
    ]
    return "Availability: current nonzero burdens are " + "; ".join(items) + "."


def _format_probability_columns(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    probability_columns = [
        column
        for column in formatted.columns
        if column.endswith("_probability")
        or column.endswith("_win")
        or column.endswith("_draw")
        or column in {"advance_probability"}
    ]
    for column in probability_columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.1%}")
    float_columns = [
        column
        for column in formatted.select_dtypes(include=["floating"]).columns
        if column not in probability_columns
    ]
    for column in float_columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
    return formatted


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    text_frame = frame.astype(str)
    rows = [
        "| " + " | ".join(_escape_markdown_cell(column) for column in text_frame.columns) + " |",
        "| " + " | ".join("---" for _ in text_frame.columns) + " |",
    ]
    for row in text_frame.itertuples(index=False):
        rows.append("| " + " | ".join(_escape_markdown_cell(value) for value in row) + " |")
    return "\n".join(rows)


def _escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def _player_label(count: int) -> str:
    return "player" if count == 1 else "players"
