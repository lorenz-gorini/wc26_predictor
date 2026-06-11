#!/usr/bin/env python
"""Fit baseline models, run holdout backtests, and forecast 2026 fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wc26_predictor.data.availability import (
    adjust_fixture_forecasts_for_team_availability,
    aggregate_team_availability_burden,
    apply_availability_to_top_scorers,
    build_default_player_availability,
    load_player_availability_csv,
    merge_player_availability,
)
from wc26_predictor.data.club_form import (
    enrich_top_scorers_with_club_form,
    load_club_top_scorers_csv,
)
from wc26_predictor.data.goalscorers import load_known_goalscorers_csv
from wc26_predictor.data.ingest_results import load_results_csv
from wc26_predictor.data.odds import load_football_data_world_cup_odds
from wc26_predictor.data.squads import load_squads_csv
from wc26_predictor.data.transfermarkt import build_transfermarkt_top_scorer_features
from wc26_predictor.data.transfermarkt_injuries import (
    aggregate_transfermarkt_team_injury_burden,
    build_transfermarkt_squad_injury_features,
)
from wc26_predictor.models.availability_impact import estimate_goal_penalty_per_burden
from wc26_predictor.models.ensemble import add_weighted_ensemble_columns
from wc26_predictor.models.market import (
    evaluate_market_models,
    market_gate_decision,
    match_validation_predictions_to_odds,
)
from wc26_predictor.models.top_scorer import NationalTeamTopScorerModel
from wc26_predictor.pipelines.baselines import (
    build_full_match_forecast_table,
    build_pairwise_advancement_probabilities,
    collect_world_cup_validation_predictions,
    evaluate_world_cup_baselines,
    fit_world_cup_ensemble_weights,
    forecast_2026_group_fixtures,
    model_configs_to_frame,
    tune_baseline_model_configs,
    write_baseline_summary,
)
from wc26_predictor.simulation.official_tournament import (
    simulate_official_knockout_match_forecasts,
    simulate_official_tournament,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root directory.",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=2000,
        help="Monte Carlo simulations for tournament expected-match estimates.",
    )
    args = parser.parse_args()

    results_path = args.project_root / "data" / "processed" / "international_results.csv"
    fixtures_path = args.project_root / "data" / "raw" / "world_cup_2026_fixtures.csv"
    goalscorers_path = args.project_root / "data" / "raw" / "goalscorers.csv"
    squads_path = args.project_root / "data" / "raw" / "world_cup_2026_squads.csv"
    availability_path = (
        args.project_root / "data" / "raw" / "world_cup_2026_player_availability.csv"
    )
    sportsgambler_availability_path = (
        args.project_root / "data" / "raw" / "sportsgambler_player_availability_overrides.csv"
    )
    soccerdata_availability_path = (
        args.project_root / "data" / "raw" / "soccerdata_player_availability_overrides.csv"
    )
    historical_availability_path = (
        args.project_root / "data" / "raw" / "historical_team_availability.csv"
    )
    club_form_path = args.project_root / "data" / "raw" / "club_top_scorers.csv"
    odds_path = args.project_root / "data" / "raw" / "odds" / "WorldCup2026.xlsx"
    transfermarkt_dir = args.project_root / "data" / "raw" / "transfermarkt_kaggle"
    transfermarkt_injuries_path = (
        args.project_root / "data" / "raw" / "transfermarkt_injuries" / "player_injuries.csv"
    )
    output_dir = args.project_root / "data" / "processed"
    report_dir = args.project_root / "reports"

    results = load_results_csv(results_path)
    fixtures = pd.read_csv(fixtures_path)
    goalscorers = (
        load_known_goalscorers_csv(goalscorers_path) if goalscorers_path.exists() else None
    )
    squads = load_squads_csv(squads_path) if squads_path.exists() else None

    model_configs, tuning_results = tune_baseline_model_configs(results)
    selected_configs = model_configs_to_frame(model_configs)
    ensemble_weights = fit_world_cup_ensemble_weights(results, configs=model_configs)
    backtest = evaluate_world_cup_baselines(results, configs=model_configs)
    validation_predictions = add_weighted_ensemble_columns(
        collect_world_cup_validation_predictions(results, configs=model_configs),
        ensemble_weights,
        output_prefix="ensemble",
    )
    market_matches = None
    market_metrics = None
    market_weights = None
    market_decision = None
    if odds_path.exists():
        odds = load_football_data_world_cup_odds(odds_path)
        market_matches = match_validation_predictions_to_odds(validation_predictions, odds)
        market_metrics, market_weights = evaluate_market_models(market_matches)
        market_decision = market_gate_decision(market_metrics)

    forecasts = forecast_2026_group_fixtures(
        results,
        fixtures,
        ensemble_weights=ensemble_weights,
        configs=model_configs,
    )
    player_availability = None
    team_availability = None
    transfermarkt_injury_features = None
    transfermarkt_team_injury_burden = None
    availability_forecasts = None
    forecasts_for_simulation = forecasts
    availability_impact = pd.DataFrame(
        [
            {
                "source": "default",
                "goal_penalty_per_burden": 0.08,
                "note": (
                    "No historical_team_availability.csv file was found; "
                    "using conservative default."
                ),
            }
        ]
    )
    goal_penalty_per_burden = 0.08
    if historical_availability_path.exists():
        historical_availability = pd.read_csv(historical_availability_path)
        goal_penalty_per_burden = estimate_goal_penalty_per_burden(
            results,
            historical_availability,
        )
        availability_impact = pd.DataFrame(
            [
                {
                    "source": str(historical_availability_path),
                    "goal_penalty_per_burden": goal_penalty_per_burden,
                    "note": "Estimated from historical team availability burden.",
                }
            ]
        )
    if squads is not None:
        default_availability = build_default_player_availability(squads, goalscorers)
        availability_override_frames = []
        if sportsgambler_availability_path.exists():
            availability_override_frames.append(
                load_player_availability_csv(sportsgambler_availability_path)
            )
        if soccerdata_availability_path.exists():
            availability_override_frames.append(load_player_availability_csv(soccerdata_availability_path))
        if availability_path.exists():
            availability_override_frames.append(load_player_availability_csv(availability_path))
        availability_overrides = (
            pd.concat(availability_override_frames, ignore_index=True)
            .drop_duplicates(["team_key", "player_key"], keep="last")
            if availability_override_frames
            else None
        )
        player_availability = merge_player_availability(
            default_availability,
            availability_overrides,
        )
        team_availability = aggregate_team_availability_burden(player_availability)
        if transfermarkt_dir.exists() and transfermarkt_injuries_path.exists():
            transfermarkt_injury_features = build_transfermarkt_squad_injury_features(
                squads=squads,
                availability=player_availability,
                transfermarkt_dir=transfermarkt_dir,
                injuries_path=transfermarkt_injuries_path,
            )
            transfermarkt_team_injury_burden = aggregate_transfermarkt_team_injury_burden(
                transfermarkt_injury_features,
            )
        availability_forecasts = adjust_fixture_forecasts_for_team_availability(
            forecasts,
            team_availability,
            goal_penalty_per_burden=goal_penalty_per_burden,
        )
        forecasts_for_simulation = forecasts.copy()
        forecasts_for_simulation["form_poisson_home_expected_goals"] = (
            availability_forecasts["availability_adjusted_home_expected_goals"]
        )
        forecasts_for_simulation["form_poisson_away_expected_goals"] = (
            availability_forecasts["availability_adjusted_away_expected_goals"]
        )
    fixture_teams = set(forecasts["home_team"]).union(forecasts["away_team"])
    advancement_probabilities = build_pairwise_advancement_probabilities(
        results,
        fixture_teams,
        ensemble_weights,
        configs=model_configs,
    )
    tournament_probabilities = simulate_official_tournament(
        results,
        forecasts_for_simulation,
        n_simulations=args.n_simulations,
        advancement_probabilities=advancement_probabilities,
    )
    knockout_forecasts = simulate_official_knockout_match_forecasts(
        results,
        forecasts_for_simulation,
        n_simulations=args.n_simulations,
        advancement_probabilities=advancement_probabilities,
    )
    full_match_forecasts = build_full_match_forecast_table(forecasts, knockout_forecasts)
    team_expected_matches = tournament_probabilities[
        [
            "team",
            "round_of_32_probability",
            "expected_knockout_matches",
            "expected_team_matches",
        ]
    ].sort_values("expected_team_matches", ascending=False, ignore_index=True)

    top_scorers = None
    availability_adjusted_top_scorers = None
    club_adjusted_top_scorers = None
    transfermarkt_adjusted_top_scorers = None
    if goalscorers is not None:
        top_scorers = (
            NationalTeamTopScorerModel()
            .fit(goalscorers, eligible_teams=fixture_teams, squads=squads)
            .predict_group_top_scorers(
                forecasts_for_simulation,
                team_expected_matches=team_expected_matches[["team", "expected_team_matches"]],
            )
        )
        top_scorers_for_enrichment = top_scorers
        if player_availability is not None:
            availability_adjusted_top_scorers = apply_availability_to_top_scorers(
                top_scorers,
                player_availability,
            )
            top_scorers_for_enrichment = availability_adjusted_top_scorers.copy()
            top_scorers_for_enrichment["expected_group_goals"] = top_scorers_for_enrichment[
                "expected_group_goals_availability_adjusted"
            ]
            top_scorers_for_enrichment["expected_tournament_goals"] = top_scorers_for_enrichment[
                "expected_tournament_goals_availability_adjusted"
            ]
        if squads is not None and club_form_path.exists():
            club_form = load_club_top_scorers_csv(club_form_path)
            club_adjusted_top_scorers = enrich_top_scorers_with_club_form(
                top_scorers=top_scorers_for_enrichment,
                squads=squads,
                club_form=club_form,
                top_n=100,
            )
        if squads is not None and transfermarkt_dir.exists():
            transfermarkt_adjusted_top_scorers = build_transfermarkt_top_scorer_features(
                top_scorers=top_scorers_for_enrichment,
                squads=squads,
                transfermarkt_dir=transfermarkt_dir,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    backtest_path = output_dir / "baseline_backtest.csv"
    tuning_results_path = output_dir / "world_cup_2026_hyperparameter_tuning.csv"
    selected_configs_path = output_dir / "world_cup_2026_selected_hyperparameters.csv"
    ensemble_weights_path = output_dir / "world_cup_2026_ensemble_weights.csv"
    market_matches_path = output_dir / "world_cup_market_validation_matches.csv"
    market_metrics_path = output_dir / "world_cup_market_validation_metrics.csv"
    market_weights_path = output_dir / "world_cup_market_ensemble_weights_lowo.csv"
    market_decision_path = output_dir / "world_cup_market_gate_decision.csv"
    forecast_path = output_dir / "world_cup_2026_baseline_forecasts.csv"
    availability_features_path = output_dir / "world_cup_2026_player_availability_features.csv"
    team_availability_path = output_dir / "world_cup_2026_team_availability_burden.csv"
    transfermarkt_injury_features_path = (
        output_dir / "world_cup_2026_transfermarkt_squad_injury_features.csv"
    )
    transfermarkt_team_injury_burden_path = (
        output_dir / "world_cup_2026_transfermarkt_team_injury_burden.csv"
    )
    availability_impact_path = output_dir / "world_cup_2026_availability_impact.csv"
    availability_forecast_path = output_dir / "world_cup_2026_availability_adjusted_forecasts.csv"
    knockout_forecast_path = output_dir / "world_cup_2026_knockout_match_forecasts.csv"
    full_match_forecast_path = output_dir / "world_cup_2026_full_match_forecasts.csv"
    summary_path = report_dir / "baseline_summary.md"
    top_scorer_path = output_dir / "world_cup_2026_top_scorer_baseline.csv"
    availability_adjusted_top_scorer_path = (
        output_dir / "world_cup_2026_top_scorer_availability_adjusted.csv"
    )
    club_adjusted_top_scorer_path = (
        output_dir / "world_cup_2026_top_scorer_club_adjusted_top100.csv"
    )
    transfermarkt_adjusted_top_scorer_path = (
        output_dir / "world_cup_2026_top_scorer_transfermarkt_adjusted_top100.csv"
    )
    expected_matches_path = output_dir / "world_cup_2026_team_expected_matches.csv"
    tournament_probabilities_path = (
        output_dir / "world_cup_2026_official_tournament_probabilities.csv"
    )

    backtest.to_csv(backtest_path, index=False)
    tuning_results.to_csv(tuning_results_path, index=False)
    selected_configs.to_csv(selected_configs_path, index=False)
    pd.DataFrame(
        {
            "model": list(ensemble_weights.as_dict().keys()),
            "weight": list(ensemble_weights.as_dict().values()),
        }
    ).to_csv(ensemble_weights_path, index=False)
    if market_matches is not None and market_metrics is not None and market_weights is not None:
        market_matches.to_csv(market_matches_path, index=False)
        market_metrics.to_csv(market_metrics_path, index=False)
        market_weights.to_csv(market_weights_path, index=False)
        pd.DataFrame([market_decision]).to_csv(market_decision_path, index=False)
    forecasts.to_csv(forecast_path, index=False)
    if (
        player_availability is not None
        and team_availability is not None
        and availability_forecasts is not None
    ):
        player_availability.to_csv(availability_features_path, index=False)
        team_availability.to_csv(team_availability_path, index=False)
        if (
            transfermarkt_injury_features is not None
            and transfermarkt_team_injury_burden is not None
        ):
            transfermarkt_injury_features.to_csv(transfermarkt_injury_features_path, index=False)
            transfermarkt_team_injury_burden.to_csv(
                transfermarkt_team_injury_burden_path,
                index=False,
            )
        availability_impact.to_csv(availability_impact_path, index=False)
        availability_forecasts.to_csv(availability_forecast_path, index=False)
    knockout_forecasts.to_csv(knockout_forecast_path, index=False)
    full_match_forecasts.to_csv(full_match_forecast_path, index=False)
    team_expected_matches.to_csv(expected_matches_path, index=False)
    tournament_probabilities.to_csv(tournament_probabilities_path, index=False)
    if top_scorers is not None:
        top_scorers.to_csv(top_scorer_path, index=False)
    if availability_adjusted_top_scorers is not None:
        availability_adjusted_top_scorers.to_csv(availability_adjusted_top_scorer_path, index=False)
    if club_adjusted_top_scorers is not None:
        club_adjusted_top_scorers.to_csv(club_adjusted_top_scorer_path, index=False)
    if transfermarkt_adjusted_top_scorers is not None:
        transfermarkt_adjusted_top_scorers.to_csv(
            transfermarkt_adjusted_top_scorer_path,
            index=False,
        )
    write_baseline_summary(
        backtest,
        forecasts,
        summary_path,
        tournament_probabilities=tournament_probabilities,
        top_scorers=top_scorers,
        availability_adjusted_top_scorers=availability_adjusted_top_scorers,
        team_availability=team_availability,
        availability_impact=availability_impact,
        transfermarkt_team_injury_burden=transfermarkt_team_injury_burden,
        club_adjusted_top_scorers=club_adjusted_top_scorers,
        transfermarkt_adjusted_top_scorers=transfermarkt_adjusted_top_scorers,
        ensemble_weights=ensemble_weights,
        selected_configs=selected_configs,
        market_metrics=market_metrics,
        market_decision=market_decision,
    )

    print(f"Backtest metrics -> {backtest_path}")
    print(f"Hyperparameter tuning results -> {tuning_results_path}")
    print(f"Selected hyperparameters -> {selected_configs_path}")
    print(f"Ensemble weights -> {ensemble_weights_path}")
    if market_matches is not None:
        print(f"Market validation matches -> {market_matches_path}")
        print(f"Market validation metrics -> {market_metrics_path}")
        print(f"Market LOWO ensemble weights -> {market_weights_path}")
        print(f"Market gate decision -> {market_decision_path}")
    print(f"Fixture forecasts -> {forecast_path}")
    if player_availability is not None:
        print(f"Player availability features -> {availability_features_path}")
        print(f"Team availability burden -> {team_availability_path}")
        if transfermarkt_injury_features is not None:
            print(f"Transfermarkt squad injury features -> {transfermarkt_injury_features_path}")
            print(f"Transfermarkt team injury burden -> {transfermarkt_team_injury_burden_path}")
        print(f"Availability impact -> {availability_impact_path}")
        print(f"Availability-adjusted forecasts -> {availability_forecast_path}")
    print(f"Knockout match forecasts -> {knockout_forecast_path}")
    print(f"Full match forecasts -> {full_match_forecast_path}")
    print(f"Team expected matches -> {expected_matches_path}")
    print(f"Official tournament probabilities -> {tournament_probabilities_path}")
    if top_scorers is not None:
        print(f"Top scorer baseline -> {top_scorer_path}")
    if availability_adjusted_top_scorers is not None:
        print(
            "Availability-adjusted top scorer baseline -> "
            f"{availability_adjusted_top_scorer_path}"
        )
    if club_adjusted_top_scorers is not None:
        print(f"Club-adjusted top scorer top 100 -> {club_adjusted_top_scorer_path}")
    if transfermarkt_adjusted_top_scorers is not None:
        print(
            "Transfermarkt-adjusted top scorer top 100 -> "
            f"{transfermarkt_adjusted_top_scorer_path}"
        )
    print(f"Summary report -> {summary_path}")


if __name__ == "__main__":
    main()
