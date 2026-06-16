from __future__ import annotations

from wc26_predictor.reporting.validation_dashboard import _dashboard_nav


def test_validation_dashboard_nav_links_to_split_dashboard_pages() -> None:
    nav = _dashboard_nav()

    assert '../model_performance_dashboard.html">Home' in nav
    assert 'href="future_matches.html">Future matches' in nav
    assert 'href="group_stage.html">Group stage' in nav
    assert 'href="next_phases.html">Next phases' in nav
    assert 'class="active" href="model_performance.html">Model performance' in nav
