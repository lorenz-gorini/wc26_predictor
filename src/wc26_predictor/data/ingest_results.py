"""Load completed international matches from local paths or URLs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc26_predictor.data.schema import validate_results_frame


def load_results_csv(path_or_url: str | Path) -> pd.DataFrame:
    """Load and validate a results CSV.

    Parameters
    ----------
    path_or_url:
        Local path or HTTP(S) URL containing the standard international results schema.
    """

    raw = pd.read_csv(path_or_url)
    return validate_results_frame(raw)


def load_completed_results_csv(path_or_url: str | Path) -> pd.DataFrame:
    """Load only completed matches from a results CSV.

    Some public result archives also include scheduled fixtures with missing
    scores. Those rows belong in raw data but must be excluded from model fitting.
    """

    raw = pd.read_csv(path_or_url)
    completed = raw.dropna(subset=["home_score", "away_score"]).copy()
    return validate_results_frame(completed)


def save_processed_results(results: pd.DataFrame, destination: str | Path) -> None:
    """Persist validated results with stable column order."""

    processed = validate_results_frame(results)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(destination_path, index=False)
