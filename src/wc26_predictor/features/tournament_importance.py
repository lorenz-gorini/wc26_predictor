"""Tournament importance weights for international football matches."""

from __future__ import annotations


def tournament_importance_weight(tournament: str) -> float:
    """Return a conservative importance weight for a tournament label.

    The scale is inspired by FIFA/football Elo practice, but intentionally
    compressed so that a small number of World Cup matches cannot dominate the
    full historical sample.
    """

    normalized = tournament.strip().lower()
    if not normalized:
        raise ValueError("Tournament name cannot be empty.")

    if "fifa world cup qualification" in normalized:
        return 1.45
    if normalized == "fifa world cup":
        return 1.80
    if "qualification" in normalized or "qualifiers" in normalized:
        return 1.35
    if "uefa euro" in normalized or "copa america" in normalized:
        return 1.55
    if "african cup" in normalized or "asian cup" in normalized or "gold cup" in normalized:
        return 1.50
    if "nations league" in normalized:
        return 1.20
    if "friendly" in normalized:
        return 0.75
    return 1.00
