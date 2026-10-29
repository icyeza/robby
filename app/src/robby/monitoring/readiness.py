"""High readiness: the flag shown with a readiness signal, set by each facility's readiness cut."""

from __future__ import annotations


def is_high(probability: float, cut_percent: int) -> bool:
    """High readiness: the probability is at or above the facility's cut."""
    return 100 * float(probability) >= cut_percent
