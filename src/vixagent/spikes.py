"""Spike-day detection and declustering into events."""

from __future__ import annotations

import pandas as pd


def spike_days(x: pd.Series, threshold: float) -> pd.Series:
    """True where x >= threshold. NaN compares False."""
    return (x >= threshold).fillna(False).astype(bool)


def events_from_days(days: pd.Series, cooldown: int) -> pd.Series:
    """Decluster: t is an event if days[t] is True and no True occurred in rows
    t-cooldown..t-1.

    Why: consecutive crisis days are one episode, not many independent
    observations. Counting them separately inflates sample size."""
    d = days.fillna(False).astype(bool)
    if cooldown == 0:
        return d.rename("event")
    prior = d.astype(float).shift(1).rolling(cooldown, min_periods=1).max().fillna(0.0)
    return (d & (prior == 0.0)).rename("event")


def backward_any(flags: pd.Series, k: int) -> pd.Series:
    """True if any flag is True in rows t-k..t inclusive. Used by the clean
    filter. Backward-looking only."""
    f = flags.fillna(False).astype(float)
    return f.rolling(k + 1, min_periods=1).max().astype(bool)
