"""Circular-shift permutation test for event-study hit rates."""

from __future__ import annotations

import numpy as np


def circular_shift_pvalue(
    src_D: np.ndarray,
    hit_D: np.ndarray,
    clean_D: np.ndarray,
    observed: float,
    horizon: int,
    n_perm: int,
    seed: int,
) -> float:
    """One-sided p-value for the event hit rate under circular shifts.

    Why circular shifts: they keep the spacing and clustering of events intact,
    so the null distribution reflects how clumpy real events are. A t-test
    would assume independent events and give falsely small p-values.

    Returns NaN if observed is NaN, or if len(src_D) < 2*horizon + 3.
    A shifted pattern with zero clean events counts as hit rate 0.0.
    """
    n = len(src_D)
    if np.isnan(observed) or n < 2 * horizon + 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    ks = rng.integers(horizon + 1, n - horizon, size=n_perm)  # upper bound exclusive => [h+1, n-h-1]
    count = 0
    for k in ks:
        m = np.roll(src_D, int(k)) & clean_D
        r = hit_D[m].mean() if m.any() else 0.0
        if r >= observed - 1e-12:
            count += 1
    return (1 + count) / (1 + n_perm)
