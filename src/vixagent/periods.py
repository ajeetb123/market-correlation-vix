"""Period slicing and eligible-day masks."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def period_mask(index: pd.DatetimeIndex, period: tuple[date, date]) -> np.ndarray:
    """Boolean array: start <= date <= end (inclusive on both ends)."""
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    return np.asarray((index >= start) & (index <= end), dtype=bool)


def eligible_mask(
    index: pd.DatetimeIndex,
    period: tuple[date, date],
    h: int,
    valid: pd.Series | np.ndarray,
) -> np.ndarray:
    """Rows usable for an h-day-ahead analysis in `period`:
      in period AND valid[t] AND (t + h) <= last row position within period.

    Why the last condition: the outcome window t+1..t+h must lie inside the
    same period. For train, this is a natural embargo so no training outcome
    uses test-period data."""
    pos = np.arange(len(index))
    in_p = period_mask(index, period)
    if not in_p.any():
        return np.zeros(len(index), dtype=bool)
    last = pos[in_p].max()
    valid_bool = np.asarray(valid, dtype=bool)
    return in_p & valid_bool & (pos + h <= last)
