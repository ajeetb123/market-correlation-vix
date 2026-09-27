"""Forward-looking quantities. This is the ONLY module allowed to look ahead.
Everything here describes the future relative to row t and must only be used
as an outcome, never as an input to a feature."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fwd_log_vix_change(vix: pd.Series, h: int) -> pd.Series:
    """ln(V_{t+h}) - ln(V_t). The last h rows are NaN.

    Why log change: it measures proportional VIX moves, so a jump from 12 to 18
    and from 24 to 36 count equally.
    """
    return (np.log(vix.shift(-h)) - np.log(vix)).rename(f"fwd_log_vix_change_{h}")


def fwd_any_within(flags: pd.Series, h: int) -> pd.Series:
    """out_t = True if any flag is True in rows t+1..t+h (strictly after t).
    Rows where the window runs past the end still compute over the rows that
    exist; eligibility (periods.eligible_mask) excludes them from analysis.

    Implemented with a cumulative sum so it is O(n)."""
    f = flags.fillna(False).to_numpy(dtype=bool).astype(np.int64)
    n = len(f)
    csum = np.concatenate([[0], np.cumsum(f)])
    idx = np.arange(n)
    lo = np.minimum(idx + 1, n)
    hi = np.minimum(idx + h + 1, n)
    return pd.Series((csum[hi] - csum[lo]) > 0, index=flags.index, name=f"fwd_any_{h}")
