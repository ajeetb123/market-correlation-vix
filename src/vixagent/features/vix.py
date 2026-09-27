"""VIX-derived features: spike ratio, momentum, and log level."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def vix_ratio(vix: pd.Series, lookback: int) -> pd.Series:
    """V_t / median(V_{t-lookback}..V_{t-1}). Median resists single outliers.

    Why shift(1): the baseline excludes today, so a spike is measured against
    the days before it rather than diluted by itself.
    """
    return (vix / vix.shift(1).rolling(lookback).median()).rename("vix_ratio")


def vix_momentum(vix: pd.Series, k: int = 5) -> pd.Series:
    """ln(V_t / V_{t-k}): recent VIX change, used as a regression control."""
    return cast(pd.Series, np.log(vix / vix.shift(k))).rename(f"vix_mom_{k}d")


def log_vix(vix: pd.Series) -> pd.Series:
    """ln(V_t): VIX level on a log scale, used as a regression control.

    Why log: VIX changes are roughly proportional to its level, so the log
    scale makes the level control closer to linear.
    """
    return cast(pd.Series, np.log(vix)).rename("log_vix")
