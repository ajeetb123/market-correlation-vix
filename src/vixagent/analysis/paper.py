"""Replication of the paper's claim about correlations in the lead-up to VIX blowups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from vixagent.features.correlation import avg_pairwise_corr
from vixagent.spikes import events_from_days

Direction = Literal["high", "low"]


def vix_doubling_events(
    vix: pd.Series, lookback: int = 63, multiple: float = 2.0, cooldown: int = 126
) -> pd.Series:
    """Paper-style VIX events: the VIX closes at least `multiple` times its minimum
    over the trailing `lookback` trading days (inclusive), declustered so a day is
    an event only if no qualifying day occurred in the prior `cooldown` days.

    Why: the paper defined events as the VIX doubling in under three months.
    Declustering keeps one multi-week episode from counting as many events.
    """
    low = vix.rolling(lookback, min_periods=lookback).min()
    return events_from_days(vix >= multiple * low, cooldown)


def leadup_equity_corr(equity_returns: pd.DataFrame, window: int = 63) -> pd.Series:
    """Average pairwise correlation among the equity styles over rows t-window..t-1.

    Why shift(1): the lead-up window ends the day before the event, so the event
    day's own crash return never enters the "before" measurement.
    """
    return avg_pairwise_corr(equity_returns, window).shift(1).rename("equity_styles")


def leadup_gold_equity_corr(
    equity_returns: pd.DataFrame, gold_returns: pd.Series, window: int = 63
) -> pd.Series:
    """Mean over equity styles of corr(gold, style) over rows t-window..t-1."""
    per_style = [equity_returns[c].rolling(window).corr(gold_returns) for c in equity_returns]
    stacked = pd.concat(per_style, axis=1).clip(-1.0, 1.0)
    return stacked.mean(axis=1, skipna=False).shift(1).rename("gold_equity")


def align_event_dates(dates: pd.Index, index: pd.Index) -> pd.Series:
    """Map event dates onto another trading calendar (the same or next available day)."""
    flags = pd.Series(False, index=index)
    pos = index.searchsorted(dates)
    flags.iloc[pos[pos < len(index)]] = True
    return flags


@dataclass
class LeadupTest:
    """One lead-up correlation test over one set of events."""

    measure: str
    direction: Direction
    event_set: str
    event_dates: list[str]
    event_values: list[float]
    mean_leadup: float
    typical: float
    p_value: float
    n_permutations: int


def leadup_test(
    stat: pd.Series,
    events: pd.Series,
    direction: Direction,
    n_perm: int,
    seed: int,
    min_shift: int,
    measure: str,
    event_set: str,
) -> LeadupTest:
    """Compare the mean lead-up value over events with circularly shifted event sets.

    typical = mean of the measure over all days where it is defined.
    p = (1 + #shifted means at least as extreme as observed) / (1 + n_perm),
    one-sided in `direction`. Shifts are drawn from [min_shift, n - min_shift].

    Why circular shifts: the 63-day correlation is highly autocorrelated and events
    cluster in time; shifting the whole event pattern preserves both, unlike
    drawing independent random dates.
    """
    valid = stat.notna().to_numpy()
    s = stat.to_numpy()[valid]
    e = events.reindex(stat.index, fill_value=False).to_numpy(dtype=bool)[valid]
    dates = stat.index[valid][e]
    observed = float(s[e].mean()) if e.any() else float("nan")
    n = len(s)
    count = 0
    if e.any() and n > 2 * min_shift:
        rng = np.random.default_rng(seed)
        for k in rng.integers(min_shift, n - min_shift + 1, size=n_perm):
            shifted = float(s[np.roll(e, int(k))].mean())
            if direction == "high" and shifted >= observed - 1e-12:
                count += 1
            if direction == "low" and shifted <= observed + 1e-12:
                count += 1
        p = (1 + count) / (1 + n_perm)
    else:
        p = float("nan")
    return LeadupTest(
        measure=measure,
        direction=direction,
        event_set=event_set,
        event_dates=[d.strftime("%Y-%m-%d") for d in dates],
        event_values=[float(x) for x in s[e]],
        mean_leadup=observed,
        typical=float(s.mean()),
        p_value=p,
        n_permutations=n_perm,
    )
