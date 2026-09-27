"""Deterministic synthetic price and VIX data for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vixagent.config import Settings, load_settings


def make_dates(n: int, start: str = "2010-01-04") -> pd.DatetimeIndex:
    """Business days, tz-naive, name='date'."""
    return pd.bdate_range(start=start, periods=n, name="date")


def correlated_returns(
    n: int, n_assets: int, corr: float, seed: int, vol: float = 0.01
) -> np.ndarray:
    """(n, n_assets) normal returns with equal pairwise correlation `corr`.
    Covariance = vol^2 * (corr * ones + (1 - corr) * I). Uses Cholesky."""
    cov = vol**2 * (corr * np.ones((n_assets, n_assets)) + (1 - corr) * np.eye(n_assets))
    chol = np.linalg.cholesky(cov)
    z = np.random.default_rng(seed).standard_normal((n, n_assets))
    return z @ chol.T


def regime_returns(
    n: int,
    n_assets: int,
    base_corr: float,
    regimes: list[tuple[int, int, float]],
    seed: int,
    vol: float = 0.01,
) -> np.ndarray:
    """Like correlated_returns(base_corr) but rows [start, end) of each regime
    use that regime's correlation. Each block uses its own sub-seed so results
    are deterministic: the base draw uses `seed`, regime i uses `seed + i + 1`."""
    out = correlated_returns(n, n_assets, base_corr, seed, vol)
    for i, (start, end, corr) in enumerate(regimes):
        lo, hi = max(start, 0), min(end, n)
        if hi > lo:
            out[lo:hi] = correlated_returns(hi - lo, n_assets, corr, seed + i + 1, vol)
    return out


def prices_from_returns(
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Price = start_price * exp(cumsum(returns)). Columns = tickers."""
    return pd.DataFrame(start_price * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers)


def make_vix(
    dates: pd.DatetimeIndex,
    seed: int,
    base: float = 15.0,
    spike_positions: list[int] | None = None,
    spike_mult: float = 1.6,
) -> pd.Series:
    """Quiet VIX: base * exp(0.01 * standard normal) each day.
    At each spike position p: level = base * spike_mult on day p,
    base * 1.35 on p+1, base * 1.15 on p+2 (then quiet again).
    Name '^VIX'. Guarantees VIX in [5, 100]."""
    n = len(dates)
    values = base * np.exp(0.01 * np.random.default_rng(seed).standard_normal(n))
    for p in spike_positions or []:
        for offset, mult in ((0, spike_mult), (1, 1.35), (2, 1.15)):
            if 0 <= p + offset < n:
                values[p + offset] = base * mult
    return pd.Series(np.clip(values, 5.0, 100.0), index=dates, name="^VIX")


def make_price_panel(
    n: int = 1500,
    seed: int = 0,
    tickers: list[str] | None = None,
    spike_positions: list[int] | None = None,
) -> pd.DataFrame:
    """Default tickers: the 10 'all' tickers from settings. Returns prices for
    all tickers plus a '^VIX' column, same index.

    Each VIX spike at position p is preceded by a high-correlation regime in
    rows [p - 10, p), so the panel contains a planted correlation-leads-VIX
    pattern that analysis tests can detect."""
    tickers = tickers or list(load_settings().universe["all"])
    dates = make_dates(n)
    regimes = [(p - 10, p, 0.9) for p in spike_positions or []]
    rets = regime_returns(n, len(tickers), 0.3, regimes, seed)
    prices = prices_from_returns(rets, dates, tickers)
    prices["^VIX"] = make_vix(dates, seed, spike_positions=spike_positions)
    return prices


def tiny_settings(tmp_path: Path, n: int = 1500) -> Settings:
    """A Settings object for synthetic data: universe risk = first 6 tickers,
    all = 10 tickers, data.start = first date, snapshot_end = last date,
    train = first 60% of dates, test = remaining dates starting the next
    business day, z_lookback = 60 (small so tests run on short series),
    n_permutations = 500, cache_path under tmp_path."""
    base = load_settings()
    dates = make_dates(n)
    tickers = list(base.universe["all"])
    split = int(0.6 * n)
    d = base.model_dump()
    d["universe"] = {"risk": tickers[:6], "all": tickers}
    d["data"]["start"] = dates[0].date()
    d["data"]["snapshot_end"] = dates[-1].date()
    d["data"]["cache_path"] = tmp_path / "prices.parquet"
    d["periods"]["train"] = (dates[0].date(), dates[split - 1].date())
    d["periods"]["test"] = (dates[split].date(), dates[-1].date())
    d["features"]["z_lookback"] = 60
    d["event_study"]["n_permutations"] = 500
    return Settings.model_validate(d)
