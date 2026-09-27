"""Download daily adjusted closes from Yahoo Finance via yfinance."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd
import yfinance as yf


class DataError(Exception):
    """Raised when downloaded or cached data is unusable."""


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull the Close field out of either yfinance column layout.

    Why both layouts: depending on version and arguments, yfinance returns
    MultiIndex columns as (field, ticker) or (ticker, field), and sometimes
    flat columns for a single ticker.
    """
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        if "Close" in cols.get_level_values(0):
            return raw["Close"]
        if "Close" in cols.get_level_values(1):
            return raw.xs("Close", axis=1, level=1)
        raise DataError("Downloaded data has no Close field")
    if len(tickers) == 1 and "Close" in cols:
        return raw[["Close"]].rename(columns={"Close": tickers[0]})
    raise DataError("Downloaded data has an unrecognized column layout")


def fetch_prices(
    tickers: list[str],
    start: date,
    end_inclusive: date,
    *,
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Download daily adjusted closes. Returns DataFrame: index tz-naive
    DatetimeIndex named 'date' (sorted), columns exactly `tickers` in order.

    Why end + 1 day: yfinance treats `end` as exclusive. Why auto_adjust=True
    explicitly: its default changed across yfinance versions, and adjusted
    closes are needed so dividends and splits do not appear as returns.
    Retries with exponential backoff (2, 4, 8 s) because Yahoo rate-limits.
    """
    end_exclusive = end_inclusive + timedelta(days=1)
    raw: pd.DataFrame | None = None
    last_error = "empty result"
    for attempt in range(1 + max_retries):
        try:
            raw = yf.download(
                tickers,
                start=start.isoformat(),
                end=end_exclusive.isoformat(),
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column",
            )
            if raw is not None and not raw.empty:
                break
            last_error = "empty result"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        raw = None
        if attempt < max_retries:
            sleep(2 ** (attempt + 1))
    if raw is None:
        raise DataError(f"Download failed after {1 + max_retries} attempts: {last_error}")

    close = _extract_close(raw, tickers)
    idx = pd.DatetimeIndex(pd.to_datetime(close.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close = close.copy()
    close.index = idx.normalize().rename("date")
    close = close.sort_index()

    missing = [t for t in tickers if t not in close.columns or close[t].isna().all()]
    if missing:
        raise DataError(f"Missing or empty tickers: {missing}")
    return close[tickers].astype(float)
