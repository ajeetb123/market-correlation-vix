"""Tests for yfinance fetching (network mocked)."""

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from vixagent.data import fetch
from vixagent.data.fetch import DataError, fetch_prices

START, END = date(2020, 1, 2), date(2020, 1, 8)
IDX = pd.DatetimeIndex(pd.bdate_range("2020-01-02", "2020-01-08"), name="Date")


def _field_ticker(tickers: list[str]) -> pd.DataFrame:
    cols = pd.MultiIndex.from_product([["Open", "Close"], tickers])
    data = np.arange(len(IDX) * len(cols), dtype=float).reshape(len(IDX), len(cols)) + 1
    return pd.DataFrame(data, index=IDX, columns=cols)


def _patch(monkeypatch: pytest.MonkeyPatch, fn: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake(*args: Any, **kwargs: Any) -> Any:
        calls.append({"args": args, **kwargs})
        return fn(len(calls))

    monkeypatch.setattr(fetch.yf, "download", fake)
    return calls


def _no_sleep(_: float) -> None:
    return None


def test_multiindex_field_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _field_ticker(["AAA", "BBB"])
    _patch(monkeypatch, lambda _: raw)
    out = fetch_prices(["BBB", "AAA"], START, END, sleep=_no_sleep)
    assert list(out.columns) == ["BBB", "AAA"]
    np.testing.assert_array_equal(out["AAA"].to_numpy(), raw[("Close", "AAA")].to_numpy())
    assert out.index.name == "date"


def test_multiindex_ticker_field(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _field_ticker(["AAA", "BBB"]).swaplevel(axis=1)
    _patch(monkeypatch, lambda _: raw)
    out = fetch_prices(["AAA", "BBB"], START, END, sleep=_no_sleep)
    np.testing.assert_array_equal(out["BBB"].to_numpy(), raw[("BBB", "Close")].to_numpy())


def test_flat_single_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = pd.DataFrame({"Open": 1.0, "Close": [10.0, 11, 12, 13, 14]}, index=IDX)
    _patch(monkeypatch, lambda _: raw)
    out = fetch_prices(["SPY"], START, END, sleep=_no_sleep)
    assert list(out.columns) == ["SPY"]
    assert out["SPY"].tolist() == [10.0, 11, 12, 13, 14]


def test_missing_ticker_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, lambda _: _field_ticker(["AAA"]))
    with pytest.raises(DataError, match="ZZZ"):
        fetch_prices(["AAA", "ZZZ"], START, END, sleep=_no_sleep)


def test_all_nan_ticker_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _field_ticker(["AAA", "BBB"])
    raw[("Close", "BBB")] = np.nan
    _patch(monkeypatch, lambda _: raw)
    with pytest.raises(DataError, match="BBB"):
        fetch_prices(["AAA", "BBB"], START, END, sleep=_no_sleep)


def test_tz_aware_index_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _field_ticker(["AAA"])
    raw.index = (IDX + pd.Timedelta(hours=16)).tz_localize("America/New_York")
    _patch(monkeypatch, lambda _: raw)
    out = fetch_prices(["AAA"], START, END, sleep=_no_sleep)
    assert out.index.tz is None
    assert (out.index == out.index.normalize()).all()
    assert out.index[0] == pd.Timestamp("2020-01-02")


def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _field_ticker(["AAA"])

    def attempt(n: int) -> pd.DataFrame:
        if n < 3:
            raise RuntimeError("rate limited")
        return raw

    calls = _patch(monkeypatch, attempt)
    sleeps: list[float] = []
    out = fetch_prices(["AAA"], START, END, sleep=sleeps.append)
    assert len(calls) == 3
    assert sleeps == [2, 4]
    assert len(out) == len(IDX)


def test_always_empty_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch(monkeypatch, lambda _: pd.DataFrame())
    with pytest.raises(DataError):
        fetch_prices(["AAA"], START, END, sleep=_no_sleep)
    assert len(calls) == 4


def test_end_is_exclusive_plus_one_day(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch(monkeypatch, lambda _: _field_ticker(["AAA"]))
    fetch_prices(["AAA"], START, END, sleep=_no_sleep)
    assert calls[0]["end"] == (END + timedelta(days=1)).isoformat()
    assert calls[0]["auto_adjust"] is True
