"""Tests for the parquet price cache."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from vixagent.config import Settings
from vixagent.data.cache import all_tickers, load_cached_prices, load_or_fetch
from vixagent.data.fetch import DataError


class FakeFetcher:
    def __init__(self, panel: pd.DataFrame) -> None:
        self.panel = panel
        self.calls = 0

    def __call__(self, tickers: list[str], start: date, end: date, **_: Any) -> pd.DataFrame:
        self.calls += 1
        return self.panel[tickers]


def test_all_tickers(small_settings: Settings) -> None:
    tickers = all_tickers(small_settings)
    assert tickers[-1] == "^VIX"
    assert tickers[:-1] == sorted(small_settings.universe["all"])


def test_first_call_fetches_and_writes(
    panel: pd.DataFrame, small_settings: Settings, tmp_path: Path
) -> None:
    fetcher = FakeFetcher(panel)
    load_or_fetch(small_settings, root=tmp_path, fetcher=fetcher)
    assert fetcher.calls == 1
    assert small_settings.data.cache_path.exists()
    assert small_settings.data.cache_path.with_suffix(".meta.json").exists()


def test_second_call_uses_cache(
    panel: pd.DataFrame, small_settings: Settings, tmp_path: Path
) -> None:
    fetcher = FakeFetcher(panel)
    first = load_or_fetch(small_settings, root=tmp_path, fetcher=fetcher)
    second = load_or_fetch(small_settings, root=tmp_path, fetcher=fetcher)
    assert fetcher.calls == 1
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_refresh_refetches(panel: pd.DataFrame, small_settings: Settings, tmp_path: Path) -> None:
    fetcher = FakeFetcher(panel)
    load_or_fetch(small_settings, root=tmp_path, fetcher=fetcher)
    load_or_fetch(small_settings, refresh=True, root=tmp_path, fetcher=fetcher)
    assert fetcher.calls == 2


def test_changed_tickers_refetch(
    panel: pd.DataFrame, small_settings: Settings, tmp_path: Path
) -> None:
    fetcher = FakeFetcher(panel.assign(XLE=panel.iloc[:, 0]))
    load_or_fetch(small_settings, root=tmp_path, fetcher=fetcher)
    d = small_settings.model_dump()
    d["universe"]["all"] = [*d["universe"]["all"][:-1], "XLE"]
    changed = Settings.model_validate(d)
    load_or_fetch(changed, root=tmp_path, fetcher=fetcher)
    assert fetcher.calls == 2


def test_relative_cache_path_resolves_against_root(
    panel: pd.DataFrame, small_settings: Settings, tmp_path: Path
) -> None:
    d = small_settings.model_dump()
    d["data"]["cache_path"] = Path("data/raw/prices.parquet")
    s = Settings.model_validate(d)
    load_or_fetch(s, root=tmp_path, fetcher=FakeFetcher(panel))
    assert (tmp_path / "data/raw/prices.parquet").exists()


def test_metadata_keys(panel: pd.DataFrame, small_settings: Settings, tmp_path: Path) -> None:
    load_or_fetch(small_settings, root=tmp_path, fetcher=FakeFetcher(panel))
    meta = json.loads(small_settings.data.cache_path.with_suffix(".meta.json").read_text())
    assert set(meta) == {"fetched_at", "yfinance_version", "tickers", "start", "end_inclusive"}


def test_load_cached_prices_missing_raises(small_settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(DataError, match=r"No cached data found\. Run `vixagent pull` first\."):
        load_cached_prices(small_settings, root=tmp_path)


def test_load_cached_prices_loads_valid_cache(
    panel: pd.DataFrame, small_settings: Settings, tmp_path: Path
) -> None:
    fetched = load_or_fetch(small_settings, root=tmp_path, fetcher=FakeFetcher(panel))
    loaded = load_cached_prices(small_settings, root=tmp_path)
    pd.testing.assert_frame_equal(fetched, loaded, check_freq=False)


def test_load_cached_prices_mismatch_raises(
    panel: pd.DataFrame, small_settings: Settings, tmp_path: Path
) -> None:
    load_or_fetch(small_settings, root=tmp_path, fetcher=FakeFetcher(panel))
    d = small_settings.model_dump()
    d["data"]["snapshot_end"] = d["periods"]["test"][1].replace(year=2030)
    with pytest.raises(DataError):
        load_cached_prices(Settings.model_validate(d), root=tmp_path)
