"""Parquet cache for the price panel, with metadata for invalidation."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from vixagent.config import Settings, find_project_root
from vixagent.data.fetch import DataError, fetch_prices
from vixagent.data.validate import validate_prices

logger = logging.getLogger(__name__)


def all_tickers(settings: Settings) -> list[str]:
    """Sorted union of all universe groups, plus vix_ticker last."""
    union = sorted({t for tickers in settings.universe.values() for t in tickers})
    return [*union, settings.data.vix_ticker]


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".meta.json")


def _expected_meta(settings: Settings) -> dict[str, object]:
    return {
        "tickers": all_tickers(settings),
        "start": settings.data.start.isoformat(),
        "end_inclusive": settings.data.snapshot_end.isoformat(),
    }


def _resolve(settings: Settings, root: Path | None) -> Path:
    path = settings.data.cache_path
    return path if path.is_absolute() else (root or find_project_root()) / path


def load_cached_prices(settings: Settings, root: Path | None = None) -> pd.DataFrame:
    """Load the cached panel without ever touching the network.
    Raises DataError("No cached data found. Run `vixagent pull` first.") if the
    parquet or metadata file is missing, or if metadata does not match settings.

    Why: the agent must never trigger a surprise download mid-conversation."""
    path = _resolve(settings, root)
    meta_path = _meta_path(path)
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if all(meta.get(k) == v for k, v in _expected_meta(settings).items()):
            return pd.read_parquet(path)
    raise DataError("No cached data found. Run `vixagent pull` first.")


def load_or_fetch(
    settings: Settings,
    refresh: bool = False,
    root: Path | None = None,
    fetcher: Callable[..., pd.DataFrame] = fetch_prices,
) -> pd.DataFrame:
    """Return the full price panel (all_tickers columns).

    Cache files: <root>/<data.cache_path> (parquet) and the same path with
    suffix '.meta.json'. Metadata keys: fetched_at (ISO UTC), yfinance_version,
    tickers, start, end_inclusive.

    Use the cache only if both files exist, refresh is False, and metadata
    tickers/start/end_inclusive equal the current settings. Otherwise call
    fetcher(all_tickers, data.start, data.snapshot_end), run validate_prices
    (log warnings via `logging`), write both files (mkdir parents), return.

    Why: Yahoo rate-limits and its history can be revised, so a fixed local
    snapshot makes results reproducible and runs fast.
    """
    path = _resolve(settings, root)
    meta_path = _meta_path(path)
    expected = _expected_meta(settings)

    if not refresh and path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if all(meta.get(k) == v for k, v in expected.items()):
            return pd.read_parquet(path)

    tickers = all_tickers(settings)
    prices = fetcher(tickers, settings.data.start, settings.data.snapshot_end)
    for warning in validate_prices(prices, settings.data.vix_ticker):
        logger.warning(warning)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    meta = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        **expected,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return prices
