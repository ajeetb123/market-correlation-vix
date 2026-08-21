#Downloads historical prices for the asset basket and the VIX index via yfinance.

from __future__ import annotations

import pandas as pd
import yfinance as yf

# Ten large-cap stocks, one per sector, no repeats, on purpose; don't have
# similar businesses so shouldn't move together;
# movement means market-wide fear
# and not a specific change to one sector
# (an earlier version had two stocks in some sectors, which let ordinary
# sector news drive part of the correlation number instead of real
# market-wide fear - fixed by making every sector appear exactly once)
# I am measuring how market wide fear affects the correlation of these unrelated securities
DEFAULT_BASKET = {
    "AAPL": "Technology",
    "JPM": "Financials",
    "XOM": "Energy",
    "JNJ": "Healthcare",
    "PG": "Consumer Staples",
    "HD": "Consumer Discretionary",
    "CAT": "Industrials",
    "LIN": "Materials",
    "VZ": "Communication Services",
    "NEE": "Utilities",
}

VIX_TICKER = "^VIX"
# ^ means index; not a tradeable stock


def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download split/dividend-adjusted daily close prices for `tickers`."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    closes = raw["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])
    return closes.ffill().dropna(how="all")


def load_basket_and_vix(
    start: str,
    end: str | None = None,
    basket: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load basket prices and VIX level, aligned to the same trading calendar.

    Returns (basket_prices, vix_level) — both indexed on the dates the basket
    and the VIX both traded, so downstream code never has to reconcile calendars.
    """
    basket = basket or DEFAULT_BASKET
    tickers = list(basket.keys())
    basket_prices = download_prices(tickers, start, end)
    vix = download_prices([VIX_TICKER], start, end)[VIX_TICKER].rename("VIX")
    aligned = basket_prices.join(vix, how="inner").dropna()
    return aligned[tickers], aligned["VIX"]
