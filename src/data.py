#Grabs historical prices for my basket of stocks + the VIX using yfinance.

from __future__ import annotations

import pandas as pd
import yfinance as yf

# 10 big well known stocks, one from each sector, no repeats, on purpose.
# they're all in totally different industries so they shouldn't really be
# moving together day to day unless something bigger (market-wide fear) is
# going on. (earlier version of this had 2 stocks in the same sector by
# accident, which meant regular sector news could bump the correlation
# number up even when nothing market-wide was happening - fixed that by
# making sure every sector only shows up once)
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
# the ^ means it's an index, not something you can actually buy/sell


def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Grabs daily close prices for `tickers`, adjusted for splits/dividends."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # raw['Adj Close'] used to be the one to use but newer yfinance folds
    # the adjustment into 'Close' automatically now (auto_adjust=True), so
    # 'Adj Close' doesn't even exist as a column anymore. took me a while
    # to figure out why my old code was throwing a KeyError lol
    closes = raw["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])
    df = closes.ffill().dropna(how="all")
    return df


def load_basket_and_vix(
    start: str,
    end: str | None = None,
    basket: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Loads the basket's prices and the VIX level, lined up on the same
    trading days so nothing later has to worry about mismatched calendars.
    """
    basket = basket or DEFAULT_BASKET
    tickers = list(basket.keys())  # just the ticker symbols, don't need the sector names here
    basket_prices = download_prices(tickers, start, end)
    vix = download_prices([VIX_TICKER], start, end)[VIX_TICKER].rename("VIX")
    # inner join so we only keep dates where BOTH the basket and the VIX
    # actually have a price - VIX and stocks are usually on the same
    # calendar anyway but better safe than sorry
    aligned = basket_prices.join(vix, how="inner").dropna()
    return aligned[tickers], aligned["VIX"]
