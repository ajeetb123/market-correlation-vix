"""Log returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """ln(P_t / P_{t-1}) per column; the first row is dropped (it is all NaN).

    Why log returns: they are additive over time and symmetric, and correlating
    returns (not price levels) avoids spurious correlation between trending series.
    """
    return np.log(prices).diff().iloc[1:]
