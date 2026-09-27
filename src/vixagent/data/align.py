"""Per-group inner alignment of prices."""

from __future__ import annotations

import pandas as pd

from vixagent.config import Group, Settings
from vixagent.periods import period_mask


def align_group(prices: pd.DataFrame, settings: Settings, group: Group) -> tuple[pd.DataFrame, int]:
    """Columns universe[group] + [vix_ticker], rows within the full period
    (data.start..snapshot_end inclusive), keeping only rows where ALL those
    columns are non-NaN. No forward fill.

    Returns (aligned_frame, rows_dropped) where rows_dropped counts rows in the
    full-period slice that had at least one NaN in these columns.

    Why no forward fill: a filled price creates a fake 0% return, which
    artificially lowers correlations on market holidays that differ across
    exchanges.
    """
    cols = [*settings.universe[group], settings.data.vix_ticker]
    in_period = prices[cols].loc[period_mask(prices.index, settings.full_period())]
    aligned = in_period.dropna(how="any")
    return aligned, len(in_period) - len(aligned)
