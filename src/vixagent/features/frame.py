"""Assemble all backward-looking features for one group and window."""

from __future__ import annotations

import pandas as pd

from vixagent.config import Group, Settings
from vixagent.features.correlation import avg_pairwise_corr
from vixagent.features.returns import log_returns
from vixagent.features.vix import log_vix, vix_momentum, vix_ratio
from vixagent.features.zscore import trailing_zscore
from vixagent.spikes import events_from_days, spike_days


def build_frame(
    aligned: pd.DataFrame, settings: Settings, group: Group, window: int
) -> pd.DataFrame:
    """Build the per-group, per-window analysis frame from aligned prices.

    Index: dates of the return series (first aligned date dropped).
    Columns:
      vix          VIX level
      log_vix      ln(vix)
      vix_mom_5d   ln(V_t / V_{t-5})
      vix_ratio    V_t / median(prior 20)
      vix_spike_day  vix_ratio >= settings.vix_spike.ratio_threshold
      vix_event    events_from_days(vix_spike_day, vix_spike.cooldown)
      avg_corr     avg_pairwise_corr(returns of group tickers, window)
      corr_z       trailing_zscore(avg_corr, features.z_lookback)
    Correlation spike days/events depend on z_threshold and are computed in
    analysis, not here. No fwd_ columns here.

    VIX features are computed on the full aligned VIX series and then
    restricted to the return dates, so the first return date still has one
    prior VIX value available as history.
    """
    returns = log_returns(aligned[settings.universe[group]])
    vix_full = aligned[settings.data.vix_ticker]
    ratio = vix_ratio(vix_full, settings.vix_spike.median_lookback)
    spike = spike_days(ratio, settings.vix_spike.ratio_threshold)
    avg_corr = avg_pairwise_corr(returns, window)
    frame = pd.DataFrame(
        {
            "vix": vix_full,
            "log_vix": log_vix(vix_full),
            "vix_mom_5d": vix_momentum(vix_full, 5),
            "vix_ratio": ratio,
            "vix_spike_day": spike,
            "vix_event": events_from_days(spike, settings.vix_spike.cooldown),
        }
    ).loc[returns.index]
    frame["avg_corr"] = avg_corr
    frame["corr_z"] = trailing_zscore(avg_corr, settings.features.z_lookback)
    frame.index.name = "date"
    return frame
