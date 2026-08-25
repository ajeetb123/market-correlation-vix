"""Runs the whole thing start to finish: downloads the data, computes the
rolling correlations, runs the tests to see if correlation spikes lead VIX
spikes, and saves all the charts into output/.

Run with:  python main.py
"""

from __future__ import annotations

import os

from src.analysis import (
    cross_correlation,
    event_study,
    granger_causality,
    identify_spike_events,
    rolling_zscore,
)
from src.correlation import (
    average_pairwise_correlation,
    correlation_snapshot,
    daily_returns,
)
from src.data import load_basket_and_vix
from src.visualize import (
    plot_avg_correlation_vs_vix,
    plot_correlation_heatmap,
    plot_cross_correlation,
    plot_event_study,
)

START_DATE = "2015-01-01"  # ~10 years - long enough to span several VIX regimes (2018, 2020, 2022, 2025)
CORR_WINDOW = 20  # trading days (~1 month) for the rolling correlation
ZSCORE_LOOKBACK = 252  # 1 trading year, for defining a "spike" relative to recent history
SPIKE_THRESHOLD = 1.5  # standard deviations, picked this kind of arbitrarily, might play with it later
EVENT_HORIZON = 10  # trading days forward when measuring VIX's reaction
MAX_LAG = 20  # trading days, for cross-correlation and Granger tests
OUTPUT_DIR = "output"  # everything gets dumped in here


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Downloading basket + VIX data from {START_DATE}...")
    prices, vix = load_basket_and_vix(START_DATE)
    returns = daily_returns(prices)
    print(f"  {len(prices)} trading days, {prices.shape[1]} tickers: {list(prices.columns)}")
    print(prices.tail(3))  # just eyeballing that the download actually worked

    print("Computing rolling average pairwise correlation...")
    # avg_corr starts a bit later than vix because the rolling window needs
    # CORR_WINDOW days before it can spit out a real first value - this
    # lines both series back up on the exact dates they both have data for
    avg_corr = average_pairwise_correlation(returns, CORR_WINDOW)
    aligned_vix = vix.reindex(avg_corr.index).dropna()
    avg_corr = avg_corr.reindex(aligned_vix.index)

    print("Saving correlation heatmaps...")
    # sanity-check heatmap for the whole period
    full_period_matrix = prices.corr()
    plot_correlation_heatmap(
        full_period_matrix,
        "Full-period pairwise correlation",
        f"{OUTPUT_DIR}/heatmap_full_period.png",
    )
    latest_matrix = correlation_snapshot(returns, CORR_WINDOW, returns.index[-1])
    plot_correlation_heatmap(
        latest_matrix,
        f"Trailing {CORR_WINDOW}-day pairwise correlation (as of {returns.index[-1].date()})",
        f"{OUTPUT_DIR}/heatmap_latest.png",
    )

    print("Saving average-correlation-vs-VIX time series...")
    plot_avg_correlation_vs_vix(avg_corr, aligned_vix, f"{OUTPUT_DIR}/avg_corr_vs_vix.png")

    print("Running cross-correlation (lead-lag) test...")
    xcorr = cross_correlation(avg_corr, aligned_vix, MAX_LAG)
    plot_cross_correlation(xcorr, f"{OUTPUT_DIR}/cross_correlation.png")
    peak_lag = xcorr.idxmax()
    print(f"  Peak correlation {xcorr.max():.3f} at lag {peak_lag:+d} trading days "
          f"({'avg correlation leads VIX' if peak_lag > 0 else 'VIX leads avg correlation' if peak_lag < 0 else 'contemporaneous'})")

    print("Running Granger causality test (does avg correlation predict VIX?)...")
    granger_pvalues = granger_causality(avg_corr, aligned_vix, MAX_LAG)
    best_lag = min(granger_pvalues, key=granger_pvalues.get)
    print(f"  Smallest p-value {granger_pvalues[best_lag]:.4f} at lag {best_lag}")

    print("Running the event study on correlation-spike days...")
    z = rolling_zscore(avg_corr, ZSCORE_LOOKBACK)
    events = identify_spike_events(z, SPIKE_THRESHOLD)
    print(f"  {len(events)} correlation-spike events identified (z > {SPIKE_THRESHOLD})")
    if len(events) > 0:
        result = event_study(aligned_vix, events, EVENT_HORIZON)
        plot_event_study(
            result["event_changes"], result["baseline_mean_change"], EVENT_HORIZON,
            f"{OUTPUT_DIR}/event_study.png",
        )
        print(
            f"  Mean {EVENT_HORIZON}d VIX change after spikes: {result['event_mean_change']:.2%} "
            f"vs. unconditional {result['baseline_mean_change']:.2%} "
            f"(t={result['t_stat']:.2f}, p={result['p_value']:.4f}, n={result['n_events']})"
        )

    print(f"\nDone. Charts written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
