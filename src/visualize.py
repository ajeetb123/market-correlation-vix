"""All the charts for this project, made with matplotlib/seaborn.

I use the same two colors everywhere so it's easy to tell what's what at a
glance: blue = correlation stuff, orange-red = VIX stuff. Found these two
colors online as a pair that's supposed to still look different if you're
colorblind.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="white", context="talk")

BLUE = "#0072B2"
VERMILLION = "#D55E00"


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str, path: str) -> None:
    """Heatmap for one n-by-n correlation matrix, centered on 0 since
    correlation can be positive (moves together), zero (no relationship),
    or negative (moves opposite).
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr_matrix,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pairwise correlation"},
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_avg_correlation_vs_vix(avg_corr: pd.Series, vix: pd.Series, path: str) -> None:
    """Puts avg correlation and VIX on two stacked charts instead of one
    chart with two y-axes - the two numbers are on totally different scales
    (0-1 vs roughly 10-80) so smushing them onto one axis makes it too easy
    to misread how big a move in one actually is compared to the other.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(avg_corr.index, avg_corr.values, color=BLUE, linewidth=1.3)
    ax1.set_ylabel("Avg pairwise correlation")
    ax1.set_title("Basket average correlation vs. VIX")

    ax2.plot(vix.index, vix.values, color=VERMILLION, linewidth=1.3)
    ax2.set_ylabel("VIX")
    ax2.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cross_correlation(xcorr: pd.Series, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    peak_lag = xcorr.idxmax()
    # only color the tallest bar differently so your eye goes straight to
    # the answer instead of having to compare every single bar height
    colors = [VERMILLION if lag == peak_lag else BLUE for lag in xcorr.index]
    ax.bar(xcorr.index, xcorr.values, color=colors, width=0.8)
    ax.axvline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("Lag, trading days (positive = avg correlation leads VIX)")
    ax.set_ylabel("Correlation")
    ax.set_title("Cross-correlation: avg pairwise correlation vs. VIX")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_event_study(event_changes: pd.Series, baseline_mean: float, horizon: int, path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    # stat="density" instead of raw counts so the shape of the histogram is
    # comparable no matter how many events got found - the number of events
    # changes depending on the spike threshold and can be a pretty small
    # sample
    sns.histplot(event_changes, bins=15, color=BLUE, ax=ax, stat="density")
    ax.axvline(
        baseline_mean,
        color=VERMILLION,
        linestyle="--",
        linewidth=2,
        label=f"Unconditional {horizon}d mean change",
    )
    ax.axvline(
        event_changes.mean(),
        color=BLUE,
        linestyle="-",
        linewidth=2,
        label="Post-spike mean change",
    )
    ax.set_xlabel(f"VIX % change, {horizon} trading days after event")
    ax.set_title("VIX behavior following correlation-spike events")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
