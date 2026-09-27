"""Report figures. Each function saves one PNG (dpi 150) and returns its path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from vixagent.analysis.event_study import EventStudyParams, run_event_study  # noqa: E402
from vixagent.analysis.frames import FrameStore  # noqa: E402
from vixagent.config import PeriodName, Preregistered, Settings  # noqa: E402
from vixagent.features.correlation import pairwise_corr_matrix  # noqa: E402
from vixagent.features.returns import log_returns  # noqa: E402

DPI = 150
HEATMAP_WINDOW = 63


def _save(fig: plt.Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_timeseries(frame_risk21: pd.DataFrame, settings: Settings, out: Path) -> Path:
    """VIX (left axis) and average correlation (right axis), VIX events as
    vertical lines, test period shaded."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frame_risk21.index, frame_risk21["vix"], color="tab:red", lw=0.8, label="VIX")
    ax.set_ylabel("VIX")
    ax2 = ax.twinx()
    ax2.plot(
        frame_risk21.index, frame_risk21["avg_corr"], color="tab:blue", lw=0.8, label="Avg corr"
    )
    ax2.set_ylabel("Average pairwise correlation")
    for d in frame_risk21.index[frame_risk21["vix_event"].to_numpy(dtype=bool)]:
        ax.axvline(d, color="gray", lw=0.6, alpha=0.3)
    t0, t1 = settings.period("test")
    ax.axvspan(
        float(mdates.date2num(t0)),
        float(mdates.date2num(t1)),
        color="gold",
        alpha=0.12,
        label="Test period",
    )
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="upper left")
    ax.set_title("VIX and Average Correlation (risk group, 21-day window)")
    return _save(fig, out)


def _heatmap_dates(vix: pd.Series, earliest: pd.Timestamp) -> list[tuple[str, pd.Timestamp]]:
    """Calm (min VIX), 2008 peak, 2020 peak. A year with no data falls back to
    the full-sample peak so the figure still renders on short samples."""
    eligible = vix.loc[earliest:]
    dates = pd.DatetimeIndex(eligible.index)
    values = eligible.to_numpy(dtype=float)
    picks = [("Calm", dates[int(values.argmin())])]
    for year in (2008, 2020):
        in_year = dates.year == year
        if in_year.any():
            year_dates = dates[in_year]
            picks.append((f"{year} peak", year_dates[int(values[in_year].argmax())]))
        else:
            picks.append(("Peak", dates[int(values.argmax())]))
    return picks


def plot_corr_heatmaps(aligned_all: pd.DataFrame, settings: Settings, out: Path) -> Path:
    """1x3 heatmaps of the all-group correlation matrix (63-day window) at the
    calmest date and the 2008 and 2020 VIX peaks."""
    tickers = settings.universe["all"]
    returns = log_returns(aligned_all[tickers])
    vix = aligned_all[settings.data.vix_ticker].loc[returns.index]
    earliest = returns.index[HEATMAP_WINDOW - 1]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (label, date) in zip(axes, _heatmap_dates(vix, earliest), strict=True):
        m = pairwise_corr_matrix(returns, date, HEATMAP_WINDOW)
        sns.heatmap(
            m, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f", square=True, cbar=False
        )
        ax.set_title(f"{label}: {date.date()} (VIX {vix.loc[date]:.1f})")
    return _save(fig, out)


def plot_event_study(
    results: dict[str, Any],
    store: FrameStore,
    settings: Settings,
    prereg: Preregistered,
    out: Path,
) -> Path:
    """Hit rate vs base rate by horizon at the preregistered group/window/z,
    clean corr_to_vix events, one panel each for train and test."""
    p = prereg.primary
    horizons = settings.grid.horizons
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    periods: tuple[PeriodName, PeriodName] = ("train", "test")
    for ax, period in zip(axes, periods, strict=True):
        res = [
            run_event_study(
                store,
                EventStudyParams(p.group, p.window, p.z_threshold, h, period, True, "corr_to_vix"),
                settings,
            )
            for h in horizons
        ]
        x = np.arange(len(horizons))
        hit = [r.hit_rate for r in res]
        base = [r.base_rate for r in res]
        ax.bar(x - 0.2, hit, 0.4, label="Hit rate after correlation event")
        ax.bar(x + 0.2, base, 0.4, label="Base rate (all eligible days)")
        for i, r in enumerate(res):
            top = np.nanmax([r.hit_rate, r.base_rate, 0.0])
            ax.text(x[i], top + 0.01, f"n={r.n_events}", ha="center", fontsize=9)
        ax.set_xticks(x, [f"h={h}" for h in horizons])
        ax.set_title(f"{period.capitalize()} period")
        ax.set_ylabel("P(VIX spike within h days)")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Event study ({p.group}, W={p.window}, z >= {p.z_threshold}, clean events)")
    return _save(fig, out)


def plot_grid_lift(results: dict[str, Any], out: Path) -> Path:
    """Two heatmaps (train, test) of event-study lift for the risk group:
    rows are (window, z) combinations, columns are horizons."""
    rows = [r for r in results["grid"] if r["group"] == "risk"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, period in zip(axes, ("train", "test"), strict=True):
        df = pd.DataFrame(
            {
                "row": [f"W={r['window']}, z={r['z_threshold']}" for r in rows],
                "col": [f"h={r['horizon']}" for r in rows],
                "lift": [np.nan if r[period]["lift"] is None else r[period]["lift"] for r in rows],
            }
        )
        table = df.pivot(index="row", columns="col", values="lift")
        table = table.reindex(
            index=list(dict.fromkeys(df["row"])), columns=list(dict.fromkeys(df["col"]))
        )
        sns.heatmap(table, ax=ax, annot=True, fmt=".2f", center=1.0, cmap="RdBu_r")
        ax.set_title(f"Lift, risk group ({period})")
        ax.set_xlabel("")
        ax.set_ylabel("")
    return _save(fig, out)
