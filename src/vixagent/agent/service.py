"""Cached facade over the analysis layer; the only thing agent tools call."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from vixagent.agent.methodology import methodology_text
from vixagent.analysis.event_study import (
    Direction,
    EventStudyParams,
    EventStudyResult,
    run_event_study,
)
from vixagent.analysis.frames import FrameStore
from vixagent.analysis.grid import GridRow, overfitting_check, run_grid
from vixagent.analysis.oos import run_oos
from vixagent.analysis.regression import run_regression
from vixagent.config import (
    Group,
    PeriodName,
    Preregistered,
    Settings,
    load_preregistered,
    load_settings,
)
from vixagent.data.cache import load_cached_prices
from vixagent.periods import period_mask
from vixagent.spikes import events_from_days, spike_days
from vixagent.utils.jsonable import to_jsonable

DATE_CAP = 50
DATA_SOURCE = "Yahoo Finance via yfinance, daily adjusted closes; VIX close from ^VIX"
FROZEN_NOTE = "Committed at git tag prereg-freeze before any results were computed."


def _cap(dates: list[str]) -> tuple[list[str], bool]:
    return dates[:DATE_CAP], len(dates) > DATE_CAP


def _capped_event_study(res: EventStudyResult) -> dict[str, Any]:
    """EventStudyResult as a dict with date and flag lists capped at DATE_CAP.

    Why: tool outputs go into the model's context; long lists waste tokens and
    invite the model to reason over raw data instead of summary statistics.
    """
    d = dataclasses.asdict(res)
    d["event_dates"], d["truncated"] = _cap(res.event_dates)
    d["hit_flags"] = res.hit_flags[:DATE_CAP]
    return d


def _grid_summary(row: GridRow | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "group": row.group,
        "window": row.window,
        "z_threshold": row.z_threshold,
        "horizon": row.horizon,
        "train_lift": row.train.lift,
        "train_n_events": row.train.n_events,
        "test_lift": row.test.lift,
        "test_n_events": row.test.n_events,
    }


class ResearchService:
    """Single entry point the tools call. Holds one FrameStore and memoizes
    every result by its arguments, so repeated tool calls are instant and
    identical. All public methods return JSON-safe dicts (via to_jsonable)."""

    def __init__(self, settings: Settings, prereg: Preregistered, prices: pd.DataFrame) -> None:
        self.settings = settings
        self.prereg = prereg
        self.store = FrameStore(prices, settings)
        self._memo: dict[tuple[Any, ...], dict[str, Any]] = {}

    @classmethod
    def from_cache(cls, root: Path | None = None) -> ResearchService:
        """Build a service from config files and the local price cache (no network)."""
        settings = load_settings()
        prereg = load_preregistered(settings)
        return cls(settings, prereg, load_cached_prices(settings, root))

    def _cached(self, key: tuple[Any, ...], fn: Callable[[], Any]) -> dict[str, Any]:
        if key not in self._memo:
            self._memo[key] = to_jsonable(fn())
        return self._memo[key]

    def describe_dataset(self) -> dict[str, Any]:
        """Tickers, date range, trading days, alignment drops, periods, and source."""

        def build() -> dict[str, Any]:
            s = self.settings
            aligned = {g: self.store.aligned(g) for g in s.universe}
            return {
                "tickers": {g: list(t) for g, t in s.universe.items()},
                "vix_ticker": s.data.vix_ticker,
                "start": s.data.start,
                "end": s.data.snapshot_end,
                "n_trading_days": {g: len(a[0]) for g, a in aligned.items()},
                "rows_dropped_in_alignment": {g: a[1] for g, a in aligned.items()},
                "periods": {n: s.period(n) for n in ("train", "test", "full")},
                "data_source": DATA_SOURCE,
            }

        return self._cached(("describe_dataset",), build)

    def correlation_summary(self, group: Group, window: int, period: PeriodName) -> dict[str, Any]:
        """Summary statistics of average correlation over non-NaN days in the period."""

        def build() -> dict[str, Any]:
            f = self.store.frame(group, window)
            in_p = f.loc[period_mask(f.index, self.settings.period(period))]
            c = in_p["avg_corr"].dropna()
            if c.empty:
                raise ValueError(f"No correlation values for {group}, W={window}, {period}")
            latest = c.index[-1]
            return {
                "group": group,
                "window": window,
                "period": period,
                "mean": c.mean(),
                "std": c.std(),
                "min": c.min(),
                "min_date": c.idxmin(),
                "max": c.max(),
                "max_date": c.idxmax(),
                "latest_value": c.iloc[-1],
                "latest_date": latest,
                "latest_z": in_p.loc[latest, "corr_z"],
            }

        return self._cached(("correlation_summary", group, window, period), build)

    def spike_events(
        self,
        kind: Literal["vix", "corr"],
        period: PeriodName,
        group: Group | None = None,
        window: int | None = None,
        z_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Declustered VIX or correlation events in the period, with their definition."""
        s = self.settings
        if kind == "corr" and (group is None or window is None or z_threshold is None):
            raise ValueError("kind='corr' requires group, window, and z_threshold")

        def build() -> dict[str, Any]:
            if kind == "vix":
                f = self.store.frame(group or "risk", window or s.grid.windows[0])
                events = f["vix_event"]
                v = s.vix_spike
                definition = (
                    f"VIX event: first day with VIX / median(prior {v.median_lookback} days) >= "
                    f"{v.ratio_threshold}, with no VIX spike day in the prior {v.cooldown} "
                    "trading days."
                )
            else:
                assert group is not None and window is not None and z_threshold is not None
                f = self.store.frame(group, window)
                events = events_from_days(
                    spike_days(f["corr_z"], z_threshold), s.corr_spike.cooldown
                )
                definition = (
                    f"Correlation event: first day with trailing z-score of {window}-day average "
                    f"pairwise correlation ({group} group, {s.features.z_lookback}-day baseline) "
                    f">= {z_threshold}, with no correlation spike day in the prior "
                    f"{s.corr_spike.cooldown} trading days."
                )
            in_p = period_mask(f.index, s.period(period))
            hits = events.to_numpy(dtype=bool) & in_p
            all_dates = [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(f.index[hits])]
            dates, truncated = _cap(all_dates)
            return {
                "kind": kind,
                "period": period,
                "n_events": len(all_dates),
                "dates": dates,
                "truncated": truncated,
                "definition": definition,
            }

        return self._cached(("spike_events", kind, period, group, window, z_threshold), build)

    def event_study(
        self,
        group: Group,
        window: int,
        z_threshold: float,
        horizon: int,
        period: PeriodName,
        clean_only: bool,
        direction: Direction,
    ) -> dict[str, Any]:
        """Event study result with event_dates and hit_flags capped at 50."""
        params = EventStudyParams(
            group, window, z_threshold, horizon, period, clean_only, direction
        )
        return self._cached(
            ("event_study", params),
            lambda: _capped_event_study(run_event_study(self.store, params, self.settings)),
        )

    def regression(
        self, group: Group, window: int, horizon: int, period: PeriodName, include_controls: bool
    ) -> dict[str, Any]:
        """Predictive regression with HAC standard errors."""
        return self._cached(
            ("regression", group, window, horizon, period, include_controls),
            lambda: run_regression(
                self.store, self.settings, group, window, horizon, period, include_controls
            ),
        )

    def oos(self, group: Group, window: int, horizon: int, z_threshold: float) -> dict[str, Any]:
        """Out-of-sample R squared and train/test event studies (dates capped)."""

        def build() -> dict[str, Any]:
            r = run_oos(self.store, self.settings, group, window, horizon, z_threshold)
            return {
                "r2_oos": r.r2_oos,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "event_study_train": _capped_event_study(r.event_study_train),
                "event_study_test": _capped_event_study(r.event_study_test),
            }

        return self._cached(("oos", group, window, horizon, z_threshold), build)

    def overfitting_check(self) -> dict[str, Any]:
        """Best in-sample grid combination vs the preregistered spec, train and test lift."""

        def build() -> dict[str, Any]:
            check = overfitting_check(run_grid(self.store, self.settings), self.prereg)
            return {
                "n_combinations": check.n_combinations,
                "best_in_sample": _grid_summary(check.best_in_sample),
                "preregistered": _grid_summary(check.preregistered),
            }

        return self._cached(("overfitting_check",), build)

    def preregistered_spec(self) -> dict[str, Any]:
        """The frozen primary specification and its note."""
        return self._cached(
            ("preregistered_spec",),
            lambda: {
                "primary": self.prereg.primary.model_dump(),
                "note": self.prereg.note.strip(),
                "frozen": FROZEN_NOTE,
            },
        )

    def methodology(self, topic: str) -> dict[str, Any]:
        """Canonical methodology text for one topic."""
        return self._cached(
            ("methodology", topic),
            lambda: {"topic": topic, "text": methodology_text(topic, self.settings, self.prereg)},
        )
