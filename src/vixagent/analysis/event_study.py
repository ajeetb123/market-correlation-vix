"""Event study: do source events raise the chance of a target spike soon after?"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from vixagent.analysis.frames import FrameStore
from vixagent.analysis.permutation import circular_shift_pvalue
from vixagent.config import Group, PeriodName, Settings
from vixagent.periods import eligible_mask
from vixagent.spikes import backward_any, events_from_days, spike_days
from vixagent.targets import fwd_any_within

Direction = Literal["corr_to_vix", "vix_to_corr"]


@dataclass(frozen=True)
class EventStudyParams:
    group: Group
    window: int
    z_threshold: float
    horizon: int
    period: PeriodName
    clean_only: bool
    direction: Direction


@dataclass
class EventStudyResult:
    n_events: int
    n_hits: int
    hit_rate: float  # NaN if n_events == 0
    base_rate: float  # NaN if no base days
    lift: float  # NaN if base_rate is 0 or NaN
    p_value: float  # NaN if not computable
    n_permutations: int
    event_dates: list[str]  # YYYY-MM-DD
    hit_flags: list[bool]
    warning: str | None


def event_study_core(
    src_events: np.ndarray,
    tgt_spike_days: np.ndarray,
    eligible: np.ndarray,
    dates: pd.DatetimeIndex,
    horizon: int,
    clean_pre_window: int,
    clean_only: bool,
    n_perm: int,
    seed: int,
) -> EventStudyResult:
    """Hit rate after source events vs the base rate over all eligible days.

    hit(t) = any target spike day in rows t+1..t+horizon (strictly after t).
    Why the clean filter uses target spike days in rows t-clean_pre_window..t:
    in a crash both series jump the same day, and without the filter that
    contemporaneous co-movement would masquerade as prediction. The base rate
    uses the same filter so events are compared with comparable days.
    """
    hit_all = fwd_any_within(pd.Series(tgt_spike_days), horizon).to_numpy()
    dirty = backward_any(pd.Series(tgt_spike_days), clean_pre_window).to_numpy()
    clean = ~dirty if clean_only else np.ones_like(dirty)
    ev = src_events & eligible & clean
    base_days = eligible & clean

    n_events = int(ev.sum())
    n_hits = int(hit_all[ev].sum())
    hit_rate = float(hit_all[ev].mean()) if n_events else float("nan")
    base_rate = float(hit_all[base_days].mean()) if base_days.any() else float("nan")
    lift = hit_rate / base_rate if base_rate > 0 else float("nan")
    p_value = circular_shift_pvalue(
        src_events[eligible],
        hit_all[eligible],
        clean[eligible],
        hit_rate,
        horizon,
        n_perm,
        seed,
    )
    return EventStudyResult(
        n_events=n_events,
        n_hits=n_hits,
        hit_rate=hit_rate,
        base_rate=base_rate,
        lift=lift,
        p_value=p_value,
        n_permutations=n_perm,
        event_dates=[d.strftime("%Y-%m-%d") for d in dates[ev]],
        hit_flags=[bool(x) for x in hit_all[ev]],
        warning="insufficient events (<5)" if n_events < 5 else None,
    )


def run_event_study(
    store: FrameStore, params: EventStudyParams, settings: Settings
) -> EventStudyResult:
    """Run the event study for one parameter set on the cached frame.

    corr_to_vix: source = declustered correlation events, target = VIX spike days.
    vix_to_corr (reverse check): source = VIX events, target = correlation spike
    days. If the reverse direction is as strong, the relationship is
    co-movement rather than a lead.
    """
    f = store.frame(params.group, params.window)
    corr_days = spike_days(f.corr_z, params.z_threshold)
    corr_events = events_from_days(corr_days, settings.corr_spike.cooldown)
    valid = f.corr_z.notna() & f.vix_ratio.notna()
    if params.direction == "corr_to_vix":
        src, tgt = corr_events, f.vix_spike_day
    else:
        src, tgt = f.vix_event, corr_days
    eligible = eligible_mask(f.index, settings.period(params.period), params.horizon, valid)
    return event_study_core(
        src.to_numpy(dtype=bool),
        tgt.to_numpy(dtype=bool),
        eligible,
        pd.DatetimeIndex(f.index),
        params.horizon,
        settings.event_study.clean_pre_window,
        params.clean_only,
        settings.event_study.n_permutations,
        settings.seed,
    )
