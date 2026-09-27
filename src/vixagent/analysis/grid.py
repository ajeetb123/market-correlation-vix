"""Exploratory parameter grid and the overfitting check."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

from vixagent.analysis.event_study import EventStudyParams, EventStudyResult, run_event_study
from vixagent.analysis.frames import FrameStore
from vixagent.config import Group, Preregistered, Settings


@dataclass
class GridRow:
    group: Group
    window: int
    z_threshold: float
    horizon: int
    train: EventStudyResult
    test: EventStudyResult


@dataclass
class OverfittingCheck:
    n_combinations: int
    best_in_sample: GridRow | None  # None if no combination has >= 5 train events
    preregistered: GridRow


def run_grid(store: FrameStore, settings: Settings) -> list[GridRow]:
    """Every combination of grid groups x windows x z_thresholds x horizons,
    corr_to_vix, clean_only=True, on train and test. Deterministic order:
    group, window, z_threshold, horizon (each in config order)."""
    g = settings.grid
    rows: list[GridRow] = []
    for group, window, z, h in product(g.groups, g.windows, g.z_thresholds, g.horizons):
        train, test = (
            run_event_study(
                store, EventStudyParams(group, window, z, h, period, True, "corr_to_vix"), settings
            )
            for period in ("train", "test")
        )
        rows.append(GridRow(group, window, z, h, train, test))
    return rows


def overfitting_check(rows: list[GridRow], prereg: Preregistered) -> OverfittingCheck:
    """Select the row with the highest train lift among rows with train
    n_events >= 5 (ties broken by grid order). Also find the row matching the
    preregistered spec.

    Why: whichever combination looks best in-sample is partly luck; its test
    performance shows how much of the in-sample result was noise."""
    best: GridRow | None = None
    for row in rows:
        lift = row.train.lift
        if row.train.n_events < 5 or math.isnan(lift):
            continue
        if best is None or lift > best.train.lift:
            best = row
    p = prereg.primary
    matches = [
        r
        for r in rows
        if (r.group, r.window, r.z_threshold, r.horizon)
        == (p.group, p.window, p.z_threshold, p.horizon)
    ]
    if not matches:
        raise ValueError("preregistered spec is not a grid combination")
    return OverfittingCheck(n_combinations=len(rows), best_in_sample=best, preregistered=matches[0])
