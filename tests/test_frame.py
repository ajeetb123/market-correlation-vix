"""Tests for the analysis frame builder."""

import pandas as pd

from vixagent.config import Settings
from vixagent.data.align import align_group
from vixagent.features.frame import build_frame

COLUMNS = [
    "vix",
    "log_vix",
    "vix_mom_5d",
    "vix_ratio",
    "vix_spike_day",
    "vix_event",
    "avg_corr",
    "corr_z",
]


def test_frame_columns_and_index(panel: pd.DataFrame, small_settings: Settings) -> None:
    aligned, _ = align_group(panel, small_settings, "risk")
    f = build_frame(aligned, small_settings, "risk", 21)
    assert list(f.columns) == COLUMNS
    assert not any(c.startswith("fwd_") for c in f.columns)
    assert f.index.equals(aligned.index[1:])
    assert f.index.name == "date"


def test_frame_detects_planted_vix_spikes(panel: pd.DataFrame, small_settings: Settings) -> None:
    aligned, _ = align_group(panel, small_settings, "all")
    f = build_frame(aligned, small_settings, "all", 21)
    event_dates = set(f.index[f.vix_event])
    assert {panel.index[p] for p in (300, 700, 1100)} <= event_dates
    assert f.corr_z.notna().sum() > 1000
