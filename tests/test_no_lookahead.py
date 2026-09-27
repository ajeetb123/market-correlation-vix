"""No-lookahead guarantees: truncation invariance and a static shift(-) check."""

import re

import numpy as np
import pandas as pd
import pytest

from vixagent.config import Settings, find_project_root
from vixagent.data.align import align_group
from vixagent.features.frame import build_frame
from vixagent.spikes import backward_any, events_from_days, spike_days


def _cut_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    rng = np.random.default_rng(123)
    cuts = rng.integers(int(0.4 * len(index)), int(0.95 * len(index)), size=5)
    return [index[c] for c in cuts]


@pytest.mark.parametrize("window", [21, 63])
def test_build_frame_has_no_lookahead(
    panel: pd.DataFrame, small_settings: Settings, window: int
) -> None:
    aligned, _ = align_group(panel, small_settings, "risk")
    full = build_frame(aligned, small_settings, "risk", window)
    for cut_date in _cut_dates(aligned.index):
        trunc = build_frame(aligned.loc[:cut_date], small_settings, "risk", window)
        pd.testing.assert_frame_equal(
            full.loc[:cut_date], trunc, check_exact=False, atol=1e-12, rtol=0, check_freq=False
        )


def test_spikes_have_no_lookahead(panel: pd.DataFrame, small_settings: Settings) -> None:
    aligned, _ = align_group(panel, small_settings, "risk")
    corr_z = build_frame(aligned, small_settings, "risk", 21).corr_z

    def pipeline(z: pd.Series) -> pd.DataFrame:
        days = spike_days(z, 2.0)
        return pd.DataFrame(
            {
                "days": days,
                "events": events_from_days(days, 20),
                "dirty": backward_any(days, 5),
            }
        )

    full = pipeline(corr_z)
    for cut_date in _cut_dates(corr_z.index):
        trunc = pipeline(corr_z.loc[:cut_date])
        pd.testing.assert_frame_equal(full.loc[:cut_date], trunc, check_freq=False)


def test_only_targets_looks_forward() -> None:
    root = find_project_root() / "src" / "vixagent"
    pattern = re.compile(r"shift\(\s*-")
    offenders = [
        p for p in root.rglob("*.py") if pattern.search(p.read_text()) and p.name != "targets.py"
    ]
    assert offenders == []


def test_truncation_check_catches_full_sample_standardization(
    panel: pd.DataFrame, small_settings: Settings
) -> None:
    """Proves the truncation test has teeth: a leaky z-score must fail it."""
    aligned, _ = align_group(panel, small_settings, "risk")
    avg_corr = build_frame(aligned, small_settings, "risk", 21).avg_corr

    def leaky_z(x: pd.Series) -> pd.DataFrame:
        return ((x - x.mean()) / x.std()).to_frame("z")

    full = leaky_z(avg_corr)
    cut_date = _cut_dates(avg_corr.index)[0]
    trunc = leaky_z(avg_corr.loc[:cut_date])
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            full.loc[:cut_date], trunc, check_exact=False, atol=1e-12, rtol=0, check_freq=False
        )
