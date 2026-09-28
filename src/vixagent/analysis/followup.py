"""Follow-up study: does high average correlation signal that a VIX spike is peaking?"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from vixagent.analysis.frames import FrameStore
from vixagent.analysis.regression import RegressionResult, design_matrix, fit_hac
from vixagent.config import Group
from vixagent.periods import eligible_mask
from vixagent.targets import fwd_log_vix_change


@dataclass
class FollowupResult:
    """Regression of the forward VIX change on correlation z over one date range."""

    period: tuple[date, date]
    include_controls: bool
    regression: RegressionResult
    coef_z: float
    t_hac_z: float
    p_one_sided: float  # H1: coefficient on z is negative


def one_sided_p_negative(t_stat: float, p_two_sided: float) -> float:
    """One-sided p-value for H1: coefficient < 0, from a two-sided HAC test.

    Why one-sided: the follow-up hypothesis is directional (high correlation
    precedes a VIX decline), fixed before the holdout data was examined.
    """
    return p_two_sided / 2 if t_stat < 0 else 1 - p_two_sided / 2


def run_followup(
    store: FrameStore,
    group: Group,
    window: int,
    horizon: int,
    period: tuple[date, date],
    include_controls: bool,
) -> FollowupResult:
    """OLS of the forward h-day log VIX change on the correlation z-score
    (plus VIX momentum and log level when include_controls) over eligible days
    in `period`, with HAC standard errors (maxlags = horizon).

    Why controls: the VIX mean-reverts after spikes on its own, and correlation
    tends to be high right after a spike. Only a negative z coefficient that
    survives the VIX's own momentum and level says correlation adds information.
    """
    f = store.frame(group, window)
    X = design_matrix(f, include_controls)
    y = fwd_log_vix_change(f["vix"], horizon)
    valid = X.notna().all(axis=1) & y.notna()
    m = eligible_mask(f.index, period, horizon, valid)
    reg = fit_hac(y[m], X[m], horizon)
    z = reg.coefs["z"]
    return FollowupResult(
        period=period,
        include_controls=include_controls,
        regression=reg,
        coef_z=z["coef"],
        t_hac_z=z["t_hac"],
        p_one_sided=one_sided_p_negative(z["t_hac"], z["p_hac"]),
    )
