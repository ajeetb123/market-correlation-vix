"""Predictive regression of forward VIX change on the correlation z-score."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

from vixagent.analysis.frames import FrameStore
from vixagent.config import Group, PeriodName, Settings
from vixagent.periods import eligible_mask
from vixagent.targets import fwd_log_vix_change


@dataclass
class RegressionResult:
    n: int
    r2: float
    hac_maxlags: int
    coefs: dict[str, dict[str, float]]  # name -> {"coef", "t_hac", "p_hac"}


def design_matrix(frame: pd.DataFrame, include_controls: bool) -> pd.DataFrame:
    """Columns: z (= corr_z) and, if include_controls, vix_mom_5d and log_vix."""
    X = pd.DataFrame({"z": frame["corr_z"]}, index=frame.index)
    if include_controls:
        X["vix_mom_5d"] = frame["vix_mom_5d"]
        X["log_vix"] = frame["log_vix"]
    return X


def fit_hac(y: pd.Series, X: pd.DataFrame, maxlags: int) -> RegressionResult:
    """OLS of y on X plus a constant with Newey-West (HAC) standard errors.

    Why HAC: when y is an h-day forward change, neighbouring rows share h-1
    days of their outcome window, so residuals are autocorrelated and plain
    OLS standard errors are too small. maxlags = h covers that overlap.
    """
    res = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    coefs = {
        str(name): {
            "coef": float(res.params[name]),
            "t_hac": float(res.tvalues[name]),
            "p_hac": float(res.pvalues[name]),
        }
        for name in res.params.index
    }
    return RegressionResult(
        n=int(res.nobs), r2=float(res.rsquared), hac_maxlags=maxlags, coefs=coefs
    )


def run_regression(
    store: FrameStore,
    settings: Settings,
    group: Group,
    window: int,
    horizon: int,
    period: PeriodName,
    include_controls: bool,
) -> RegressionResult:
    """OLS of fwd_log_vix_change_h on the design matrix over eligible days,
    with Newey-West (HAC) standard errors, maxlags = horizon.

    Why HAC: h-day forward windows overlap, so residuals are autocorrelated
    and ordinary standard errors are too small.
    Why controls: tests whether correlation adds information beyond VIX
    momentum and level."""
    f = store.frame(group, window)
    X = design_matrix(f, include_controls)
    y = fwd_log_vix_change(f["vix"], horizon)
    valid = X.notna().all(axis=1) & y.notna()
    m = eligible_mask(f.index, settings.period(period), horizon, valid)
    return fit_hac(y[m], X[m], horizon)
