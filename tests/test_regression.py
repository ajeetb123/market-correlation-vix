"""Tests for the predictive regression and out-of-sample evaluation."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from vixagent.analysis.frames import FrameStore
from vixagent.analysis.oos import r2_oos, run_oos
from vixagent.analysis.regression import design_matrix, fit_hac, run_regression
from vixagent.config import Settings


def _signal(n: int = 2000, seed: int = 3, beta: float = 0.5) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = beta * x + rng.normal(0, 0.5, n)
    return pd.Series(y, name="y"), pd.DataFrame({"z": x})


def test_fit_hac_recovers_coefficient() -> None:
    y, X = _signal()
    res = fit_hac(y, X, 10)
    assert abs(res.coefs["z"]["coef"] - 0.5) < 0.05
    assert set(res.coefs) == {"const", "z"}
    assert res.n == 2000


def test_hac_maxlags_recorded() -> None:
    y, X = _signal(n=300)
    assert fit_hac(y, X, 7).hac_maxlags == 7


def test_hac_widens_errors_for_overlapping_targets() -> None:
    """Overlapping h-day sums make residuals autocorrelated; HAC must account for it."""
    n, h = 3000, 10
    rng = np.random.default_rng(11)
    e = rng.standard_normal(n + h)
    y = np.array([e[t + 1 : t + h + 1].sum() for t in range(n)])
    x = np.zeros(n)
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + shocks[t]
    X = pd.DataFrame({"z": x})
    hac = fit_hac(pd.Series(y), X, h)
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    assert abs(hac.coefs["z"]["t_hac"]) < abs(float(ols.tvalues["z"]))


@pytest.mark.parametrize(
    ("include_controls", "names"),
    [(False, {"const", "z"}), (True, {"const", "z", "vix_mom_5d", "log_vix"})],
)
def test_run_regression_on_panel(
    panel: pd.DataFrame, small_settings: Settings, include_controls: bool, names: set[str]
) -> None:
    store = FrameStore(panel, small_settings)
    res = run_regression(store, small_settings, "risk", 21, 10, "full", include_controls)
    assert set(res.coefs) == names
    assert res.n > 0
    assert res.hac_maxlags == 10


def test_design_matrix_columns(panel: pd.DataFrame, small_settings: Settings) -> None:
    f = FrameStore(panel, small_settings).frame("risk", 21)
    assert list(design_matrix(f, False).columns) == ["z"]
    assert list(design_matrix(f, True).columns) == ["z", "vix_mom_5d", "log_vix"]


def test_r2_oos_hand_example() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert r2_oos(y, y, 0.0) == 1.0
    assert r2_oos(y, np.array([0.0, 0.0, 0.0]), 0.0) == 0.0


def _split_r2(y: pd.Series, X: pd.DataFrame) -> float:
    cut = int(0.6 * len(y))
    fit = sm.OLS(y[:cut], sm.add_constant(X[:cut])).fit()
    pred = fit.predict(sm.add_constant(X[cut:], has_constant="add"))
    return r2_oos(y[cut:].to_numpy(), np.asarray(pred), float(y[:cut].mean()))


def test_r2_oos_positive_for_signal() -> None:
    y, X = _signal()
    assert _split_r2(y, X) > 0


def test_r2_oos_near_zero_for_noise() -> None:
    y, X = _signal(beta=0.0)
    assert _split_r2(y, X) < 0.02


def test_run_oos_on_panel(panel: pd.DataFrame, small_settings: Settings) -> None:
    res = run_oos(FrameStore(panel, small_settings), small_settings, "risk", 21, 10, 2.0)
    assert res.n_train > 0
    assert res.n_test > 0
    assert np.isfinite(res.r2_oos)
    assert res.event_study_train.n_permutations == small_settings.event_study.n_permutations
