"""Out-of-sample evaluation: fit on train, score on test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm

from vixagent.analysis.event_study import EventStudyParams, EventStudyResult, run_event_study
from vixagent.analysis.frames import FrameStore
from vixagent.analysis.regression import design_matrix
from vixagent.config import Group, PeriodName, Settings
from vixagent.periods import eligible_mask
from vixagent.targets import fwd_log_vix_change


@dataclass
class OOSResult:
    """Out-of-sample R squared plus train and test event studies."""

    r2_oos: float
    n_train: int
    n_test: int
    event_study_train: EventStudyResult
    event_study_test: EventStudyResult


def r2_oos(y_test: np.ndarray, y_pred: np.ndarray, train_mean: float) -> float:
    """1 - SSE(model) / SSE(historical mean). Positive = beats the benchmark.

    Why the train mean as benchmark: a forecaster standing at the end of the
    training period only knows the training average. Beating it is the
    minimum bar for a useful predictor.
    """
    y_test, y_pred = np.asarray(y_test, float), np.asarray(y_pred, float)
    sse_model = float(np.sum((y_test - y_pred) ** 2))
    sse_bench = float(np.sum((y_test - train_mean) ** 2))
    return 1.0 - sse_model / sse_bench if sse_bench > 0 else float("nan")


def run_oos(
    store: FrameStore,
    settings: Settings,
    group: Group,
    window: int,
    horizon: int,
    z_threshold: float,
) -> OOSResult:
    """Fit the base spec (correlation z-score only) on train eligible days and
    predict test eligible days.

    The question is whether the correlation z-score alone forecasts VIX
    changes better than the historical average. Also runs the clean
    corr_to_vix event study separately on train and on test.
    """
    f = store.frame(group, window)
    X = design_matrix(f, include_controls=False)
    y = fwd_log_vix_change(f["vix"], horizon)
    valid = X.notna().all(axis=1) & y.notna()
    tr = eligible_mask(f.index, settings.period("train"), horizon, valid)
    te = eligible_mask(f.index, settings.period("test"), horizon, valid)
    y_tr, y_te = y[tr], y[te]
    model = sm.OLS(y_tr, sm.add_constant(X[tr], has_constant="add")).fit()
    y_pred = model.predict(sm.add_constant(X[te], has_constant="add"))
    score = r2_oos(y_te.to_numpy(), np.asarray(y_pred), float(y_tr.mean()))

    def _es(period: PeriodName) -> EventStudyResult:
        params = EventStudyParams(group, window, z_threshold, horizon, period, True, "corr_to_vix")
        return run_event_study(store, params, settings)

    return OOSResult(
        r2_oos=score,
        n_train=int(tr.sum()),
        n_test=int(te.sum()),
        event_study_train=_es("train"),
        event_study_test=_es("test"),
    )
