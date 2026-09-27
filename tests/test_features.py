"""Tests for backward-looking feature functions."""

import math

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic import correlated_returns, make_dates
from vixagent.features.correlation import avg_pairwise_corr, pairwise_corr_matrix
from vixagent.features.returns import log_returns
from vixagent.features.vix import log_vix, vix_momentum, vix_ratio
from vixagent.features.zscore import trailing_zscore


def _frame(arr: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        arr, index=make_dates(len(arr)), columns=[f"A{i}" for i in range(arr.shape[1])]
    )


def test_log_returns_hand_example() -> None:
    prices = pd.DataFrame({"x": [100.0, 102.0, 99.0]}, index=make_dates(3))
    out = log_returns(prices)
    assert len(out) == 2
    assert out["x"].tolist() == pytest.approx([math.log(1.02), math.log(99 / 102)])


def test_identical_columns_give_one() -> None:
    col = np.random.default_rng(0).standard_normal(100)
    out = avg_pairwise_corr(_frame(np.column_stack([col, col, col])), 20).dropna()
    assert len(out) > 0
    assert out.to_numpy() == pytest.approx(1.0, abs=1e-9)


def test_recovers_true_correlation() -> None:
    rets = _frame(correlated_returns(n=3000, n_assets=3, corr=0.8, seed=1))
    out = avg_pairwise_corr(rets, 252).dropna()
    assert abs(out.mean() - 0.8) < 0.05


@pytest.mark.parametrize("n_assets", [2, 10])
def test_nan_warmup(n_assets: int) -> None:
    window = 30
    rets = _frame(correlated_returns(n=200, n_assets=n_assets, corr=0.5, seed=2))
    out = avg_pairwise_corr(rets, window)
    assert out.iloc[: window - 1].isna().all()
    assert not math.isnan(out.iloc[window - 1])


def test_needs_two_assets() -> None:
    with pytest.raises(ValueError):
        avg_pairwise_corr(_frame(np.ones((10, 1))), 5)


def test_matches_brute_force() -> None:
    window = 40
    rets = _frame(correlated_returns(n=300, n_assets=5, corr=0.4, seed=3))
    out = avg_pairwise_corr(rets, window)
    iu = np.triu_indices(5, k=1)
    for t in np.random.default_rng(4).integers(window - 1, 300, size=5):
        block = rets.iloc[t - window + 1 : t + 1].to_numpy()
        expected = np.corrcoef(block, rowvar=False)[iu].mean()
        assert out.iloc[t] == pytest.approx(expected, abs=1e-10)


def test_zero_variance_pair_is_nan() -> None:
    arr = correlated_returns(n=60, n_assets=3, corr=0.3, seed=5)
    arr[:, 2] = 0.0
    assert avg_pairwise_corr(_frame(arr), 20).isna().all()


def test_pairwise_corr_matrix() -> None:
    rets = _frame(correlated_returns(n=100, n_assets=4, corr=0.5, seed=6))
    m = pairwise_corr_matrix(rets, rets.index[80], 30)
    np.testing.assert_allclose(m.to_numpy(), m.to_numpy().T)
    np.testing.assert_allclose(np.diag(m.to_numpy()), 1.0)
    with pytest.raises(ValueError):
        pairwise_corr_matrix(rets, rets.index[10], 30)


def test_trailing_zscore_hand_example() -> None:
    x = pd.Series([1.0, 2, 3, 4, 10], name="x")
    z = trailing_zscore(x, 3)
    assert z.iloc[:3].isna().all()
    assert z.iloc[3] == pytest.approx(2.0)
    assert z.iloc[4] == pytest.approx(7.0)


def test_constant_series_zscore_is_nan() -> None:
    z = trailing_zscore(pd.Series([5.0] * 20, name="x"), 5)
    assert z.isna().all()
    assert not np.isinf(z).any()


def test_vix_ratio_hand_example() -> None:
    out = vix_ratio(pd.Series([10.0, 10, 10, 15]), 3)
    assert out.iloc[:3].isna().all()
    assert out.iloc[3] == pytest.approx(1.5)


def test_vix_ratio_uses_median() -> None:
    out = vix_ratio(pd.Series([10.0, 60, 10, 15]), 3)
    assert out.iloc[3] == pytest.approx(1.5)


def test_vix_momentum_hand_example() -> None:
    v = pd.Series([10.0, 11, 12, 13, 14, 20])
    out = vix_momentum(v, 5)
    assert out.iloc[:5].isna().all()
    assert out.iloc[5] == pytest.approx(math.log(2.0))


def test_log_vix() -> None:
    assert log_vix(pd.Series([math.e])).iloc[0] == pytest.approx(1.0)
