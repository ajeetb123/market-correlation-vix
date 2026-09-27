"""Tests for period and eligibility masks."""

import numpy as np

from tests.fixtures.synthetic import make_dates
from vixagent.periods import eligible_mask, period_mask


def test_full_period_matches_spec_example() -> None:
    dates = make_dates(30)
    m = eligible_mask(dates, (dates[0].date(), dates[-1].date()), 5, np.ones(30, dtype=bool))
    assert list(np.flatnonzero(m)) == list(range(25))


def test_invalid_rows_excluded() -> None:
    dates = make_dates(30)
    valid = np.ones(30, dtype=bool)
    valid[:4] = False
    m = eligible_mask(dates, (dates[0].date(), dates[-1].date()), 5, valid)
    assert list(np.flatnonzero(m)) == list(range(4, 25))


def test_adjacent_periods_embargo() -> None:
    dates = make_dates(30)
    valid = np.ones(30, dtype=bool)
    train = eligible_mask(dates, (dates[0].date(), dates[14].date()), 5, valid)
    test = eligible_mask(dates, (dates[15].date(), dates[-1].date()), 5, valid)
    assert np.flatnonzero(train).max() == 9
    assert np.flatnonzero(train).max() + 5 <= 14
    assert np.flatnonzero(test).min() == 15


def test_empty_period() -> None:
    dates = make_dates(30)
    period = (dates[-1].date().replace(year=2099), dates[-1].date().replace(year=2100))
    assert not eligible_mask(dates, period, 5, np.ones(30, dtype=bool)).any()
    assert not period_mask(dates, period).any()
