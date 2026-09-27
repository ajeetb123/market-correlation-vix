"""Real-network smoke test. Deselected by default; run with `pytest -m network`."""

from datetime import date

import pytest

from vixagent.data.fetch import fetch_prices


@pytest.mark.network
def test_real_fetch_small() -> None:
    df = fetch_prices(["SPY", "^VIX"], date(2024, 1, 2), date(2024, 1, 31))
    assert list(df.columns) == ["SPY", "^VIX"] and len(df) >= 19
