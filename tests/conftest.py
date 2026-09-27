"""Shared pytest fixtures."""

from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures.synthetic import make_price_panel, tiny_settings
from vixagent.config import Settings


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_price_panel(seed=0, spike_positions=[300, 700, 1100])


@pytest.fixture
def small_settings(tmp_path: Path) -> Settings:
    return tiny_settings(tmp_path)
