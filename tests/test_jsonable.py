"""Tests for strict JSON conversion."""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from vixagent.utils.jsonable import to_jsonable


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.float64(1.5), 1.5),
        (np.int64(3), 3),
        (np.bool_(True), True),
        (float("nan"), None),
        (float("inf"), None),
        (pd.Timestamp("2020-03-16"), "2020-03-16"),
        (pd.NaT, None),
        (np.array([1.0, np.nan]), [1.0, None]),
    ],
)
def test_scalar_conversions(value: Any, expected: Any) -> None:
    assert to_jsonable(value) == expected


@dataclass
class _Row:
    when: pd.Timestamp
    value: np.float64
    count: np.int64


def test_nested_dataclass() -> None:
    obj = {"row": _Row(pd.Timestamp("2008-10-24"), np.float64(0.25), np.int64(7))}
    assert to_jsonable(obj) == {"row": {"when": "2008-10-24", "value": 0.25, "count": 7}}


def test_mixed_structure_is_strict_json() -> None:
    obj = {
        "a": (np.float32(1.0), np.nan),
        "b": [pd.Series([1, 2]), {"c": np.inf}],
        3: {np.bool_(False)},
    }
    json.dumps(to_jsonable(obj), allow_nan=False)


def test_bool_stays_bool() -> None:
    out = to_jsonable(True)
    assert out is True


def test_dataframe_rejected() -> None:
    with pytest.raises(TypeError):
        to_jsonable(pd.DataFrame({"x": [1]}))
