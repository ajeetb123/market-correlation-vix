"""Convert analysis outputs into strictly JSON-safe Python objects."""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel


def to_jsonable(obj: Any) -> Any:
    """Recursively convert `obj` so json.dumps(result, allow_nan=False) succeeds.

    Why: numpy scalars are not JSON serializable, and NaN/inf serialize to
    tokens that are not valid JSON. Tool results sent to the model and files
    written to disk must be valid JSON.
    """
    if obj is None or obj is pd.NaT:
        return None
    if isinstance(obj, np.datetime64) and np.isnat(obj):
        return None
    if isinstance(obj, bool | str):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, pd.Timestamp | dt.datetime | dt.date | np.datetime64):
        return pd.Timestamp(obj).strftime("%Y-%m-%d")
    if isinstance(obj, np.ndarray | pd.Series | pd.Index):
        return [to_jsonable(x) for x in list(obj)]
    if isinstance(obj, BaseModel):
        return to_jsonable(obj.model_dump())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [to_jsonable(x) for x in obj]
    raise TypeError(f"to_jsonable: unsupported type {type(obj).__name__}")
