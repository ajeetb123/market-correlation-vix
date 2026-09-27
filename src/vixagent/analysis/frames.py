"""Cached analysis frames per (group, window)."""

from __future__ import annotations

import pandas as pd

from vixagent.config import Group, Settings
from vixagent.data.align import align_group
from vixagent.features.frame import build_frame


class FrameStore:
    """Builds and caches analysis frames so each (group, window) pair is built
    once per process.

    Why: the grid, report, and agent all request the same frames repeatedly;
    rebuilding rolling correlations each time is wasteful."""

    def __init__(self, prices: pd.DataFrame, settings: Settings) -> None:
        self.prices = prices
        self.settings = settings
        self._aligned: dict[Group, tuple[pd.DataFrame, int]] = {}
        self._frames: dict[tuple[Group, int], pd.DataFrame] = {}

    def aligned(self, group: Group) -> tuple[pd.DataFrame, int]:
        """Cached align_group result."""
        if group not in self._aligned:
            self._aligned[group] = align_group(self.prices, self.settings, group)
        return self._aligned[group]

    def frame(self, group: Group, window: int) -> pd.DataFrame:
        """Cached build_frame result."""
        key = (group, window)
        if key not in self._frames:
            aligned, _ = self.aligned(group)
            self._frames[key] = build_frame(aligned, self.settings, group, window)
        return self._frames[key]
