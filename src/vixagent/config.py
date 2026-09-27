"""Load and validate project configuration from YAML."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Group = Literal["risk", "all"]
PeriodName = Literal["train", "test", "full"]


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: this file) until a directory containing
    pyproject.toml is found. Raises FileNotFoundError if none."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"No pyproject.toml found above {here}")


class DataConfig(BaseModel):
    start: date
    snapshot_end: date
    vix_ticker: str
    cache_path: Path


class PeriodsConfig(BaseModel):
    train: tuple[date, date]
    test: tuple[date, date]


class FeaturesConfig(BaseModel):
    z_lookback: int = Field(gt=1)


class VixSpikeConfig(BaseModel):
    median_lookback: int = Field(gt=0)
    ratio_threshold: float = Field(gt=1.0)
    cooldown: int = Field(ge=0)


class CorrSpikeConfig(BaseModel):
    cooldown: int = Field(ge=0)


class EventStudyConfig(BaseModel):
    clean_pre_window: int = Field(ge=0)
    n_permutations: int = Field(gt=0)


class GridConfig(BaseModel):
    windows: list[int]
    horizons: list[int]
    z_thresholds: list[float]
    groups: list[Group]


class Settings(BaseModel):
    seed: int
    data: DataConfig
    universe: dict[Group, list[str]]
    periods: PeriodsConfig
    features: FeaturesConfig
    vix_spike: VixSpikeConfig
    corr_spike: CorrSpikeConfig
    event_study: EventStudyConfig
    grid: GridConfig

    @model_validator(mode="after")
    def _check(self) -> Settings:
        """Reject configurations that would silently produce invalid research.

        Why: a train period that overlaps the test period, or a group too small
        to average correlations over, would yield results that look valid but
        are not. Failing at load time is cheaper than debugging later.
        """
        if not self.data.start < self.data.snapshot_end:
            raise ValueError("data.start must be before data.snapshot_end")
        for name in ("train", "test"):
            p_start, p_end = getattr(self.periods, name)
            if p_start > p_end:
                raise ValueError(f"periods.{name}: start must be <= end")
        if not self.periods.train[1] < self.periods.test[0]:
            raise ValueError("periods.train must end before periods.test starts")
        if self.periods.test[1] > self.data.snapshot_end:
            raise ValueError("periods.test must end on or before data.snapshot_end")
        if self.periods.train[0] < self.data.start:
            raise ValueError("periods.train must start on or after data.start")
        for key in ("risk", "all"):
            if key not in self.universe:
                raise ValueError(f"universe must define group '{key}'")
        for group, tickers in self.universe.items():
            if len(tickers) < 5:
                raise ValueError(f"universe.{group} needs at least 5 tickers")
            if len(set(tickers)) != len(tickers):
                raise ValueError(f"universe.{group} has duplicate tickers")
        grid = self.grid
        for name, values in (
            ("windows", grid.windows),
            ("horizons", grid.horizons),
            ("z_thresholds", grid.z_thresholds),
        ):
            if not values:
                raise ValueError(f"grid.{name} must be non-empty")
            if any(v <= 0 for v in values):
                raise ValueError(f"grid.{name} values must be positive")
        if not grid.groups:
            raise ValueError("grid.groups must be non-empty")
        return self

    def full_period(self) -> tuple[date, date]:
        """The whole dataset: data.start through data.snapshot_end."""
        return (self.data.start, self.data.snapshot_end)

    def period(self, name: PeriodName) -> tuple[date, date]:
        """Inclusive (start, end) dates for a named period."""
        if name == "train":
            return self.periods.train
        if name == "test":
            return self.periods.test
        if name == "full":
            return self.full_period()
        raise ValueError(f"unknown period: {name}")


class PrimarySpec(BaseModel):
    group: Group
    window: int
    z_threshold: float
    horizon: int
    clean_only: bool
    direction: Literal["corr_to_vix", "vix_to_corr"]
    alpha: float = Field(gt=0, lt=1)


class Preregistered(BaseModel):
    primary: PrimarySpec
    note: str


def load_settings(path: Path | None = None) -> Settings:
    """Load settings. Default path: <project root>/config/settings.yaml."""
    path = path or find_project_root() / "config" / "settings.yaml"
    with open(path) as fh:
        return Settings.model_validate(yaml.safe_load(fh))


def load_preregistered(settings: Settings, path: Path | None = None) -> Preregistered:
    """Load the preregistered spec. Default path: <project root>/config/preregistered.yaml.

    Raises ValueError unless primary.window, horizon, z_threshold, and group are
    each members of the corresponding settings.grid list.

    Why: the overfitting check compares the preregistered spec against the best
    grid combination, which requires the spec to be one of the grid rows.
    """
    path = path or find_project_root() / "config" / "preregistered.yaml"
    with open(path) as fh:
        prereg = Preregistered.model_validate(yaml.safe_load(fh))
    p, g = prereg.primary, settings.grid
    checks = (
        ("window", p.window, g.windows),
        ("horizon", p.horizon, g.horizons),
        ("z_threshold", p.z_threshold, g.z_thresholds),
        ("group", p.group, g.groups),
    )
    for name, value, allowed in checks:
        if value not in allowed:
            raise ValueError(f"preregistered {name}={value} is not in grid {allowed}")
    return prereg
