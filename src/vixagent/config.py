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
    """Data source and cache settings."""

    start: date
    snapshot_end: date
    vix_ticker: str
    cache_path: Path


class PeriodsConfig(BaseModel):
    """Inclusive train and test date ranges."""

    train: tuple[date, date]
    test: tuple[date, date]


class FeaturesConfig(BaseModel):
    """Feature construction settings."""

    z_lookback: int = Field(gt=1)


class VixSpikeConfig(BaseModel):
    """VIX spike definition: ratio to prior median, threshold, and cooldown."""

    median_lookback: int = Field(gt=0)
    ratio_threshold: float = Field(gt=1.0)
    cooldown: int = Field(ge=0)


class CorrSpikeConfig(BaseModel):
    """Correlation spike declustering settings."""

    cooldown: int = Field(ge=0)


class EventStudyConfig(BaseModel):
    """Event study clean-filter window and permutation count."""

    clean_pre_window: int = Field(ge=0)
    n_permutations: int = Field(gt=0)


class GridConfig(BaseModel):
    """Parameter values for the exploratory grid."""

    windows: list[int]
    horizons: list[int]
    z_thresholds: list[float]
    groups: list[Group]


class Settings(BaseModel):
    """Validated contents of config/settings.yaml."""

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
    """The preregistered primary test specification."""

    group: Group
    window: int
    z_threshold: float
    horizon: int
    clean_only: bool
    direction: Literal["corr_to_vix", "vix_to_corr"]
    alpha: float = Field(gt=0, lt=1)


class Preregistered(BaseModel):
    """Validated contents of config/preregistered.yaml."""

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


class FollowupPeriods(BaseModel):
    """Holdout (untouched before the follow-up freeze) and exploration date ranges."""

    holdout: tuple[date, date]
    exploration: tuple[date, date]


class FollowupConfig(BaseModel):
    """Validated contents of config/followup.yaml."""

    universe: list[str]
    data: DataConfig
    periods: FollowupPeriods


def load_followup(base: Settings, path: Path | None = None) -> tuple[Settings, FollowupConfig]:
    """Load config/followup.yaml and derive Settings for the sector universe.

    Both groups ('risk' and 'all') are set to the sector list so the existing
    frame and feature code can be reused unchanged. Settings periods map
    train = holdout and test = exploration only to satisfy validation; the
    follow-up code always passes explicit date ranges.
    """
    path = path or find_project_root() / "config" / "followup.yaml"
    with open(path) as fh:
        cfg = FollowupConfig.model_validate(yaml.safe_load(fh))
    d = base.model_dump()
    d["universe"] = {"risk": list(cfg.universe), "all": list(cfg.universe)}
    d["data"].update(cfg.data.model_dump())
    d["periods"] = {"train": cfg.periods.holdout, "test": cfg.periods.exploration}
    return Settings.model_validate(d), cfg


class FollowupPrimary(BaseModel):
    """The preregistered follow-up test."""

    window: int = Field(ge=10, le=126)
    horizon: int = Field(ge=1, le=60)
    include_controls: bool
    alpha: float = Field(gt=0, lt=1)


class FollowupPrereg(BaseModel):
    """Validated contents of config/preregistered_followup.yaml."""

    hypothesis: str
    primary: FollowupPrimary
    note: str


def load_followup_prereg(path: Path | None = None) -> FollowupPrereg:
    """Load config/preregistered_followup.yaml. Raises FileNotFoundError if absent."""
    path = path or find_project_root() / "config" / "preregistered_followup.yaml"
    with open(path) as fh:
        return FollowupPrereg.model_validate(yaml.safe_load(fh))
