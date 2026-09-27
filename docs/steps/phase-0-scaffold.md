# Phase 0: Scaffold

Goal: an installable, linted, type-checked, tested, CI-green empty project with config loading and JSON safety in place.

Read first: `CLAUDE.md`, `docs/SPEC.md` sections 2, 3, 13.

Every step ends with **Verify** (a command and its expected result) and **Commit** (the exact message). Do not proceed past a failing Verify.

---

## Step 0.1: Create the repo and directory tree

```bash
mkdir vix-research-agent && cd vix-research-agent
git init -b main
mkdir -p config src/vixagent/{data,features,analysis,report,agent,evals,utils} \
  tests/fixtures evals/cases evals/results reports/figures docs/steps docs/walkthroughs \
  .github/workflows app
```

Create empty package markers (each with a one-line module docstring):
```
src/vixagent/__init__.py              """VIX research agent."""   plus  __version__ = "0.1.0"
src/vixagent/data/__init__.py
src/vixagent/features/__init__.py
src/vixagent/analysis/__init__.py
src/vixagent/report/__init__.py
src/vixagent/agent/__init__.py
src/vixagent/evals/__init__.py
src/vixagent/utils/__init__.py
tests/__init__.py
tests/fixtures/__init__.py
```

`tests/__init__.py` and `tests/fixtures/__init__.py` exist so tests can `from tests.fixtures.synthetic import ...`.

Add `.gitkeep` to `evals/cases`, `reports/figures`, `docs/walkthroughs`.

Copy `CLAUDE.md` to the repo root and `SPEC.md`, `PLAN.md`, `EVALS.md`, `00-FOUNDATION.md`, `README.md` (docs index), and `steps/*.md` into `docs/`.

**Verify**: `find . -name "__init__.py" | wc -l` prints `10`.
**Commit**: none yet (commit after Step 0.3).

---

## Step 0.2: `pyproject.toml`

Write exactly:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vixagent"
version = "0.1.0"
description = "Preregistered test of whether cross-asset correlation spikes lead VIX spikes, with a tool-using research agent and eval harness."
readme = "README.md"
requires-python = ">=3.11"
authors = [{ name = "Ajeet Bondugula" }]
dependencies = [
  "yfinance>=0.2.54",
  "pandas>=2.2",
  "numpy>=1.26",
  "scipy>=1.12",
  "statsmodels>=0.14",
  "matplotlib>=3.8",
  "seaborn>=0.13",
  "pyarrow>=15",
  "anthropic>=0.40",
  "pydantic>=2.6",
  "pyyaml>=6",
  "typer>=0.12",
  "rich>=13",
  "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8",
  "pytest-cov>=5",
  "ruff>=0.5",
  "mypy>=1.10",
  "pandas-stubs",
  "types-PyYAML",
  "jsonschema>=4",
]
app = ["streamlit>=1.35"]

[project.scripts]
vixagent = "vixagent.cli:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not network'"
markers = ["network: hits the real network; deselected by default"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
```

Note: `jsonschema` is a dev-only addition used by the Phase 3 schema-validity test. It is pre-approved.

**Verify**: nothing yet (install happens in 0.4).

---

## Step 0.3: `.gitignore`, `.env.example`

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
*.egg-info/
build/
dist/
.env
data/
runs/
evals/results/*/
.DS_Store
```

`.env.example`:
```
ANTHROPIC_API_KEY=
VIX_AGENT_MODEL=claude-sonnet-5
VIX_JUDGE_MODEL=claude-sonnet-5
```

**Verify**: `git status` does not list `.env` (it doesn't exist yet; fine).
**Commit**: `chore: initialize repository layout and project metadata`

---

## Step 0.4: Virtual environment and minimal CLI

Create `src/vixagent/cli.py`:
```python
"""Command-line interface for vixagent."""

import typer

from vixagent import __version__

app = typer.Typer(help="VIX research agent CLI.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
```

Install:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

**Verify**: `vixagent version` prints `0.1.0`.
**Commit**: `feat: add minimal Typer CLI entry point`

---

## Step 0.5: Config files

Create `config/settings.yaml` and `config/preregistered.yaml` **exactly** as in SPEC section 3, including comments.

**Verify**: `python -c "import yaml; print(yaml.safe_load(open('config/settings.yaml'))['universe']['risk'])"` prints the 6 risk tickers.

---

## Step 0.6: `src/vixagent/config.py`

Requirements:

```python
"""Load and validate project configuration from YAML."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Group = Literal["risk", "all"]


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: this file) until a directory containing
    pyproject.toml is found. Raises FileNotFoundError if none."""


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
        # raise ValueError with a clear message if any fails:
        # 1. data.start < data.snapshot_end
        # 2. each period: start <= end
        # 3. periods.train[1] < periods.test[0]
        # 4. periods.test[1] <= data.snapshot_end, periods.train[0] >= data.start
        # 5. every universe group has >= 5 tickers, no duplicates
        # 6. both "risk" and "all" keys exist
        # 7. every grid list non-empty and all values positive
        ...

    def full_period(self) -> tuple[date, date]:
        return (self.data.start, self.data.snapshot_end)

    def period(self, name: Literal["train", "test", "full"]) -> tuple[date, date]: ...


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
    """Default path: <project root>/config/settings.yaml."""


def load_preregistered(settings: Settings, path: Path | None = None) -> Preregistered:
    """Default path: <project root>/config/preregistered.yaml.
    Raises ValueError unless primary.window, horizon, z_threshold, and group are
    each members of the corresponding settings.grid list."""
```

Relative `cache_path` values are resolved against the project root by callers (not in the model).

**Verify**: `python -c "from vixagent.config import load_settings, load_preregistered as lp; s=load_settings(); print(lp(s).primary)"` prints the primary spec.
**Commit**: `feat: add validated configuration loading`

---

## Step 0.7: `src/vixagent/utils/jsonable.py`

Implement exactly this logic (order of checks matters: `bool` is a subclass of `int`, and `pd.NaT` must be caught before date handling):

```python
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
```

Note: `pd.DataFrame` is intentionally unsupported. Tools must not return tables of raw data.

**Verify**: covered by tests in 0.9.
**Commit**: `feat: add to_jsonable utility for strict JSON output`

---

## Step 0.8: README placeholder

`README.md`:
```markdown
# VIX Research Agent

Work in progress.

<!-- RESULTS:START -->
<!-- RESULTS:END -->

<!-- EVALS:START -->
<!-- EVALS:END -->
```

**Commit**: `docs: add README placeholder with generated-content markers`

---

## Step 0.9: Tests

`tests/test_config.py`:
1. `test_loads_real_config`: `load_settings()` succeeds; `len(s.universe["risk"]) == 6`; `s.periods.train[1] < s.periods.test[0]`.
2. `test_loads_preregistered`: `load_preregistered(load_settings()).primary.window == 21`.
3. `test_rejects_train_after_test`: load the real YAML dict, set `periods.train = ["2007-05-01", "2020-01-01"]`, `Settings.model_validate(d)` raises `ValidationError`.
4. `test_rejects_small_group`: set `universe.risk` to 3 tickers, raises.
5. `test_rejects_prereg_not_in_grid`: write a temp preregistered YAML with `window: 30` using `tmp_path`, `load_preregistered(s, path)` raises `ValueError`.
6. `test_find_project_root`: returned path contains `pyproject.toml`.

`tests/test_jsonable.py`:
1. Each of: `np.float64(1.5)`, `np.int64(3)`, `np.bool_(True)`, `float("nan")` to `None`, `float("inf")` to `None`, `pd.Timestamp("2020-03-16")` to `"2020-03-16"`, `pd.NaT` to `None`, `np.array([1.0, np.nan])` to `[1.0, None]`.
2. Nested dict with a dataclass containing a Timestamp and numpy values converts correctly.
3. `json.dumps(to_jsonable(x), allow_nan=False)` succeeds for a mixed nested structure.
4. `True` stays `True` (not `1`).
5. `pd.DataFrame(...)` raises `TypeError`.

**Verify**: `pytest -q` shows all tests passing.
**Commit**: `test: cover config validation and JSON conversion`

---

## Step 0.10: CI workflow

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Lint
        run: |
          ruff check .
          ruff format --check .
      - name: Type check
        run: mypy src
      - name: Test
        run: pytest -q
```

**Commit**: `ci: add lint, type check, and test workflow`

---

## Step 0.11: Definition of Done and first push

```bash
ruff format .
ruff check .
ruff format --check .
mypy src
pytest -q
```

All must succeed. Then (human confirms repo visibility first):
```bash
gh repo create vix-research-agent --public --source=. --remote=origin --push
```

**Verify**: `gh run watch` (or the Actions tab) shows CI green on both Python versions.

---

## Step 0.12: Walkthrough and stop

Write `docs/walkthroughs/phase-0.md` per the CLAUDE.md format, covering `config.py`, `utils/jsonable.py`, `cli.py`, `pyproject.toml`, and `ci.yml`. Include a worked trace of `to_jsonable({"x": np.float64(float("nan")), "d": pd.Timestamp("2020-03-16")})`.

Append the Phase 0 Checkpoint Questions from `docs/PLAN.md`.

**Commit**: `docs: add phase 0 walkthrough`

**STOP.** Report: files created, test count, CI status. Wait for the human.
