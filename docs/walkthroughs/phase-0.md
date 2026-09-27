# Phase 0 Walkthrough: Scaffold

Phase 0 builds the empty shell of the project: an installable Python package (`vixagent`), a configuration loader that refuses bad settings, a JSON-safety helper, a one-command CLI, and a CI workflow. There is no research logic yet.

## How this repo differs from the step file

The step file assumes a brand-new `vix-research-agent` repo. This project reuses the existing `market-correlation-vix` repo instead:

- **No `git init`.** History from the v1 scripts is preserved. The v1 code (`main.py`, `src/*.py`, `output/`, `requirements.txt`) was deleted in the first commit. It is still recoverable from git history.
- **No `gh repo create` and no push.** Commits are local only, so the "CI green on first push" check has not been run yet.
- **`.gitignore` uses `/data/` and `/runs/` instead of `data/` and `runs/`.** The unanchored `data/` pattern also matches `src/vixagent/data/`, which would have silently left the whole data package out of git. The leading slash anchors the pattern to the repo root, which is what the spec intended.
- **`[tool.ruff] extend-exclude = ["docs"]`.** ruff 0.16 formats Python code blocks inside Markdown files, so `ruff format .` was rewriting the spec's step files. The spec docs should never be edited by a formatter.
- **Removed `python_version = "3.11"` from `[tool.mypy]`.** With Python 3.12, pip installs numpy 2.5, whose type stubs use the 3.12-only `type X = ...` statement. Pinning mypy to 3.11 makes it reject those stubs. Without the pin, mypy checks against whichever interpreter runs it, so the 3.11 CI job checks 3.11 and the 3.12 job checks 3.12.
- **`cli.py` has an `@app.callback()`.** Typer collapses an app with a single command into that bare command, so `vixagent version` failed with "unexpected extra argument". The empty callback forces Typer to keep a command group. It will be harmless once more commands exist.
- The venv uses Python 3.12 (`/opt/homebrew/bin/python3.12`) because 3.11 is not installed on this Mac. The spec allows any 3.11+.

---

## `pyproject.toml`

**Purpose.** It declares the package, its dependencies, the `vixagent` command, and the settings for pytest, ruff, and mypy, all in one file.

**Key sections**
- `[project] dependencies`: installed on every `pip install`. These are what the code needs at runtime.
- `[project.optional-dependencies] dev` / `app`: installed only when asked, as in `pip install -e ".[dev]"`. Test and lint tools aren't needed to *run* the pipeline.
- `[project.scripts] vixagent = "vixagent.cli:app"`: pip creates a `vixagent` executable that calls the Typer `app` object.
- `[tool.setuptools.packages.find] where = ["src"]`: the "src layout". The package lives in `src/vixagent/`, so Python can't accidentally import it from the working directory without installing it. Tests therefore always run against the installed package.
- `[tool.pytest.ini_options] addopts = "-m 'not network'"`: tests marked `@pytest.mark.network` are skipped unless you pass `-m network` explicitly.

**Non-obvious decisions**
- `-e` (editable) install links the installed package to your source files, so edits take effect without reinstalling.
- Dependencies set minimum versions (`>=`) rather than exact pins. This fixes the v1 problem of completely unpinned requirements while still allowing security updates. The parquet cache metadata (Phase 1) records the actual yfinance version used.

## `src/vixagent/config.py`

**Purpose.** Loads `config/settings.yaml` and `config/preregistered.yaml` into typed pydantic models, and raises an error immediately if the configuration would produce invalid research.

**Key functions**
- `find_project_root()`: walks up the directories until it finds `pyproject.toml`, so code works no matter which directory you run it from.
- `Settings` (pydantic model): typed view of `settings.yaml`. `_check` enforces cross-field rules.
- `Settings.period(name)`: returns the `(start, end)` dates for `"train"`, `"test"`, or `"full"`.
- `load_settings()` / `load_preregistered()`: read YAML and validate it. `load_preregistered` also requires the primary spec to be one of the grid combinations.

**Worked trace: rejecting an overlapping train period**
1. The test loads the real YAML dict and sets `periods.train = ["2007-05-01", "2020-01-01"]`.
2. `Settings.model_validate(d)`: pydantic first converts types. The strings `"2007-05-01"` become `date(2007, 5, 1)`, and `train` becomes a `tuple[date, date]`.
3. `_check` runs after all fields are parsed (`mode="after"`). It gets to `self.periods.train[1] < self.periods.test[0]`, which is `date(2020, 1, 1) < date(2019, 1, 1)`, which is `False`.
4. It raises `ValueError("periods.train must end before periods.test starts")`, and pydantic wraps it in a `ValidationError`.

**Non-obvious decisions**
- **Why validate at load time?** A train period that overlaps the test period doesn't crash anything. It silently leaks test data into training and produces results that look fine but are invalid. Failing loudly on startup is the only reliable guard.
- **Why must the preregistered spec be in the grid?** Phase 2's overfitting check puts the preregistered row next to the best in-sample grid row. If the spec weren't a grid row, that comparison couldn't be made.
- `cache_path` stays a relative `Path`. Callers resolve it against the project root, so the model doesn't depend on the current directory.

## `src/vixagent/utils/jsonable.py`

**Purpose.** `to_jsonable(obj)` recursively converts analysis outputs (numpy numbers, pandas timestamps, NaN, dataclasses) into plain Python values that `json.dumps(..., allow_nan=False)` accepts.

**Worked trace:** `to_jsonable({"x": np.float64(float("nan")), "d": pd.Timestamp("2020-03-16")})`
1. The top-level object is a `dict`. None of the scalar checks match, so it reaches the `dict` branch and recurses on each value, converting keys with `str(k)`.
2. `"x"`: `np.float64(nan)`.
   - `obj is pd.NaT`: no. `np.datetime64`: no. `bool | str`: no. `np.bool_`: no. `int | np.integer`: no.
   - `float | np.floating`: yes. `f = float(obj) = nan`, and `math.isfinite(nan)` is `False`, so it returns `None`.
3. `"d"`: `pd.Timestamp("2020-03-16")`.
   - It fails every check up to `float`, then matches `pd.Timestamp | dt.datetime | ...` and returns `"2020-03-16"`.
4. The result is `{"x": None, "d": "2020-03-16"}`, and `json.dumps` of that gives `{"x": null, "d": "2020-03-16"}`.

**Non-obvious decisions**
- **Why the order of checks matters:**
  - `bool` is a subclass of `int`, so `True` must be caught before the `int` branch or it would become `1`.
  - `pd.NaT` pretends to be a datetime, so it must be caught before the date branch or `strftime` would fail.
- **Why `pd.DataFrame` raises `TypeError`:** agent tools must return summaries, not raw tables. Refusing DataFrames enforces that at the type level.
- **Why not just `json.dumps(default=str)`?** That would turn NaN into the token `NaN` (invalid JSON) and numbers into strings. The model and the eval graders need real numbers and real `null`s.

## `src/vixagent/cli.py`

**Purpose.** It's the Typer app behind the `vixagent` command. For now it only has `version`. Later phases add `pull`, `report`, `ask`, `chat`, and `eval`.

**Non-obvious decision.** See the `@app.callback()` note at the top of this file.

## `.github/workflows/ci.yml`

**Purpose.** On every push and pull request, GitHub Actions installs the package on Python 3.11 and 3.12, then runs `ruff check`, `ruff format --check`, `mypy src`, and `pytest -q`.

**Non-obvious decisions**
- **Matrix of 3.11 and 3.12:** catches syntax or dependency differences between the two supported versions, such as the numpy stub issue above.
- **No network tests and no secrets:** CI must be deterministic and free. Yahoo rate limits would make network tests flaky, and an API key in CI could leak or cost money. Network tests are marked and deselected by default through `addopts`.

## Status

- `ruff check .`, `ruff format --check .`, `mypy src`, and `pytest -q` (21 tests) all pass locally.
- CI has not run because nothing has been pushed.

---

## Checkpoint Questions (answer in `phase-0-answers.md`)

1. What is the difference between `dependencies` and `[project.optional-dependencies]` in `pyproject.toml`?
2. Why does `json.dumps` fail on a `numpy.float64` or produce invalid JSON for `NaN`?
3. What does CI check on every push, and why are network tests excluded?
