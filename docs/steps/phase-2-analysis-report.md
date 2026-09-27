# Phase 2: Analysis and Report

Goal: implement the event study, permutation test, regressions, out-of-sample test, and overfitting check; generate figures and results files; publish v0.1.

Read first: `CLAUDE.md`, `docs/SPEC.md` sections 8 and 9, and your Phase 1 walkthrough.

Module map:
```
src/vixagent/analysis/event_study.py   EventStudyParams, EventStudyResult, event_study_core, run_event_study
src/vixagent/analysis/permutation.py   circular_shift_pvalue
src/vixagent/analysis/regression.py    RegressionResult, design_matrix, run_regression
src/vixagent/analysis/oos.py           OOSResult, run_oos
src/vixagent/analysis/grid.py          GridRow, OverfittingCheck, run_grid, overfitting_check
src/vixagent/analysis/frames.py        FrameStore (builds and caches frames per group/window)
src/vixagent/report/figures.py
src/vixagent/report/results.py
src/vixagent/report/readme.py
```

---

## Step 2.1: `analysis/frames.py`

```python
class FrameStore:
    """Builds and caches analysis frames so each (group, window) pair is built
    once per process.

    Why: the grid, report, and agent all request the same frames repeatedly;
    rebuilding rolling correlations each time is wasteful."""

    def __init__(self, prices: pd.DataFrame, settings: Settings) -> None: ...

    def aligned(self, group: Group) -> tuple[pd.DataFrame, int]:
        """Cached align_group result."""

    def frame(self, group: Group, window: int) -> pd.DataFrame:
        """Cached build_frame result."""
```

This module holds state but does no I/O, so it may live in `analysis/`.

**Tests**: two calls with the same args return the identical object (`is`); different windows return different frames.
**Commit**: `feat: add cached frame store`

---

## Step 2.2: Event study core (`analysis/event_study.py`)

```python
Direction = Literal["corr_to_vix", "vix_to_corr"]
PeriodName = Literal["train", "test", "full"]


@dataclass(frozen=True)
class EventStudyParams:
    group: Group
    window: int
    z_threshold: float
    horizon: int
    period: PeriodName
    clean_only: bool
    direction: Direction


@dataclass
class EventStudyResult:
    n_events: int
    n_hits: int
    hit_rate: float          # NaN if n_events == 0
    base_rate: float         # NaN if no base days
    lift: float              # NaN if base_rate is 0 or NaN
    p_value: float           # NaN if not computable
    n_permutations: int
    event_dates: list[str]   # YYYY-MM-DD
    hit_flags: list[bool]
    warning: str | None


def event_study_core(
    src_events: np.ndarray,       # bool, source events (already declustered)
    tgt_spike_days: np.ndarray,   # bool, target spike days (not declustered)
    eligible: np.ndarray,         # bool, from periods.eligible_mask
    dates: pd.DatetimeIndex,
    horizon: int,
    clean_pre_window: int,
    clean_only: bool,
    n_perm: int,
    seed: int,
) -> EventStudyResult:
```

Algorithm (all arrays same length, positions aligned to `dates`):
1. `hit_all = fwd_any_within(pd.Series(tgt_spike_days), horizon).to_numpy()` (import from `targets.py`).
2. `dirty = backward_any(pd.Series(tgt_spike_days), clean_pre_window).to_numpy()`.
3. `clean = ~dirty if clean_only else np.ones_like(dirty)`.
4. `ev = src_events & eligible & clean`; `base_days = eligible & clean`.
5. `n_events = int(ev.sum())`, `n_hits = int(hit_all[ev].sum())`.
6. `hit_rate = hit_all[ev].mean()` if `n_events` else NaN; `base_rate = hit_all[base_days].mean()` if `base_days.any()` else NaN; `lift = hit_rate / base_rate` if `base_rate > 0` else NaN.
7. `p_value = circular_shift_pvalue(src_events[eligible], hit_all[eligible], clean[eligible], hit_rate, horizon, n_perm, seed)`.
8. `warning = "insufficient events (<5)"` if `n_events < 5` else `None`.

Why the clean filter uses **target** spike days in both directions: for `corr_to_vix` it removes correlation events where the VIX was already spiking; for `vix_to_corr` it removes VIX events where correlation was already spiking. That's exactly SPEC 8.2, expressed once.

```python
def run_event_study(store: FrameStore, params: EventStudyParams, settings: Settings) -> EventStudyResult:
```
1. `f = store.frame(params.group, params.window)`.
2. `corr_days = spike_days(f.corr_z, params.z_threshold)`; `corr_events = events_from_days(corr_days, settings.corr_spike.cooldown)`.
3. `valid = f.corr_z.notna() & f.vix_ratio.notna()`.
4. If `corr_to_vix`: `src = corr_events`, `tgt = f.vix_spike_day`. Else `src = f.vix_event`, `tgt = corr_days`.
5. `eligible = eligible_mask(f.index, settings.period(params.period), params.horizon, valid)`.
6. Call `event_study_core` with `settings.event_study.clean_pre_window`, `settings.event_study.n_permutations`, `settings.seed`.

**Commit**: `feat: add event study core and runner`

---

## Step 2.3: Permutation test (`analysis/permutation.py`)

```python
def circular_shift_pvalue(
    src_D: np.ndarray, hit_D: np.ndarray, clean_D: np.ndarray,
    observed: float, horizon: int, n_perm: int, seed: int,
) -> float:
    """One-sided p-value for the event hit rate under circular shifts.

    Why circular shifts: they keep the spacing and clustering of events intact,
    so the null distribution reflects how clumpy real events are. A t-test
    would assume independent events and give falsely small p-values.

    Returns NaN if observed is NaN, or if len(src_D) < 2*horizon + 3.
    A shifted pattern with zero clean events counts as hit rate 0.0.
    """
    n = len(src_D)
    if np.isnan(observed) or n < 2 * horizon + 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    ks = rng.integers(horizon + 1, n - horizon, size=n_perm)  # upper bound exclusive => [h+1, n-h-1]
    count = 0
    for k in ks:
        m = np.roll(src_D, int(k)) & clean_D
        r = hit_D[m].mean() if m.any() else 0.0
        if r >= observed - 1e-12:
            count += 1
    return (1 + count) / (1 + n_perm)
```

**Tests** (`tests/test_event_study.py`):
1. **SPEC 8.2 hand example**: 30 rows, `eligible = eligible_mask(dates, (dates[0], dates[-1]), 5, all True)`, `src` True at 5 and 15, `tgt` True at 8 and 25, `clean_only=False`, `n_perm=100`. Assert `n_events == 2`, `n_hits == 1`, `hit_rate == 0.5`, `base_rate == pytest.approx(0.4)`, `lift == pytest.approx(1.25)`.
2. **Clean filter**: same data, `tgt` also True at 15; `clean_only=True` removes event 15 (VIX spiking at t), leaving `n_events == 1`.
3. **Planted lead**: n = 3000, seed 7. Choose 40 target spike positions spaced at least 40 apart. For each, set a source event at `position - rng.integers(3, 8)`. `h=10`, `clean_only=True`, `n_perm=1000`. Assert `lift > 2` and `p_value < 0.05`.
4. **No relationship, type I error**: for seeds 0..19, 30 random source events and 40 random target spike days (positions drawn independently), `h=10`, `n_perm=500`. Assert the fraction with `p_value < 0.05` is `<= 0.2`.
5. **Determinism**: same inputs and seed produce the identical p-value twice.
6. **Warning**: fewer than 5 events sets the warning string.
7. **Zero events**: `hit_rate`, `lift`, `p_value` are NaN; no exception.
8. **run_event_study** on the synthetic panel with spikes: returns a result; `direction="vix_to_corr"` also runs.

**Commit**: `feat: add circular-shift permutation test`

---

## Step 2.4: Regression (`analysis/regression.py`)

```python
@dataclass
class RegressionResult:
    n: int
    r2: float
    hac_maxlags: int
    coefs: dict[str, dict[str, float]]   # name -> {"coef", "t_hac", "p_hac"}


def design_matrix(frame: pd.DataFrame, include_controls: bool) -> pd.DataFrame:
    """Columns: z (= corr_z) and, if include_controls, vix_mom_5d and log_vix."""


def run_regression(
    store: FrameStore, settings: Settings, group: Group, window: int,
    horizon: int, period: PeriodName, include_controls: bool,
) -> RegressionResult:
    """OLS of fwd_log_vix_change_h on the design matrix over eligible days,
    with Newey-West (HAC) standard errors, maxlags = horizon.

    Why HAC: h-day forward windows overlap, so residuals are autocorrelated
    and ordinary standard errors are too small.
    Why controls: tests whether correlation adds information beyond VIX
    momentum and level."""
```
Steps:
1. `f = store.frame(group, window)`; `X = design_matrix(f, include_controls)`; `y = fwd_log_vix_change(f.vix, horizon)`.
2. `valid = X.notna().all(axis=1) & y.notna()`; `m = eligible_mask(f.index, settings.period(period), horizon, valid)`.
3. `res = sm.OLS(y[m], sm.add_constant(X[m])).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})`.
4. `coefs = {name: {"coef": res.params[name], "t_hac": res.tvalues[name], "p_hac": res.pvalues[name]} for name in res.params.index}`. Names: `const`, `z`, and optionally `vix_mom_5d`, `log_vix`.
5. Return with `n = int(res.nobs)`, `r2 = float(res.rsquared)`.

To make the synthetic tests independent of frames, also expose:
```python
def fit_hac(y: pd.Series, X: pd.DataFrame, maxlags: int) -> RegressionResult: ...
```
and have `run_regression` call it.

**Tests** (`tests/test_regression.py`):
1. `fit_hac` on `x ~ N(0,1)`, `y = 0.5x + N(0, 0.5)`, n=2000, seed 3: `coefs["z"]["coef"]` within 0.05 of 0.5 (name the column `z`).
2. `hac_maxlags == maxlags` passed.
3. HAC t-stat is smaller in magnitude than the plain OLS t-stat when `y` is built from overlapping sums (`y_t = sum of e_{t+1..t+10}`, `x` = AR(1) with phi 0.9). This demonstrates why HAC matters.
4. `run_regression` on the synthetic panel runs for both `include_controls` values and returns the expected coefficient names.

**Commit**: `feat: add predictive regression with HAC errors`

---

## Step 2.5: Out-of-sample (`analysis/oos.py`)

```python
@dataclass
class OOSResult:
    r2_oos: float
    n_train: int
    n_test: int
    event_study_train: EventStudyResult
    event_study_test: EventStudyResult


def r2_oos(y_test: np.ndarray, y_pred: np.ndarray, train_mean: float) -> float:
    """1 - SSE(model) / SSE(historical mean). Positive = beats the benchmark."""


def run_oos(
    store: FrameStore, settings: Settings, group: Group, window: int,
    horizon: int, z_threshold: float,
) -> OOSResult:
```
Steps:
1. Base spec only (`include_controls=False`): regress on `z` alone. State this in the docstring: the OOS test asks whether the correlation z-score alone forecasts VIX changes better than the historical average.
2. Train mask = `eligible_mask(..., settings.period("train"), horizon, valid)`; test mask similarly.
3. Fit plain OLS on train (`sm.OLS(y_tr, sm.add_constant(X_tr)).fit()`); predict on `sm.add_constant(X_te, has_constant="add")`.
4. `train_mean = y_tr.mean()`; compute `r2_oos`.
5. Event studies: `EventStudyParams(group, window, z_threshold, horizon, "train"/"test", True, "corr_to_vix")`.

**Tests** (`tests/test_oos.py`):
1. `r2_oos` hand example: `y=[1,2,3]`, `pred=[1,2,3]`, `train_mean=0` gives 1.0; `pred = [train_mean]*3` gives 0.0.
2. Synthetic signal (same data as regression test 1, split 60/40): `r2_oos > 0`.
3. Pure noise (`y` independent of `x`): `r2_oos < 0.02`.
4. `run_oos` on the synthetic panel runs; `n_train > 0`, `n_test > 0`.

**Commit**: `feat: add out-of-sample evaluation`

---

## Step 2.6: Grid and overfitting check (`analysis/grid.py`)

```python
@dataclass
class GridRow:
    group: Group
    window: int
    z_threshold: float
    horizon: int
    train: EventStudyResult
    test: EventStudyResult


@dataclass
class OverfittingCheck:
    n_combinations: int
    best_in_sample: GridRow | None     # None if no combination has >= 5 train events
    preregistered: GridRow


def run_grid(store: FrameStore, settings: Settings) -> list[GridRow]:
    """Every combination of grid groups x windows x z_thresholds x horizons,
    corr_to_vix, clean_only=True, on train and test. Deterministic order:
    group, window, z_threshold, horizon (each in config order)."""


def overfitting_check(rows: list[GridRow], prereg: Preregistered) -> OverfittingCheck:
    """Select the row with the highest train lift among rows with train
    n_events >= 5 (ties broken by grid order). Also find the row matching the
    preregistered spec.

    Why: whichever combination looks best in-sample is partly luck; its test
    performance shows how much of the in-sample result was noise."""
```

**Tests** (`tests/test_grid.py`):
1. `run_grid` with real grid lists on the synthetic panel returns `2 * 2 * 2 * 3 = 24` rows in the documented order.
2. `overfitting_check` on hand-made rows: picks the highest train lift, ignores a higher-lift row with 3 events, finds the preregistered row.
3. All lifts NaN or too few events: `best_in_sample is None`.

**Commit**: `feat: add parameter grid and overfitting check`

---

## Step 2.7: Results assembly (`report/results.py`)

```python
def build_results(store: FrameStore, settings: Settings, prereg: Preregistered) -> dict[str, Any]:
    """Compute everything the report needs and return a JSON-safe dict
    (passed through to_jsonable). Keys, in order:
      dataset:  {tickers: {risk: [...], all: [...]}, vix_ticker, start, end,
                 n_trading_days: {risk, all}, rows_dropped: {risk, all}, periods}
      preregistered: {spec, train, test, reverse_test}
         reverse_test = same group/window/z/horizon, period test, direction vix_to_corr
      regression: {base, controls}   (full period, prereg group/window/horizon)
      oos:  run_oos with prereg group/window/horizon/z
      overfitting_check
      grid: list of rows
      data_snapshot_end
      generated_at: UTC ISO timestamp (the ONLY non-deterministic field)
    """


def render_results_md(results: dict[str, Any]) -> str:
    """Markdown with sections: Dataset, Headline, Reverse Direction Check,
    Regression, Out-of-Sample, Overfitting Check, Full Grid (exploratory),
    Limitations. Rates as percentages with 1 decimal, lifts 2 decimals,
    p-values 3 decimals, coefficients 4 decimals. NaN renders as 'n/a'."""


def headline_sentence(results: dict[str, Any]) -> str:
```

Headline template (fill from `preregistered.test` and `spec`):
> Preregistered test ({group} group, {window}-day window, z >= {z}, {horizon}-day horizon, clean events): on the {test_start} to {test_end} test period, {n} correlation events had a VIX-spike hit rate of {hit} vs a base rate of {base} (lift {lift}, permutation p = {p}). {verdict}

Verdict:
- `n_events < 5`: "Too few events for a reliable test."
- `p < alpha`: "Significant at alpha = {alpha}."
- else: "Not significant at alpha = {alpha}: no evidence of a lead effect."

Limitations section is static text listing the six items in SPEC section 9.

**Tests** (`tests/test_results.py`, synthetic panel + small settings):
1. `json.dumps(build_results(...), allow_nan=False)` succeeds.
2. All required keys present.
3. Two builds are equal after removing `generated_at`.
4. `headline_sentence` produces each of the three verdicts with hand-made dicts.
5. `render_results_md` contains every section heading and "n/a" for a NaN.

**Commit**: `feat: assemble results JSON and markdown`

---

## Step 2.8: Figures (`report/figures.py`)

At the very top, before importing pyplot:
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
```

Functions (each saves one PNG at dpi 150, calls `plt.close(fig)`, returns the path):
```python
def plot_timeseries(frame_risk21: pd.DataFrame, settings: Settings, out: Path) -> Path
def plot_corr_heatmaps(aligned_all: pd.DataFrame, settings: Settings, out: Path) -> Path
def plot_event_study(results: dict[str, Any], store: FrameStore, settings: Settings,
                     prereg: Preregistered, out: Path) -> Path
def plot_grid_lift(results: dict[str, Any], out: Path) -> Path
```

Details:
- `timeseries.png` (figsize 12x5): VIX on left axis; `avg_corr` on right axis; thin vertical lines at `vix_event` dates (alpha 0.3); `axvspan` shading for the test period; legend; title "VIX and Average Correlation (risk group, 21-day window)".
- `corr_heatmaps.png` (figsize 18x6): compute returns of the `all` group; pick dates programmatically: min VIX over full sample, max VIX within 2008, max VIX within 2020. For each, `pairwise_corr_matrix(returns, date, 63)`; `sns.heatmap(..., cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f", square=True)`; subplot titles like "Calm: 2017-11-03 (VIX 9.1)". If a date has fewer than 63 prior rows, use the first date that has 63.
- `event_study.png` (figsize 12x5): two panels (train, test). For horizons 5, 10, 20 run the event study at the preregistered group/window/z with `clean_only=True`, `corr_to_vix`. Grouped bars: hit rate and base rate. Annotate `n=...` above each pair.
- `grid_lift.png` (figsize 14x5): two heatmaps (train, test) for group `risk`, rows labeled `W={window}, z={z}`, columns `h={horizon}`, `annot=True`, `fmt=".2f"`, `center=1.0`, `cmap="RdBu_r"`. NaN cells show blank.

**Tests** (`tests/test_figures.py`): on synthetic data into `tmp_path`, each function creates a non-empty PNG. (Do not test pixels.)

**Commit**: `feat: add report figures`

---

## Step 2.9: README markers (`report/readme.py`)

```python
def replace_between_markers(text: str, name: str, content: str) -> str:
    """Replace everything between <!-- {name}:START --> and <!-- {name}:END -->
    with '\n' + content + '\n'. Raises ValueError if either marker is missing
    or END precedes START."""


def update_readme(readme_path: Path, results: dict[str, Any], evals_summary: str | None) -> None:
    """RESULTS block: headline sentence, a one-line reverse-direction summary,
    the out-of-sample R squared, and a link to reports/results.md.
    EVALS block: evals_summary if provided, else 'Evals not yet run.'"""
```

**Tests** (`tests/test_readme.py`): replacement works; content outside markers is unchanged; missing marker raises; running twice is idempotent.

**Commit**: `feat: add README result injection`

---

## Step 2.10: `vixagent report`

```python
@app.command()
def report() -> None:
    """Compute all results, write figures and results files, update README."""
```
Flow:
1. Load settings and prereg; `prices = load_or_fetch(settings)` (no refresh; if no cache, it fetches).
2. `store = FrameStore(prices, settings)`; `results = build_results(...)`.
3. Write `reports/results.json` (`json.dumps(results, indent=2, allow_nan=False)`), `reports/results.md`.
4. Generate the four figures into `reports/figures/`.
5. `update_readme(root / "README.md", results, evals summary from evals/results/latest.md if it exists)`.
6. Print the headline sentence and elapsed time.

Expose the core as `generate_report(settings, prereg, prices, root: Path) -> dict` so tests can run it into `tmp_path`.

**Tests** (`tests/test_report_cli.py`): `generate_report` on synthetic data into `tmp_path` (with a README containing markers) creates all 6 output files and updates README; running twice yields identical `results.json` except `generated_at`.

**Commit**: `feat: add report command`

---

## Step 2.11: Definition of Done, walkthrough, stop

```bash
ruff format . && ruff check . && ruff format --check . && mypy src && pytest -q
pytest -q --cov=vixagent --cov-report=term-missing
```
Coverage for `analysis/` at least 85%.

`docs/walkthroughs/phase-2.md` must include:
- The SPEC 8.2 hand example traced through `event_study_core` step by step (show `hit_all`, `eligible`, `ev`, `base_days` arrays).
- One permutation iteration traced with `k = 7` on the same example.
- How `fit_hac` differs from plain OLS and what test 3 in Step 2.4 demonstrated.
- How `r2_oos` is computed on a 3-row example.

**Commit**: `docs: add phase 2 walkthrough`

**STOP.** Report test count and coverage. Do **not** run `vixagent report` on real data yourself unless the human asks; the human does it in Step 2.12 so they see the result first.

---

## Step 2.12: Human-only steps

1. `vixagent report`.
2. Before opening any figure, read the headline sentence in the terminal. Write it down in `docs/walkthroughs/phase-2-answers.md`.
3. Open all four figures and `reports/results.md`. Sanity checks:
   - `timeseries.png`: VIX peaks in late 2008 and March 2020; average correlation rises in both.
   - `corr_heatmaps.png`: the crisis panels are redder (higher correlations) than the calm panel.
   - Event counts are plausible (tens, not thousands).
4. If anything looks wrong, describe it to the agent. Do **not** change `preregistered.yaml`.
5. Publish v0.1:
   ```bash
   git add reports README.md && git commit -m "docs: add first generated results"
   git tag v0.1 && git push && git push --tags
   ```
6. Answer the Phase 2 Checkpoint Questions.
