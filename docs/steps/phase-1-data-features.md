# Phase 1: Data, Features, Spikes

Goal: download and cache prices, build every backward-looking feature, detect spikes and events, define eligible days, and **prove** there is no lookahead. End by freezing the preregistration.

Read first: `CLAUDE.md`, `docs/SPEC.md` sections 4 to 8.1, 12, 15.

Module map for this phase:
```
src/vixagent/data/fetch.py        fetch_prices, DataError
src/vixagent/data/validate.py     validate_prices
src/vixagent/data/cache.py        load_or_fetch
src/vixagent/data/align.py        align_group
src/vixagent/features/returns.py  log_returns
src/vixagent/features/correlation.py  avg_pairwise_corr, pairwise_corr_matrix
src/vixagent/features/zscore.py   trailing_zscore
src/vixagent/features/vix.py      vix_ratio, vix_momentum, log_vix
src/vixagent/features/frame.py    build_frame
src/vixagent/targets.py           fwd_log_vix_change, fwd_any_within
src/vixagent/spikes.py            spike_days, events_from_days, backward_any
src/vixagent/periods.py           eligible_mask, period_mask
tests/fixtures/synthetic.py
```

---

## Step 1.1: Synthetic fixtures (`tests/fixtures/synthetic.py`)

Build these first; every later test uses them.

```python
def make_dates(n: int, start: str = "2010-01-04") -> pd.DatetimeIndex:
    """Business days, tz-naive, name='date'."""
    return pd.bdate_range(start=start, periods=n, name="date")


def correlated_returns(
    n: int, n_assets: int, corr: float, seed: int, vol: float = 0.01
) -> np.ndarray:
    """(n, n_assets) normal returns with equal pairwise correlation `corr`.
    Covariance = vol^2 * (corr * ones + (1 - corr) * I). Uses Cholesky."""


def regime_returns(
    n: int, n_assets: int, base_corr: float,
    regimes: list[tuple[int, int, float]], seed: int, vol: float = 0.01,
) -> np.ndarray:
    """Like correlated_returns(base_corr) but rows [start, end) of each regime
    use that regime's correlation. Draw each block with its own sub-seed
    (seed + i) so results are deterministic."""


def prices_from_returns(
    returns: np.ndarray, dates: pd.DatetimeIndex, tickers: list[str], start_price: float = 100.0
) -> pd.DataFrame:
    """Price = start_price * exp(cumsum(returns)). Columns = tickers."""


def make_vix(
    dates: pd.DatetimeIndex, seed: int, base: float = 15.0,
    spike_positions: list[int] | None = None, spike_mult: float = 1.6,
) -> pd.Series:
    """Quiet VIX: base * exp(0.01 * standard normal) each day.
    At each spike position p: level = base * spike_mult on day p,
    base * 1.35 on p+1, base * 1.15 on p+2 (then quiet again).
    Name '^VIX'. Guarantees VIX in [5, 100]."""


def make_price_panel(
    n: int = 1500, seed: int = 0, tickers: list[str] | None = None,
    spike_positions: list[int] | None = None,
) -> pd.DataFrame:
    """Default tickers: the 10 'all' tickers from settings. Returns prices for
    all tickers plus a '^VIX' column, same index."""


def tiny_settings(tmp_path: Path) -> Settings:
    """A Settings object for synthetic data: universe risk = first 6 tickers,
    all = 10 tickers, data.start = first date, snapshot_end = last date,
    train = first 60% of dates, test = remaining dates starting the next
    business day, z_lookback = 60 (small so tests run on short series),
    n_permutations = 500, cache_path under tmp_path. Build it by loading the
    real settings, model_dump(), editing, and model_validate()."""
```

Also add `tests/conftest.py` with fixtures `panel` (make_price_panel with seed 0 and spikes at positions [300, 700, 1100]) and `small_settings` (tiny_settings(tmp_path)).

**Verify**: `python -c "from tests.fixtures.synthetic import make_price_panel as m; p=m(); print(p.shape, p['^VIX'].between(5,100).all())"` prints `(1500, 11) True`.
**Commit**: `test: add synthetic data fixtures`

---

## Step 1.2: `data/fetch.py`

```python
class DataError(Exception):
    """Raised when downloaded or cached data is unusable."""


def fetch_prices(
    tickers: list[str],
    start: date,
    end_inclusive: date,
    *,
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Download daily adjusted closes. Returns DataFrame: index tz-naive
    DatetimeIndex named 'date' (sorted), columns exactly `tickers` in order."""
```

Algorithm:
1. `end_exclusive = end_inclusive + timedelta(days=1)` (yfinance `end` is exclusive).
2. Attempt up to `1 + max_retries` times. Each attempt:
   ```python
   raw = yf.download(tickers, start=start.isoformat(), end=end_exclusive.isoformat(),
                     auto_adjust=True, progress=False, threads=False, group_by="column")
   ```
   Treat an exception **or** `raw is None or raw.empty` as a failed attempt. On failure, `sleep(2 ** (attempt + 1))` (2, 4, 8 seconds) then retry. After the last failure raise `DataError` including the last error message.
3. `_extract_close(raw, tickers)`:
   - If `raw.columns` is a `MultiIndex`: if `"Close"` is in level 0 use `raw["Close"]`; elif `"Close"` is in level 1 use `raw.xs("Close", axis=1, level=1)`; else raise `DataError`.
   - Else (flat): if exactly one ticker and `"Close"` in columns, return `raw[["Close"]].rename(columns={"Close": tickers[0]})`; else raise `DataError`.
4. Normalize index: `idx = pd.DatetimeIndex(pd.to_datetime(close.index))`; if `idx.tz is not None`: `idx = idx.tz_localize(None)`; `idx = idx.normalize()`; name `"date"`; sort.
5. Missing check: any ticker not in columns, or whose column is all NaN, goes into a list; if non-empty raise `DataError(f"Missing or empty tickers: {missing}")`.
6. Return `close[tickers].astype(float)`.

Import yfinance as `import yfinance as yf` at module level so tests can monkeypatch `vixagent.data.fetch.yf.download`.

**Tests** (`tests/test_fetch.py`, monkeypatch `yf.download`, pass `sleep=lambda s: None`):
1. MultiIndex `(field, ticker)` input with fields Open/Close for 2 tickers: returns Close values, columns in requested order.
2. MultiIndex `(ticker, field)` input: same result.
3. Flat single-ticker input: works.
4. Missing ticker: `DataError` naming it.
5. All-NaN ticker: `DataError`.
6. tz-aware index (`America/New_York`): output index is tz-naive and normalized.
7. Retry: fake raises twice then returns valid data; result correct; fake called 3 times; recorded sleeps `[2, 4]`.
8. Always empty: `DataError` after 4 calls.
9. `end` argument passed to `yf.download` equals `end_inclusive + 1 day`.

**Commit**: `feat: add yfinance price fetching with normalization and retries`

---

## Step 1.3: `data/validate.py`

```python
def validate_prices(prices: pd.DataFrame, vix_ticker: str) -> list[str]:
    """Raise DataError on hard problems; return warning strings for soft ones.

    Errors: index not monotonic increasing; duplicate index values; any
    non-NaN value <= 0; any non-NaN VIX value outside [5, 100].
    Warnings: any non-VIX column with |daily log return| > 0.5 (list ticker
    and date); any gap > 5 calendar days between consecutive index dates.
    """
```

**Tests** (`tests/test_validate.py`): clean synthetic panel returns `[]`; each error condition raises; a planted 70% jump returns a warning mentioning the ticker; a planted 10-day gap returns a warning.

**Commit**: `feat: add price data validation`

---

## Step 1.4: `data/align.py`

```python
def align_group(
    prices: pd.DataFrame, settings: Settings, group: Group
) -> tuple[pd.DataFrame, int]:
    """Columns universe[group] + [vix_ticker], rows within the full period
    (data.start..snapshot_end inclusive), keeping only rows where ALL those
    columns are non-NaN. No forward fill.

    Returns (aligned_frame, rows_dropped) where rows_dropped counts rows in the
    full-period slice that had at least one NaN in these columns.

    Why no forward fill: a filled price creates a fake 0% return, which
    artificially lowers correlations on market holidays that differ across
    exchanges.
    """
```

**Tests** (`tests/test_align.py`): panel with NaNs planted on 3 dates in one ticker: 3 rows dropped, no NaN remains, column order is group tickers then VIX; rows outside the period are excluded.

**Commit**: `feat: add group alignment without forward filling`

---

## Step 1.5: `data/cache.py`

```python
def all_tickers(settings: Settings) -> list[str]:
    """Sorted union of all universe groups, plus vix_ticker last."""


def load_or_fetch(
    settings: Settings,
    refresh: bool = False,
    root: Path | None = None,
    fetcher: Callable[..., pd.DataFrame] = fetch_prices,
) -> pd.DataFrame:
    """Return the full price panel (all_tickers columns).

    Cache files: <root>/<data.cache_path> (parquet) and the same path with
    suffix '.meta.json'. Metadata keys: fetched_at (ISO UTC), yfinance_version,
    tickers, start, end_inclusive.

    Use the cache only if both files exist, refresh is False, and metadata
    tickers/start/end_inclusive equal the current settings. Otherwise call
    fetcher(all_tickers, data.start, data.snapshot_end), run validate_prices
    (log warnings via `logging`), write both files (mkdir parents), return.
    """
```

**Tests** (`tests/test_cache.py`, fake fetcher counting calls, `tmp_path` root):
1. First call fetches and writes both files.
2. Second call loads cache; fetcher not called again; frames equal (`pd.testing.assert_frame_equal`, `check_freq=False`).
3. `refresh=True` refetches.
4. Changing a ticker in settings triggers refetch.
5. Metadata JSON contains the five keys.

**Commit**: `feat: add parquet cache with metadata`

---

## Step 1.6: Feature functions

### `features/returns.py`
```python
def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """ln(P_t / P_{t-1}) per column; the first row is dropped (it is all NaN)."""
    return np.log(prices).diff().iloc[1:]
```

### `features/correlation.py`
```python
def avg_pairwise_corr(returns: pd.DataFrame, window: int) -> pd.Series:
    """Mean of all pairwise rolling Pearson correlations over the trailing
    `window` rows (inclusive of the current row). NaN until `window` returns
    exist, and NaN if any pair is undefined (zero variance).

    Why pairwise with itertools.combinations: it is equivalent to averaging the
    upper triangle of the rolling correlation matrix, but simpler to verify.
    """
    cols = list(returns.columns)
    if len(cols) < 2:
        raise ValueError("need at least 2 assets")
    pairs = [returns[a].rolling(window).corr(returns[b]) for a, b in combinations(cols, 2)]
    stacked = pd.concat(pairs, axis=1).clip(-1.0, 1.0)
    return stacked.mean(axis=1, skipna=False).rename(f"avg_corr_{window}")


def pairwise_corr_matrix(returns: pd.DataFrame, end_date: pd.Timestamp, window: int) -> pd.DataFrame:
    """Correlation matrix of the last `window` rows ending at end_date
    (inclusive). Raises ValueError if fewer than `window` rows exist."""
```

### `features/zscore.py`
```python
def trailing_zscore(x: pd.Series, lookback: int) -> pd.Series:
    """z_t = (x_t - mean(x_{t-L}..x_{t-1})) / std(x_{t-L}..x_{t-1}), ddof=1.

    Why shift(1): the baseline must exclude today so today's value is compared
    against history only. Why not full-sample stats: that leaks future data.
    A std at or below 1e-12 yields NaN (not inf).
    """
    past = x.shift(1)
    mu = past.rolling(lookback).mean()
    sd = past.rolling(lookback).std()
    return ((x - mu) / sd.where(sd > 1e-12)).rename(f"z_{x.name}")
```

### `features/vix.py`
```python
def vix_ratio(vix: pd.Series, lookback: int) -> pd.Series:
    """V_t / median(V_{t-lookback}..V_{t-1}). Median resists single outliers."""
    return vix / vix.shift(1).rolling(lookback).median()


def vix_momentum(vix: pd.Series, k: int = 5) -> pd.Series:
    """ln(V_t / V_{t-k}): recent VIX change, used as a regression control."""
    return np.log(vix / vix.shift(k))


def log_vix(vix: pd.Series) -> pd.Series:
    return np.log(vix)
```

**Tests** (`tests/test_features.py`):
1. `log_returns`: hand example prices [100, 102, 99] gives [ln(1.02), ln(99/102)]; output has one fewer row.
2. `avg_pairwise_corr`, identical columns: all non-NaN values `== pytest.approx(1.0, abs=1e-9)`.
3. Three assets, `correlated_returns(n=3000, corr=0.8, seed=1)`, window 252: mean of non-NaN values within 0.05 of 0.8.
4. Works for 2 and for 10 assets; first `window - 1` values NaN, value at position `window - 1` not NaN.
5. Matches a brute-force check: for 5 random positions, compute `returns.iloc[t-W+1:t+1].corr()` upper-triangle mean with numpy; equal within 1e-10.
6. `pairwise_corr_matrix` is symmetric with ones on the diagonal; raises when too few rows.
7. `trailing_zscore` hand example: x = [1, 2, 3, 4, 10], lookback 3. At position 3: past = [1,2,3], mean 2, std 1, z = (4-2)/1 = 2.0. At position 4: past = [2,3,4], mean 3, std 1, z = 7.0. Positions 0 to 2 NaN.
8. Constant series: z is all NaN (no inf).
9. `vix_ratio` hand example with lookback 3: [10, 10, 10, 15] gives NaN, NaN, NaN, 1.5.
10. `vix_momentum` hand example.

**Commit**: `feat: add return, correlation, z-score, and VIX features`

---

## Step 1.7: `targets.py` (the only forward-looking module)

```python
"""Forward-looking quantities. This is the ONLY module allowed to look ahead.
Everything here describes the future relative to row t and must only be used
as an outcome, never as an input to a feature."""


def fwd_log_vix_change(vix: pd.Series, h: int) -> pd.Series:
    """ln(V_{t+h}) - ln(V_t). The last h rows are NaN."""
    return (np.log(vix.shift(-h)) - np.log(vix)).rename(f"fwd_log_vix_change_{h}")


def fwd_any_within(flags: pd.Series, h: int) -> pd.Series:
    """out_t = True if any flag is True in rows t+1..t+h (strictly after t).
    Rows where the window runs past the end still compute over the rows that
    exist; eligibility (periods.eligible_mask) excludes them from analysis.

    Implemented with a cumulative sum so it is O(n)."""
    f = flags.fillna(False).to_numpy(dtype=bool).astype(np.int64)
    n = len(f)
    csum = np.concatenate([[0], np.cumsum(f)])
    idx = np.arange(n)
    lo = np.minimum(idx + 1, n)
    hi = np.minimum(idx + h + 1, n)
    return pd.Series((csum[hi] - csum[lo]) > 0, index=flags.index, name=f"fwd_any_{h}")
```

**Tests** (`tests/test_targets.py`):
1. `fwd_log_vix_change` on [10, 20, 40], h=1: [ln 2, ln 2, NaN].
2. Last `h` rows NaN for h in {1, 5, 10}.
3. `fwd_any_within`: flags True at positions 8 and 25 of 30, h=5: True exactly at positions 3..7 and 20..24.
4. A flag at position t does **not** make position t True (strictly after).

**Commit**: `feat: add forward-looking targets module`

---

## Step 1.8: `spikes.py`

```python
def spike_days(x: pd.Series, threshold: float) -> pd.Series:
    """True where x >= threshold. NaN compares False."""
    return (x >= threshold).fillna(False).astype(bool)


def events_from_days(days: pd.Series, cooldown: int) -> pd.Series:
    """Decluster: t is an event if days[t] is True and no True occurred in rows
    t-cooldown..t-1.

    Why: consecutive crisis days are one episode, not many independent
    observations. Counting them separately inflates sample size."""
    d = days.fillna(False).astype(bool)
    if cooldown == 0:
        return d.rename("event")
    prior = d.astype(float).shift(1).rolling(cooldown, min_periods=1).max().fillna(0.0)
    return (d & (prior == 0.0)).rename("event")


def backward_any(flags: pd.Series, k: int) -> pd.Series:
    """True if any flag is True in rows t-k..t inclusive. Used by the clean
    filter. Backward-looking only."""
    f = flags.fillna(False).astype(float)
    return f.rolling(k + 1, min_periods=1).max().astype(bool)
```

**Tests** (`tests/test_spikes.py`):
1. SPEC section 7 example: spike days at 10, 11, 12, 40, 45 in 60 rows, cooldown 20: events exactly at 10 and 40.
2. Cooldown 0: events equal spike days.
3. A spike day exactly `cooldown + 1` rows after the previous spike day is an event; exactly `cooldown` rows after is not.
4. `backward_any` flag at 10, k=5: True at 10..15, False at 9 and 16.
5. NaN input treated as False everywhere.

**Commit**: `feat: add spike detection and declustering`

---

## Step 1.9: `features/frame.py`

```python
def build_frame(aligned: pd.DataFrame, settings: Settings, group: Group, window: int) -> pd.DataFrame:
    """Build the per-group, per-window analysis frame from aligned prices.

    Index: dates of the return series (first aligned date dropped).
    Columns:
      vix          VIX level
      log_vix      ln(vix)
      vix_mom_5d   ln(V_t / V_{t-5})
      vix_ratio    V_t / median(prior 20)
      vix_spike_day  vix_ratio >= settings.vix_spike.ratio_threshold
      vix_event    events_from_days(vix_spike_day, vix_spike.cooldown)
      avg_corr     avg_pairwise_corr(returns of group tickers, window)
      corr_z       trailing_zscore(avg_corr, features.z_lookback)
    Correlation spike days/events depend on z_threshold and are computed in
    analysis, not here. No fwd_ columns here.
    """
```

**Commit**: `feat: add analysis frame builder`

---

## Step 1.10: `periods.py`

```python
def period_mask(index: pd.DatetimeIndex, period: tuple[date, date]) -> np.ndarray:
    """Boolean array: start <= date <= end."""


def eligible_mask(
    index: pd.DatetimeIndex, period: tuple[date, date], h: int, valid: pd.Series | np.ndarray
) -> np.ndarray:
    """Rows usable for an h-day-ahead analysis in `period`:
      in period AND valid[t] AND (t + h) <= last row position within period.

    Why the last condition: the outcome window t+1..t+h must lie inside the
    same period. For train, this is a natural embargo so no training outcome
    uses test-period data."""
```

Implementation: `pos = np.arange(len(index))`; `in_p = period_mask(...)`; if `not in_p.any()` return all False; `last = pos[in_p].max()`; return `in_p & valid_bool & (pos + h <= last)`.

**Tests** (`tests/test_periods.py`):
1. 30 rows, period = all, h=5, all valid: eligible exactly rows 0..24 (matches SPEC 8.2 example).
2. Invalid rows (NaN feature) excluded.
3. Two adjacent periods split at row 15 with h=5: train eligible rows end at 10; test rows start at 15.
4. Empty period returns all False.

**Commit**: `feat: add period and eligibility masks`

---

## Step 1.11: No-lookahead guarantees (critical)

`tests/test_no_lookahead.py`:

**Test A: truncation invariance.**
```python
@pytest.mark.parametrize("window", [21, 63])
def test_build_frame_has_no_lookahead(panel, small_settings, window):
    aligned, _ = align_group(panel, small_settings, "risk")
    full = build_frame(aligned, small_settings, "risk", window)
    rng = np.random.default_rng(123)
    # choose 5 cut positions between 40% and 95% of the frame
    cuts = rng.integers(int(0.4 * len(aligned)), int(0.95 * len(aligned)), size=5)
    for c in cuts:
        cut_date = aligned.index[c]
        trunc = build_frame(aligned.loc[:cut_date], small_settings, "risk", window)
        a = full.loc[:cut_date]
        pd.testing.assert_frame_equal(a, trunc, check_exact=False, atol=1e-12, rtol=0, check_freq=False)
```
Also truncation tests for `spike_days`/`events_from_days` on `corr_z` with threshold 2.0, and `backward_any`.

**Test B: static check.**
```python
def test_only_targets_looks_forward():
    root = find_project_root() / "src" / "vixagent"
    pattern = re.compile(r"shift\(\s*-")
    offenders = [p for p in root.rglob("*.py")
                 if pattern.search(p.read_text()) and p.name != "targets.py"]
    assert offenders == []
```

**Test C: demonstrate the test catches a bug.** A local function using full-sample standardization `(x - x.mean()) / x.std()` fails the truncation comparison (assert that `assert_frame_equal` raises). This proves Test A has teeth.

**Verify**: `pytest -q tests/test_no_lookahead.py` passes.
**Commit**: `test: enforce no-lookahead via truncation and static checks`

---

## Step 1.12: `vixagent pull`

Add to `cli.py`:
```python
@app.command()
def pull(refresh: bool = typer.Option(False, help="Ignore cache and redownload.")) -> None:
    """Download (or load cached) prices, validate, and print a summary."""
```
Behavior:
1. `load_dotenv()`, `settings = load_settings()`, `prices = load_or_fetch(settings, refresh)`.
2. For each group, `align_group`, then print a rich table with: group, tickers, first date, last date, trading days, rows dropped.
3. Print validation warnings (if any) in yellow.
4. Exit code 1 with a red message on `DataError`.

Add `tests/test_network.py`:
```python
@pytest.mark.network
def test_real_fetch_small():
    df = fetch_prices(["SPY", "^VIX"], date(2024, 1, 2), date(2024, 1, 31))
    assert list(df.columns) == ["SPY", "^VIX"] and len(df) >= 19
```

**Verify**: `pytest -q` passes (network test deselected). The human then runs `pytest -q -m network` and `vixagent pull`.
**Commit**: `feat: add pull command and network smoke test`

---

## Step 1.13: Definition of Done, coverage, walkthrough

```bash
ruff format . && ruff check . && ruff format --check . && mypy src && pytest -q
pytest -q --cov=vixagent --cov-report=term-missing
```
Coverage must be at least 85% for `features/`, `spikes.py`, `targets.py`, `periods.py`.

Write `docs/walkthroughs/phase-1.md`. Required worked traces:
- `trailing_zscore` on `[1, 2, 3, 4, 10]` with lookback 3.
- `events_from_days` on the SPEC section 7 example, showing the `prior` column row by row for rows 9 to 13 and 38 to 46.
- How Test C proves Test A would catch full-sample standardization.

**Commit**: `docs: add phase 1 walkthrough`

**STOP.** Report test count, coverage numbers, and ask the human to perform the manual steps below.

---

## Step 1.14: Human-only steps (agent waits)

1. `pytest -q -m network` passes.
2. `vixagent pull` prints sensible numbers: `risk` starts in May 2007, ends 2026-06-30, roughly 4,800 trading days, a small number of dropped rows.
3. **Decide the universe.** If the GMU study used different assets, edit `config/settings.yaml` now and rerun steps 1 and 2.
4. Read `config/preregistered.yaml` and confirm you accept it as your primary test.
5. Freeze it:
   ```bash
   git add config/ && git commit -m "chore: freeze preregistered specification"
   git tag prereg-freeze && git push --tags
   ```
6. Answer the Phase 1 Checkpoint Questions in `docs/walkthroughs/phase-1-answers.md`.

From this commit on, `config/preregistered.yaml` never changes.
