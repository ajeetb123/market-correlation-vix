# Phase 1 Walkthrough: Data, Features, Spikes

Phase 1 downloads and caches prices, builds every backward-looking feature, detects spikes and declusters them into events, defines which days are eligible for analysis, and proves with automated tests that no feature looks into the future.

**Status:** 91 tests (90 unit tests plus 1 network test, which is deselected by default). Definition of Done passes. Coverage is 100% on `features/`, `spikes.py`, `targets.py`, and `periods.py`, and 96% overall.

**Real data pull** (`vixagent pull`, yfinance 1.7.0):

| group | first date | last date | trading days | rows dropped |
|---|---|---|---|---|
| risk | 2007-05-01 | 2026-06-30 | 4822 | 1 |
| all | 2007-05-01 | 2026-06-30 | 4822 | 1 |

There were no validation warnings. The one dropped row is **2026-05-25 (Memorial Day)**: Yahoo has a VIX value for that date but no ETF prices. Inner alignment drops the row rather than forward-filling it, which is exactly what it is designed to do.

## Deviations and interpretations

- **`tests/test_periods.py`, adjacent periods.** The step file says "split at row 15 ... train eligible rows end at 10; test rows start at 15". Those numbers are only consistent if row 15 belongs to both periods, and the config validator forbids overlapping periods. The test uses non-overlapping periods (train = rows 0..14, test = rows 15..29). With `h = 5`, train eligibility therefore ends at row 9 (9 + 5 = 14, the last train row) and test starts at 15. The embargo logic is the same.
- **`regime_returns` sub-seeds.** The base draw uses `seed`, and regime `i` uses `seed + i + 1`, so no regime reuses the base draw's random numbers.
- **`make_price_panel` plants a lead.** Each VIX spike at row `p` is preceded by a high-correlation regime (0.9, versus 0.3 normally) in rows `p-10 .. p-1`. That gives Phase 2 analysis tests a known lead to detect on the shared `panel` fixture.
- **`build_frame` computes VIX features before dropping the first row.** The VIX ratio, momentum, and log level are computed on the full aligned VIX series and then restricted to the return dates, so the first return date can use the previous day's VIX as history. This is still backward-looking, and the truncation test confirms it.
- **`period_mask` / `eligible_mask` accept `pd.Index`** rather than `pd.DatetimeIndex`, because pandas-stubs types `DataFrame.index` as a plain `Index`. Behavior is unchanged.
- **Small `typing.cast` calls** were added around `np.log(...)` and `raw["Close"]`. pandas-stubs types `np.log(DataFrame)` as an ndarray. The casts affect type checking only.

---

## `data/fetch.py`

**Purpose.** Downloads daily adjusted closes with yfinance and returns a clean DataFrame: tz-naive dates named `date`, and exactly the requested tickers in order.

**Key functions**
- `fetch_prices(tickers, start, end_inclusive)`: download with retries and normalization.
- `_extract_close(raw, tickers)`: finds the `Close` field in any of yfinance's three column layouts.
- `DataError`: the single exception type for unusable data.

**Worked trace (retry test).** The fake `yf.download` raises on calls 1 and 2 and returns data on call 3.
- Attempt 0 raises, so `sleep(2**1 = 2)`.
- Attempt 1 raises, so `sleep(2**2 = 4)`.
- Attempt 2 returns a non-empty frame, and the loop breaks.

The recorded sleeps are `[2, 4]` and the fake was called 3 times.

**Non-obvious decisions**
- `end = end_inclusive + 1 day`, because yfinance's `end` is exclusive. Without it, 2026-06-30 would be silently missing.
- `auto_adjust=True` is passed explicitly. Its default changed between yfinance versions. Unadjusted prices would turn dividends and splits into fake returns.
- `sleep` is injectable so tests don't actually wait 14 seconds.

## `data/validate.py`

**Purpose.** Rejects corrupt data (errors) and flags suspicious data (warnings).

**Errors vs warnings.**
- A non-positive price or a VIX of 120 is impossible, so the data is broken and the function raises.
- A 70% daily move or a 10-day gap *can* be real (a crash, 9/11, a holiday cluster), so the function reports it and lets a human decide.

## `data/align.py`

**Purpose.** For one group, keeps the group's tickers plus VIX, restricted to the full period, and keeps only rows where *every* column has a value.

**Non-obvious decision: no forward fill.** Suppose EFA (international stocks) has a holiday that US markets don't. Forward-filling its price gives EFA a 0% return that day, while the other assets move. That artificially lowers every EFA pair's correlation. Dropping the row avoids the fake data point.

## `data/cache.py`

**Purpose.** Downloads once, then serves a parquet snapshot. The snapshot is refetched only if the tickers or date range in `settings.yaml` change, or if you pass `--refresh`.

**Non-obvious decision.** The cache is keyed on the metadata (`tickers`, `start`, `end_inclusive`), not on the file's age. Research results must be reproducible from a fixed snapshot. Yahoo occasionally revises history, so re-downloading "fresh" data would change results without anyone changing code. `yfinance_version` and `fetched_at` are recorded so the snapshot's origin is known.

## `features/returns.py`, `features/correlation.py`

**Purpose.**
- `log_returns` turns prices into `ln(P_t / P_{t-1})`.
- `avg_pairwise_corr` averages the rolling Pearson correlation of every pair of assets over the last `W` rows.

With 6 assets there are C(6,2) = 15 pairs, and with 10 assets there are 45.

**Non-obvious decisions**
- **Correlate returns, never prices.** Two unrelated assets that both trended up for years have highly correlated *prices*. This was one of the pitfalls in the v1 project.
- **Pairwise loop instead of a rolling correlation matrix.** The two are mathematically identical. The loop is easier to verify, and `test_matches_brute_force` checks it against `numpy.corrcoef` on 5 random windows to within 1e-10.
- **`skipna=False` in the mean.** If one pair's correlation is undefined (a zero-variance asset), the average is NaN rather than a silently smaller average.

## `features/zscore.py`

**Purpose.** Measures how unusual today's value is compared with the previous `L` days only.

**Worked trace:** `trailing_zscore([1, 2, 3, 4, 10], lookback=3)`

| pos | x | past = x.shift(1) | rolling window of `past` | mu | sd | z |
|---|---|---|---|---|---|---|
| 0 | 1 | NaN | (not enough) | NaN | NaN | NaN |
| 1 | 2 | 1 | (not enough) | NaN | NaN | NaN |
| 2 | 3 | 2 | [NaN, 1, 2] | NaN | NaN | NaN |
| 3 | 4 | 3 | [1, 2, 3] | 2 | 1 | (4-2)/1 = **2.0** |
| 4 | 10 | 4 | [2, 3, 4] | 3 | 1 | (10-3)/1 = **7.0** |

Row 4's value (10) is not part of its own baseline. Without `shift(1)`, the baseline at row 4 would be [3, 4, 10], and the spike would partly hide itself.

**Non-obvious decision.** A standard deviation at or below 1e-12 becomes NaN, not a tiny number. A constant series would otherwise produce `inf` or wildly large z-scores.

## `features/vix.py`

- `vix_ratio`: `V_t / median(previous 20 days)`. A spike day is `ratio >= 1.30`. The **median** is used because one freak day in the baseline would drag a mean up and hide the next spike. `test_vix_ratio_uses_median` shows `[10, 60, 10, 15]` still gives 1.5.
- `vix_momentum`: `ln(V_t / V_{t-5})`, and `log_vix`: `ln(V_t)`. These are regression controls for Phase 2.

## `targets.py` (the only forward-looking module)

- `fwd_log_vix_change(V, h)`: `ln(V_{t+h}) - ln(V_t)`. This is the regression outcome.
- `fwd_any_within(flags, h)`: True if any flag occurs in rows `t+1 .. t+h`. This is the event-study "hit".

**Worked trace:** `fwd_any_within` with flags at rows 8 and 25 of 30, and `h = 5`.
- `csum` is the running count of flags: `csum[k]` is the number of flags in rows `0..k-1`.
- For row `t`, it takes `csum[min(t+6, 30)] - csum[min(t+1, 30)]`, the number of flags in `t+1 .. t+5`.
- Row 3 covers rows 4..8, which includes 8, so it is True.
- Row 8 covers rows 9..13, so it is False. A flag at `t` itself never counts.

The True rows are 3..7 and 20..24.

**Why isolate forward-looking code?** If `shift(-h)` could appear anywhere, one typo in a feature would leak the future. With all of it in one file, a single grep-based test enforces the rule.

## `spikes.py`

- `spike_days`: marks days where the value is at or above the threshold. NaN counts as not a spike.
- `events_from_days`: declusters, so a day is an event only if there was no spike day in the previous `cooldown` rows.
- `backward_any`: True if a flag occurred in rows `t-k .. t`. Phase 2's clean filter uses it.

**Worked trace: SPEC section 7 example.** Spike days are at 10, 11, 12, 40, and 45, with cooldown 20. Here `prior_t = max(days over rows t-20 .. t-1)`, computed as `shift(1).rolling(20, min_periods=1).max()`.

| row | spike day? | rows in prior window | prior | event = spike day AND prior == 0 |
|---|---|---|---|---|
| 9 | F | 0..8 | 0 | F |
| 10 | **T** | 0..9 | 0 | **T** |
| 11 | T | 0..10 (has 10) | 1 | F |
| 12 | T | 0..11 | 1 | F |
| 13 | F | 0..12 | 1 | F |
| 38 | F | 18..37 (12 has aged out) | 0 | F |
| 39 | F | 19..38 | 0 | F |
| 40 | **T** | 20..39 | 0 | **T** |
| 41 | F | 21..40 (has 40) | 1 | F |
| 42–44 | F | ... | 1 | F |
| 45 | T | 25..44 (has 40) | 1 | F |
| 46 | F | 26..45 | 1 | F |

The events are rows 10 and 40. The 11/12 cluster and row 45 are absorbed into their episodes.

**Why decluster?** The COVID crash produced a long run of consecutive spike days. Counting each as an independent event would claim many observations from one crisis, which shrinks p-values without adding information.

## `features/frame.py`

`build_frame(aligned, settings, group, window)` combines everything above into one DataFrame per (group, window). Its columns are `vix`, `log_vix`, `vix_mom_5d`, `vix_ratio`, `vix_spike_day`, `vix_event`, `avg_corr`, and `corr_z`. There are no `fwd_` columns. Correlation spike days are left to Phase 2 because they depend on the z-threshold being tested.

## `periods.py`

- `period_mask`: `start <= date <= end`.
- `eligible_mask`: in the period, the feature is valid, **and** `t + h` is still on or before the period's last row.

**Worked trace:** 30 rows, the whole range as the period, and `h = 5`. The last row is 29. Row 24 is eligible (24 + 5 = 29), and row 25 is not (30 > 29). Eligible rows are 0..24, which is 25 days, the denominator of the SPEC 8.2 base rate.

**Why the `t + h` condition?** For the train period, it guarantees that no training outcome window reaches into test data. It acts as an automatic embargo.

## `tests/test_no_lookahead.py`

- **Test A (truncation invariance).** Build the frame on all data. Build it again on data chopped off at 5 random dates. Every value on or before the chop date must match to within 1e-12. If any feature used a future value, chopping the future would change it. This runs for windows 21 and 63, and again for spike days, events, and `backward_any`.
- **Test B (static check).** Grep `src/vixagent` for `shift(-`. Only `targets.py` may contain it.
- **Test C (proof that Test A has teeth).** A deliberately leaky z-score, `(x - x.mean()) / x.std()` over the whole sample, must *fail* Test A's comparison. On the synthetic panel, the first cut is at row 612 (2012-05-09):
  - Full-sample mean and std of `avg_corr` are 0.3119 and 0.1033.
  - Truncated mean and std are 0.2826 and 0.0926.
  - On 2011-12-06 (before the cut), `avg_corr` = 0.2932. That gives a full-sample z of **-0.18** and a truncated z of **+0.11**.

  The value for a 2011 date changed sign depending on data from after 2012-05-09, which is exactly the lookahead Test A exists to catch. `assert_frame_equal` raises, and Test C asserts that it does.

## `cli.py`: `vixagent pull`

Loads `.env`, loads settings, calls `load_or_fetch`, validates, aligns each group, and prints the summary table and any warnings in yellow. A `DataError` prints in red and exits with code 1. It is tested with the fetcher and settings monkeypatched, so no network is needed.

---

## Human-only steps (not done by the agent)

1. ~~`pytest -q -m network` passes.~~ Done: it passed.
2. ~~`vixagent pull` prints sensible numbers.~~ Done: see the table at the top.
3. **Decide the universe.** The spec uses ETFs: `risk` = SPY, QQQ, IWM, EFA, EEM, HYG, and `all` adds TLT, IEF, LQD, GLD. The v1 project used 10 single stocks. If the GMU study used a different list, edit `config/settings.yaml` *now*, before freezing.
4. Read `config/preregistered.yaml` and confirm you accept it as your primary test.
5. Freeze it with a commit and the `prereg-freeze` tag. From then on it never changes.
6. Answer the questions below in `docs/walkthroughs/phase-1-answers.md`.

## Checkpoint Questions (answer in `phase-1-answers.md`)

1. In your own words: why does `x.shift(1).rolling(252).mean()` avoid lookahead but `x.rolling(252).mean()` applied to a full-sample standardization would not?
2. Walk through the no-lookahead truncation test. What bug would it catch?
3. Why declustering? What goes wrong statistically if every crisis day counts as an event?
4. Why is the VIX baseline a median and not a mean?
5. Why is `preregistered.yaml` frozen *before* any results are computed?
