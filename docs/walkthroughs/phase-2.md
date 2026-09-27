# Phase 2 Walkthrough: Analysis and Report

Phase 2 adds everything that turns features into evidence:
- the event study and its circular-shift permutation p-value
- a predictive regression with HAC standard errors
- an out-of-sample test
- the 24-combination grid with the overfitting check
- the report command, which writes `results.json`, `results.md`, and four figures, and injects the headline into the README

**Status:** 131 unit tests pass (1 network test deselected). Definition of Done passes. Coverage is 100% on every `analysis/` module, 97–100% on `report/`, and 96% overall.

`vixagent report` has **not** been run on real data. Per Step 2.11, you run it first so you read the headline before anyone else does.

## Deviations and interpretations

- **`test_run_event_study_*`.** The step file asks that `vix_to_corr` "also runs" on the synthetic panel. The panel deliberately raises correlation in the 10 days *before* each planted VIX spike. With `clean_only=True`, every reverse-direction event therefore has a correlation spike in its `t-5..t` window and is correctly filtered out, leaving 0 events. The test was split into two:
  - The forward direction must find a lift above 1.
  - The reverse direction, with the clean filter off, must find exactly the 3 planted VIX events.
- **Report tests live in one file,** `tests/test_results.py`. It covers results, figures, README injection, and `generate_report`, and reuses one module-scoped `build_results` call because a full build is the slowest thing in the suite.
- **`generate_report` lives in `report/__init__.py`,** so the CLI stays thin and tests can call it with a `tmp_path` root.
- **Heatmap date fallback.** If the sample contains no 2008 (or 2020) data, as with the synthetic 2010–2015 panel, that panel uses the full-sample VIX peak and is labeled "Peak". On real data, the 2008 and 2020 peaks always exist.
- **`reverse_test`** runs the opposite of the preregistered direction, which is `vix_to_corr`.

---

## `analysis/frames.py`: `FrameStore`

**Purpose.** Builds each `(group, window)` frame once per process and hands back the same object afterwards.

**Why.** One report needs:
- the preregistered frame about 10 times
- 4 grid frames, 48 times in total
- figures that reuse both

Rebuilding 15 or 45 rolling correlations each time is pure waste. The class holds state but does no I/O, so it can live in `analysis/`.

## `analysis/event_study.py`

**Purpose.** Answers: after a source event, is a target spike within `h` days more likely than on a typical day?

**Key functions**
- `event_study_core(...)`: the pure array computation, with no pandas frames or settings. It is easy to hand-check.
- `run_event_study(store, params, settings)`: picks source and target from the frame by direction, builds the eligible mask, and calls the core.

**Worked trace: SPEC 8.2 hand example.** There are 30 rows, source events at rows 5 and 15, target spike days at rows 8 and 25, `h = 5`, and `clean_only = False`.

1. `hit_all = fwd_any_within(tgt, 5)`: True at rows **3–7** (their windows `t+1..t+5` contain 8) and **20–24** (contain 25). That is 10 True rows.
2. `eligible = eligible_mask(..., h=5)`: rows **0–24**, which is 25 rows. Row 24 is the last one whose window `25..29` fits inside the period.
3. `clean` is all True, because `clean_only = False`.
4. `ev = src & eligible & clean`: rows **5** and **15**. `base_days = eligible & clean`: rows **0–24**.
5. `n_events = 2`. `hit_all[5] = True` and `hit_all[15] = False`, so `n_hits = 1` and `hit_rate = 0.5`.
6. `base_rate` = mean of `hit_all` over rows 0–24 = **10 / 25 = 0.4**.
7. `lift = 0.5 / 0.4 = 1.25`.

**Why the clean filter uses target spike days in both directions.**
- For `corr_to_vix`, it removes a correlation event if the VIX was already spiking in `t-5..t`.
- For `vix_to_corr`, it removes a VIX event if correlation was already spiking.

In both cases it strips out same-day co-movement, which would otherwise look like prediction. The base rate uses the same filter, so events are compared against comparable days.

## `analysis/permutation.py`: `circular_shift_pvalue`

**Purpose.** Answers: if the same pattern of events were slid to a random point on the timeline, how often would it score at least as well as the real one?

**Worked trace: one iteration, `k = 7`, on the same example.** Restricted to eligible rows 0–24:
- `src_D` is True at rows 5 and 15.
- `hit_D` is True at rows 3–7 and 20–24.
- `clean_D` is all True.

`np.roll(src_D, 7)` moves the events to rows **12** and **22**. `hit_D[12] = False` and `hit_D[22] = True`, so the shifted hit rate is 0.5. That is at least the observed 0.5, so `count` goes up by 1.

After `n_perm` iterations, `p = (1 + count) / (1 + n_perm)`. The `+1` counts the observed pattern as one of the permutations, so `p` can never be exactly 0.

**Non-obvious decisions**
- **Why shift instead of shuffle?** Market events cluster. A shift keeps the spacing between events exactly, so the null distribution is as clumpy as reality. A t-test (or a random scatter) treats events as independent, which makes real clusters look like skill.
- **Why `k` is drawn from `[h+1, n-h-1]`:**
  - A shift of `h` or less would leave each event overlapping its own original outcome window.
  - A shift of `n-h` or more wraps it back around to the same place.
- **The seed** is `settings.seed`. `test_pvalue_is_deterministic` checks that the same inputs give the same p-value.
- **Test evidence:**
  - Planted lead (events 3–7 days before 40 spikes): lift > 2 and p < 0.05.
  - Independent random events across 20 seeds: at most 20% have p < 0.05. That shows the test doesn't cry wolf.

## `analysis/regression.py`

**Purpose.** A continuous version of the question: does today's correlation z-score predict the log VIX change over the next `h` days?

**Key functions**
- `design_matrix(frame, include_controls)`: the column `z`, plus `vix_mom_5d` and `log_vix` when controls are included.
- `fit_hac(y, X, maxlags)`: OLS with Newey-West standard errors.
- `run_regression(...)`: builds `y` from `targets.fwd_log_vix_change`, restricts to eligible days, and fits.

**How `fit_hac` differs from plain OLS.** The coefficient is **identical**. Only the standard error changes.
- Plain OLS assumes each row's error is independent.
- With a 10-day forward target, row `t` covers days `t+1..t+10` and row `t+1` covers `t+2..t+11`. They share 9 days, so their errors are strongly correlated.
- Newey-West adds the autocovariances up to `maxlags = h` into the variance estimate.

**What test 3 demonstrated.**
- `y` was built as overlapping 10-day sums of pure noise.
- `x` was an unrelated AR(1) with persistence 0.9.
- The true coefficient is therefore **0**.

| | coef | SE | t | p |
|---|---|---|---|---|
| plain OLS | 0.0256 | 0.0240 | 1.07 | 0.287 |
| HAC (maxlags 10) | 0.0256 | about 2.2x larger | **0.48** | **0.632** |

HAC roughly halved the t-statistic. On real data, where the effect is subtle, that difference decides whether a spurious "significant" result gets reported.

**Why controls?** VIX momentum and level are known predictors of future VIX changes. If `z` loses significance once they are included, correlation added nothing new.

## `analysis/oos.py`

**Purpose.** Fits the base regression on train only, predicts test, and scores the predictions against the dumbest honest benchmark: the train-period mean.

**Worked trace: `r2_oos` with 3 rows.** `y_test = [1, 2, 3]`.
- Perfect predictions `[1, 2, 3]` with `train_mean = 0`: SSE(model) = 0, and SSE(benchmark) = 1² + 2² + 3² = 14. `R2_oos = 1 - 0/14 = 1.0`.
- Predictions equal to the benchmark `[0, 0, 0]`: SSE(model) = 14 = SSE(benchmark), so `R2_oos = 1 - 14/14 = 0.0`.
- Anything worse than the benchmark is negative.

**Non-obvious decision.** The benchmark is the **train** mean, not the test mean. At the end of 2018, a forecaster couldn't know the 2019–2026 average. Using the test mean would itself be lookahead.

## `analysis/grid.py`

**Purpose.** Runs the clean `corr_to_vix` event study on train and test for all 2 × 2 × 2 × 3 = 24 combinations. It then picks the best train lift (at least 5 train events, ties broken by grid order) and reports that row's test lift next to the preregistered row.

**Why.** Picking the best of 24 on train is exactly the cherry-picking that preregistration prevents. Showing how much that pick decays on test measures how much of its in-sample lift was luck.

## `report/results.py`, `report/figures.py`, `report/readme.py`, `vixagent report`

- **`build_results`**: computes everything and passes it through `to_jsonable`. `generated_at` is the only non-deterministic field, and a test checks that two builds match once it is removed.
- **`headline_sentence`**: the template from Step 2.7. The verdict is one of:
  - "Too few events" when there are fewer than 5 events
  - "Significant" when p < alpha
  - "Not significant ... no evidence of a lead effect" otherwise
- **`render_results_md`**: the eight sections, with fixed number formats and NaN rendered as `n/a`.
- **Figures**: `timeseries.png`, `corr_heatmaps.png`, `event_study.png`, and `grid_lift.png`. The Agg backend is set before pyplot is imported, so no display is needed.
- **`replace_between_markers` / `update_readme`**: only text between `<!-- NAME:START -->` and `<!-- NAME:END -->` changes. Running it twice is a no-op. README numbers are never typed by hand.

---

## Human-only steps

1. Run `vixagent report`. Read the headline sentence in the terminal **before** opening anything else, and write it in `docs/walkthroughs/phase-2-answers.md`.
2. Open the four figures and `reports/results.md`. Sanity checks:
   - The VIX peaks in late 2008 and March 2020, and correlation rises in both.
   - The crisis heatmaps are redder than the calm one.
   - Event counts are in the tens.
3. Commit `reports/` and `README.md`, then tag `v0.1`.
4. Answer the checkpoint questions.

## Checkpoint Questions (answer in `phase-2-answers.md`)

1. State the headline result in one sentence, including `n_events`, lift, and p-value. Is it significant at alpha 0.05?
2. Explain the hand-checkable event-study example (SPEC 8.2) and compute the base rate by hand.
3. Why a circular-shift permutation test instead of a t-test?
4. Why HAC standard errors with `maxlags = h`?
5. What did the reverse-direction check show, and what would it mean if it were as strong as the forward direction?
6. In the overfitting check, how much did the best in-sample combination decay out of sample? Why is that expected?
