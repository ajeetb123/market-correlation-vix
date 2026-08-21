# Market Correlation vs. VIX

**Status:** 🚧 work in progress. The core pipeline runs end-to-end and its
statistical functions are verified (see [Verification](#verification)
below), but the methodology has known gaps — see [Open items](#open-items)
— and the basket, thresholds, and event-study design are still being
refined.

A Python rebuild of an Excel-based research project: does a breakdown in
diversification across a basket of stocks - i.e., a spike in how correlated
they all are with each other — tend to *lead* a spike in the VIX, or are the
two more or less contemporaneous?

## Background

This project extends earlier published research: [Derek Horstmeyer, Hugh
Holtman, and Ajeet Bondugula, "What Signals a Coming Market/VIX Blow
Up?"](https://www.advisorperspectives.com/articles/2025/10/06/what-signals-market-vix-blow-up),
*Advisor Perspectives*, October 6, 2025.

That study looked at a broad cross-asset universe — equities, commodities,
and debt — around six historical volatility events where the VIX roughly
doubled in under three months (1998, 2002, 2008, 2010, 2011, and 2020). For
each event, it built a correlation matrix for the three months *before* the
spike and compared it to a correlation matrix from *during* the spike. The
takeaway across all six events: equity-class correlations tend to run
unusually high heading into a VIX blowup, gold-vs-equity correlation tends
to turn unusually negative, and diversification broadly breaks down once the
event is underway.

That earlier work compared two static snapshots (before vs. during) across
historical crises. This project turns the same underlying idea into a
continuous, day-by-day question on a single equity basket: instead of two
snapshots, it builds an unbroken rolling correlation series and tests
directly, with statistics rather than a visual before/after comparison,
whether *rises* in that series measurably *precede* rises in the VIX.

## Hypothesis

In normal markets, different companies' stock prices move for their own
reasons: an earnings beat here, an oil price move there, a product recall
somewhere else. Diversification works because those reasons are mostly
independent, so a basket of unrelated names shouldn't move in lockstep.

During market-wide stress, that breaks down. Investors stop differentiating
between individual names and sell (or buy) everything together, driven by a
single macro factor — this is the "correlations go to 1 in a crisis" effect.
The VIX, meanwhile, is a forward-looking measure of expected S&P 500
volatility derived from options prices.

The hypothesis this project tests: **a rise in average pairwise correlation
across a diversified stock basket is an early signal of rising fear, and
tends to precede — not just coincide with — a spike in the VIX.** If true,
average correlation would be a useful leading indicator. If the two series
are simply contemporaneous (or correlation *lags* VIX), that's a much weaker
and less useful result.

## Data

- **Basket** — 10 large-cap U.S. stocks, **one per sector, no repeats**:
  `AAPL` (Technology), `JPM` (Financials), `XOM` (Energy), `JNJ`
  (Healthcare), `PG` (Consumer Staples), `HD` (Consumer Discretionary),
  `CAT` (Industrials), `LIN` (Materials), `VZ` (Communication Services),
  `NEE` (Utilities). Every sector appears exactly once on purpose — an
  earlier version of this basket had two stocks in each of three sectors,
  which meant part of the "average correlation" number could just be
  ordinary sector news (e.g. two tech stocks reacting to the same rate
  headline) rather than genuine market-wide co-movement. See
  [`src/data.py`](src/data.py) for the reasoning in full.
- **VIX** — `^VIX`, the CBOE Volatility Index.
- **Source** — [yfinance](https://github.com/ranaroussi/yfinance), daily
  split/dividend-adjusted closes, 2015-01-01 to present.

## Method

1. **[`src/data.py`](src/data.py)** downloads adjusted daily closes for the
   basket and the VIX and aligns them to a common trading calendar.
2. **[`src/correlation.py`](src/correlation.py)** computes daily returns,
   then a rolling 20-trading-day (~1 month) pairwise correlation matrix. Each
   day's matrix is collapsed to a single **average pairwise correlation** —
   the mean of every unique off-diagonal pair, not just correlation-to-the-mean,
   so no single ticker dominates the number.
3. **[`src/visualize.py`](src/visualize.py)** renders the results: seaborn
   heatmaps of the correlation matrix, and average correlation plotted
   against the VIX.
4. **[`src/analysis.py`](src/analysis.py)** tests the lead-lag hypothesis
   three ways:
   - **Cross-correlation** of the average-correlation series against the VIX
     at lags of ±20 trading days, to see where the relationship peaks.
   - **Granger causality** test — does average correlation's own recent
     history help predict the VIX beyond what the VIX's own history already
     predicts?
   - **Event study** — flag days where average correlation spikes more than
     1.5 standard deviations above its trailing 1-year mean, then compare the
     VIX's change over the following 10 trading days after those events
     against its unconditional 10-day change (Welch's t-test).

Run the whole pipeline with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Charts are written to `output/`.

## Verification

Before trusting any number below, every piece of the pipeline was checked
against an independent source of truth rather than just read back:

- **`average_pairwise_correlation`** — recomputed with a separate,
  brute-force implementation (a plain Python loop over each 20-day window,
  no pandas vectorization) and compared to the pipeline's output across all
  2,906 overlapping trading days. Max difference: `8.2e-15`, i.e.
  floating-point noise, not a discrepancy.
- **`cross_correlation`'s lag sign convention** — tested on synthetic data
  built so `x` was constructed to lead `y` by a known 5 trading days. The
  function correctly reported its peak at lag +5.
- **`granger_causality`'s direction** — tested on a synthetic series where
  `x` truly drives `y` one day later: the test correctly returned p≈0 for
  "x causes y" and p>0.05 (not significant) for the reverse "y causes x".
- **`identify_spike_events`** — tested against a series with six manually
  planted spikes at known positions; it recovered exactly those six, no
  more, no fewer.
- **`event_study`** — tested on synthetic data with a forced, real VIX jump
  10 days after each planted spike; it correctly detected the effect
  (p=0.014).
- **The downloaded data itself** — spot-checked against publicly recorded
  VIX closes on well-known dates (Aug 24 2015, Feb 5 2018, Mar 16 2020, Aug
  5 2024, Apr 8 2025); every value matched exactly.
- **The one-per-sector redesign worked as intended** — with no two stocks
  sharing a sector, the full-period heatmap now has no artificially
  dominant pair: the highest is JPM/CAT at 0.58, well below the 0.88 a
  same-sector pair (JPM/BAC) hit in the earlier version of this basket.

## Open items

Real gaps, not yet addressed:

- [ ] `event_study`'s "baseline" 10-day VIX change is computed over *all*
  trading days, which technically includes the spike days themselves —
  it isn't excluded from its own control group. With only 16 events in
  ~2,900 days the contamination is small, but it should be excluded
  properly rather than approximately.
- [ ] `identify_spike_events`'s `min_gap` de-duplication measures spacing
  in *calendar* days (`(d - kept[-1]).days`) but is meant to represent
  *trading*-day spacing — close enough most of the time, but not exact
  across weekends/holidays.
- [ ] Granger causality and cross-correlation are both run on the *level*
  of average correlation and VIX rather than their day-to-day changes;
  both series are highly autocorrelated, which can inflate apparent
  lead-lag relationships. Re-run on differenced series as a robustness
  check.
- [ ] Basket is single-country equities only. The original research this
  extends also covered commodities, debt, and gold-vs-equity correlation
  specifically — not yet reproduced here.
- [ ] Only one rolling window (20 days), one z-score lookback (252 days),
  and one spike threshold (1.5σ) have been tried. No sensitivity check yet
  on whether the results hold across nearby parameter choices.

## Results

*(From the run committed to this repo, 2015-01-01 through the run date.
Re-running `main.py` refreshes both the numbers below and the charts.)*

**Average correlation vs. VIX over time** — visually, the two do move
together, most clearly during the March 2020 and 2022 stress periods:

![Average correlation vs VIX](output/avg_corr_vs_vix.png)

**Cross-correlation** peaked at **lag −4** (correlation 0.56), meaning in
this sample the VIX moved slightly *ahead of* average correlation rather
than behind it — the opposite of the hypothesis's direction, though only by
a few days:

![Cross-correlation](output/cross_correlation.png)

**Granger causality** found no lag at which average correlation
significantly predicted the VIX beyond the VIX's own history (smallest
p-value 0.42, far above conventional significance thresholds).

**Event study**: 16 correlation-spike events were identified over the
sample. The VIX's mean 10-day forward change after those events (**−2.4%**)
was actually *lower* than its unconditional 10-day change (+2.3%) — the
opposite of the hypothesized direction this time — and, as before, not
statistically significant (t=−0.72, p=0.48).

![Event study](output/event_study.png)

**Full-period correlation heatmap**, as a sanity check — with no two stocks
sharing a sector, no single pair dominates; the highest is JPM/CAT at 0.58,
consistent with genuine cross-sector co-movement rather than one pair of
similar businesses inflating the picture:

![Correlation heatmap](output/heatmap_full_period.png)

### Honest takeaway

This basket/period does **not** provide evidence that correlation spikes
lead VIX spikes — under the original basket the event study leaned
(insignificantly) in the hypothesized direction, and under this
one-per-sector basket it flipped to leaning (still insignificantly) the
*other* way. That instability between two reasonable basket choices is
itself informative: whatever relationship exists here is not strong or
robust enough to survive a change in which 10 stocks you pick. See [Open
items](#open-items) above for the remaining methodological gaps, and
[Verification](#verification) for what's already been checked and is *not*
the explanation.

## Project layout

```
main.py              # runs the full pipeline end to end
src/
  data.py             # yfinance download + alignment
  correlation.py      # returns, rolling correlation matrices, average pairwise correlation
  analysis.py         # cross-correlation, Granger causality, event study
  visualize.py         # seaborn/matplotlib charts
output/                # generated charts (committed) and CSVs (gitignored)
```
