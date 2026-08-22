# Market Correlation vs. VIX

🚧 **Work in progress.** Does a spike in how correlated a basket of unrelated
stocks are with each other tend to *lead* a spike in the VIX, or do the two
just move together? A Python rebuild of an Excel-based research project,
extending [Horstmeyer, Holtman & Bondugula, "What Signals a Coming
Market/VIX Blow Up?"](https://www.advisorperspectives.com/articles/2025/10/06/what-signals-market-vix-blow-up)
(*Advisor Perspectives*, Oct 2025) — that piece compared static
before/during correlation snapshots across 6 historical crises; this project
turns the same idea into a continuous, statistically-tested daily question.

## The hypothesis

In calm markets, companies move for their own reasons — a drug trial here,
an oil price there — so unrelated stocks shouldn't move in lockstep. In a
panic, investors stop differentiating and sell everything together, so
correlation rises. **The question: does that rise in correlation happen
*before* the VIX spikes, making it a leading indicator — or just alongside it?**

## Data & method

10 large-cap U.S. stocks, **one per sector, no repeats** (`AAPL`, `JPM`,
`XOM`, `JNJ`, `PG`, `HD`, `CAT`, `LIN`, `VZ`, `NEE`) plus `^VIX`, daily since
2015-01-01, via [yfinance](https://github.com/ranaroussi/yfinance). Every
sector appears exactly once so a correlated move can't just be ordinary
sector news — see [`src/data.py`](src/data.py).

[`src/correlation.py`](src/correlation.py) turns prices into daily returns,
then a rolling 20-day pairwise correlation matrix collapsed into one
"average correlation" number per day. [`src/analysis.py`](src/analysis.py)
tests the timing three ways — cross-correlation across ±20-day lags,
Granger causality, and an event study on detected correlation spikes.
[`src/visualize.py`](src/visualize.py) draws the charts below.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Results

The two series clearly move together — both spike hard in March 2020, and
track each other's smaller bumps through 2018 and 2022:

![Average correlation vs VIX](output/avg_corr_vs_vix.png)

The real test is *timing*, not just co-movement. The chart below repeats
that same comparison at every possible offset from −20 to +20 trading days —
if the hypothesis were true, the tallest bar would sit on the positive
(right) side. Instead it peaks at **lag −4**: VIX moved slightly *ahead of*
correlation here, not behind it:

![Cross-correlation](output/cross_correlation.png)

A Granger causality test asks the same timing question a stricter way —
does correlation's recent history actually improve a prediction of VIX,
beyond what VIX's own history already gives you? No: the best p-value
across all 20 lags was 0.42, far from significant.

The event study looks at what happens *after* the 16 days correlation
spiked hardest — VIX's average change over the next 10 trading days,
compared to a normal 10-day stretch. This chart is the most literal
"leading indicator" test of the three:

![Event study](output/event_study.png)

Post-spike VIX changes actually averaged **lower** than normal (−2.4% vs.
+2.3%), though not by a statistically meaningful margin (p=0.48, n=16 —
too small a sample to trust either way).

The heatmap below is a sanity check on the basket itself, not the
hypothesis — confirming the one-per-sector redesign worked: no pair
dominates (highest is JPM/CAT at 0.58), unlike an earlier version of this
basket where a same-sector pair hit 0.88 purely from shared-industry news:

![Correlation heatmap](output/heatmap_full_period.png)

**Bottom line:** these three charts don't support "correlation leads VIX"
in this basket and period — they look closer to simultaneous, if anything
tilted the other way. That's a real, useful negative result, not a failed
one.

## What's next

- **Test on changes, not levels.** Both series are highly autocorrelated,
  which can inflate apparent lead-lag relationships in cross-correlation
  and Granger tests. Re-run on day-to-day differences as a robustness check.
- **Go back to the original research's full scope.** This project is
  equities-only; the study it extends also covered commodities, debt, and
  specifically gold-vs-equity correlation. Adding those back in is the
  most direct way to reconnect the two.
- **Check parameter sensitivity.** Only one rolling window (20 days), one
  z-score lookback (252 days), and one spike threshold (1.5σ) have been
  tried. Next: see whether the "no clear lead" finding holds across nearby
  parameter choices, or if it's an artifact of these specific settings.
- **Try other baskets.** International equities, small-caps, or a
  cross-asset basket instead of just 10 U.S. large-caps.

Two known bugs from earlier verification passes — an event-study baseline
that included its own event days, and a calendar-day vs. trading-day mixup
in spike de-duplication — have since been fixed and re-verified; see commit
history for details.

## Project layout

```
main.py         # runs the full pipeline end to end
src/
  data.py        # yfinance download + basket definition
  correlation.py # returns, rolling correlation, average pairwise correlation
  analysis.py    # cross-correlation, Granger causality, event study
  visualize.py   # seaborn/matplotlib charts
output/          # generated charts (committed) and CSVs (gitignored)
```
