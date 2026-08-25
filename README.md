# Market Correlation vs. VIX

🚧 **Work in progress.** This project is an extension of a previous paper I co-authored [an article](https://www.advisorperspectives.com/articles/2025/10/06/what-signals-market-vix-blow-up) about what signals a coming market "blow up." It takes one idea from that article and builds a small Python pipeline around it.

## The idea

When the market panics, investors tend to stop evaluating companies individually and sell everything at once. As a result, stocks that normally have little to do with each other, start moving together, since the selling is driven by broad fear rather than anything specific to either company.

The question this project asks: does that rise in correlation happen *before* the VIX spikes, or does it happen at the same time?**

The VIX is often described as the market's fear index, where it rises when investors expect bigger price swings ahead. If a correlation spike consistently happens before the VIX spikes, that would be meaningful, since it could work as an early warning sign. If the two move together instead, correlation is just another symptom of the same panic.

## What the project does

1. Selects 10 well-known, large-cap stocks, one from each of 10 different industries (Apple, JPMorgan, Exxon, Johnson & Johnson, Procter & Gamble, Home Depot, Caterpillar, Linde, Verizon, NextEra), so that under normal conditions they shouldn't move together. Only a market-wide event should cause the stocks to move together. See [`src/data.py`](src/data.py).
2. Downloads daily prices for all 10 stocks plus the VIX, back to 2015, using [yfinance](https://github.com/ranaroussi/yfinance).
3. Converts prices into daily percent changes rather than working with raw prices, to more easily spot when stocks are actually moving together instead of just all drifting upward over time. See [`src/correlation.py`](src/correlation.py).
4. For each day, computes the average correlation across every pair of stocks over the trailing 20 trading days.
5. Compares that number against the VIX using three different methods to test for a lead/lag relationship (explained below). See [`src/analysis.py`](src/analysis.py).
6. Generates charts summarizing the results. See [`src/visualize.py`](src/visualize.py).

## The three tests, explained simply

- **Cross-correlation**: compares the two series at every possible time offset, from 20 days behind to 20 days ahead, and finds which offset lines them up best. If correlation truly leads the VIX, the best match should fall on the positive side.
- **Granger causality**: a statistical test that asks: given the VIX's own recent history, does also knowing the correlation number's recent history improve a prediction of the VIX? A small p-value is evidence that it does.
- **Event study**: identifies days when the correlation number spiked well above its normal range (measured with a z-score — essentially, how many standard deviations above its own recent average a value is), then checks how the VIX behaved over the following 10 trading days, compared to a typical 10-day stretch.

## Results so far

Correlation doesn't appear to lead the VIX in this dataset. All three tests point in roughly the same direction — closer to "happens at the same time," or even "the VIX moves slightly first" — rather than "correlation spikes first as a warning sign."

![Average correlation vs VIX](output/avg_corr_vs_vix.png)

The cross-correlation test peaks at lag **−4**, meaning the VIX tended to move slightly *before* correlation in this data, not after:

![Cross-correlation](output/cross_correlation.png)

For Granger causality, the best (smallest) p-value across every lag tested was 0.42 — far from statistically significant.

The event study is the most direct test of whether this would actually work as a warning sign. Post-spike VIX changes averaged *lower* than a typical stretch (−2.4% vs. +2.3%), though the sample is small (16 events), so this result shouldn't be read as strong evidence either way:

![Event study](output/event_study.png)

This heatmap is a sanity check on the basket itself, not a test of the hypothesis — just confirming the basket behaves the way I expected:

![Correlation heatmap](output/heatmap_full_period.png)

These numbers came out higher than expected — almost every pair is above 0.8, including stocks in unrelated industries like Exxon and Home Depot. I haven't fully tracked down why. It may be specific to how this particular chart is calculated compared to the others, or it may simply reflect that large-cap stocks broadly trended upward together over the past decade. Worth investigating further before drawing conclusions from it.

**Bottom line:** the current results don't support "correlation leads the VIX" in this basket and time period. That's a real, useful finding, even though it isn't the result I originally expected.

## Known issues with the code

This hasn't been reviewed by anyone with more experience in statistics or software, so here is an honest list of what's still rough, roughly in order of importance:

- **No error handling around the data download.** If yfinance is unavailable, rate-limited, or the network drops mid-download, the program crashes with a raw Python error instead of a clear message about what went wrong.
- **No automated tests.** The core calculations were checked by hand against sample data with known correct answers (that process caught the two bugs mentioned below), but that verification was never saved as reusable test code. Any future change could reintroduce a bug without anything catching it automatically.
- **Dependency versions aren't pinned.** `requirements.txt` only specifies minimum versions. yfinance in particular changes behavior between releases fairly often, so installing this fresh in the future could break the pipeline, or worse, silently change the results.
- **No local caching of downloaded data.** Every run re-downloads roughly 10 years of history for 11 tickers. This is slow, and it also means that if yfinance revises historical data later, results could change between runs without any indication of why.
- **All parameters are hardcoded.** The date range, the correlation window, the z-score threshold, and so on are all constants at the top of `main.py`. Testing different settings currently requires editing the source code rather than passing options at run time.
- **No numerical output is saved, only charts.** There's currently no way to compare exact results across runs, or dig further into the underlying numbers, without re-running the pipeline and reading values off the console output.
- **Limited handling of edge cases.** If a run returns unusually little data, several calculations would likely fail with a confusing error rather than a clear "not enough data" message.
- **The cross-correlation lag range is probably off by one.** `cross_correlation()` in `src/analysis.py` loops over `range(-max_lag, max_lag)`, which leaves out the `+max_lag` endpoint (it should almost certainly be `range(-max_lag, max_lag + 1)`). So the cross-correlation chart is actually missing its last bar on the positive side. Doesn't change the overall conclusion here since the peak is nowhere near that edge, but I should fix this.

## Limitations of the approach itself

- **The basket may have survivorship bias.** The 10 stocks were chosen based on how large and well-known they are today, then applied retroactively back to 2015. That uses information that wasn't available at the time — a more rigorous version of this test would use a basket that was actually large and established as of 2015, not one selected with hindsight.
- **The same hypothesis was tested three different ways, without adjusting for that.** Running cross-correlation, Granger causality, and an event study on the same question increases the odds that one of them shows a "significant" result purely by chance, compared to running a single test. All three agreeing here (all pointing toward "no lead") is reassuring, but if only one test had come back significant, that result should be treated with caution.
- **Granger causality assumes the data is stationary** (roughly speaking, that it isn't drifting with a long-term trend), and it's not clear that either the correlation series or the VIX actually meets that assumption. If they don't, the resulting p-values could be misleading rather than simply noisy. This is listed as a next step below, but it likely matters more than originally assumed.

## Next steps

- **Test on day-to-day changes rather than raw levels.** Both series carry over strongly from one day to the next, which can create a misleading lead/lag signal in both the cross-correlation and Granger tests. Repeating the analysis on daily differences would be a useful check.
- **Extend to the original article's full scope.** The article this project is based on also covered commodities, debt, and gold-versus-equity correlation specifically; this project currently only looks at equities. Adding those back in would better match the source material.
- **Test sensitivity to parameter choices.** Only one rolling window (20 days), one z-score threshold (1.5), and one lookback period (252 days) have been tried so far. The next step is checking whether the "no clear lead" finding holds across nearby parameter choices, or whether it's specific to these exact settings.
- **Try other baskets.** International stocks, smaller companies, or a mix of asset classes instead of 10 U.S. large-cap stocks.
- **Address the engineering gaps listed above** — automated tests, local data caching, pinned dependencies, configurable parameters, and saved numerical output.

Two earlier bugs — an event-study baseline that accidentally included its own event days, and a mixup between calendar days and trading days when de-duplicating spike events — were identified and fixed; see the commit history for details.

## How to run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Project layout

```
main.py         # runs the full pipeline end to end
src/
  data.py        # yfinance download + basket definition
  correlation.py # returns, rolling correlation, average pairwise correlation
  analysis.py    # cross-correlation, Granger causality, event study
  visualize.py   # matplotlib/seaborn charts
output/          # generated charts, committed so results are visible without re-running
```
