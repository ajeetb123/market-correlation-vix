# Market Correlation vs. Vix
This extends [a paper I co-authored](https://www.advisorperspectives.com/articles/2025/10/06/what-signals-market-vix-blow-up) about what signals a coming market blowup: in a panic, unrelated stocks start moving together because everyone sells out of fear instead of judging companies individually. I wanted to test that with real, continuous data instead of static snapshots, so I picked 10 large-cap stocks with little in common (Apple, JPMorgan, Exxon, Johnson & Johnson, Procter & Gamble, Home Depot, Caterpillar, Linde, Verizon, NextEra, one per sector by design) plus the VIX, back to 2015, and checked whether a spike in their correlation actually precedes a VIX spike, or just accompanies it. 

The pipeline downloads daily prices via yfinance, converts them to daily percent changes (correlating on raw price is misleading, since unrelated stocks that both trended upward for years would appear corrrelated for no meaningful reason), then computes the basket's average pairwise correlation each day. That series is tested against the VIX three ways: cross-correlation across a range of lages, Granger causality, and an event study on correlation-spike days.

![Average correlation vs VIX](output/avg_corr_vs_vix.png)

The two series clearly move together, both spike sharply in 2020. But co-movement isn't the same as leading. The cross-correlation test peaks at lag -4, meaning the VIX moved slightly before correlation in this data, not after:

![Cross-correlation](output/cross_correlation.png)

Granger causality points the same way (the best p-value across every lag tested was 0.42, far from significant), and the event study came out slightly negative rather than positive (-2.4% vs. a baseline of +2.3%, though on only 16 events, so not a result I'd lean on heavily):

![Event study](output/event_study.png)

Taken together, the evidence doesn't support correlation leading the VIX here, if anything the relationship runs the other way. Not the result I expected, but a legitimate one.

This heatmap was meant as a basic sanity check on the basket, but the values came out higher than expected, nearly everything above 0.8, including unrelated pairs like Exxon and Home Depot. I haven't isolated the cause yet, it may be specific to how this chart is built relative to the others, or it may simply reflect a decade of broad upward drift across large-cap stocks:

![Correlation heatmap](output/heatmap_full_period.png)

Known limitations in the implementation: no error handling around the data download, no automated tests (the calculations were verified by hand against synthetic data at one point, which caught two real bugs, but that check was never preserved as reusable code), unpinned dependencies, no local caching so every run re-downloads the full history, and all parameters hardcoded rather than configurable. There is also likely an off-by-one in the cross-correlation loop in `src/analysis.py`, it should probably extend one lag further than it does, though this doesn't affect the reported result since the actual peak is well away from that edge.

There are also open questions about the design itself: the basket carries some survivorship bias, since it's built from companies known today to be large and stable and applied retroactively to 2015. The same hypothesis is tested three separate ways without correcting for that, which inflates the odds of a spurious "significant" result somewhere. And Granger causality assumes stationary inputs, an assumption I haven't verified for either series here.

Next steps: rerun on day-to-day changes rather than levels, extend to the commodities, debt, and gold-versus-equity comparisons the original paper covered, and test whether the result holds across a range of window and threshold settings rather than just the one tried so far.

## How to run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
