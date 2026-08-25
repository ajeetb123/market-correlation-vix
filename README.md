# Market Correlation vs. VIX

Work in progress.

I got the idea for this after reading [an article](https://www.advisorperspectives.com/articles/2025/10/06/what-signals-market-vix-blow-up) by Horstmeyer, Holtman & Bondugula about what tends to signal a coming market blowup. Their main point was that stocks which normally have nothing to do with each other start moving together right before things get bad, because everyone stops caring about individual companies and just sells out of fear. I wanted to actually test that with real data instead of just reading about it, so I picked 10 big companies that don't have much to do with each other (Apple, JPMorgan, Exxon, Johnson & Johnson, Procter & Gamble, Home Depot, Caterpillar, Linde, Verizon, NextEra, one from each sector on purpose) and the VIX going back to 2015, and checked whether a spike in how correlated those stocks are actually comes before a VIX spike, or if it's just the same thing happening at the same time.

The code downloads daily prices with yfinance, turns them into daily percent changes (correlating on raw price is a trap, since two totally unrelated stocks that both went up for years will look "correlated" even though it means nothing), then for every day computes the average correlation across every pair of stocks over the last 20 trading days. That one number per day is what actually gets compared against the VIX, three different ways: a cross-correlation across a range of lags, a Granger causality test, and an event study on the days where correlation spiked.

Here's the main chart, correlation on top, VIX on bottom:

![Average correlation vs VIX](output/avg_corr_vs_vix.png)

They clearly move together, both spike hard in 2020. But moving together isn't the same as one leading the other, which is the actual question. The cross-correlation chart checks that directly, and it peaks at lag -4, meaning in this data the VIX moved slightly before correlation did, not after:

![Cross-correlation](output/cross_correlation.png)

Granger causality agrees, best p-value across every lag I tried was 0.42, nowhere near significant. And the event study, which looks at what the VIX actually did in the 10 days after a correlation spike, came out slightly negative instead of positive (about -2.4% vs. the normal +2.3%, though that's only 16 events so it's pretty noisy):

![Event study](output/event_study.png)

So overall, it doesn't look like correlation leads the VIX here, more like they move together or the VIX moves first if anything. That's not the answer I was hoping for going in, but it's still a real result.

This heatmap was supposed to just be a sanity check on the basket, but the numbers came out way higher than I expected, almost everything above 0.8, even totally unrelated stuff like Exxon and Home Depot:

![Correlation heatmap](output/heatmap_full_period.png)

I haven't figured out exactly why yet. Might be something with how this specific chart gets built compared to the others, or it could just be that mega-cap stocks all trended up together over 10 years. Need to dig into that more before I trust it.

Stuff I know is still rough: there's no error handling if the download fails, no real automated tests (I checked the math by hand against made-up data at one point and that's how I caught two actual bugs, but I never saved that as code I can rerun), the dependency versions in requirements.txt aren't pinned so this could break on a fresh install, every run re-downloads everything instead of saving it locally, and all the settings (date range, window size, thresholds) are hardcoded constants instead of something you can change without editing the code. I'm also pretty sure there's an off-by-one in the cross-correlation loop in `src/analysis.py`, it should probably go one lag further than it does, though it doesn't change the overall answer here since the actual peak isn't anywhere near that edge.

There's also some bigger-picture stuff that bugs me about the setup itself. I picked today's well-known big companies and applied them back to 2015, which is a little unfair since I only know they turned out to be big and stable because of hindsight. I also ran the same hypothesis three different ways without really accounting for the fact that testing something three times makes it more likely one of them looks "significant" just by chance. And Granger causality technically assumes the data doesn't have long-term drift baked into it, which I'm not sure is true here, so that p-value might be less trustworthy than it looks.

Next up I want to try this on day-to-day changes instead of raw levels, go back and add in the other asset types (commodities, debt, gold vs. equities) that the original article covered, and try a few different window/threshold settings to see if the "no lead" result holds up or if I just got unlucky with these specific numbers.

## How to run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Everything runs from `main.py`, plus a few files under `src/`: `data.py` downloads the prices, `correlation.py` does the returns/correlation math, `analysis.py` has the three lead-lag tests, and `visualize.py` makes the charts. `output/` has the generated charts committed so you can see them without rerunning anything.
