# Market Correlation vs. VIX

🚧 **still a work in progress.** This is a school-adjacent side project I started after reading [an article](https://www.advisorperspectives.com/articles/2025/10/06/what-signals-market-vix-blow-up) by Horstmeyer, Holtman & Bondugula about what signals a coming market "blow up." I'm basically taking one idea from that article and turning it into my own little Python thing.

## The idea, in normal words

When the market panics, people stop caring about individual companies and just sell everything at once. So stocks that normally have nothing to do with each other (like an oil company and a healthcare company) start moving together, because everyone's selling for the same reason (fear), not for reasons specific to that company.

The question I'm trying to answer: **does that "everything moving together" thing happen BEFORE the VIX spikes, or does it just happen at the same time as the VIX spike?**

The VIX is basically the market's "fear gauge" — it goes up when people expect big swings coming. If correlation spiking is something that happens *before* the VIX spikes, that'd be kind of a big deal, because it means you could maybe use it as an early warning sign. If it just happens at the same time, it's a lot less useful as a warning sign, it's just... two things that panic does at once.

## What I actually did

1. Picked 10 well-known, large stocks that are each in a different industry (Apple, JPMorgan, Exxon, Johnson & Johnson, Procter & Gamble, Home Depot, Caterpillar, Linde, Verizon, NextEra), so none of them should be moving together for normal reasons — only if something market-wide is happening. See [`src/data.py`](src/data.py).
2. Downloaded daily prices for all 10 plus the VIX, going back to 2015, using [yfinance](https://github.com/ranaroussi/yfinance).
3. Turned the prices into daily percent changes instead of using raw prices. This part actually matters — two totally unrelated stocks that both went up for 10 years straight would *look* correlated on raw price even though it means nothing about how they trade day to day. See [`src/correlation.py`](src/correlation.py).
4. For every day, calculated the average correlation across every pair of stocks over the past 20 trading days. This gives one number per day: "how much is everything moving together right now."
5. Compared that number against the VIX three different ways to check if one leads the other (details below). See [`src/analysis.py`](src/analysis.py).
6. Made a bunch of charts. See [`src/visualize.py`](src/visualize.py).

## The three tests, explained without a stats textbook

- **Cross-correlation** — I lined up the two series at every possible time offset from -20 to +20 trading days and checked which offset makes them match up best. If correlation really does lead the VIX, the best match should be on the positive side.
- **Granger causality** — a statistical test that basically asks: "if I already know the VIX's own recent history, does *also* knowing the correlation number's recent history help me guess the VIX any better?" A small p-value there is evidence that yes, it helps.
- **Event study** — I found the days where the correlation number spiked way above its own normal range (using a z-score, which is just "how many standard deviations away from its own recent average is this"), and then checked what the VIX did over the next 10 trading days after those spikes, compared to a random 10-day stretch.

## What I found so far

Short version: it doesn't really look like correlation leads the VIX in this data. All three tests point roughly the same direction — more "happens around the same time" or even "VIX moves slightly first," not "correlation spikes first as a warning sign."

![Average correlation vs VIX](output/avg_corr_vs_vix.png)

The cross-correlation test peaks at lag **−4**, meaning the VIX moved slightly *before* correlation here, not after:

![Cross-correlation](output/cross_correlation.png)

Granger causality: best p-value across every lag I tried was 0.42 — nowhere close to significant.

The event study is the most literal "does this actually warn you" test — and post-spike VIX changes actually averaged *lower* than a normal stretch (−2.4% vs. +2.3%), though the sample is tiny (16 events) so I don't trust that number much either way:

![Event study](output/event_study.png)

Sanity-check heatmap on the basket itself (not really testing the hypothesis, just checking the basket looks how I expected):

![Correlation heatmap](output/heatmap_full_period.png)

Ngl these numbers are way higher than I expected — almost everything's above 0.8, even stuff like Exxon and Home Depot. Not totally sure why yet since these are all different sectors. Might be something with how I'm calculating this one specific chart vs. the other ones, or it could just be that mega-cap stocks all trended up together over the last 10 years. Need to look into this more before I trust it.

**Bottom line so far:** not much evidence that correlation leads the VIX in this basket/period. That's still a real answer even if it's not the exciting one I was hoping for.

## Stuff I know is wrong or shaky with the code (being honest here)

I haven't had anyone who's actually good at stats or software look this over, so here's what I'm pretty sure is rough around the edges, worst first:

- **No error handling on the download step.** If yfinance is down, rate-limits me, or my wifi drops mid-download, the whole thing just crashes with a big ugly Python error instead of telling me what actually went wrong.
- **I never actually wrote automated tests.** I *did* check the math by hand against some fake data I made up (that's what fixed the two bugs mentioned below), but I never saved that checking as actual test code I can just re-run. So if I break something later, I have no way to automatically catch it — I'd have to redo all that checking by hand again.
- **No pinned versions.** My `requirements.txt` just says "any version newer than X" for everything. yfinance especially changes how it behaves between versions pretty often, so there's a real chance this whole thing breaks (or worse, silently gives different numbers) if someone installs it fresh in a year.
- **Every run re-downloads everything.** There's no saving the data locally, so every single run pulls ~10 years of data for 11 tickers again. Slow, and also means if yfinance's historical numbers ever get revised, I'd never notice — the results could quietly change between runs without me knowing why.
- **Everything is hardcoded.** The date range, the 20-day window, the z-score cutoff, all of it — they're just constants at the top of `main.py`. To try different settings I have to go edit the code directly instead of just passing an option when I run it.
- **The numbers themselves never get saved anywhere**, only the charts. So I can't easily go back and compare exact numbers between two runs, or dig into the data further, without re-running the whole thing and reading it off the printed output.
- **A couple of the charts and the underlying math could break on weird/edge-case data** (like if a run somehow comes back with barely any data) — I don't have any graceful "not enough data" message, it'd just throw a confusing error.

## Stuff about the *idea* itself that might be shaky, not just the code

- **The basket might have survivorship bias.** I picked today's well-known big companies and applied them back to 2015. But that's using hindsight — I know AAPL and JPM turned out to be huge, stable companies. Someone in 2015 didn't necessarily know that. A more honest test would use a basket that was actually big/stable back in 2015, not one picked with 2026 knowledge.
- **I tested the same idea three different ways and didn't account for that.** Running cross-correlation, Granger causality, AND an event study on the same hypothesis means there's a higher chance one of them randomly comes back "significant" just by luck, compared to only running one test. All three agreeing here (all pointing "no lead") makes me feel better about it, but if only one of them had come back significant I'd want to be careful about reading too much into that.
- **Granger causality assumes the data is "stationary"** (basically, that it doesn't have long-term trends/drift baked in), and I'm not sure the correlation series or the VIX actually qualify. If they're not stationary, the Granger test's p-value can be misleading rather than just noisy. I flagged this as a to-do below but I think it matters more than I originally gave it credit for.

## What I'd want to do next

- **Test on day-to-day changes instead of raw levels.** Both series carry over a lot from the previous day (they're "autocorrelated"), which can fake out a lead-lag relationship in both the cross-correlation and Granger tests. Redo it on daily differences as a double-check.
- **Go back to the original article's full scope.** That piece also looked at commodities, debt, and specifically gold-vs-equity correlation — this project is stocks-only right now. Adding those back in is the most direct way to actually match what I'm building off of.
- **Try different settings.** I've only ever tried one rolling window (20 days), one z-score cutoff (1.5), one lookback (252 days). I want to see if "no clear lead" still holds up across a range of nearby settings, or if I just got unlucky/lucky with these exact numbers.
- **Try other baskets.** International stocks, smaller companies, or mixing in other asset types instead of just 10 U.S. large companies.
- **Actually add the missing engineering stuff above** — tests, saving the raw data locally, pinning versions, making the settings changeable without editing code, saving numbers not just charts.

Two actual bugs from an earlier pass — an event-study baseline that accidentally included its own event days, and a calendar-day vs. trading-day mixup when de-duplicating spikes — got caught and fixed; see the commit history.

## How to run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Project layout

```
main.py         # runs the whole pipeline start to finish
src/
  data.py        # yfinance download + basket definition
  correlation.py # returns, rolling correlation, average pairwise correlation
  analysis.py    # cross-correlation, Granger causality, event study
  visualize.py   # matplotlib/seaborn charts
output/          # generated charts, committed so you can see them without re-running
```
