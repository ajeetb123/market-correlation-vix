# Testing the Paper's Lead-up Claim

Equity styles: VFINX, NAESX, VIVAX, VIGRX. Gold: GC=F. Data through 2026-06-30.

## Preregistered hypotheses

- **equity_styles**: Over the 63 trading days before a VIX-doubling event, the average pairwise correlation of daily returns among US large-cap, small-cap, value, and growth equity (VFINX, NAESX, VIVAX, VIGRX) is higher than its average over all days.
- **gold_equity**: Over the same 63-day lead-up windows, the average correlation between gold (GC=F front-month futures) and those four equity styles is lower (more negative) than its average over all days.

Events: A VIX-doubling event is a day when the VIX closes at least 2x its minimum over the trailing 63 trading days (inclusive), declustered so that a day counts only if no qualifying day occurred in the prior 126 trading days. The lead-up window is the 63 trading days ending the day before the event.

## Primary verdicts (events not in the paper)

- Claim: equity-style correlation (large, small, value, growth) is unusually high in the 3 months before a VIX doubling. Across 12 events the paper did not study, the lead-up average was 0.835 vs 0.876 on a typical day (one-sided permutation p = 0.949). Not supported at alpha = 0.025.
- Claim: gold-equity correlation is unusually negative in the 3 months before a VIX doubling. Across 11 events the paper did not study, the lead-up average was -0.003 vs 0.002 on a typical day (one-sided permutation p = 0.493). Not supported at alpha = 0.025.

## All tests

| measure | events | n | lead-up mean | typical | p-value |
|---|---|---|---|---|---|
| equity_styles | new | 12 | 0.835 | 0.876 | 0.949 |
| equity_styles | paper | 6 | 0.914 | 0.876 | 0.114 |
| equity_styles | all | 18 | 0.861 | 0.876 | 0.785 |
| gold_equity | new | 11 | -0.003 | 0.002 | 0.493 |
| gold_equity | paper | 5 | -0.112 | 0.002 | 0.144 |
| gold_equity | all | 16 | -0.037 | 0.002 | 0.221 |

Only the `new` rows are confirmatory. The `paper` rows re-check the paper's own events with different data; `all` pools both.

## Per-event lead-up values

| event date | set | equity styles | gold-equity |
|---|---|---|---|
| 1994-03-31 | new | 0.854 | n/a |
| 1998-08-14 | paper | 0.847 | n/a |
| 2001-09-17 | new | 0.824 | -0.254 |
| 2002-07-16 | paper | 0.885 | -0.248 |
| 2006-06-13 | new | 0.939 | -0.001 |
| 2007-08-09 | new | 0.962 | 0.156 |
| 2008-09-29 | paper | 0.953 | -0.367 |
| 2010-05-06 | paper | 0.962 | 0.455 |
| 2011-08-04 | paper | 0.964 | 0.017 |
| 2014-10-13 | new | 0.944 | -0.316 |
| 2015-08-21 | new | 0.958 | -0.182 |
| 2018-02-05 | new | 0.847 | 0.135 |
| 2018-10-10 | new | 0.704 | 0.202 |
| 2019-08-05 | new | 0.891 | -0.087 |
| 2020-02-24 | paper | 0.871 | -0.417 |
| 2021-12-01 | new | 0.776 | -0.070 |
| 2024-08-05 | new | 0.616 | 0.205 |
| 2026-03-06 | new | 0.706 | 0.184 |
