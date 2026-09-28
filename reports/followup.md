# Follow-up Study Results

Universe: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY. Data through 2026-06-30.

## Hypothesis (preregistered)

When the average pairwise correlation of the nine sector ETFs is unusually high (its trailing 252-day z-score over a 21-day window), the VIX falls more over the next 10 trading days than its own 5-day momentum and log level predict. Formally: in the regression of the forward 10-day log VIX change on the correlation z-score, 5-day VIX momentum, and log VIX, the coefficient on the z-score is negative.

## Verdict

Follow-up test (sector ETFs, 21-day window, 10-day horizon, with VIX momentum and level controls): on the untouched 1998-12-22 to 2007-04-30 holdout, the coefficient on the correlation z-score was -0.0078 (HAC t = -1.26, one-sided p = 0.1038, n = 1816). Not supported at alpha = 0.05.

## Regressions (coefficient on correlation z-score)

| | period | n | coef z | t (HAC) | one-sided p |
|---|---|---|---|---|---|
| holdout, preregistered spec | 1998-12-22 to 2007-04-30 | 1816 | -0.0078 | -1.26 | 0.1038 |
| holdout, other spec (secondary) | 1998-12-22 to 2007-04-30 | 1816 | -0.0128 | -2.24 | 0.0125 |
| exploration, preregistered spec | 2007-05-01 to 2026-06-30 | 4812 | -0.0205 | -3.23 | 0.0006 |

Only the first row is the confirmatory test. The exploration period was already seen when the hypothesis was formed, so it cannot confirm anything.
