# Results

Generated 2026-09-28T20:29:39+00:00 from data through 2026-06-30.

## Dataset

| group | tickers | trading days | rows dropped |
|---|---|---|---|
| risk | SPY, QQQ, IWM, EFA, EEM, HYG | 4822 | 1 |
| all | SPY, QQQ, IWM, EFA, EEM, HYG, TLT, IEF, LQD, GLD | 4822 | 1 |

VIX ticker `^VIX`, 2007-05-01 to 2026-06-30. Train 2007-05-01 to 2018-12-31, test 2019-01-01 to 2026-06-30.

## Headline

Preregistered test (risk group, 21-day window, z >= 2.0, 10-day horizon, clean events): on the 2019-01-01 to 2026-06-30 test period, 2 correlation events had a VIX-spike hit rate of 0.0% vs a base rate of 18.8% (lift 0.00, permutation p = 1.000). Too few events for a reliable test.

| | events | hits | hit rate | base rate | lift | p-value |
|---|---|---|---|---|---|---|
| train | 0 | 0 | n/a | 14.2% | n/a | n/a |
| test | 2 | 0 | 0.0% | 18.8% | 0.00 | 1.000 |

## Reverse Direction Check

Same parameters on the test period with the roles swapped (VIX events, correlation spike days as hits). If this is as strong as the forward direction, the relationship is co-movement rather than a lead.

| | events | hits | hit rate | base rate | lift | p-value |
|---|---|---|---|---|---|---|
| reverse (test) | 24 | 0 | 0.0% | 2.2% | 0.00 | 1.000 |

## Regression

Forward 10-day log VIX change on the correlation z-score, full period, Newey-West (HAC) standard errors.

### Base

n = 4539, R^2 = 0.0260, HAC maxlags = 10

| term | coef | t (HAC) | p (HAC) |
|---|---|---|---|
| const | -0.0005 | -0.07 | 0.948 |
| z | -0.0279 | -4.59 | 0.000 |

### With controls (VIX momentum and log level)

n = 4539, R^2 = 0.0965, HAC maxlags = 10

| term | coef | t (HAC) | p (HAC) |
|---|---|---|---|
| const | 0.3287 | 5.40 | 0.000 |
| z | -0.0139 | -2.10 | 0.035 |
| vix_mom_5d | -0.1826 | -3.90 | 0.000 |
| log_vix | -0.1129 | -5.43 | 0.000 |

## Out-of-Sample

Base regression fit on train (2656 days), evaluated on test (1873 days): R^2_oos = 0.0210 (positive means it beats the train-period mean).

| | events | hits | hit rate | base rate | lift | p-value |
|---|---|---|---|---|---|---|
| train | 0 | 0 | n/a | 14.2% | n/a | n/a |
| test | 2 | 0 | 0.0% | 18.8% | 0.00 | 1.000 |

## Overfitting Check

24 combinations tested on train.

| | spec | train lift | test lift | train events | test events |
|---|---|---|---|---|---|
| preregistered | risk, W=21, z=2.0, h=10 | n/a | 0.00 | 0 | 2 |
| best in-sample | all, W=63, z=1.5, h=10 | 2.76 | 0.00 | 5 | 3 |

## Full Grid (exploratory)

| spec | train events | train lift | train p | test events | test lift | test p |
|---|---|---|---|---|---|---|
| risk, W=21, z=1.5, h=5 | 1 | 0.00 | 1.000 | 7 | 0.00 | 1.000 |
| risk, W=21, z=1.5, h=10 | 1 | 0.00 | 1.000 | 7 | 0.00 | 1.000 |
| risk, W=21, z=1.5, h=20 | 1 | 0.00 | 1.000 | 6 | 0.49 | 0.902 |
| risk, W=21, z=2.0, h=5 | 0 | n/a | n/a | 2 | 0.00 | 1.000 |
| risk, W=21, z=2.0, h=10 | 0 | n/a | n/a | 2 | 0.00 | 1.000 |
| risk, W=21, z=2.0, h=20 | 0 | n/a | n/a | 2 | 0.00 | 1.000 |
| risk, W=63, z=1.5, h=5 | 1 | 0.00 | 1.000 | 3 | 6.96 | 0.003 |
| risk, W=63, z=1.5, h=10 | 1 | 0.00 | 1.000 | 3 | 3.54 | 0.013 |
| risk, W=63, z=1.5, h=20 | 1 | 0.00 | 1.000 | 3 | 1.95 | 0.083 |
| risk, W=63, z=2.0, h=5 | 0 | n/a | n/a | 0 | n/a | n/a |
| risk, W=63, z=2.0, h=10 | 0 | n/a | n/a | 0 | n/a | n/a |
| risk, W=63, z=2.0, h=20 | 0 | n/a | n/a | 0 | n/a | n/a |
| all, W=21, z=1.5, h=5 | 17 | 1.64 | 0.280 | 10 | 1.04 | 0.388 |
| all, W=21, z=1.5, h=10 | 17 | 0.83 | 0.655 | 10 | 0.53 | 0.735 |
| all, W=21, z=1.5, h=20 | 17 | 1.31 | 0.213 | 10 | 0.88 | 0.641 |
| all, W=21, z=2.0, h=5 | 12 | 2.32 | 0.132 | 8 | 1.30 | 0.313 |
| all, W=21, z=2.0, h=10 | 12 | 2.34 | 0.048 | 8 | 0.66 | 0.631 |
| all, W=21, z=2.0, h=20 | 12 | 1.23 | 0.347 | 8 | 0.73 | 0.713 |
| all, W=63, z=1.5, h=5 | 5 | 2.73 | 0.113 | 3 | 0.00 | 1.000 |
| all, W=63, z=1.5, h=10 | 5 | 2.76 | 0.056 | 3 | 0.00 | 1.000 |
| all, W=63, z=1.5, h=20 | 5 | 1.45 | 0.230 | 3 | 1.95 | 0.122 |
| all, W=63, z=2.0, h=5 | 7 | 1.95 | 0.180 | 4 | 2.61 | 0.114 |
| all, W=63, z=2.0, h=10 | 7 | 0.99 | 0.415 | 4 | 1.33 | 0.333 |
| all, W=63, z=2.0, h=20 | 7 | 0.52 | 0.761 | 4 | 0.73 | 0.672 |

## Limitations

- Single data source: all prices come from Yahoo Finance via yfinance.
- Daily closing data only; intraday dynamics are not captured.
- The VIX spike threshold (ratio >= 1.30 vs the prior 20-day median) is a modeling choice.
- 24 grid combinations were tested, so exploratory results face multiple-testing risk.
- Correlation spikes and VIX spikes can share a common cause; a lead is not causation.
- Results are not a trading strategy and are not investment advice.
