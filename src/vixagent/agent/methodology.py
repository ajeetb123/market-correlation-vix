"""Canonical methodology text served by the get_methodology tool."""

from __future__ import annotations

from collections.abc import Callable
from math import comb

from vixagent.config import Preregistered, Settings
from vixagent.report.results import LIMITATIONS

TOPICS: tuple[str, ...] = (
    "data",
    "correlation",
    "zscore",
    "vix_spike",
    "event_study",
    "permutation_test",
    "regression",
    "out_of_sample",
    "lookahead",
    "limitations",
)


def _data(s: Settings, p: Preregistered) -> str:
    groups = "; ".join(f"{g}: {', '.join(t)}" for g, t in s.universe.items())
    return (
        f"Daily adjusted closing prices from Yahoo Finance (via yfinance) for two asset groups "
        f"({groups}), plus the VIX close ({s.data.vix_ticker}), from {s.data.start} to "
        f"{s.data.snapshot_end}. For each group, only dates where every ticker and the VIX have "
        f"a price are kept (inner alignment). Nothing is forward filled, because a filled price "
        f"would create a fake 0% return and distort correlations. The training period is "
        f"{s.periods.train[0]} to {s.periods.train[1]} and the test period is "
        f"{s.periods.test[0]} to {s.periods.test[1]}."
    )


def _correlation(s: Settings, p: Preregistered) -> str:
    pairs = ", ".join(f"{comb(len(t), 2)} pairs in the {g} group" for g, t in s.universe.items())
    return (
        "Prices are converted to daily log returns, ln(P_t / P_{t-1}). For every pair of assets "
        "in a group, the Pearson correlation of returns is computed over the trailing W trading "
        "days, including today. The feature is the average of those pairwise correlations "
        f"({pairs}). The preregistered window is W = {p.primary.window}; "
        f"the exploratory grid also uses {', '.join(str(w) for w in s.grid.windows)}. Returns, "
        "not prices, are correlated because trending prices look correlated even when unrelated."
    )


def _zscore(s: Settings, p: Preregistered) -> str:
    return (
        "The correlation z-score compares today's average correlation with its own recent "
        f"history: z_t = (C_t - mean) / std, where the mean and sample std (ddof 1) are taken "
        f"over the previous {s.features.z_lookback} trading days, excluding today. A "
        "full-sample mean and std would let later data (for example 2020) change earlier "
        "z-scores (for example 2009), which is lookahead bias. A near-zero std gives a missing "
        "value rather than an infinite z-score."
    )


def _vix_spike(s: Settings, p: Preregistered) -> str:
    v = s.vix_spike
    return (
        f"A VIX spike day is a day where the VIX divided by the median of the prior "
        f"{v.median_lookback} trading days (excluding today) is at least {v.ratio_threshold}. "
        "The median is used because one extreme day in the baseline would pull a mean up and "
        "hide the next spike. Spike days are declustered into events: a spike day is a VIX "
        f"event only if no VIX spike day occurred in the previous {v.cooldown} trading days, so "
        "a multi-day crisis counts once instead of many times."
    )


def _event_study(s: Settings, p: Preregistered) -> str:
    e = s.event_study
    return (
        f"Correlation events are declustered days where the z-score is at or above the "
        f"threshold (cooldown {s.corr_spike.cooldown} days). An event is a hit if a VIX spike "
        "day occurs in rows t+1 to t+h, strictly after the event. The hit rate over events is "
        "compared with the base rate: the same hit definition averaged over all eligible days. "
        "Lift is hit rate divided by base rate. The clean filter drops events (and base days) "
        f"with a target spike day in the prior {e.clean_pre_window} days, because same-day "
        "co-movement in a crash would otherwise look like prediction. Eligible days must have "
        "t + h inside the same period, which embargoes train from test."
    )


def _permutation_test(s: Settings, p: Preregistered) -> str:
    return (
        "The p-value comes from a circular-shift permutation test. The event pattern over "
        "eligible days is slid along the timeline by a random offset between h+1 and n-h-1 "
        "(wrapping around the end), the clean filter is reapplied, and the hit rate is "
        f"recomputed, {s.event_study.n_permutations} times with seed {s.seed}. The one-sided "
        "p-value is (1 + number of shifted hit rates >= observed) / (1 + number of shifts). "
        "Shifting preserves the spacing and clustering of events; a t-test would wrongly treat "
        "clustered market events as independent."
    )


def _regression(s: Settings, p: Preregistered) -> str:
    return (
        "The target is the forward h-day log VIX change, ln(V_{t+h}) - ln(V_t). The base "
        "specification regresses it on the correlation z-score. The controls specification adds "
        "5-day VIX momentum, ln(V_t / V_{t-5}), and the log VIX level, to test whether "
        "correlation adds information beyond the VIX itself. Standard errors are Newey-West "
        "(HAC) with maxlags = h, because overlapping h-day outcome windows make residuals "
        f"autocorrelated. The preregistered horizon is h = {p.primary.horizon}."
    )


def _out_of_sample(s: Settings, p: Preregistered) -> str:
    return (
        f"The base regression (z-score only) is fit on the training period "
        f"({s.periods.train[0]} to {s.periods.train[1]}) and used to predict the test period "
        f"({s.periods.test[0]} to {s.periods.test[1]}). R2_oos = 1 - SSE(model) / "
        "SSE(training-period mean forecast). A positive value means the model beat the "
        "historical average; zero or negative means it did not. The event study is also run on "
        "train and test separately."
    )


def _lookahead(s: Settings, p: Preregistered) -> str:
    return (
        "Lookahead bias means using information that was not available at the time. Three "
        "safeguards prevent it. First, every feature baseline uses only past data (shift(1) "
        "before rolling statistics). Second, only one module, targets.py, may look forward, and "
        "its outputs are named fwd_. Third, an automated test recomputes every feature on data "
        "truncated at random dates and requires identical values before each cut; a static "
        "test also checks that negative shifts appear only in targets.py."
    )


def _limitations(s: Settings, p: Preregistered) -> str:
    items = " ".join(LIMITATIONS)
    return (
        f"{items} The preregistration covers one specification; every other parameter "
        "combination is exploratory."
    )


_TEXT: dict[str, Callable[[Settings, Preregistered], str]] = {
    "data": _data,
    "correlation": _correlation,
    "zscore": _zscore,
    "vix_spike": _vix_spike,
    "event_study": _event_study,
    "permutation_test": _permutation_test,
    "regression": _regression,
    "out_of_sample": _out_of_sample,
    "lookahead": _lookahead,
    "limitations": _limitations,
}


def methodology_text(topic: str, settings: Settings, prereg: Preregistered) -> str:
    """Canonical explanation for each topic, consistent with docs/SPEC.md.
    Every number is formatted from settings/prereg, never hardcoded, so the
    text cannot drift from the code. Each text is at most 200 words.
    Raises KeyError for unknown topics."""
    return _TEXT[topic](settings, prereg)
