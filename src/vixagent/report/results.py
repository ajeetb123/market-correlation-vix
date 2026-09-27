"""Assemble all results into a JSON-safe dict and render them as Markdown."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from vixagent.analysis.event_study import (
    Direction,
    EventStudyParams,
    EventStudyResult,
    run_event_study,
)
from vixagent.analysis.frames import FrameStore
from vixagent.analysis.grid import overfitting_check, run_grid
from vixagent.analysis.oos import run_oos
from vixagent.analysis.regression import run_regression
from vixagent.config import PeriodName, Preregistered, Settings
from vixagent.utils.jsonable import to_jsonable

LIMITATIONS = [
    "Single data source: all prices come from Yahoo Finance via yfinance.",
    "Daily closing data only; intraday dynamics are not captured.",
    "The VIX spike threshold (ratio >= 1.30 vs the prior 20-day median) is a modeling choice.",
    "24 grid combinations were tested, so exploratory results face multiple-testing risk.",
    "Correlation spikes and VIX spikes can share a common cause; a lead is not causation.",
    "Results are not a trading strategy and are not investment advice.",
]


def build_results(store: FrameStore, settings: Settings, prereg: Preregistered) -> dict[str, Any]:
    """Compute everything the report needs and return a JSON-safe dict
    (passed through to_jsonable). Keys, in order:
      dataset, preregistered {spec, train, test, reverse_test}, regression
      {base, controls} (full period), oos, overfitting_check, grid,
      data_snapshot_end, generated_at (the ONLY non-deterministic field).
    """
    p = prereg.primary
    groups = list(settings.universe)
    aligned = {g: store.aligned(g) for g in groups}

    def es(period: PeriodName, direction: Direction) -> EventStudyResult:
        params = EventStudyParams(
            p.group, p.window, p.z_threshold, p.horizon, period, p.clean_only, direction
        )
        return run_event_study(store, params, settings)

    reverse: Direction = "vix_to_corr" if p.direction == "corr_to_vix" else "corr_to_vix"
    rows = run_grid(store, settings)
    results = {
        "dataset": {
            "tickers": {g: list(settings.universe[g]) for g in groups},
            "vix_ticker": settings.data.vix_ticker,
            "start": settings.data.start,
            "end": settings.data.snapshot_end,
            "n_trading_days": {g: len(aligned[g][0]) for g in groups},
            "rows_dropped": {g: aligned[g][1] for g in groups},
            "periods": {name: settings.period(name) for name in ("train", "test", "full")},
        },
        "preregistered": {
            "spec": p.model_dump(),
            "train": es("train", p.direction),
            "test": es("test", p.direction),
            "reverse_test": es("test", reverse),
        },
        "regression": {
            "base": run_regression(store, settings, p.group, p.window, p.horizon, "full", False),
            "controls": run_regression(store, settings, p.group, p.window, p.horizon, "full", True),
        },
        "oos": run_oos(store, settings, p.group, p.window, p.horizon, p.z_threshold),
        "overfitting_check": overfitting_check(rows, prereg),
        "grid": rows,
        "data_snapshot_end": settings.data.snapshot_end,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out: dict[str, Any] = to_jsonable(results)
    return out


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _num(x: float | None, digits: int) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def headline_sentence(results: dict[str, Any]) -> str:
    """One-sentence headline for the preregistered test on the test period."""
    spec = results["preregistered"]["spec"]
    test = results["preregistered"]["test"]
    start, end = results["dataset"]["periods"]["test"]
    alpha = spec["alpha"]
    p = test["p_value"]
    if test["n_events"] < 5:
        verdict = "Too few events for a reliable test."
    elif p is not None and p < alpha:
        verdict = f"Significant at alpha = {alpha}."
    else:
        verdict = f"Not significant at alpha = {alpha}: no evidence of a lead effect."
    clean = "clean events" if spec["clean_only"] else "all events"
    return (
        f"Preregistered test ({spec['group']} group, {spec['window']}-day window, "
        f"z >= {spec['z_threshold']}, {spec['horizon']}-day horizon, {clean}): "
        f"on the {start} to {end} test period, {test['n_events']} correlation events had a "
        f"VIX-spike hit rate of {_pct(test['hit_rate'])} vs a base rate of "
        f"{_pct(test['base_rate'])} (lift {_num(test['lift'], 2)}, permutation p = "
        f"{_num(p, 3)}). {verdict}"
    )


def _es_row(label: str, r: dict[str, Any]) -> str:
    return (
        f"| {label} | {r['n_events']} | {r['n_hits']} | {_pct(r['hit_rate'])} | "
        f"{_pct(r['base_rate'])} | {_num(r['lift'], 2)} | {_num(r['p_value'], 3)} |"
    )


_ES_HEADER = [
    "| | events | hits | hit rate | base rate | lift | p-value |",
    "|---|---|---|---|---|---|---|",
]


def _coef_table(reg: dict[str, Any]) -> list[str]:
    lines = [
        f"n = {reg['n']}, R^2 = {_num(reg['r2'], 4)}, HAC maxlags = {reg['hac_maxlags']}",
        "",
        "| term | coef | t (HAC) | p (HAC) |",
        "|---|---|---|---|",
    ]
    for name, c in reg["coefs"].items():
        lines.append(
            f"| {name} | {_num(c['coef'], 4)} | {_num(c['t_hac'], 2)} | {_num(c['p_hac'], 3)} |"
        )
    return lines


def _grid_label(row: dict[str, Any]) -> str:
    return f"{row['group']}, W={row['window']}, z={row['z_threshold']}, h={row['horizon']}"


def render_results_md(results: dict[str, Any]) -> str:
    """Markdown with sections: Dataset, Headline, Reverse Direction Check,
    Regression, Out-of-Sample, Overfitting Check, Full Grid (exploratory),
    Limitations. Rates as percentages with 1 decimal, lifts 2 decimals,
    p-values 3 decimals, coefficients 4 decimals. NaN renders as 'n/a'."""
    ds = results["dataset"]
    pre = results["preregistered"]
    spec = pre["spec"]
    oos = results["oos"]
    ofc = results["overfitting_check"]
    lines = [
        "# Results",
        "",
        f"Generated {results['generated_at']} from data through {results['data_snapshot_end']}.",
        "",
        "## Dataset",
        "",
        "| group | tickers | trading days | rows dropped |",
        "|---|---|---|---|",
    ]
    for g, tickers in ds["tickers"].items():
        lines.append(
            f"| {g} | {', '.join(tickers)} | {ds['n_trading_days'][g]} | {ds['rows_dropped'][g]} |"
        )
    lines += [
        "",
        f"VIX ticker `{ds['vix_ticker']}`, {ds['start']} to {ds['end']}. "
        f"Train {ds['periods']['train'][0]} to {ds['periods']['train'][1]}, "
        f"test {ds['periods']['test'][0]} to {ds['periods']['test'][1]}.",
        "",
        "## Headline",
        "",
        headline_sentence(results),
        "",
        *_ES_HEADER,
        _es_row("train", pre["train"]),
        _es_row("test", pre["test"]),
        "",
        "## Reverse Direction Check",
        "",
        "Same parameters on the test period with the roles swapped (VIX events, correlation "
        "spike days as hits). If this is as strong as the forward direction, the relationship "
        "is co-movement rather than a lead.",
        "",
        *_ES_HEADER,
        _es_row("reverse (test)", pre["reverse_test"]),
        "",
        "## Regression",
        "",
        f"Forward {spec['horizon']}-day log VIX change on the correlation z-score, full period, "
        "Newey-West (HAC) standard errors.",
        "",
        "### Base",
        "",
        *_coef_table(results["regression"]["base"]),
        "",
        "### With controls (VIX momentum and log level)",
        "",
        *_coef_table(results["regression"]["controls"]),
        "",
        "## Out-of-Sample",
        "",
        f"Base regression fit on train ({oos['n_train']} days), evaluated on test "
        f"({oos['n_test']} days): R^2_oos = {_num(oos['r2_oos'], 4)} "
        "(positive means it beats the train-period mean).",
        "",
        *_ES_HEADER,
        _es_row("train", oos["event_study_train"]),
        _es_row("test", oos["event_study_test"]),
        "",
        "## Overfitting Check",
        "",
        f"{ofc['n_combinations']} combinations tested on train.",
        "",
        "| | spec | train lift | test lift | train events | test events |",
        "|---|---|---|---|---|---|",
    ]
    for label, row in (
        ("preregistered", ofc["preregistered"]),
        ("best in-sample", ofc["best_in_sample"]),
    ):
        if row is None:
            lines.append(f"| {label} | none with >= 5 train events | | | | |")
            continue
        lines.append(
            f"| {label} | {_grid_label(row)} | {_num(row['train']['lift'], 2)} | "
            f"{_num(row['test']['lift'], 2)} | {row['train']['n_events']} | "
            f"{row['test']['n_events']} |"
        )
    lines += [
        "",
        "## Full Grid (exploratory)",
        "",
        "| spec | train events | train lift | train p | test events | test lift | test p |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in results["grid"]:
        tr, te = row["train"], row["test"]
        lines.append(
            f"| {_grid_label(row)} | {tr['n_events']} | {_num(tr['lift'], 2)} | "
            f"{_num(tr['p_value'], 3)} | {te['n_events']} | {_num(te['lift'], 2)} | "
            f"{_num(te['p_value'], 3)} |"
        )
    lines += ["", "## Limitations", "", *[f"- {item}" for item in LIMITATIONS], ""]
    return "\n".join(lines)
