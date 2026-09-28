"""Render the paper-replication results (from the JSON-safe results dict)."""

from __future__ import annotations

from typing import Any

LABELS = {
    "equity_styles": "Equity-style correlation (large, small, value, growth) is unusually high",
    "gold_equity": "Gold-equity correlation is unusually negative",
}


def _fmt(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def primary_verdicts(results: dict[str, Any]) -> list[str]:
    """One sentence per primary test (events the paper did not study)."""
    alpha = results["preregistered"]["alpha_per_test"]
    out = []
    for t in results["tests"]:
        if t["event_set"] != results["preregistered"]["primary_event_set"]:
            continue
        p = t["p_value"]
        ok = p is not None and p < alpha
        out.append(
            f"{LABELS[t['measure']]} in the 3 months before a VIX doubling: across "
            f"{len(t['event_dates'])} events the paper did not study, the lead-up average was "
            f"{_fmt(t['mean_leadup'])} vs {_fmt(t['typical'])} on a typical day "
            f"(one-sided permutation p = {_fmt(p)}). "
            + (f"Supported at alpha = {alpha}." if ok else f"Not supported at alpha = {alpha}.")
        )
    return out


def render_paper_md(results: dict[str, Any]) -> str:
    """Markdown report: hypotheses, primary verdicts, every test, and per-event values."""
    pre = results["preregistered"]
    lines = [
        "# Testing the Paper's Lead-up Claim",
        "",
        f"Equity styles: {', '.join(results['equity_styles'])}. Gold: {results['gold']}. "
        f"Data through {results['data_snapshot_end']}.",
        "",
        "## Preregistered hypotheses",
        "",
        *[f"- **{k}**: {v.strip()}" for k, v in pre["hypotheses"].items()],
        "",
        f"Events: {pre['event_definition'].strip()}",
        "",
        "## Primary verdicts (events not in the paper)",
        "",
        *[f"- {v}" for v in primary_verdicts(results)],
        "",
        "## All tests",
        "",
        "| measure | events | n | lead-up mean | typical | p-value |",
        "|---|---|---|---|---|---|",
    ]
    for t in results["tests"]:
        lines.append(
            f"| {t['measure']} | {t['event_set']} | {len(t['event_dates'])} | "
            f"{_fmt(t['mean_leadup'])} | {_fmt(t['typical'])} | {_fmt(t['p_value'])} |"
        )
    lines += [
        "",
        "Only the `new` rows are confirmatory. The `paper` rows re-check the paper's own events "
        "with different data; `all` pools both.",
        "",
        "## Per-event lead-up values",
        "",
        "| event date | set | equity styles | gold-equity |",
        "|---|---|---|---|",
    ]
    for row in results["events"]:
        lines.append(
            f"| {row['date']} | {row['set']} | {_fmt(row['equity_styles'])} | "
            f"{_fmt(row['gold_equity'])} |"
        )
    lines.append("")
    return "\n".join(lines)
