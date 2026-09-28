"""Render follow-up study results (from the JSON-safe results dict)."""

from __future__ import annotations

from typing import Any


def verdict_sentence(results: dict[str, Any]) -> str:
    """One-sentence verdict for the preregistered follow-up test on the holdout."""
    p = results["preregistered"]["primary"]
    r = results["holdout_primary"]
    start, end = r["period"]
    spec = "with VIX momentum and level controls" if p["include_controls"] else "without controls"
    supported = r["p_one_sided"] < p["alpha"] and r["coef_z"] < 0
    verdict = (
        f"Supported at alpha = {p['alpha']}."
        if supported
        else f"Not supported at alpha = {p['alpha']}."
    )
    return (
        f"Follow-up test (sector ETFs, {p['window']}-day window, {p['horizon']}-day horizon, "
        f"{spec}): on the untouched {start} to {end} holdout, the coefficient on the "
        f"correlation z-score was {r['coef_z']:+.4f} (HAC t = {r['t_hac_z']:+.2f}, one-sided "
        f"p = {r['p_one_sided']:.4f}, n = {r['regression']['n']}). {verdict}"
    )


def _row(label: str, r: dict[str, Any]) -> str:
    return (
        f"| {label} | {r['period'][0]} to {r['period'][1]} | {r['regression']['n']} | "
        f"{r['coef_z']:+.4f} | {r['t_hac_z']:+.2f} | {r['p_one_sided']:.4f} |"
    )


def render_followup_md(results: dict[str, Any]) -> str:
    """Markdown report: hypothesis, verdict, and the three regressions."""
    pre = results["preregistered"]
    lines = [
        "# Follow-up Study Results",
        "",
        f"Universe: {', '.join(results['universe'])}. Data through {results['data_snapshot_end']}.",
        "",
        "## Hypothesis (preregistered)",
        "",
        pre["hypothesis"].strip(),
        "",
        "## Verdict",
        "",
        verdict_sentence(results),
        "",
        "## Regressions (coefficient on correlation z-score)",
        "",
        "| | period | n | coef z | t (HAC) | one-sided p |",
        "|---|---|---|---|---|---|",
        _row("holdout, preregistered spec", results["holdout_primary"]),
        _row("holdout, other spec (secondary)", results["holdout_secondary"]),
        _row("exploration, preregistered spec", results["exploration_same_spec"]),
        "",
        "Only the first row is the confirmatory test. The exploration period was already "
        "seen when the hypothesis was formed, so it cannot confirm anything.",
        "",
    ]
    return "\n".join(lines)
