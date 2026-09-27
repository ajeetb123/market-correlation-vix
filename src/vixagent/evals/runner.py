"""Run eval cases through the agent, grade them, and summarize."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic

from vixagent.agent.loop import ToolCallRecord, run_agent
from vixagent.agent.service import ResearchService
from vixagent.agent.transcript import TranscriptWriter
from vixagent.evals.cases import EvalCase
from vixagent.evals.graders import (
    GraderResult,
    collect_sources,
    grade_date,
    grade_grounding,
    grade_numeric,
    grade_tools,
)
from vixagent.evals.judge import run_judge
from vixagent.evals.references import resolve, resolve_truth
from vixagent.utils.jsonable import to_jsonable

CATEGORIES = ("factual", "methodology", "pushback", "trap", "out_of_scope")


@dataclass
class CaseRun:
    """One run of one eval case: answer, tool calls, grades, and cost."""

    case_id: str
    category: str
    repeat: int
    answer: str
    tool_calls: list[ToolCallRecord]
    graders: list[GraderResult]
    passed: bool | None  # None if any grader errored or the agent call raised
    input_tokens: int
    output_tokens: int
    duration_s: float
    error: str | None


def _grade(
    case: EvalCase,
    answer: str,
    calls: list[ToolCallRecord],
    service: ResearchService,
    judge_client: Any,
    judge_model: str,
) -> list[GraderResult]:
    results: list[GraderResult] = []
    for name in case.graders:
        if name == "grounding":
            sources = collect_sources(calls, case.question, service.settings, service.prereg)
            results.append(grade_grounding(answer, sources))
        elif name == "numeric":
            results.append(grade_numeric(answer, [resolve(r, service) for r in case.expected]))
        elif name == "date":
            results.append(grade_date(answer, [str(resolve(r, service)) for r in case.expected]))
        elif name == "tools":
            results.append(grade_tools([c.name for c in calls], case.required_tools))
        elif name == "judge":
            assert case.rubric is not None
            rubric = case.rubric
            if case.truth is not None:
                rubric = rubric.replace("{truth}", resolve_truth(case.truth, service))
            try:
                v = run_judge(judge_client, judge_model, case.question, rubric, calls, answer)
                results.append(GraderResult("judge", v.passed, float(bool(v.passed)), v.reason))
            except anthropic.APIError as exc:
                results.append(GraderResult("judge", None, 0.0, f"judge API error: {exc}"))
    return results


def run_case(
    case: EvalCase,
    repeat: int,
    service: ResearchService,
    agent_client: Any,
    judge_client: Any,
    agent_model: str,
    judge_model: str,
    runs_dir: Path,
) -> CaseRun:
    """1. run_agent(case.question, fresh history, with a TranscriptWriter).
    2. Run each listed grader (grounding, numeric, date, tools, judge), with
       '{truth}' in the rubric replaced by resolve_truth(...).
    3. passed = all graders passed; None if any grader passed is None.
    Catch anthropic.APIError from the agent: error set, passed None."""
    started = time.perf_counter()
    transcript = TranscriptWriter(runs_dir, f"eval-{case.id}-r{repeat}")
    try:
        result = run_agent(
            case.question,
            client=agent_client,
            service=service,
            model=agent_model,
            transcript=transcript,
        )
    except anthropic.APIError as exc:
        return CaseRun(
            case_id=case.id,
            category=case.category,
            repeat=repeat,
            answer="",
            tool_calls=[],
            graders=[],
            passed=None,
            input_tokens=0,
            output_tokens=0,
            duration_s=time.perf_counter() - started,
            error=f"agent API error: {exc}",
        )
    graders = _grade(case, result.final_text, result.tool_calls, service, judge_client, judge_model)
    if any(g.passed is None for g in graders):
        passed: bool | None = None
    else:
        passed = all(g.passed for g in graders)
    return CaseRun(
        case_id=case.id,
        category=case.category,
        repeat=repeat,
        answer=result.final_text,
        tool_calls=result.tool_calls,
        graders=graders,
        passed=passed,
        input_tokens=result.usage["input_tokens"],
        output_tokens=result.usage["output_tokens"],
        duration_s=time.perf_counter() - started,
        error=None,
    )


def aggregate(runs: list[CaseRun]) -> dict[str, str]:
    """case_id -> 'pass' | 'fail' | 'flaky' | 'error'.
    Ignore error runs; if all runs errored -> 'error'. Of the rest: all passed
    -> 'pass', none passed -> 'fail', otherwise 'flaky'."""
    by_case: dict[str, list[bool | None]] = defaultdict(list)
    for r in runs:
        by_case[r.case_id].append(r.passed)
    outcomes: dict[str, str] = {}
    for case_id, results in by_case.items():
        valid = [p for p in results if p is not None]
        if not valid:
            outcomes[case_id] = "error"
        elif all(valid):
            outcomes[case_id] = "pass"
        elif not any(valid):
            outcomes[case_id] = "fail"
        else:
            outcomes[case_id] = "flaky"
    return outcomes


def _reason(runs: list[CaseRun]) -> str:
    for r in runs:
        if r.error:
            return r.error
        for g in r.graders:
            if g.passed is not True:
                return f"{g.name}: {g.detail}"
    return ""


def render_summary(runs: list[CaseRun], outcomes: dict[str, str], meta: dict[str, Any]) -> str:
    """Markdown: overall pass rate, per-category pass rates, one line per
    failing/flaky/error case with the first failing grader's detail, token
    totals, and run metadata. Uses no tables wider than 3 columns."""
    total = len(outcomes)
    counts = {k: sum(v == k for v in outcomes.values()) for k in ("pass", "fail", "flaky", "error")}
    pct = 100 * counts["pass"] / total if total else 0.0
    category = {r.case_id: r.category for r in runs}
    lines = [
        "# Eval Summary",
        "",
        f"Overall: {counts['pass']}/{total} pass ({pct:.1f}%), fail {counts['fail']}, "
        f"flaky {counts['flaky']}, error {counts['error']}",
        "",
        "| category | pass | total |",
        "|---|---|---|",
    ]
    for cat in CATEGORIES:
        ids = [c for c in outcomes if category[c] == cat]
        if ids:
            n_pass = sum(outcomes[c] == "pass" for c in ids)
            lines.append(f"| {cat} | {n_pass} | {len(ids)} |")
    problems = sorted(c for c, o in outcomes.items() if o != "pass")
    lines += ["", "## Failures, flaky, and errors", ""]
    if problems:
        for c in problems:
            case_runs = [r for r in runs if r.case_id == c]
            lines.append(f"- **{c}** ({outcomes[c]}): {_reason(case_runs)}")
    else:
        lines.append("None.")
    lines += [
        "",
        "## Tokens (agent)",
        "",
        f"- input: {sum(r.input_tokens for r in runs)}",
        f"- output: {sum(r.output_tokens for r in runs)}",
        "",
        "## Meta",
        "",
        f"- agent model: {meta.get('agent_model')}",
        f"- judge model: {meta.get('judge_model')}",
        f"- repeats: {meta.get('repeats')}",
        f"- data snapshot end: {meta.get('data_snapshot_end')}",
        f"- run timestamp: {meta.get('timestamp')}",
        "",
    ]
    return "\n".join(lines)


def run_suite(
    cases: list[EvalCase],
    repeats: int,
    root: Path,
    service: ResearchService,
    agent_client: Any,
    judge_client: Any,
    agent_model: str,
    judge_model: str,
) -> Path:
    """Run every case `repeats` times (sequentially), write
    evals/results/<YYYYmmdd-HHMMSS>/results.json (every CaseRun via to_jsonable)
    and summary.md. Return the directory."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = root / "evals" / "results" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = root / "runs"
    runs = [
        run_case(case, rep, service, agent_client, judge_client, agent_model, judge_model, runs_dir)
        for case in cases
        for rep in range(1, repeats + 1)
    ]
    outcomes = aggregate(runs)
    meta = {
        "agent_model": agent_model,
        "judge_model": judge_model,
        "repeats": repeats,
        "data_snapshot_end": service.settings.data.snapshot_end.isoformat(),
        "timestamp": stamp,
    }
    payload = {"meta": meta, "outcomes": outcomes, "runs": runs}
    (out_dir / "results.json").write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")
    (out_dir / "summary.md").write_text(render_summary(runs, outcomes, meta))
    return out_dir
