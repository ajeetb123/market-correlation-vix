"""LLM-as-judge grading with robust JSON parsing, plus calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vixagent.agent.loop import ToolCallRecord

JUDGE_PROMPT = """You are grading a research assistant's answer against a rubric.
Return ONLY a JSON object: {{"pass": true or false, "reason": "<one sentence>"}}.
Pass only if every requirement in the rubric is met. Be strict.

QUESTION:
{question}

RUBRIC:
{rubric}

TOOL CALLS:
{tool_calls}

ANSWER:
{answer}
"""

OUTPUT_CHARS = 2000


@dataclass
class JudgeVerdict:
    passed: bool | None  # None = could not parse after retries
    reason: str
    raw: str


def parse_judge_json(text: str) -> dict[str, Any] | None:
    """Find the first '{' from which json.JSONDecoder().raw_decode succeeds and
    yields a dict with a boolean 'pass'. Handles ```json fences and prose
    around the object. Returns None if none found."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("pass"), bool):
            return obj
        start = text.find("{", start + 1)
    return None


def render_tool_calls(tool_calls: list[ToolCallRecord]) -> str:
    """One JSON line per call with the output truncated to 2,000 characters."""
    lines = []
    for c in tool_calls:
        out = json.dumps(c.output)
        if len(out) > OUTPUT_CHARS:
            out = out[:OUTPUT_CHARS] + "...(truncated)"
        lines.append(json.dumps({"name": c.name, "input": c.input, "output": out}))
    return "\n".join(lines) or "(none)"


def run_judge(
    client: Any,
    model: str,
    question: str,
    rubric: str,
    tool_calls: list[ToolCallRecord],
    answer: str,
    max_retries: int = 2,
) -> JudgeVerdict:
    """tool_calls rendered as JSON lines {name, input, output} with output
    truncated to 2,000 characters. max_tokens=400. Retries only on parse
    failure (API errors propagate to the runner).

    Why an unparseable reply is None, not False: a judge malfunction is not
    evidence that the agent failed, so it is reported separately."""
    prompt = JUDGE_PROMPT.format(
        question=question, rubric=rubric, tool_calls=render_tool_calls(tool_calls), answer=answer
    )
    raw = ""
    for _ in range(1 + max_retries):
        resp = client.messages.create(
            model=model, max_tokens=400, messages=[{"role": "user", "content": prompt}]
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        parsed = parse_judge_json(raw)
        if parsed is not None:
            return JudgeVerdict(parsed["pass"], str(parsed.get("reason", "")), raw)
    return JudgeVerdict(None, "could not parse judge output", raw)


def calibrate(client: Any, model: str, path: Path) -> tuple[int, int, list[str]]:
    """Run every item in judge_calibration.yaml; return (agreements, total,
    disagreement descriptions).

    Why: a judge is only trustworthy if it agrees with labels we already know."""
    items = yaml.safe_load(path.read_text())
    agree = 0
    disagreements: list[str] = []
    for item in items:
        calls = [
            ToolCallRecord(c["name"], c.get("input", {}), c.get("output", {}), False, 0.0)
            for c in item.get("tool_calls", [])
        ]
        verdict = run_judge(client, model, item["question"], item["rubric"], calls, item["answer"])
        if verdict.passed is item["label"]:
            agree += 1
        else:
            disagreements.append(
                f"{item['id']}: label {item['label']}, judge {verdict.passed} ({verdict.reason})"
            )
    return agree, len(items), disagreements
