"""The agent's tool-use loop over the Anthropic Messages API."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from vixagent.agent.prompts import build_system_prompt
from vixagent.agent.service import ResearchService
from vixagent.agent.tools import TOOL_SCHEMAS, execute_tool
from vixagent.agent.transcript import TranscriptWriter

MAX_ITERATIONS = 12
LIMIT_MESSAGE = (
    "I stopped after reaching the limit of 12 tool-use steps without a "
    "final answer. Please ask a narrower question."
)


@dataclass
class ToolCallRecord:
    name: str
    input: dict[str, Any]
    output: dict[str, Any]
    is_error: bool
    duration_ms: float


@dataclass
class AgentResult:
    final_text: str
    tool_calls: list[ToolCallRecord]
    iterations: int
    usage: dict[str, int]
    messages: list[dict[str, Any]]
    stop_reason: str  # "end_turn" | "max_tokens" | "iteration_limit" | other API value


def run_agent(
    question: str,
    *,
    client: Any,
    service: ResearchService,
    model: str,
    history: list[dict[str, Any]] | None = None,
    max_iterations: int = MAX_ITERATIONS,
    max_tokens: int = 4096,
    transcript: TranscriptWriter | None = None,
) -> AgentResult:
    """Answer one question, letting the model call tools until it gives a final answer.

    Why the client is injected: tests pass a scripted fake, so the loop's
    message bookkeeping is tested without network calls or an API key.
    API rules this enforces: the assistant message containing tool_use blocks
    is appended before the tool results, and every tool_use id gets a
    tool_result in the very next user message, in the same order.
    """
    system = build_system_prompt(service.settings)
    messages: list[dict[str, Any]] = list(history or [])
    messages.append({"role": "user", "content": question})
    if transcript:
        transcript.write("user", question)
    calls: list[ToolCallRecord] = []
    usage = {"input_tokens": 0, "output_tokens": 0}

    for i in range(1, max_iterations + 1):
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        usage["input_tokens"] += resp.usage.input_tokens
        usage["output_tokens"] += resp.usage.output_tokens
        messages.append({"role": "assistant", "content": resp.content})
        if transcript:
            transcript.write("assistant", [b.model_dump() for b in resp.content])

        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            if resp.stop_reason == "max_tokens":
                text += "\n\n[Answer truncated: max_tokens reached.]"
            if transcript:
                transcript.write("final", text)
                transcript.write("usage", usage)
            return AgentResult(text, calls, i, usage, messages, resp.stop_reason)

        results: list[dict[str, Any]] = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            t0 = time.perf_counter()
            out, is_err = execute_tool(service, block.name, block.input)
            dur = (time.perf_counter() - t0) * 1000
            calls.append(ToolCallRecord(block.name, dict(block.input), out, is_err, dur))
            if transcript:
                transcript.write(
                    "tool_call", {"id": block.id, "name": block.name, "input": block.input}
                )
                transcript.write("tool_result", {"id": block.id, "is_error": is_err, "output": out})
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(out, allow_nan=False),
                    "is_error": is_err,
                }
            )
        messages.append({"role": "user", "content": results})

    # Close the turn with an assistant message so the history stays valid for the next question.
    messages.append({"role": "assistant", "content": [{"type": "text", "text": LIMIT_MESSAGE}]})
    if transcript:
        transcript.write("final", LIMIT_MESSAGE)
        transcript.write("usage", usage)
    return AgentResult(LIMIT_MESSAGE, calls, max_iterations, usage, messages, "iteration_limit")
