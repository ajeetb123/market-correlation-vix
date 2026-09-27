"""Scripted stand-ins for the Anthropic client. No network, no API key."""

import copy
from types import SimpleNamespace
from typing import Any

from anthropic.types import Message, TextBlock, ToolUseBlock, Usage


def _usage() -> Usage:
    return Usage.model_construct(input_tokens=10, output_tokens=5)


def text_msg(text: str, stop_reason: str = "end_turn") -> Message:
    return Message.model_construct(
        id="msg_fake",
        type="message",
        role="assistant",
        model="fake",
        content=[TextBlock.model_construct(type="text", text=text)],
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=_usage(),
    )


def tool_msg(*calls: tuple[str, str, dict[str, Any]], preface: str | None = None) -> Message:
    """calls = (id, name, input) tuples; stop_reason 'tool_use'."""
    content: list[Any] = []
    if preface:
        content.append(TextBlock.model_construct(type="text", text=preface))
    for call_id, name, tool_input in calls:
        content.append(
            ToolUseBlock.model_construct(type="tool_use", id=call_id, name=name, input=tool_input)
        )
    return Message.model_construct(
        id="msg_fake",
        type="message",
        role="assistant",
        model="fake",
        content=content,
        stop_reason="tool_use",
        stop_sequence=None,
        usage=_usage(),
    )


class FakeClient:
    """Returns scripted responses in order. Records a deep copy of each
    request's kwargs in self.requests."""

    def __init__(self, responses: list[Message]) -> None:
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)
        self._responses = list(responses)

    def _create(self, **kwargs: Any) -> Message:
        self.requests.append(copy.deepcopy(kwargs))
        return self._responses.pop(0)
