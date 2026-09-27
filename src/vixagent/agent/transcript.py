"""JSONL transcripts of agent runs, with API-key redaction."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def slugify(text: str, max_len: int = 40) -> str:
    """Lowercase, non-alphanumerics to '-', collapsed, trimmed to max_len."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "run"


def redact(line: str) -> str:
    """Replace the current ANTHROPIC_API_KEY value (if set and at least 8 chars)
    with '[REDACTED]'."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return line.replace(key, "[REDACTED]") if len(key) >= 8 else line


def _serialize(data: Any) -> Any:
    if hasattr(data, "model_dump"):
        return data.model_dump()
    if isinstance(data, list):
        return [_serialize(x) for x in data]
    if isinstance(data, dict):
        return {k: _serialize(v) for k, v in data.items()}
    return data


class TranscriptWriter:
    """Append-only JSONL log of one agent run at runs/<YYYYmmdd-HHMMSS>-<slug>.jsonl.

    Events: {"event": "user"|"assistant"|"tool_call"|"tool_result"|"final"|"usage",
             "ts": ISO-8601 UTC, "data": ...}
    SDK content blocks are serialized with .model_dump(); dicts pass through.
    Every line is passed through redact() before writing."""

    def __init__(self, runs_dir: Path, question: str) -> None:
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        path = runs_dir / f"{stamp}-{slugify(question)}.jsonl"
        n = 1
        while path.exists():
            n += 1
            path = runs_dir / f"{stamp}-{slugify(question)}-{n}.jsonl"
        self._path = path

    @property
    def path(self) -> Path:
        """Location of the JSONL file."""
        return self._path

    def write(self, event: str, data: Any) -> None:
        """Append one event as a single redacted JSON line."""
        record = {
            "event": event,
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "data": _serialize(data),
        }
        with self._path.open("a") as fh:
            fh.write(redact(json.dumps(record, default=str)) + "\n")
