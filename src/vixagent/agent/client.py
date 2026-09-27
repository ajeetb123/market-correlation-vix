"""Anthropic client factory and model IDs."""

from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

DEFAULT_MODEL = "claude-sonnet-5"


class MissingAPIKeyError(RuntimeError):
    """Raised when ANTHROPIC_API_KEY is not configured."""


def agent_model() -> str:
    """Model ID for the research agent (env VIX_AGENT_MODEL)."""
    return os.environ.get("VIX_AGENT_MODEL", DEFAULT_MODEL)


def judge_model() -> str:
    """Model ID for the eval judge (env VIX_JUDGE_MODEL)."""
    return os.environ.get("VIX_JUDGE_MODEL", DEFAULT_MODEL)


def make_client() -> anthropic.Anthropic:
    """Return an Anthropic client, reading the key from the environment or .env.

    Why a factory: callers (and tests) inject the client into run_agent, so the
    loop never constructs one itself and tests never need a real key.
    """
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return anthropic.Anthropic(max_retries=3, timeout=120.0)
