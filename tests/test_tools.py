"""Tests for agent tool schemas and dispatch."""

import json
from typing import Any, get_args

import jsonschema
import pandas as pd
import pytest

from vixagent.agent import tools as tools_mod
from vixagent.agent.methodology import TOPICS
from vixagent.agent.service import ResearchService
from vixagent.agent.tools import TOOL_SCHEMAS, TOOLS, TopicArg, execute_tool
from vixagent.config import Settings, load_preregistered, load_settings

SPEC_NAMES = [
    "describe_dataset",
    "get_correlation_summary",
    "get_spike_events",
    "run_event_study",
    "run_predictive_regression",
    "run_out_of_sample_test",
    "run_overfitting_check",
    "get_preregistered_spec",
    "get_methodology",
]

VALID_INPUTS: dict[str, dict[str, Any]] = {
    "describe_dataset": {},
    "get_correlation_summary": {"group": "risk", "window": 21, "period": "test"},
    "get_spike_events": {
        "kind": "corr",
        "period": "full",
        "group": "risk",
        "window": 21,
        "z_threshold": 2.0,
    },
    "run_event_study": {
        "group": "risk",
        "window": 21,
        "z_threshold": 2.0,
        "horizon": 10,
        "period": "test",
        "clean_only": True,
        "direction": "corr_to_vix",
    },
    "run_predictive_regression": {
        "group": "risk",
        "window": 21,
        "horizon": 10,
        "period": "full",
        "include_controls": True,
    },
    "run_out_of_sample_test": {"group": "risk", "window": 21, "horizon": 10, "z_threshold": 2.0},
    "run_overfitting_check": {},
    "get_preregistered_spec": {},
    "get_methodology": {"topic": "lookahead"},
}


@pytest.fixture
def service(panel: pd.DataFrame, small_settings: Settings) -> ResearchService:
    return ResearchService(small_settings, load_preregistered(load_settings()), panel)


def test_registry_matches_spec() -> None:
    assert len(TOOLS) == 9
    assert list(TOOLS) == SPEC_NAMES
    assert [s["name"] for s in TOOL_SCHEMAS] == list(TOOLS)


def test_topic_enum_matches_methodology() -> None:
    assert get_args(TopicArg) == TOPICS


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["name"])
def test_schemas_are_valid(schema: dict[str, Any]) -> None:
    jsonschema.Draft202012Validator.check_schema(schema["input_schema"])
    assert schema["input_schema"]["type"] == "object"
    assert schema["description"]


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_valid_inputs_execute(service: ResearchService, name: str) -> None:
    out, is_error = execute_tool(service, name, VALID_INPUTS[name])
    assert is_error is False, out
    json.dumps(out, allow_nan=False)


def test_out_of_range_window(service: ResearchService) -> None:
    args = {**VALID_INPUTS["get_correlation_summary"], "window": 5}
    out, is_error = execute_tool(service, "get_correlation_summary", args)
    assert is_error is True
    assert any("window" in d for d in out["details"])
    assert any("greater than or equal to 10" in d for d in out["details"])


def test_extra_argument_rejected(service: ResearchService) -> None:
    args = {"group": "risk", "windw": 21, "period": "test"}
    out, is_error = execute_tool(service, "get_correlation_summary", args)
    assert is_error is True
    assert any("windw" in d for d in out["details"])


def test_corr_events_need_threshold(service: ResearchService) -> None:
    args = {"kind": "corr", "period": "full", "group": "risk", "window": 21}
    _, is_error = execute_tool(service, "get_spike_events", args)
    assert is_error is True


def test_unknown_tool(service: ResearchService) -> None:
    out, is_error = execute_tool(service, "predict_the_market", {})
    assert is_error is True
    assert "Unknown tool" in out["error"]


def test_handler_exception_is_caught(
    service: ResearchService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(svc: ResearchService, args: Any) -> dict[str, Any]:
        raise RuntimeError("kaboom")

    spec = TOOLS["describe_dataset"]
    monkeypatch.setitem(
        tools_mod.TOOLS,
        "describe_dataset",
        tools_mod.ToolSpec(spec.name, spec.description, spec.args_model, boom),
    )
    out, is_error = execute_tool(service, "describe_dataset", {})
    assert is_error is True
    assert out["error"] == "RuntimeError: kaboom"
