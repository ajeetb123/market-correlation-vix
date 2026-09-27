"""Agent tools: pydantic argument models, JSON Schemas, and dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vixagent.agent.service import ResearchService

GroupArg = Literal["risk", "all"]
PeriodArg = Literal["train", "test", "full"]
TopicArg = Literal[
    "data",
    "correlation",
    "zscore",
    "vix_spike",
    "event_study",
    "permutation_test",
    "regression",
    "out_of_sample",
    "lookahead",
    "limitations",
]

_GROUP = "Asset group: 'risk' (equities + high yield) or 'all' (adds Treasuries, IG credit, gold)."
_WINDOW = "Rolling correlation window in trading days (10 to 126)."
_PERIOD = "'train', 'test', or 'full'. Call describe_dataset for exact dates."
_Z = "Correlation z-score threshold for a spike day (0.5 to 4.0)."
_HORIZON = "Forward horizon in trading days (1 to 60)."
_FORBID = ConfigDict(extra="forbid")


class NoArgs(BaseModel):
    """This tool takes no arguments."""

    model_config = _FORBID


class CorrSummaryArgs(BaseModel):
    """Arguments for get_correlation_summary."""

    model_config = _FORBID
    group: GroupArg = Field(description=_GROUP)
    window: int = Field(ge=10, le=126, description=_WINDOW)
    period: PeriodArg = Field(description=_PERIOD)


class SpikeEventsArgs(BaseModel):
    """Arguments for get_spike_events."""

    model_config = _FORBID
    kind: Literal["vix", "corr"] = Field(
        description="'vix' for VIX spike events, 'corr' for correlation spike events."
    )
    period: PeriodArg = Field(description=_PERIOD)
    group: GroupArg | None = Field(default=None, description=_GROUP + " Required for kind='corr'.")
    window: int | None = Field(
        default=None, ge=10, le=126, description=_WINDOW + " Required for kind='corr'."
    )
    z_threshold: float | None = Field(
        default=None, ge=0.5, le=4.0, description=_Z + " Required for kind='corr'."
    )

    @model_validator(mode="after")
    def _corr_needs_params(self) -> SpikeEventsArgs:
        if self.kind == "corr" and None in (self.group, self.window, self.z_threshold):
            raise ValueError("kind='corr' requires group, window, and z_threshold")
        return self


class EventStudyArgs(BaseModel):
    """Arguments for run_event_study."""

    model_config = _FORBID
    group: GroupArg = Field(description=_GROUP)
    window: int = Field(ge=10, le=126, description=_WINDOW)
    z_threshold: float = Field(ge=0.5, le=4.0, description=_Z)
    horizon: int = Field(ge=1, le=60, description=_HORIZON)
    period: PeriodArg = Field(description=_PERIOD)
    clean_only: bool = Field(
        description="Drop events where the target was already spiking in the prior 5 days."
    )
    direction: Literal["corr_to_vix", "vix_to_corr"] = Field(
        description="'corr_to_vix' is the main test; 'vix_to_corr' is the reverse check."
    )


class RegressionArgs(BaseModel):
    """Arguments for run_predictive_regression."""

    model_config = _FORBID
    group: GroupArg = Field(description=_GROUP)
    window: int = Field(ge=10, le=126, description=_WINDOW)
    horizon: int = Field(ge=1, le=60, description=_HORIZON)
    period: PeriodArg = Field(description=_PERIOD)
    include_controls: bool = Field(
        description="Add 5-day VIX momentum and log VIX level as control variables."
    )


class OOSArgs(BaseModel):
    """Arguments for run_out_of_sample_test."""

    model_config = _FORBID
    group: GroupArg = Field(description=_GROUP)
    window: int = Field(ge=10, le=126, description=_WINDOW)
    horizon: int = Field(ge=1, le=60, description=_HORIZON)
    z_threshold: float = Field(ge=0.5, le=4.0, description=_Z)


class MethodologyArgs(BaseModel):
    """Arguments for get_methodology."""

    model_config = _FORBID
    topic: TopicArg = Field(description="Methodology topic to explain.")


Handler = Callable[[ResearchService, Any], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    """A tool's name, description, argument model, and handler."""

    name: str
    description: str
    args_model: type[BaseModel]
    handler: Handler

    def schema(self) -> dict[str, Any]:
        """Anthropic tool definition generated from the pydantic model.

        Why generate it: the schema the model sees and the validation the code
        performs then come from one source and cannot disagree.
        """
        s = self.args_model.model_json_schema()
        s.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": s}


def _spec(name: str, description: str, args_model: type[BaseModel], handler: Handler) -> ToolSpec:
    return ToolSpec(name, description, args_model, handler)


_TOOL_LIST = [
    _spec(
        "describe_dataset",
        "Describe the dataset: tickers in each asset group, VIX ticker, date range, trading "
        "days, rows dropped during alignment, and the train/test/full periods. Use this first "
        "when asked what data exists or whether something is in scope.",
        NoArgs,
        lambda svc, a: svc.describe_dataset(),
    ),
    _spec(
        "get_correlation_summary",
        "Summary statistics of the rolling average pairwise correlation for an asset group and "
        "window over a period: mean, std, min and max with dates, and the latest value with its "
        "trailing z-score.",
        CorrSummaryArgs,
        lambda svc, a: svc.correlation_summary(a.group, a.window, a.period),
    ),
    _spec(
        "get_spike_events",
        "List declustered spike events. kind='vix': days where VIX / median of the prior 20 "
        "days crosses the spike threshold. kind='corr': days where the trailing z-score of "
        "average correlation crosses z_threshold (requires group, window, z_threshold). Returns "
        "the count, up to 50 dates, and the exact definition.",
        SpikeEventsArgs,
        lambda svc, a: svc.spike_events(a.kind, a.period, a.group, a.window, a.z_threshold),
    ),
    _spec(
        "run_event_study",
        "Test whether source events are followed by target spikes within `horizon` trading "
        "days more often than the base rate. direction='corr_to_vix' is the main test; "
        "'vix_to_corr' is the reverse check. Returns n_events, hit_rate, base_rate, lift, and a "
        "circular-shift permutation p-value. clean_only=true drops events where the target was "
        "already spiking in the prior 5 days.",
        EventStudyArgs,
        lambda svc, a: svc.event_study(
            a.group, a.window, a.z_threshold, a.horizon, a.period, a.clean_only, a.direction
        ),
    ),
    _spec(
        "run_predictive_regression",
        "OLS of the forward h-day log change in VIX on the correlation z-score (optionally with "
        "VIX momentum and level controls), with Newey-West HAC standard errors. Returns "
        "coefficients, HAC t-stats and p-values, R squared, and n.",
        RegressionArgs,
        lambda svc, a: svc.regression(a.group, a.window, a.horizon, a.period, a.include_controls),
    ),
    _spec(
        "run_out_of_sample_test",
        "Fit the z-score-only regression on the training period and evaluate on the test "
        "period (out-of-sample R squared vs the historical-mean forecast), plus the event study "
        "on train and test separately.",
        OOSArgs,
        lambda svc, a: svc.oos(a.group, a.window, a.horizon, a.z_threshold),
    ),
    _spec(
        "run_overfitting_check",
        "Run the full exploratory grid (groups x windows x z thresholds x horizons) on training "
        "data, pick the best in-sample lift, and report how it performs on the test period next "
        "to the preregistered spec. Use when the user wants to search parameters.",
        NoArgs,
        lambda svc, a: svc.overfitting_check(),
    ),
    _spec(
        "get_preregistered_spec",
        "Return the preregistered primary specification, fixed before any results were "
        "computed. The headline result is this spec on the test period.",
        NoArgs,
        lambda svc, a: svc.preregistered_spec(),
    ),
    _spec(
        "get_methodology",
        "Return the canonical explanation of a methodology topic. Use for 'how is X defined' or "
        "'why was Y done' questions.",
        MethodologyArgs,
        lambda svc, a: svc.methodology(a.topic),
    ),
]

TOOLS: dict[str, ToolSpec] = {t.name: t for t in _TOOL_LIST}
TOOL_SCHEMAS: list[dict[str, Any]] = [t.schema() for t in TOOLS.values()]


def execute_tool(
    service: ResearchService, name: str, raw_input: Any
) -> tuple[dict[str, Any], bool]:
    """Validate and run one tool. Returns (output, is_error). Never raises.

    - Unknown name: ({"error": f"Unknown tool '{name}'. Available: [...]"}, True)
    - ValidationError: ({"error": "Invalid arguments", "details": [f"{loc}: {msg}", ...]}, True)
      Pydantic messages already state the bound (e.g. 'Input should be greater
      than or equal to 10').
    - Any other exception: ({"error": f"{type(e).__name__}: {e}"}, True)
    - Success: (handler output, False)

    Why never raise: an exception would abort the whole agent loop. Returning
    the error as a tool_result lets the model read it and correct its call.
    """
    spec = TOOLS.get(name)
    if spec is None:
        return {"error": f"Unknown tool '{name}'. Available: {list(TOOLS)}"}, True
    try:
        args = spec.args_model.model_validate(raw_input or {})
    except ValidationError as exc:
        details = [
            f"{'.'.join(str(p) for p in err['loc']) or 'input'}: {err['msg']}"
            for err in exc.errors()
        ]
        return {"error": "Invalid arguments", "details": details}, True
    try:
        return cast(dict[str, Any], spec.handler(service, args)), False
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}, True
