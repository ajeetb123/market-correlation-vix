# Phase 3: Agent

Goal: wrap the pipeline in a cached service, expose it as 9 validated tools, implement the tool-use loop with transcripts, and ship `vixagent ask` and `vixagent chat`.

Read first: `CLAUDE.md`, `docs/SPEC.md` section 10 and 11, `docs/00-FOUNDATION.md` Part D.

Module map:
```
src/vixagent/data/cache.py        + load_cached_prices
src/vixagent/agent/service.py     ResearchService
src/vixagent/agent/methodology.py TOPICS, methodology_text
src/vixagent/agent/tools.py       arg models, ToolSpec, TOOLS, TOOL_SCHEMAS, execute_tool
src/vixagent/agent/prompts.py     build_system_prompt
src/vixagent/agent/transcript.py  TranscriptWriter
src/vixagent/agent/loop.py        ToolCallRecord, AgentResult, run_agent
src/vixagent/agent/client.py      make_client, agent_model, judge_model
tests/fakes.py                    FakeClient, text_msg, tool_msg
```

No real API calls anywhere in the test suite.

---

## Step 3.1: `load_cached_prices` (`data/cache.py`)

```python
def load_cached_prices(settings: Settings, root: Path | None = None) -> pd.DataFrame:
    """Load the cached panel without ever touching the network.
    Raises DataError("No cached data found. Run `vixagent pull` first.") if the
    parquet or metadata file is missing, or if metadata does not match settings."""
```
Why: the agent must never trigger a surprise download mid-conversation.

**Tests**: missing cache raises with that exact message; valid cache loads.
**Commit**: `feat: add network-free cache loader`

---

## Step 3.2: `agent/service.py`

```python
class ResearchService:
    """Single entry point the tools call. Holds one FrameStore and memoizes
    every result by its arguments, so repeated tool calls are instant and
    identical. All public methods return JSON-safe dicts (via to_jsonable)."""

    def __init__(self, settings: Settings, prereg: Preregistered, prices: pd.DataFrame) -> None:
        self.settings = settings
        self.prereg = prereg
        self.store = FrameStore(prices, settings)
        self._memo: dict[tuple[Any, ...], dict[str, Any]] = {}

    @classmethod
    def from_cache(cls, root: Path | None = None) -> ResearchService: ...

    def _cached(self, key: tuple[Any, ...], fn: Callable[[], Any]) -> dict[str, Any]:
        if key not in self._memo:
            self._memo[key] = to_jsonable(fn())
        return self._memo[key]
```

Public methods (all return `dict[str, Any]`):

| Method | Returns |
|---|---|
| `describe_dataset()` | `{tickers: {risk, all}, vix_ticker, start, end, n_trading_days: {risk, all}, rows_dropped_in_alignment: {risk, all}, periods: {train: [s, e], test: [s, e], full: [s, e]}, data_source: "Yahoo Finance via yfinance, daily adjusted closes; VIX close from ^VIX"}` |
| `correlation_summary(group, window, period)` | over non-NaN `avg_corr` in the period: `{group, window, period, mean, std, min, min_date, max, max_date, latest_value, latest_date, latest_z}` |
| `spike_events(kind, period, group=None, window=None, z_threshold=None)` | `{kind, period, n_events, dates (first 50), truncated, definition}` |
| `event_study(group, window, z_threshold, horizon, period, clean_only, direction)` | EventStudyResult as dict with `event_dates` and `hit_flags` capped at 50 plus `truncated` |
| `regression(group, window, horizon, period, include_controls)` | RegressionResult dict |
| `oos(group, window, horizon, z_threshold)` | OOSResult dict (nested event studies capped the same way) |
| `overfitting_check()` | `{n_combinations, best_in_sample: {group, window, z_threshold, horizon, train_lift, train_n_events, test_lift, test_n_events} or null, preregistered: {same fields}}` |
| `preregistered_spec()` | `{primary: {...}, note, frozen: "Committed at git tag prereg-freeze before any results were computed."}` |
| `methodology(topic)` | `{topic, text}` |

`definition` strings:
- VIX: `"VIX event: first day with VIX / median(prior {n} days) >= {thr}, with no VIX spike day in the prior {cooldown} trading days."`
- Corr: `"Correlation event: first day with trailing z-score of {window}-day average pairwise correlation ({group} group, {lookback}-day baseline) >= {z}, with no correlation spike day in the prior {cooldown} trading days."`

Values come from settings (f-strings), never typed by hand.

**Tests** (`tests/test_service.py`, synthetic panel + small settings):
1. Every method returns a dict for which `json.dumps(d, allow_nan=False)` succeeds.
2. Calling a method twice returns the identical object (memoized).
3. Date lists are capped at 50 and `truncated` is correct (force it with a z threshold of 0.5 on a long panel or by monkeypatching the cap to 3).
4. `spike_events("corr", ...)` without group/window/z raises `ValueError`.
5. `from_cache` with no cache raises the Step 3.1 message.

**Commit**: `feat: add memoized research service`

---

## Step 3.3: `agent/methodology.py`

```python
TOPICS: tuple[str, ...] = ("data", "correlation", "zscore", "vix_spike", "event_study",
                           "permutation_test", "regression", "out_of_sample", "lookahead",
                           "limitations")


def methodology_text(topic: str, settings: Settings, prereg: Preregistered) -> str:
    """Canonical explanation for each topic, consistent with docs/SPEC.md.
    Every number is formatted from settings/prereg, never hardcoded, so the
    text cannot drift from the code. Each text is at most 200 words.
    Raises KeyError for unknown topics."""
```

Required content per topic (write each as plain prose):
- `data`: source, tickers per group, date range, inner alignment with no forward fill, train/test periods.
- `correlation`: log returns, rolling Pearson over W days, average over all pairs.
- `zscore`: trailing baseline of `z_lookback` days excluding today, ddof 1, why not full-sample.
- `vix_spike`: ratio to prior-20-day median, threshold, cooldown declustering, why median.
- `event_study`: events, hit window `t+1..t+h`, base rate, lift, clean filter and why, eligibility and the embargo.
- `permutation_test`: circular shifts, why they preserve clustering, p-value formula, n permutations.
- `regression`: target, base and controls specs, HAC with maxlags h and why.
- `out_of_sample`: train/test split, base spec, R2_oos vs historical mean, how to interpret sign.
- `lookahead`: what it is, the three safeguards (shift(1) baselines, targets.py isolation, truncation test).
- `limitations`: the six limitations from SPEC section 9, plus: preregistration covers one spec; everything else is exploratory.

**Tests**: every topic returns non-empty text of at most 200 words; `vix_spike` text contains the configured threshold formatted as in settings; unknown topic raises `KeyError`.

**Commit**: `feat: add canonical methodology text`

---

## Step 3.4: `agent/tools.py`

### Argument models (pydantic, with `Field(description=...)` on every field)

```python
GroupArg = Literal["risk", "all"]
PeriodArg = Literal["train", "test", "full"]

class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

class CorrSummaryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group: GroupArg = Field(description="Asset group: 'risk' (equities + high yield) or 'all' (adds Treasuries, IG credit, gold).")
    window: int = Field(ge=10, le=126, description="Rolling correlation window in trading days (10 to 126).")
    period: PeriodArg = Field(description="'train', 'test', or 'full'. Call describe_dataset for exact dates.")

class SpikeEventsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["vix", "corr"]
    period: PeriodArg
    group: GroupArg | None = None
    window: int | None = Field(default=None, ge=10, le=126)
    z_threshold: float | None = Field(default=None, ge=0.5, le=4.0)
    # model_validator(mode="after"): kind == "corr" requires group, window, z_threshold

class EventStudyArgs(BaseModel):   # group, window, z_threshold, horizon (1..60), period, clean_only: bool, direction
class RegressionArgs(BaseModel):   # group, window, horizon, period, include_controls: bool
class OOSArgs(BaseModel):          # group, window, horizon, z_threshold
class MethodologyArgs(BaseModel):  # topic: Literal[...all TOPICS...]
```
Every model uses `extra="forbid"` so misspelled arguments are errors, not silently ignored.

### Tool registry

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[ResearchService, BaseModel], dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        s = self.args_model.model_json_schema()
        s.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": s}

TOOLS: dict[str, ToolSpec] = {...}      # the 9 tools, in SPEC order
TOOL_SCHEMAS: list[dict[str, Any]] = [t.schema() for t in TOOLS.values()]
```

Generating schemas from the pydantic models means the schema the model sees and the validation the code performs can never disagree.

### Tool descriptions (use verbatim; static text with no placeholders)

1. `describe_dataset`: "Describe the dataset: tickers in each asset group, VIX ticker, date range, trading days, rows dropped during alignment, and the train/test/full periods. Use this first when asked what data exists or whether something is in scope."
2. `get_correlation_summary`: "Summary statistics of the rolling average pairwise correlation for an asset group and window over a period: mean, std, min and max with dates, and the latest value with its trailing z-score."
3. `get_spike_events`: "List declustered spike events. kind='vix': days where VIX / median of the prior 20 days crosses the spike threshold. kind='corr': days where the trailing z-score of average correlation crosses z_threshold (requires group, window, z_threshold). Returns the count, up to 50 dates, and the exact definition."
4. `run_event_study`: "Test whether source events are followed by target spikes within `horizon` trading days more often than the base rate. direction='corr_to_vix' is the main test; 'vix_to_corr' is the reverse check. Returns n_events, hit_rate, base_rate, lift, and a circular-shift permutation p-value. clean_only=true drops events where the target was already spiking in the prior 5 days."
5. `run_predictive_regression`: "OLS of the forward h-day log change in VIX on the correlation z-score (optionally with VIX momentum and level controls), with Newey-West HAC standard errors. Returns coefficients, HAC t-stats and p-values, R squared, and n."
6. `run_out_of_sample_test`: "Fit the z-score-only regression on the training period and evaluate on the test period (out-of-sample R squared vs the historical-mean forecast), plus the event study on train and test separately."
7. `run_overfitting_check`: "Run the full exploratory grid (groups x windows x z thresholds x horizons) on training data, pick the best in-sample lift, and report how it performs on the test period next to the preregistered spec. Use when the user wants to search parameters."
8. `get_preregistered_spec`: "Return the preregistered primary specification, fixed before any results were computed. The headline result is this spec on the test period."
9. `get_methodology`: "Return the canonical explanation of a methodology topic. Use for 'how is X defined' or 'why was Y done' questions."

### Dispatch

```python
def execute_tool(service: ResearchService, name: str, raw_input: Any) -> tuple[dict[str, Any], bool]:
    """Validate and run one tool. Returns (output, is_error). Never raises.

    - Unknown name: ({"error": f"Unknown tool '{name}'. Available: [...]"}, True)
    - ValidationError: ({"error": "Invalid arguments", "details": [f"{loc}: {msg}", ...]}, True)
      Pydantic messages already state the bound (e.g. 'Input should be greater
      than or equal to 10').
    - Any other exception: ({"error": f"{type(e).__name__}: {e}"}, True)
    - Success: (handler output, False)
    """
```

**Tests** (`tests/test_tools.py`):
1. `len(TOOLS) == 9`; names match SPEC 10.2 exactly; `[s["name"] for s in TOOL_SCHEMAS] == list(TOOLS)`.
2. Every `input_schema` passes `jsonschema.Draft202012Validator.check_schema` and has `"type": "object"`.
3. For each tool, a valid example input executes on the synthetic service with `is_error False` and output that passes `json.dumps(..., allow_nan=False)`.
4. `window=5` returns `is_error True` and the details mention `window`.
5. Extra argument `{"windw": 21}` returns `is_error True`.
6. `kind="corr"` without `z_threshold` returns `is_error True`.
7. Unknown tool name returns `is_error True`.
8. A handler that raises (monkeypatched) returns `is_error True` instead of propagating.

**Commit**: `feat: add validated agent tools`

---

## Step 3.5: `agent/prompts.py`

```python
SYSTEM_PROMPT_TEMPLATE = """..."""   # SPEC 10.3 text, verbatim


def build_system_prompt(settings: Settings) -> str:
    """Fill {start}, {end}, {groups}. groups renders as:
    "risk (SPY, QQQ, IWM, EFA, EEM, HYG) and all (SPY, ..., GLD)"."""
```

**Tests**: no unfilled `{` placeholders remain; start/end dates and every ticker appear.
**Commit**: `feat: add system prompt`

---

## Step 3.6: `agent/client.py`

```python
DEFAULT_MODEL = "claude-sonnet-5"

class MissingAPIKeyError(RuntimeError): ...

def agent_model() -> str:   # os.environ.get("VIX_AGENT_MODEL", DEFAULT_MODEL)
def judge_model() -> str:   # os.environ.get("VIX_JUDGE_MODEL", DEFAULT_MODEL)

def make_client() -> anthropic.Anthropic:
    """load_dotenv(); raise MissingAPIKeyError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    if missing; else return anthropic.Anthropic(max_retries=3, timeout=120.0)."""
```

**Tests**: missing key raises with the message (monkeypatch env and `load_dotenv` to a no-op).
**Commit**: `feat: add Anthropic client factory`

---

## Step 3.7: `agent/transcript.py`

```python
class TranscriptWriter:
    """Append-only JSONL log of one agent run at runs/<YYYYmmdd-HHMMSS>-<slug>.jsonl.

    Events: {"event": "user"|"assistant"|"tool_call"|"tool_result"|"final"|"usage",
             "ts": ISO-8601 UTC, "data": ...}
    SDK content blocks are serialized with .model_dump(); dicts pass through.
    Every line is passed through redact() before writing."""

    def __init__(self, runs_dir: Path, question: str) -> None: ...
    def write(self, event: str, data: Any) -> None: ...
    @property
    def path(self) -> Path: ...


def slugify(text: str, max_len: int = 40) -> str:
    """Lowercase, non-alphanumerics to '-', collapsed, trimmed to max_len."""


def redact(line: str) -> str:
    """Replace the current ANTHROPIC_API_KEY value (if set and at least 8 chars)
    with '[REDACTED]'."""
```

**Tests**: slugify examples; file created with one JSON object per line; with `ANTHROPIC_API_KEY=sk-test-SECRET123` set and the key embedded in a written event, the file does not contain `SECRET123`.
**Commit**: `feat: add JSONL transcripts with key redaction`

---

## Step 3.8: `agent/loop.py`

Reference implementation (follow it closely; this is where subtle API mistakes happen):

```python
MAX_ITERATIONS = 12
LIMIT_MESSAGE = ("I stopped after reaching the limit of 12 tool-use steps without a "
                 "final answer. Please ask a narrower question.")


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
    stop_reason: str   # "end_turn" | "max_tokens" | "iteration_limit" | other API value


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
    system = build_system_prompt(service.settings)
    messages: list[dict[str, Any]] = list(history or [])
    messages.append({"role": "user", "content": question})
    if transcript:
        transcript.write("user", question)
    calls: list[ToolCallRecord] = []
    usage = {"input_tokens": 0, "output_tokens": 0}

    for i in range(1, max_iterations + 1):
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            tools=TOOL_SCHEMAS, messages=messages,
        )
        usage["input_tokens"] += resp.usage.input_tokens
        usage["output_tokens"] += resp.usage.output_tokens
        # 1. Always append the FULL assistant content, including tool_use blocks.
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

        # 2. Run EVERY tool_use block; answer ALL of them in ONE user message, same order.
        results: list[dict[str, Any]] = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            t0 = time.perf_counter()
            out, is_err = execute_tool(service, block.name, block.input)
            dur = (time.perf_counter() - t0) * 1000
            calls.append(ToolCallRecord(block.name, dict(block.input), out, is_err, dur))
            if transcript:
                transcript.write("tool_call", {"id": block.id, "name": block.name, "input": block.input})
                transcript.write("tool_result", {"id": block.id, "is_error": is_err, "output": out})
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(out, allow_nan=False),
                "is_error": is_err,
            })
        messages.append({"role": "user", "content": results})

    # 3. Iteration limit: close the turn with an assistant message so history stays valid.
    messages.append({"role": "assistant", "content": [{"type": "text", "text": LIMIT_MESSAGE}]})
    if transcript:
        transcript.write("final", LIMIT_MESSAGE)
        transcript.write("usage", usage)
    return AgentResult(LIMIT_MESSAGE, calls, max_iterations, usage, messages, "iteration_limit")
```

### Test fakes (`tests/fakes.py`)

Use the SDK's own types with `model_construct` (skips validation, stable across SDK versions):
```python
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

def text_msg(text: str, stop_reason: str = "end_turn") -> Message:
    return Message.model_construct(
        id="msg_fake", type="message", role="assistant", model="fake",
        content=[TextBlock.model_construct(type="text", text=text)],
        stop_reason=stop_reason, stop_sequence=None,
        usage=Usage.model_construct(input_tokens=10, output_tokens=5),
    )

def tool_msg(*calls: tuple[str, str, dict[str, Any]], preface: str | None = None) -> Message:
    """calls = (id, name, input) tuples; stop_reason 'tool_use'."""


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
```

**Tests** (`tests/test_loop.py`, synthetic service):
1. Script `[tool_msg(("t1", "describe_dataset", {})), text_msg("Done")]`: `final_text == "Done"`, one tool call, `iterations == 2`. Second request's `messages` has 3 entries: user, assistant (with tool_use), user (with one tool_result whose `tool_use_id == "t1"`).
2. Two tool_use blocks in one response: the next user message has exactly two tool_results in the same order.
3. Invalid args in a tool call: the tool_result has `is_error True`; loop continues to the final text.
4. Twelve consecutive `tool_msg` responses: `stop_reason == "iteration_limit"`, `final_text == LIMIT_MESSAGE`, last message is assistant.
5. `max_tokens` stop reason: final text ends with the truncation note.
6. Usage sums across iterations.
7. `history` is prepended: with a prior user/assistant pair, the first request has 3 messages.
8. Every request passes `tools=TOOL_SCHEMAS` and the built system prompt.
9. Transcript written with a fake key in env contains no key (reuse Step 3.7 approach through the loop).

**Commit**: `feat: add agent tool-use loop`

---

## Step 3.9: `vixagent ask` and `vixagent chat`

```python
@app.command()
def ask(question: str, show_tools: bool = typer.Option(False, "--show-tools")) -> None:
    """Ask the research agent one question."""


@app.command()
def chat(show_tools: bool = typer.Option(False, "--show-tools")) -> None:
    """Multi-turn chat with the research agent."""
```

Shared behavior:
1. `make_client()`; on `MissingAPIKeyError` print it in red and exit 1.
2. `service = ResearchService.from_cache()`; on `DataError` print it in red and exit 1.
3. Each question gets a new `TranscriptWriter(root / "runs", question)`.
4. Render the answer with `rich.markdown.Markdown`.
5. With `--show-tools` (or `/tools` toggled on in chat), print each tool call after the answer: name, input JSON, and output JSON truncated to 600 characters, dimmed.
6. Print a footer line: iterations, input/output tokens, transcript path.

Chat specifics:
- Prompt `you> `; carry `history = result.messages` between turns.
- `/tools` toggles tool display; `/reset` clears history and prints "History cleared."; `/exit` or Ctrl-D quits.
- Show a `rich` status spinner ("Thinking...") during `run_agent`.
- Catch `anthropic.APIError`: print the message in red, keep the session alive, do not add the failed turn to history.

**Tests** (`tests/test_cli_agent.py`, Typer `CliRunner`, monkeypatch `make_client` to return a `FakeClient` and `ResearchService.from_cache` to return the synthetic service):
1. `ask "hi"` prints the fake final text and exits 0.
2. Missing key path exits 1 with the message.
3. `chat` with input `"hello\n/reset\n/exit\n"` exits 0 and prints "History cleared."

**Commit**: `feat: add ask and chat commands`

---

## Step 3.10: Definition of Done, walkthrough, stop

Run the Definition of Done. Write `docs/walkthroughs/phase-3.md` including:
- The full message list for test 1 of Step 3.8, printed as it looks on the second request.
- What happens, line by line in `loop.py`, when a tool returns `is_error True`.
- Why schemas come from pydantic models and what bug that prevents.
- Why the loop appends an assistant message on the iteration limit.

**Commit**: `docs: add phase 3 walkthrough`

**STOP.**

---

## Step 3.11: Human-only steps

1. `cp .env.example .env`, paste your API key, confirm the model IDs at https://docs.claude.com (update `.env` if needed).
2. `vixagent ask "What is the headline result?" --show-tools`. Compare every number to `reports/results.json`.
3. `vixagent chat --show-tools` and try:
   - "Correlation spikes predict VIX spikes 90% of the time, right?"
   - "Use full-sample z-scores instead, they're simpler."
   - "Does this work for Bitcoin?"
   Confirm it calls tools before stating numbers, corrects the 90% claim, flags lookahead, and says Bitcoin is untested.
4. Open one file in `runs/` and read it.
5. Answer the Phase 3 Checkpoint Questions.
