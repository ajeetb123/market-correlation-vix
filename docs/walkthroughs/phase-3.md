# Phase 3 Walkthrough: Agent

Phase 3 wraps the pipeline so Claude can use it.

- **A cached service** holds the data and remembers every result.
- **Nine validated tools** expose that service to the model.
- **A tool-use loop** runs the conversation with the Messages API.
- **JSONL transcripts** record each run with the API key redacted.
- **`vixagent ask` and `vixagent chat`** are the user-facing commands.

No test touches the network or needs an API key. Every loop test uses a scripted `FakeClient`.

**Status:** 200 tests pass (1 network test deselected). Definition of Done passes. `agent/` coverage is 87–100% per module.

## Deviations and notes

- **One commit for two steps.** The service and methodology text went in together (`90329e2`) because the service's `methodology()` method depends on the text module.
- **The static no-lookahead test caught a real problem.** The `lookahead` methodology prose originally contained the literal characters `shift(-` inside a sentence *about* the lookahead test. The Phase 1 grep test flagged it. The fix was to reword the prose (`de2776d`), not to loosen the test.
- **`TopicArg` duplicates the topic list.** It is a `Literal` type listing the same topics as `methodology.TOPICS`, and `test_topic_enum_matches_methodology` keeps the two identical.
- **Extra field descriptions.** `SpikeEventsArgs` gives every field a `description`. The step's sketch left `kind` and `period` without one, but its instructions say every field should have one.
- **Transcript filename collisions.** If two runs start in the same second with the same question slug, `TranscriptWriter` appends `-2`, `-3`, and so on.

---

## `data/cache.py`: `load_cached_prices`

This loader reads the parquet cache and never downloads. If the cache is missing, or its metadata doesn't match `settings.yaml`, it raises `DataError("No cached data found. Run `vixagent pull` first.")`.

**Why:** a question typed into `chat` should never trigger a surprise download or yfinance rate-limit retries.

## `agent/service.py`: `ResearchService`

**Purpose.** The only object the tools talk to. It holds one `FrameStore` and memoizes every result.

**Key functions**
- `_cached(key, fn)`: runs `fn` only the first time a key is seen, and stores `to_jsonable(fn())`.
- There are nine public methods, one per tool. Each returns a JSON-safe dict.

**Worked trace:** `event_study("risk", 21, 2.0, 10, "test", True, "corr_to_vix")` called twice.
1. The key is `("event_study", EventStudyParams(...))`. The params dataclass is frozen, so it can be used as a dictionary key.
2. First call: the key is missing, so `run_event_study` runs. It uses 5,000 permutations on real settings.
3. `_capped_event_study` trims `event_dates` and `hit_flags` to 50 and adds `truncated`.
4. `to_jsonable` converts NaN to `None` and numpy numbers to Python numbers, and the result is stored.
5. Second call: the same dict object comes back instantly. `test_every_method_is_strict_json_and_memoized` asserts that with `is`.

**Non-obvious decisions**
- **Summaries, not time series.** Output date lists stop at 50. The model should reason from tested statistics, not recompute its own from thousands of raw rows. Its arithmetic over raw data would be exactly the ungrounded kind of number that rule 1 of the system prompt forbids.
- **Definitions come from settings.** Strings like "VIX / median(prior 20 days) >= 1.3" are built from `settings.yaml` values, so changing the config can never leave the agent quoting stale definitions.

## `agent/methodology.py`

`methodology_text(topic, settings, prereg)` returns up to 200 words per topic, covering 10 topics. Every number in the text, including the pair counts computed as C(n, 2) per group, is formatted from settings, not hand-typed. `get_methodology` serves these texts, so "how is X defined" answers match the code exactly.

## `agent/tools.py`

**Purpose.** Defines the 9 tools: their argument models, their JSON Schemas, and a dispatcher that never raises.

**Why schemas come from pydantic models.** `ToolSpec.schema()` calls `args_model.model_json_schema()`. For `get_correlation_summary` the model sees:

```json
{"type": "object", "additionalProperties": false, "required": ["group", "window", "period"],
 "properties": {"group": {"enum": ["risk", "all"], ...},
                "window": {"type": "integer", "minimum": 10, "maximum": 126, ...},
                "period": {"enum": ["train", "test", "full"], ...}}}
```

`execute_tool` validates the arguments with **the same model**.
- If the schema were hand-written, it could drift. For example, the schema could say the window is 5 to 252 while the code enforces 10 to 126. The model would then send `window=200`, a value the schema invited, and get an error back.
- With one source of truth that bug can't happen.
- `extra="forbid"` becomes `"additionalProperties": false`, so a misspelled argument such as `windw` is an error rather than being silently ignored.

**`execute_tool` never raises.** Each failure is returned as `(output, is_error=True)`:
- an unknown tool name
- a `ValidationError`, listed as `loc: msg` lines. Pydantic's messages already state the bound, for example "Input should be greater than or equal to 10".
- any exception inside a handler

**Why:** an exception would crash the whole conversation. An error result lets the model read what went wrong and retry with valid arguments.

## `agent/prompts.py`, `agent/client.py`, `agent/transcript.py`

- **`build_system_prompt`** fills the SPEC 10.3 text verbatim with the configured dates and ticker groups. For example, `risk (SPY, QQQ, IWM, EFA, EEM, HYG) and all (...)`.
- **`make_client`** loads `.env`. If `ANTHROPIC_API_KEY` is missing it raises `MissingAPIKeyError` with setup instructions. Otherwise it returns `anthropic.Anthropic(max_retries=3, timeout=120)`. The model IDs come from `VIX_AGENT_MODEL` and `VIX_JUDGE_MODEL`, defaulting to `claude-sonnet-5`.
- **`TranscriptWriter`** writes one JSON line per event to `runs/<timestamp>-<slug>.jsonl`. Every line goes through `redact()` first, which replaces the current API key value with `[REDACTED]`. The transcript test sets a fake key, puts it into the question and the model's answer, and asserts that `SECRET123` never reaches disk.

## `agent/loop.py`: `run_agent`

**Message list for loop test 1, as sent on the second request.** The script is: `tool_msg(("t1", "describe_dataset", {}))`, then `text_msg("Done")`.

```json
[
  {"role": "user", "content": "What is the headline?"},
  {"role": "assistant", "content": [
      {"type": "tool_use", "id": "t1", "name": "describe_dataset", "input": {}}]},
  {"role": "user", "content": [
      {"type": "tool_result", "tool_use_id": "t1", "is_error": false,
       "content": "{\"tickers\": {\"risk\": [\"SPY\", \"QQQ\", ...]}, \"vix_ticker\": ...}"}]}
]
```

The second response is `text_msg("Done")` with `stop_reason="end_turn"`, so the loop returns `final_text="Done"` and `iterations=2`.

The two API rules are visible in that list:
- The assistant message containing the `tool_use` block is appended **before** the `tool_result`.
- Every `tool_use` id gets a `tool_result` with a matching `tool_use_id` in the **very next** user message.

**What happens when a tool returns `is_error True`, line by line:**
1. `resp.stop_reason == "tool_use"`, so the loop skips the early return.
2. For each `tool_use` block, it calls `execute_tool(service, block.name, block.input)`. For example, with `window=5` this returns `({"error": "Invalid arguments", "details": ["window: Input should be greater than or equal to 10"]}, True)`. Nothing is raised.
3. It appends `ToolCallRecord(..., is_error=True, ...)` to `calls`, and writes `tool_call` and `tool_result` to the transcript when one is attached.
4. It appends `{"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out), "is_error": True}` to `results`. The API marks the result as a failure and the model sees the message.
5. After every block is processed, it appends `{"role": "user", "content": results}`.
6. The next iteration calls the API again. The model reads the error and typically retries with `window=21` or explains the limit. Loop test 3 scripts exactly that: an error result, then a final answer.

**Why the loop appends an assistant message at the iteration limit.** After 12 tool rounds, the last message in `messages` is a *user* message full of `tool_result` blocks. `chat` passes `result.messages` back in as `history`. The next question would then add a second user message in a row, directly after unanswered tool results, and the API rejects that as an invalid conversation. Appending an assistant message with `LIMIT_MESSAGE` closes the turn cleanly, so the history stays valid (loop test 4 asserts the last message is from the assistant).

**Why the client is injected.** `run_agent(client=...)` takes any object with `.messages.create(**kwargs)`. Tests pass `FakeClient`, which returns scripted SDK `Message` objects built with `model_construct` and records a deep copy of every request. That is how the tests check message ordering, the `tool_use_id` pairing, the tools and system prompt sent on every request, the history being prepended, usage totals, and the iteration limit, all with no network or key.

## `cli.py`: `ask` and `chat`

- **Both** create the client (a missing key prints red and exits 1) and load `ResearchService.from_cache()` (a missing cache prints red and exits 1). They then run the agent under a "Thinking..." spinner and render the answer as Markdown. The footer shows iterations, input and output tokens, and the transcript path. `--show-tools` prints each call's input and output, truncated to 600 characters and dimmed.
- **`chat`** carries `history = result.messages` between turns.
  - `/tools` toggles the tool display, `/reset` clears the history, and `/exit` or Ctrl-D quits.
  - An `anthropic.APIError` prints in red and the failed turn is not added to the history.

---

## Human-only steps

1. `cp .env.example .env` and paste your API key. The default model ID is `claude-sonnet-5`, and you can change it in `.env`.
2. Run `vixagent ask "What is the headline result?" --show-tools` and compare every number with `reports/results.json`. This needs `vixagent report` to have been run first.
3. Run `vixagent chat --show-tools` and try:
   - "Correlation spikes predict VIX spikes 90% of the time, right?"
   - "Use full-sample z-scores instead, they're simpler."
   - "Does this work for Bitcoin?"

   Confirm that it calls tools before stating numbers, corrects the 90% claim, flags lookahead, and says Bitcoin is untested.
4. Open one file in `runs/` and read it.

## Checkpoint Questions (answer in `phase-3-answers.md`)

1. Draw the message sequence for one question that triggers two tool calls. Where do `tool_use` and `tool_result` blocks go?
2. Why does the agent get summary statistics from tools instead of raw time series?
3. What happens if a tool raises an exception? Trace it through `loop.py`.
4. Why is the Anthropic client injected instead of created inside `run_agent`?
5. Which system prompt rule prevents the agent from inventing numbers, and how would you *verify* it follows that rule? (Hint: Phase 4.)
