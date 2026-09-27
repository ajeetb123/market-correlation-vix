# Phase 5 Walkthrough: One Question, End to End

Phase 5 includes:
- the final README: summary, the Result and Eval generated blocks, "Why this is hard", a Mermaid architecture diagram, Quickstart, limitations, and a repository map
- a cleanup pass: every public class and function in `src/` now has a docstring, and there are no TODOs
- a fresh-clone check

## Status and what's left for you

- **Fresh-clone check (done, adapted).** Pushing isn't allowed yet, so the repo was cloned from the local path rather than GitHub. A brand-new Python 3.12 venv ran `pip install -e ".[dev]"`, `ruff check`, `ruff format --check`, `mypy src`, and `pytest -q`: 254 passed, and `git status` in the clone was clean. `vixagent pull` and `vixagent report` were not run in the clone, because you run the report first.
- **README notes.**
  - The Quickstart URL points at `ajeetb123/market-correlation-vix`, the existing repo, not the spec's `vix-research-agent`.
  - The Limitations bullets are copied verbatim from `reports/results.md`. They contain configuration values (the 1.30 threshold, the 24 combinations) but no result numbers.
- **Step 5.3, example transcript.** This needs a real `chat` session from you. The README has an HTML comment where it goes.
- **Step 5.4, Streamlit demo.** It's optional and was not built. Per the step file, it's only built if you want it.
- **"What I learned".** This section contains only the placeholder comment. You write it.
- **Human steps before v1.0:** run the report, add your API key, run the evals, write the transcript example and "What I learned", tag, push, and release.

---

## Trace: `vixagent ask "What is the headline result?"`

The model decides which tools to call. This trace follows a typical sequence: `get_preregistered_spec`, then `run_event_study` with the preregistered parameters on the test period.

### 1. CLI setup (`cli.py`)
1. `ask(question, show_tools)` is the Typer command.
2. `_agent_setup()`:
   1. `make_client()` (`agent/client.py`): `load_dotenv()` reads `.env`. It raises `MissingAPIKeyError` if `ANTHROPIC_API_KEY` is absent, which prints in red and exits 1. Otherwise it returns `anthropic.Anthropic(max_retries=3, timeout=120)`.
   2. `ResearchService.from_cache()` (`agent/service.py`):
      - `load_settings()`, which calls `find_project_root()` and then `Settings.model_validate()`. That runs `Settings._check`, which checks date ordering, group sizes, and the grid.
      - `load_preregistered(settings)`, which checks that the spec is a grid combination.
      - `load_cached_prices(settings)` (`data/cache.py`): `_resolve`, `_meta_path`, `_expected_meta` (which calls `all_tickers`), and a metadata comparison. Then `pd.read_parquet`. It never touches the network. If there is no cache, it raises `DataError`, which prints in red and exits 1.
      - `ResearchService.__init__`, which creates `FrameStore(prices, settings)` and an empty memo dict.
   3. `find_project_root()` gives the directory for `runs/`.
3. `TranscriptWriter(root / "runs", question)` (`agent/transcript.py`) calls `slugify(question)`, which gives `runs/<stamp>-what-is-the-headline-result.jsonl`.
4. `console.status("Thinking...")` wraps `run_agent(...)`, with `model=agent_model()`.

### 2. First model call (`agent/loop.py: run_agent`)
5. `build_system_prompt(service.settings)` (`agent/prompts.py`) fills the SPEC 10.3 text with the dates and ticker groups.
6. `messages = [{"role": "user", "content": question}]`, then `transcript.write("user", ...)`. That goes through `_serialize` and then `redact`, which appends one JSON line.
7. Iteration 1 calls `client.messages.create(model, max_tokens=4096, system, tools=TOOL_SCHEMAS, messages)`. `TOOL_SCHEMAS` was built at import time by `ToolSpec.schema()`, which calls `model_json_schema()` on each argument model.
8. The response has `stop_reason="tool_use"` and `content = [text?, tool_use(get_preregistered_spec)]`. Usage is added up, and the full assistant content is appended to `messages` and to the transcript.
9. For the `tool_use` block, `execute_tool(service, "get_preregistered_spec", {})` (`agent/tools.py`) runs:
   - `TOOLS["get_preregistered_spec"]`, then `NoArgs.model_validate({})`, then the handler, then `service.preregistered_spec()`.
   - That goes through `_cached(("preregistered_spec",), fn)`, then `fn()`, which calls `prereg.primary.model_dump()`, and then `to_jsonable(...)` (`utils/jsonable.py`).
   - It returns `(output, False)`.
10. A `ToolCallRecord` is recorded, `tool_call` and `tool_result` are written to the transcript, and `{"type": "tool_result", "tool_use_id": ..., "content": json.dumps(output), "is_error": False}` is built. One user message holding all the results is appended.

### 3. Second model call: the event study
11. Iteration 2 calls `client.messages.create(...)` again. The response is `tool_use(run_event_study, {group: "risk", window: 21, z_threshold: 2.0, horizon: 10, period: "test", clean_only: true, direction: "corr_to_vix"})`.
12. `execute_tool` runs `EventStudyArgs.model_validate(input)`, which checks bounds and forbids extra fields. The handler then calls `service.event_study(...)`.
13. `ResearchService.event_study` builds a frozen `EventStudyParams(...)` and calls `_cached(("event_study", params), fn)`. The key is new, so `fn()` runs.
14. `run_event_study(store, params, settings)` (`analysis/event_study.py`):
    1. `store.frame("risk", 21)` (`analysis/frames.py`):
       - `store.aligned("risk")` calls `align_group(prices, settings, "risk")` (`data/align.py`), which uses `period_mask(index, settings.full_period())` (`periods.py`) and then `dropna`. It returns `(aligned, rows_dropped)`.
       - `build_frame(aligned, settings, "risk", 21)` (`features/frame.py`) runs, in order:
         - `log_returns` (`features/returns.py`)
         - `vix_ratio` (`features/vix.py`), then `spike_days` (`spikes.py`) to get `vix_spike_day`
         - `avg_pairwise_corr(returns, 21)` (`features/correlation.py`)
         - `log_vix` and `vix_momentum`
         - `events_from_days(vix_spike_day, 20)` to get `vix_event`
         - `trailing_zscore(avg_corr, 252)` (`features/zscore.py`) to get `corr_z`

       The frame is cached, so later calls reuse it.
    2. `spike_days(f.corr_z, 2.0)`, then `events_from_days(corr_days, 20)`, gives the correlation events.
    3. `valid = corr_z.notna() & vix_ratio.notna()`.
    4. For direction `corr_to_vix`, the source is the correlation events and the target is `vix_spike_day`.
    5. `settings.period("test")`, then `eligible_mask(index, period, 10, valid)` (`periods.py`): in the test period, valid, and `t + 10` inside the period.
    6. `event_study_core(...)`:
       - `fwd_any_within(tgt, 10)` (`targets.py`, the only forward-looking code) gives `hit_all`.
       - `backward_any(tgt, 5)` (`spikes.py`) gives the `dirty` flags, and `clean = ~dirty`.
       - `ev = src & eligible & clean` and `base_days = eligible & clean`.
       - It computes `n_events`, `n_hits`, `hit_rate`, `base_rate`, and `lift`.
       - `circular_shift_pvalue(src[eligible], hit_all[eligible], clean[eligible], hit_rate, 10, 5000, seed)` (`analysis/permutation.py`) uses `np.random.default_rng(seed)` and 5,000 `np.roll` shifts to give `p_value`.
       - It returns an `EventStudyResult`, with a warning if there are fewer than 5 events.
15. Back in the service, `_capped_event_study(result)` caps `event_dates` and `hit_flags` at 50 and adds `truncated`. Then `to_jsonable` runs and the result is memoized and returned.
16. `execute_tool` returns `(output, False)`. The `ToolCallRecord`, the transcript lines, and the `tool_result` user message are added as in step 10.

### 4. Final answer
17. Iteration 3 calls `client.messages.create(...)`, and the response has `stop_reason="end_turn"`. The text blocks are joined, and `transcript.write("final", text)` and `transcript.write("usage", usage)` are written. The function returns `AgentResult(final_text, tool_calls, iterations=3, usage, messages, "end_turn")`.
18. `_render(result, transcript, show_tools)` (`cli.py`):
    - `console.print(Markdown(final_text))`
    - with `--show-tools`, each call's name, input, and output, truncated to 600 characters and dimmed
    - a footer with iterations, token counts, and the transcript path

The numbers in the rendered answer are the same numbers `vixagent report` wrote to `reports/results.json` under `preregistered.test`. Both paths call `run_event_study` with the same frozen parameters, the same cached frame, and the same seed. That match is the Phase 3 acceptance check you run by hand, and `fact-004` in the eval harness checks it automatically.

---

## Checkpoint Questions (answer out loud, timed)

1. Explain the whole project in 60 seconds as you would in an interview.
2. What is the single biggest limitation, and what would you do next to address it?
3. If an interviewer asks "did AI write this?", what is your honest answer, and which parts can you explain line by line?
