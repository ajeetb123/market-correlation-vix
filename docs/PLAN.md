# PLAN: Phased Build

Put this file at `docs/PLAN.md`. Section numbers refer to `docs/SPEC.md`.

This is the overview. The detailed procedure for each phase (files, signatures, algorithms, tests, verify commands, commit messages) is in `docs/steps/phase-N-*.md`. Where a step file is more specific than the task list below, follow the step file.

Each phase ends with: Definition of Done passing (AGENTS.md), a walkthrough in `docs/walkthroughs/phase-N.md`, and a **STOP** for the human. The human answers the Checkpoint Questions in their own words before approving the next phase. If the human can't answer one, the agent explains that part again before moving on.

---

## Phase 0: Scaffold

**Tasks**
1. `git init`, create the layout in SPEC section 2 (empty modules with docstrings are fine).
2. `pyproject.toml` per SPEC section 3.
3. `.gitignore` per AGENTS.md; `.env.example` per SPEC section 3.
4. `config/settings.yaml` and `config/preregistered.yaml` exactly as in SPEC section 3.
5. `config.py`: pydantic models + loader + validation rules from SPEC section 3.
6. `utils/jsonable.py`: `to_jsonable(obj)` recursively converts numpy scalars/arrays, `pd.Timestamp` (to `YYYY-MM-DD`), `NaN`/`inf` (to `None`), dataclasses, tuples, and dicts.
7. CI workflow per SPEC section 13.
8. Placeholder `README.md` containing the four marker comments from SPEC section 9.

**Tests**
- `config.py` loads the real YAML; invalid configs (train end after test start, empty group, primary not in grid) raise.
- `to_jsonable` handles every type above, and its output passes `json.dumps(..., allow_nan=False)`.

**Acceptance**
- `pip install -e ".[dev]"` works in a fresh venv.
- Definition of Done passes. CI is green on the first push.

**Checkpoint Questions**
1. What is the difference between `dependencies` and `[project.optional-dependencies]` in `pyproject.toml`?
2. Why does `json.dumps` fail on a `numpy.float64` or produce invalid JSON for `NaN`?
3. What does CI check on every push, and why are network tests excluded?

---

## Phase 1: Data, Features, Spikes

**Tasks**
1. `data/fetch.py`, `data/cache.py`, `data/validate.py`, alignment helper (SPEC section 4).
2. `features/returns.py`, `features/correlation.py`, `features/zscore.py`, VIX ratio (SPEC section 5).
3. `targets.py` (SPEC section 6).
4. `spikes.py` (SPEC section 7).
5. `periods.py` with eligible-day logic (SPEC section 8.1).
6. `tests/fixtures/synthetic.py`.
7. `vixagent pull` CLI command.
8. Every test listed under Data, Features, Targets, and Spikes in SPEC section 12, **including the no-lookahead truncation test and the `shift(-` static check**.

**Manual step (human)**
- Run `vixagent pull` once with network. Confirm the summary looks right (tickers, date range, row counts).
- If replacing the asset universe with the GMU study's list, do it now.
- Commit `config/preregistered.yaml`. **It is frozen from this commit on.** Tag the commit `prereg-freeze`.

**Acceptance**
- Definition of Done passes.
- `vixagent pull` prints: tickers per group, first and last date, trading days, rows dropped in alignment, and any validation warnings.
- Coverage at least 85% on `features/`, `spikes.py`, `targets.py`.

**Checkpoint Questions**
1. In your own words: why does `x.shift(1).rolling(252).mean()` avoid lookahead but `x.rolling(252).mean()` applied to a full-sample standardization would not?
2. Walk through the no-lookahead truncation test. What bug would it catch?
3. Why declustering? What goes wrong statistically if every crisis day counts as an event?
4. Why is the VIX baseline a median and not a mean?
5. Why is `preregistered.yaml` frozen *before* any results are computed?

---

## Phase 2: Analysis and Report (first GitHub publish, v0.1)

**Tasks**
1. `analysis/event_study.py`, `analysis/permutation.py` (SPEC sections 8.2 and 8.3).
2. `analysis/regression.py` (8.4), `analysis/oos.py` (8.5), `analysis/grid.py` (8.6).
3. `report/figures.py`, `report/results.py`, `report/readme.py` (SPEC section 9).
4. `vixagent report` CLI command.
5. Every test listed under Event study and permutation, and Regression and OOS, in SPEC section 12.

**Manual step (human)**
- Run `vixagent report`. Open all four figures and `results.md`. Read the headline result before reading anything else.
- Push to GitHub, tag `v0.1`.

**Acceptance**
- Definition of Done passes. Coverage at least 85% on `analysis/`.
- `vixagent report` produces the four PNGs, `results.json`, `results.md`, and updates the README markers.
- Running `vixagent report` twice produces identical `results.json` except `generated_at`.
- Headline section reports the preregistered spec on the **test** period, whatever the outcome.

**Checkpoint Questions**
1. State the headline result in one sentence, including `n_events`, lift, and p-value. Is it significant at alpha 0.05?
2. Explain the hand-checkable event-study example (SPEC 8.2) and compute the base rate by hand.
3. Why a circular-shift permutation test instead of a t-test?
4. Why HAC standard errors with `maxlags = h`?
5. What did the reverse-direction check show, and what would it mean if it were as strong as the forward direction?
6. In the overfitting check, how much did the best in-sample combination decay out of sample? Why is that expected?

---

## Phase 3: Agent

**Tasks**
1. `agent/service.py` (SPEC section 10.1).
2. `agent/methodology.py`: canonical text for every topic in the `get_methodology` enum. Each entry is consistent with SPEC and under 200 words.
3. `agent/tools.py`: all nine tools with JSON Schemas, pydantic arg models, dispatch, bounds, and output caps (SPEC section 10.2).
4. `agent/prompts.py` (SPEC section 10.3, verbatim).
5. `agent/loop.py` (SPEC section 10.4) with injected client.
6. `agent/transcript.py` (SPEC section 10.5).
7. `vixagent ask` and `vixagent chat` CLI commands (SPEC section 11).
8. Every test listed under Agent in SPEC section 12, using a fake client. No real API calls in tests.

**Manual step (human)**
- Put your API key in `.env`. Run `vixagent chat --show-tools` and try:
  - "What is the headline result?"
  - "Correlation spikes predict VIX spikes 90% of the time, right?"
  - "Use full-sample z-scores instead, they're simpler."
- Confirm the agent calls tools before stating numbers and pushes back on the last two.

**Acceptance**
- Definition of Done passes.
- `vixagent ask "What is the headline result?"` returns an answer whose numbers match `reports/results.json`.
- Transcripts appear in `runs/` and contain no API key (a test greps a transcript written with a fake key).

**Checkpoint Questions**
1. Draw the message sequence for one question that triggers two tool calls. Where do `tool_use` and `tool_result` blocks go?
2. Why does the agent get summary statistics from tools instead of raw time series?
3. What happens if a tool raises an exception? Trace it through `loop.py`.
4. Why is the Anthropic client injected instead of created inside `run_agent`?
5. Which system prompt rule prevents the agent from inventing numbers, and how would you *verify* it follows that rule? (Hint: Phase 4.)

---

## Phase 4: Eval Harness

Follow `docs/EVALS.md` exactly.

**Tasks**
1. `evals/cases.py`, `evals/references.py`, `evals/graders.py`, `evals/judge.py`, `evals/runner.py`.
2. All 24 case files in `evals/cases/` and `evals/judge_calibration.yaml` as specified in EVALS.md.
3. `vixagent eval` CLI command.
4. Unit tests for graders with canned answers (no API calls).

**Manual step (human)**
- Run `vixagent eval --calibrate-judge`. It must agree with all 6 labels before the judge is trusted.
- Run `vixagent eval --repeats 3`. Read every failing transcript. Decide with the agent whether each failure is an agent problem (fix prompt or tool descriptions) or a case problem (fix the case). **Never** change a case just to make it pass; write down the reason for every case change in `evals/CHANGELOG.md`.
- Re-run after fixes. Copy the final summary to `evals/results/latest.md`.

**Acceptance**
- Definition of Done passes.
- Judge calibration 6/6.
- `evals/results/latest.md` exists with per-category pass rates, flaky cases, and total token cost.
- README EVALS markers updated.

**Checkpoint Questions**
1. Why are the reference answers computed at eval time instead of hardcoded?
2. The agent and the reference functions share the same analysis code. What does the eval therefore *not* test, and what tests cover that instead?
3. How does the grounding grader decide whether a number in the answer is supported?
4. Why calibrate the judge before trusting it?
5. Pick one failing case from your first run. What was the root cause and how did you fix it?

---

## Phase 5: Polish and Publish (v1.0)

**Tasks**
1. Final README per SPEC section 14 (the agent does **not** write "What I learned").
2. Mermaid architecture diagram.
3. Abridged real transcript from `runs/` showing pushback.
4. Optional: `app/streamlit_app.py` with a chat panel, tool-call expander, and the four figures. Install via `pip install -e ".[app]"`, run with `streamlit run app/streamlit_app.py`.
5. Final pass: remove dead code, confirm every public function has a docstring, confirm no TODOs remain except "What I learned".

**Manual step (human)**
- Write "What I learned" yourself.
- Tag `v1.0`, pin the repo on your GitHub profile, add it to your resume and portfolio site.

**Acceptance**
- Definition of Done passes. CI green.
- A fresh clone plus the Quickstart steps works end to end on a clean machine.
- README contains no hand-typed result numbers.

**Checkpoint Questions**
1. Explain the whole project in 60 seconds as you would in an interview.
2. What is the single biggest limitation, and what would you do next to address it?
3. If an interviewer asks "did AI write this?", what is your honest answer, and which parts can you explain line by line?

---

## Suggested resume bullet (fill in real numbers after Phase 4)

> **VIX Research Agent** (Python, Anthropic API, statsmodels): Built a preregistered event-study pipeline testing whether cross-asset correlation spikes lead VIX spikes (permutation tests, HAC regressions, out-of-sample validation), plus a tool-using LLM research agent evaluated on a 24-case harness for numeric grounding and pushback on flawed premises (X% pass rate).
