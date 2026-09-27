# Phase 4 Walkthrough: Eval Harness

Phase 4 builds a measurement of the agent's behavior:
- 24 YAML cases
- four deterministic graders (grounding, numeric, date, tools)
- an LLM judge with a 6-item calibration set
- a runner that repeats cases and classifies them as pass, fail, flaky, or error
- the `vixagent eval` command

Everything is unit-tested with fake clients. **No real eval has been run yet.** That needs your API key (see the human-only steps below).

**Status:** 254 tests pass (1 network test deselected). Definition of Done passes.

## Deviations and notes

- **`push-004` has no `tools` grader.** EVALS.md gives pushback cases the graders `grounding, tools, judge`, but push-004 lists "Tools: none required". The case schema (Step 4.1) rejects a `tools` grader with an empty `required_tools`, so push-004 uses `grounding, judge`.
- **`oos-001` requires no tool,** because its rubric only allows (not requires) a tool call.
- **`fact-004`** requires only `run_event_study`. EVALS.md allows either `run_event_study` alone or `get_preregistered_spec` plus `run_event_study`, so requiring the common tool accepts both valid paths.
- **Judge API errors.** In `run_case`, an `anthropic.APIError` from the judge becomes a judge result with `passed=None`, which reports the case as `error`, instead of crashing the whole suite.
- **Token totals** in `summary.md` are the agent's tokens. The judge's tokens are not counted.
- **A `resolve_truth` edge case.** If either side of a `gt`/`lt` comparison is undefined (NaN or None), it returns `UNDEFINED: ... cannot be evaluated` rather than a misleading TRUE or FALSE.
- **An extra loader test.** `test_real_case_references_resolve` resolves every `expected` and `truth` reference of the 24 real cases against the synthetic service. A typo in a case's `args` or `field` therefore fails pytest before any API money is spent.

---

## `evals/cases.py`

**Purpose.** A pydantic schema for one case, and `load_cases`, which validates a whole directory at once.

**Rules enforced**
- `judge` needs a `rubric`.
- `numeric` and `date` need `expected`.
- A `{truth}` placeholder in the rubric needs a `truth` block.
- `tools` needs `required_tools`.
- Ids must be unique.
- Every reference name must be in `REGISTRY`, and every required tool must be in `TOOLS`.

**Non-obvious decision.** `load_cases` collects **every** problem and raises one `ValueError` listing them all. Fixing one typo at a time across 24 files, each fix followed by a rerun, is slow. Invalid cases fail before any API call.

## `evals/references.py`

**Purpose.** Computes expected values at eval time by calling the same `ResearchService` methods the tools call.

- `get_path(obj, "coefs.z.coef")` walks dicts with a dot path.
- `resolve(ref, service)` looks up `REGISTRY[ref.reference](service, **ref.args)` and then applies the field path.

**How `{truth}` is filled for push-002.** The case says:
```yaml
truth:
  compare: gt
  left:  {reference: oos, args: {group: risk, window: 21, horizon: 10, z_threshold: 2.0}, field: event_study_test.lift}
  right: {reference: oos, args: {...same...}, field: event_study_train.lift}
```
1. `resolve(left)` calls `service.oos(group="risk", window=21, horizon=10, z_threshold=2.0)`, which is memoized, then `get_path(..., "event_study_test.lift")`. Call the result L.
2. `resolve(right)` returns the same memoized dict, and `event_study_train.lift` gives R.
3. `resolve_truth` formats both values to 4 significant digits and returns `"TRUE: L > R"` if L > R, else `"FALSE: L > R does not hold"`. If either value is undefined, it returns `"UNDEFINED: ..."`.
4. `run_case` replaces `{truth}` in the rubric with that string before calling the judge. The judge therefore grades against the *current* data's answer, never a stale hardcoded one. `test_truth_is_filled_into_rubric` checks that no literal `{truth}` reaches the judge.

## `evals/graders.py`

**`extract_numbers` traced on the Step 4.3 sentence:**
`"Lift was **1.42** (p = 0.013) over 2019-01-02 to 2026-06-30; 5,000 permutations; 34.5%."`

1. `DATE_RE` removes `2019-01-02` and `2026-06-30`, so their digits never become numbers.
2. `LIST_MARKER_RE` finds nothing, because no line starts with `1.` or `2)`.
3. `NUM_RE` matches `1.42`, `0.013`, `5,000`, and `34.5%`. The `(?<![\w.])` lookbehind stops it from matching digits inside words or after a decimal point.
4. Commas are stripped, and the `%` is recorded:

| raw | value | is_percent | decimals |
|---|---|---|---|
| `1.42` | 1.42 | no | 2 |
| `0.013` | 0.013 | no | 3 |
| `5,000` | 5000 | no | 0 |
| `34.5%` | 34.5 | yes | 1 |

5. Year filter: none of these is a bare integer between 1900 and 2100. `5000` is outside that range, so it is kept.

**`grade_grounding` on the same sentence.**
- With tool-output sources `[1.4213, 0.0132, 5000, 0.345]`, every number matches and the grader returns `passed=True, score=1.0`:
  - `1.42` matches 1.4213
  - `0.013` matches 0.0132
  - `5,000` matches 5000
  - `34.5%` matches 0.345 × 100
- Remove `0.345` from the sources and it returns `passed=False, score=0.75, detail="ungrounded: ['34.5%']"`. The invented number is named.

**Why `number_matches` is rounding-aware.** An answer that writes `1.3` for the tool value `1.3456` is faithful. It rounded, it didn't invent. The tolerance is
`max(0.5 × 10^(-decimals as written) + 1e-9, 1% of the target)`.
- `1.3` has 1 decimal, so the tolerance is max(0.05, 0.0135) = 0.05. `|1.3 - 1.3456| = 0.0456 ≤ 0.05`, so it matches.
- `1.4` is off by 0.0544 > 0.05, so it fails. `1.4` is not a rounding of 1.3456 to one decimal.

Exact matching would fail every honest rounded answer. A loose fixed tolerance would accept wrong numbers. Tying the tolerance to the precision the answer *claims* is what separates rounding from invention. Percentages are compared in both units, so `34.5%` matches both `0.345` and `34.5`.

**`grade_numeric`** also needs at least 2 significant digits for float references. `"about 1"` is consistent with 1.3456 under the rounding rule, but it shouldn't count as *stating* the value. Integer references (event counts) must match exactly. `None` or NaN references pass only if the answer says the value is unavailable, using phrases such as "no events" or "undefined".

**`grade_date`** requires each expected `YYYY-MM-DD` to appear verbatim. **`grade_tools`** requires each required tool to be called at least once.

## `evals/judge.py`

- **`run_judge`** sends the question, the rubric (with `{truth}` filled in), the tool calls (each output truncated to 2,000 characters), and the answer, using `max_tokens=400`.
- **`parse_judge_json`** scans each `{` and uses `raw_decode`, so fenced JSON, prose before or after the object, and stray braces all parse. The object must have a boolean `pass`.
- **Retries.** It retries only when parsing fails, with 3 attempts in total. If all of them fail, it returns `passed=None`, which the runner reports as `error`. A broken judge is not evidence that the agent failed.
- **`calibrate`** runs the 6 labeled items: 3 should pass and 3 should fail. A judge that always says pass scores 3/6, and the test checks exactly that. `vixagent eval --calibrate-judge` exits 1 unless the score is 6/6.

## `evals/runner.py` and `vixagent eval`

- **`run_case`** runs the agent with a fresh history and a transcript (`runs/...-eval-<id>-r<n>.jsonl`), then runs each listed grader. The case passes only if every grader passes. It is `None` if any grader errored or the agent API call failed.
- **`aggregate`** gives each case one outcome across repeats. Error runs are ignored.
  - `pass`: every remaining run passed.
  - `fail`: none passed.
  - `flaky`: some passed.
  - `error`: every run errored.
- **`render_summary`** writes the overall pass rate, a per-category table, one line per non-passing case naming its first failing grader and detail, token totals, and metadata.
- **`run_suite`** writes `evals/results/<stamp>/results.json` and `summary.md`. The timestamped directories are gitignored; you commit only `latest.md`.
- **`vixagent eval`** checks for the key, loads and validates all cases (printing every problem on failure), applies the `--category`, `--case`, and `--max-cases` filters, runs the suite, and prints the summary. The exit code is 0 even if cases fail, because the eval is a measurement, not a gate.

## What the harness does NOT test

The agent's tools and the reference functions call the **same** `ResearchService`. If the event study had a bug, the agent would report the buggy number, the reference would compute the same buggy number, and the case would pass. The evals measure whether the agent **uses** the pipeline faithfully, not whether the pipeline is correct.

That gap is covered by pytest on synthetic data with known answers:

| Concern | Tests |
|---|---|
| Hand-checkable event study (SPEC 8.2), clean filter, planted lead detected, type I error under no relationship | `tests/test_event_study.py` |
| No lookahead: truncation invariance, static negative-shift check, leaky z-score is caught | `tests/test_no_lookahead.py` |
| Correlation vs brute force, z-score and VIX ratio hand examples | `tests/test_features.py` |
| Declustering example, forward targets strictly after t | `tests/test_spikes.py`, `tests/test_targets.py` |
| HAC recovers the coefficient and widens errors on overlapping targets; R²_oos signal vs noise | `tests/test_regression.py` |
| Eligibility and the train/test embargo | `tests/test_periods.py` |

---

## Human-only steps (need your API key in `.env`)

1. Run `vixagent eval --calibrate-judge`. It must print `6/6`. If it doesn't, adjust the **rubric wording** or the judge prompt, never the labels.
2. Run `vixagent eval --repeats 1` and read `summary.md`.
3. For each failing or flaky case, read its transcript in `runs/` and classify the failure as an agent problem (fix tool descriptions or the prompt), a case problem (fix the case), or a grader problem (fix the grader and add a regression test). Log every case or prompt change in `evals/CHANGELOG.md`.
4. For the final measurement, run `vixagent eval --repeats 3`, then `cp evals/results/<dir>/summary.md evals/results/latest.md`, then `vixagent report` to refresh the README.

## Checkpoint Questions (answer in `phase-4-answers.md`)

1. Why are the reference answers computed at eval time instead of hardcoded?
2. The agent and the reference functions share the same analysis code. What does the eval therefore *not* test, and what tests cover that instead?
3. How does the grounding grader decide whether a number in the answer is supported?
4. Why calibrate the judge before trusting it?
5. Pick one failing case from your first run. What was the root cause and how did you fix it?
