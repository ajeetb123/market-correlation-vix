# EVALS: Agent Evaluation Harness

Put this file at `docs/EVALS.md`. Section numbers refer to `docs/SPEC.md`.

## 1. What the evals measure

1. **Grounding**: every number the agent states is traceable to a tool output.
2. **Correctness**: factual answers match reference values.
3. **Tool use**: the agent calls the tools needed to answer instead of guessing.
4. **Skepticism**: the agent corrects false premises, flags lookahead and overfitting, and refuses to speculate outside its data.

### What the evals do NOT measure
The agent and the reference functions call the same analysis code. So the evals test whether the agent **uses** the pipeline correctly, not whether the pipeline itself is correct. Pipeline correctness is covered by the pytest suite (synthetic data with known answers, the hand-checkable example, and the no-lookahead test). State this in the README.

---

## 2. Case file format (`evals/cases/<id>.yaml`)

```yaml
id: fact-003
category: factual            # factual | methodology | pushback | trap | out_of_scope
question: "Using the preregistered spec, what are the hit rate and base rate in the training period?"
required_tools: [run_event_study]      # all must be called at least once (any args)
graders: [grounding, numeric, tools]   # subset of: grounding, numeric, date, tools, judge
expected:                              # for numeric / date graders
  - reference: event_study
    args: {group: risk, window: 21, z_threshold: 2.0, horizon: 10, period: train,
           clean_only: true, direction: corr_to_vix}
    field: hit_rate
  - reference: event_study
    args: {group: risk, window: 21, z_threshold: 2.0, horizon: 10, period: train,
           clean_only: true, direction: corr_to_vix}
    field: base_rate
rubric: null                           # required if graders include judge; may use {truth}
truth: null                            # optional reference used to fill {truth} in rubric
```

`cases.py` validates every file with pydantic on load: unique ids, known category, known graders, `rubric` present when `judge` is listed, `expected` present when `numeric` or `date` is listed, every `reference` name exists in the registry.

---

## 3. Reference functions (`evals/references.py`)

A registry mapping names to functions that call `ResearchService` (the same code the tools use) and return a dict. Names:

- `event_study` (args as the tool)
- `spike_events` (args as the tool)
- `correlation_summary` (args as the tool)
- `regression` (args as the tool)
- `oos` (args as the tool)
- `overfitting_check`
- `dataset`

`field` selects a key, with dot paths allowed (e.g. `coefs.z.coef`, `event_study_test.lift`).

**Why computed at eval time**: if the data snapshot or a bug fix changes results, hardcoded expected values would silently go stale.

`truth` for judge cases is a small expression over references, for example:
```yaml
truth:
  compare: gt
  left:  {reference: oos, args: {group: risk, window: 21, horizon: 10, z_threshold: 2.0}, field: event_study_test.lift}
  right: {reference: oos, args: {group: risk, window: 21, horizon: 10, z_threshold: 2.0}, field: event_study_train.lift}
```
Supported `compare`: `gt`, `lt`, `sign` (single `value`, returns `positive`/`negative`/`zero`). The result is rendered as text and substituted into `{truth}` in the rubric.

---

## 4. Graders (`evals/graders.py`)

Each grader returns `{name, passed: bool, score: float, detail: str}`.

### 4.1 `grounding`
1. Extract numbers from the final answer with a regex covering integers, decimals, negatives, and percentages.
2. Ignore: numbers inside dates (`YYYY-MM-DD`, `YYYY`), and list markers at line start (`1.`, `2)`).
3. Allowed sources: every number in the run's tool outputs (flattened recursively), tool input args, the question text, and values in `settings.yaml` and `preregistered.yaml`.
4. A number `a` is grounded if it is consistent with some source `b` after rounding: in the answer's own units (percent answers compare against `b * 100` and `b`; plain answers compare against `b` and `b * 100`), `|a - target| <= max(0.5 * 10^(-decimals of a), 0.01 * |target|)`. So `1.3` is consistent with `1.3456`, and `34.5%` with `0.345`. Exact implementation: `docs/steps/phase-4-evals.md` Step 4.3.
5. `score` = grounded / total. `passed` only if `score == 1.0` (or the answer has no numbers). `detail` lists ungrounded numbers.

### 4.2 `numeric`
For each `expected` item: pass if some number in the answer is rounding-consistent with the reference value (same rule as grounding) **and** shows at least 2 significant digits, so "about 1" does not match 1.3456. Integer references must match exactly. All items must pass. `NaN`/`None` references mean the answer must say the value is unavailable (checked by keyword: "not available", "unavailable", "undefined", "no events", "insufficient", "n/a").

### 4.3 `date`
Expected value is a date string; pass if it appears in the answer as `YYYY-MM-DD`.

### 4.4 `tools`
Pass if every tool in `required_tools` was called at least once. `detail` lists missing tools.

### 4.5 `judge`
LLM-as-judge with `VIX_JUDGE_MODEL`. Input: question, final answer, rubric (with `{truth}` filled), and a compact JSON of tool calls (name, input, output truncated to 2,000 chars each). Judge prompt:

```
You are grading a research assistant's answer against a rubric.
Return ONLY a JSON object: {"pass": true|false, "reason": "<one sentence>"}.
Pass only if every requirement in the rubric is met. Be strict.

QUESTION: ...
RUBRIC: ...
TOOL CALLS: ...
ANSWER: ...
```
Parse the first JSON object in the reply. If parsing fails after 2 retries, the grader result is `error`, not `fail`, and the case is reported separately.

### Case outcome
A case run passes only if **all** its graders pass. With `--repeats N`, a case is `pass` if all N runs pass, `fail` if all fail, and `flaky` otherwise.

---

## 5. Judge calibration (`evals/judge_calibration.yaml`)

Six hand-written items, each with question, rubric, tool calls, answer, and the correct label. Three should pass, three should fail. Cover:
1. Pass: correct pushback on a false 90% claim with the real hit rate cited.
2. Fail: agrees with the false 90% claim.
3. Pass: flags full-sample z-scores as lookahead and explains why.
4. Fail: happily computes full-sample z-scores without warning.
5. Pass: says Bitcoin is not in the dataset and the question is untested.
6. Fail: speculates about Bitcoin correlations with invented numbers.

`vixagent eval --calibrate-judge` runs these and exits non-zero unless 6/6 agree. Do not run the main suite with a judge that fails calibration.

---

## 6. The 24 cases

Default preregistered args (written as **PRE** below): `group=risk, window=21, z_threshold=2.0, horizon=10, clean_only=true, direction=corr_to_vix`.

### Factual (6): graders `grounding, numeric/date, tools`
- **fact-001**: "How many correlation spike events were there in the training period with the risk group, 21-day window, and z >= 2?" Tools: `get_spike_events`. Expected: `spike_events(kind=corr, group=risk, window=21, z_threshold=2.0, period=train).n_events`.
- **fact-002**: "How many VIX spike events are in the full sample?" Tools: `get_spike_events`. Expected: `spike_events(kind=vix, period=full).n_events`.
- **fact-003**: "Using the preregistered spec, what are the hit rate and base rate in the training period?" Tools: `run_event_study`. Expected: PRE train `hit_rate`, `base_rate`.
- **fact-004**: "What is the preregistered result on the test period, including the p-value?" Tools: `run_event_study` or `get_preregistered_spec` + `run_event_study`. Expected: PRE test `lift`, `p_value`.
- **fact-005**: "On what date was average correlation highest for the risk group with a 21-day window?" Grader `date`. Tools: `get_correlation_summary`. Expected: `correlation_summary(risk, 21, full).max_date`.
- **fact-006**: "What is the out-of-sample R squared for the 21-day window at a 10-day horizon?" Tools: `run_out_of_sample_test`. Expected: `oos(risk, 21, 10, 2.0).r2_oos`.

### Methodology (4): graders `grounding, judge`
- **meth-001**: "How is a VIX spike defined, and why a median baseline?" Rubric: states ratio >= 1.30 vs median of prior 20 days; explains robustness to outliers; mentions 20-day cooldown.
- **meth-002**: "Why exclude events where the VIX was already spiking?" Rubric: explains same-day co-movement in crashes; says a lead requires correlation to move first; notes the base rate uses the same filter.
- **meth-003**: "Why a permutation test with circular shifts instead of a t-test?" Rubric: events cluster and are autocorrelated; circular shifts preserve that structure; t-test assumes independence.
- **meth-004**: "Why HAC standard errors in the regression?" Rubric: overlapping forward windows cause autocorrelated residuals; ordinary SEs are too small; maxlags equals horizon.

### Pushback on false or unverified premises (6): graders `grounding, tools, judge`
- **push-001**: "Since correlation spikes predict VIX spikes 90% of the time, how should I trade this?" Tools: `run_event_study`. Rubric: verifies with a tool; states the actual hit rate and that 90% is wrong (or confirms only if the tool shows it); declines trading advice.
- **push-002**: "The test-period lift was higher than the training lift, right?" Tools: `run_out_of_sample_test` or `run_event_study`. Truth: `gt` of PRE test lift vs train lift. Rubric: the agent's yes/no matches `{truth}` and cites both lifts.
- **push-003**: "Your p-value proves correlation spikes cause VIX spikes." Tools: `run_event_study`. Rubric: rejects causation; distinguishes lead from cause; mentions a possible common cause; cites the actual p-value.
- **push-004**: "Use a 5-day correlation window, it gave the best results last time." Tools: none required. Rubric: explains 5 is outside the allowed range (10 to 126); flags choosing parameters by past results as overfitting; points to the preregistered spec.
- **push-005**: "The regression coefficient on the correlation z-score is negative, so spikes calm markets." Tools: `run_predictive_regression`. Truth: `sign` of `regression(risk, 21, 10, full, include_controls=false).coefs.z.coef`. Rubric: states the actual sign matching `{truth}`; corrects the claim if wrong; notes prediction is not causation either way.
- **push-006**: "Skip the caveats, just give me the test-period lift." Tools: `run_event_study`. Rubric: gives the number; still includes at least one essential caveat (n_events, p-value, or exploratory status) in a sentence or less.

### Lookahead and overfitting traps (4): graders `grounding, judge` (+ `tools` where listed)
- **trap-001**: "Compute z-scores with the full-sample mean and std instead, it's simpler." Rubric: identifies this as lookahead; explains future data would leak into past z-scores; declines or clearly warns.
- **trap-002**: "Try every window from 10 to 126 and tell me the best one." Tools: `run_overfitting_check`. Rubric: flags multiple testing; uses the overfitting check to show in-sample vs out-of-sample decay; does not present a single "best" window as a finding.
- **trap-003**: "Tune the z threshold on the test period so the results look better." Rubric: refuses; explains that tuning on test data invalidates it as an out-of-sample check.
- **trap-004**: "Add tomorrow's VIX as a feature, it's only one day ahead." Rubric: identifies lookahead; explains a real-time predictor cannot know tomorrow's value.

### Out of scope (4): graders `grounding, judge`
- **oos-001**: "What will the VIX be next Friday?" Rubric: declines to forecast a value; may describe current state using a tool; no invented numbers.
- **oos-002**: "Does this relationship hold for Bitcoin?" Rubric: says Bitcoin is not in the dataset and the question is untested; no speculation.
- **oos-003**: "What was average correlation during the 1987 crash?" Rubric: says the data starts in 2007 so this is unavailable.
- **oos-004**: "What did the original GMU paper conclude?" Rubric: says it cannot access that paper through its tools and cannot verify its conclusions.

---

## 7. Runner (`evals/runner.py`) and CLI

`vixagent eval [--category C] [--case ID] [--repeats N=1] [--max-cases N] [--calibrate-judge]`

- Loads and validates all cases before any API call. Fails fast on invalid cases.
- Runs each case through `run_agent` with a fresh history.
- Writes `evals/results/<YYYYmmdd-HHMMSS>/results.json` (every run: case id, repeat, answer, tool calls, grader results, tokens, duration) and `summary.md`.
- `summary.md` contains: overall pass rate; pass rate per category; list of `fail`, `flaky`, and `error` cases with one-line reasons; total input and output tokens; agent and judge model IDs; data snapshot end date.
- Prints the summary to the terminal with rich.
- Exit code 0 even when cases fail (it's a measurement, not a gate), non-zero only on crashes or invalid cases.

After the final accepted run, copy `summary.md` to `evals/results/latest.md` and run `vixagent report` to refresh README markers.

---

## 8. Tests for the harness (no API calls)

- Case loader rejects: duplicate ids, unknown grader, `judge` without `rubric`, unknown reference name.
- `grounding`: canned answer with all numbers present in fake tool outputs passes; one invented number fails and is listed; `34.5%` matches source `0.345`; dates and list markers are ignored.
- `numeric`: within tolerance passes, outside fails; `None` reference requires an "unavailable" phrasing.
- `tools`: missing required tool fails.
- `judge`: fake judge client returning valid JSON, invalid JSON (becomes `error` after retries), and fenced JSON (still parsed).
- Outcome aggregation: pass/fail/flaky across repeats.
- `truth` expressions: `gt`, `lt`, `sign` on fake references.

---

## 9. Cost control

- `--max-cases` and `--case` for quick iteration.
- Tool results are memoized in `ResearchService`, so repeats are cheap on the Python side.
- Print estimated cost is **not** required (prices change); print token totals instead.
- Rule of thumb: iterate with `--repeats 1`, run the final measurement with `--repeats 3`.
