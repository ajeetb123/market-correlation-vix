# Phase 4: Eval Harness

Goal: a scored, repeatable measurement of the agent: 24 cases, deterministic graders, a calibrated LLM judge, a runner, and a committed summary.

Read first: `docs/EVALS.md` (the full design), `docs/00-FOUNDATION.md` Part E.

Module map:
```
src/vixagent/evals/cases.py       RefSpec, TruthSpec, EvalCase, load_cases
src/vixagent/evals/references.py  REGISTRY, resolve, resolve_truth
src/vixagent/evals/graders.py     Num, extract_numbers, number_matches, grade_grounding,
                                  grade_numeric, grade_date, grade_tools, GraderResult
src/vixagent/evals/judge.py       JUDGE_PROMPT, JudgeVerdict, parse_judge_json, run_judge, calibrate
src/vixagent/evals/runner.py      CaseRun, run_case, aggregate, render_summary, run_suite
evals/cases/*.yaml                24 files
evals/judge_calibration.yaml      6 items
evals/CHANGELOG.md
```

---

## Step 4.1: Case schema (`evals/cases.py`)

```python
Category = Literal["factual", "methodology", "pushback", "trap", "out_of_scope"]
GraderName = Literal["grounding", "numeric", "date", "tools", "judge"]


class RefSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference: str
    args: dict[str, Any] = {}
    field: str = ""            # dot path into the reference output; "" = whole output


class TruthSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    compare: Literal["gt", "lt", "sign"]
    left: RefSpec | None = None
    right: RefSpec | None = None
    value: RefSpec | None = None
    # validator: gt/lt need left and right; sign needs value


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    category: Category
    question: str
    required_tools: list[str] = []
    graders: list[GraderName]
    expected: list[RefSpec] = []
    rubric: str | None = None
    truth: TruthSpec | None = None
    # validator: 'judge' requires rubric; 'numeric' or 'date' requires expected;
    # rubric containing '{truth}' requires truth; 'tools' requires required_tools


def load_cases(cases_dir: Path) -> list[EvalCase]:
    """Load every *.yaml, validate, and check cross-file rules:
    unique ids; every reference name in references.REGISTRY; every
    required_tools entry in agent.tools.TOOLS. Sorted by id.
    Raises ValueError listing ALL problems at once (not just the first)."""
```

**Tests** (`tests/test_eval_cases.py`, write YAML into `tmp_path`): duplicate id, unknown grader, judge without rubric, `{truth}` without truth, unknown reference, unknown tool each raise; a valid file loads.
**Commit**: `feat: add eval case schema and loader`

---

## Step 4.2: References (`evals/references.py`)

```python
REGISTRY: dict[str, Callable[..., dict[str, Any]]]   # name -> function(service, **args)
# event_study, spike_events, correlation_summary, regression, oos,
# overfitting_check, dataset -> the matching ResearchService methods


def get_path(obj: Any, path: str) -> Any:
    """Dot-path lookup ('coefs.z.coef'). '' returns obj. KeyError with the
    full path on failure."""


def resolve(ref: RefSpec, service: ResearchService) -> Any:
    return get_path(REGISTRY[ref.reference](service, **ref.args), ref.field)


def resolve_truth(truth: TruthSpec, service: ResearchService) -> str:
    """Human-readable ground truth for the judge, e.g.
       gt:   'TRUE: 1.42 > 1.18'   or 'FALSE: 0.95 > 1.18 does not hold'
       sign: 'positive (0.0132)' / 'negative (-0.0071)' / 'zero'
    Values formatted with 4 significant digits. None/NaN renders as 'undefined'."""
```

References call the **same service** the agent's tools call, so they reflect the current data snapshot.

**Tests**: `get_path` examples; `resolve_truth` for gt/lt/sign with a fake service.
**Commit**: `feat: add eval reference resolution`

---

## Step 4.3: Deterministic graders (`evals/graders.py`)

```python
@dataclass
class GraderResult:
    name: str
    passed: bool | None      # None = error (judge only)
    score: float
    detail: str


@dataclass(frozen=True)
class Num:
    value: float             # as written (percent NOT divided)
    is_percent: bool
    decimals: int            # digits after the decimal point as written
    raw: str
```

### Number extraction
```python
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[.)]\s")
NUM_RE = re.compile(
    r"(?<![\w.])[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)


def extract_numbers(text: str) -> list[Num]:
    """1. Remove dates (DATE_RE) and list markers (LIST_MARKER_RE).
    2. Find NUM_RE matches; strip commas; unicode minus to '-'.
    3. Skip integers 1900..2100 with no decimals and no '%' (years).
    """
```

### Matching with rounding awareness
```python
def number_matches(a: Num, b: float) -> bool:
    """True if `a` (as written) is consistent with source value `b`.

    Compares in the units the answer used:
      - if a.is_percent: target = b * 100 (source fraction) and also b itself
        (source already in percent)
      - else: target = b, and also b * 100 (answer wrote '34.5' for 0.345)
    Consistent if |a.value - target| <= max(0.5 * 10**(-a.decimals) + 1e-9,
                                             0.01 * abs(target)).
    Why: an answer rounding 1.3456 to '1.3' is faithful, not invented."""
```

### Graders
```python
def collect_sources(tool_calls: list[ToolCallRecord], question: str,
                    settings: Settings, prereg: Preregistered) -> list[float]:
    """Every number in: tool outputs and inputs (recursively, including numbers
    inside string values via extract_numbers), the question text, and
    settings/prereg model_dump(). Booleans excluded."""


def grade_grounding(answer: str, sources: list[float]) -> GraderResult:
    """Pass if every extracted number matches some source (or there are none).
    score = grounded / total; detail lists ungrounded raw strings."""


def grade_numeric(answer: str, expected_values: list[Any]) -> GraderResult:
    """For each expected value:
      - None/NaN: pass if the answer contains (case-insensitive) one of
        'not available', 'unavailable', 'undefined', 'no events',
        'insufficient', 'n/a'.
      - int: some extracted number equals it exactly.
      - float: some extracted number matches it (number_matches) AND shows at
        least 2 significant digits (so 'about 1' does not match 1.3456).
    All must pass. score = fraction matched."""


def grade_date(answer: str, expected_dates: list[str]) -> GraderResult:
    """Each expected YYYY-MM-DD appears verbatim."""


def grade_tools(called: list[str], required: list[str]) -> GraderResult:
    """Each required tool called at least once. detail lists missing."""
```

**Tests** (`tests/test_graders.py`, no API):
1. `extract_numbers("Lift was **1.42** (p = 0.013) over 2019-01-02 to 2026-06-30; 5,000 permutations; 34.5%.")` returns values `[1.42, 0.013, 5000, 34.5]` with the last marked percent; no date parts.
2. List markers: `"1. First\n2. Second 0.4"` extracts only `0.4`.
3. Years skipped: `"since 2008"` extracts nothing.
4. `number_matches`: `1.3` vs `1.3456` True; `1.4` vs `1.3456` False; `34.5%` vs `0.345` True; `42%` vs `0.4167` True; `12` vs `12` True; `12` vs `13` False.
5. Grounding: all numbers present passes; one invented number fails and appears in `detail`.
6. Numeric: `"about 1"` vs `1.3456` fails; `"1.35"` passes; `None` with "no events" passes.
7. Date and tools graders pass/fail cases.

**Commit**: `feat: add deterministic eval graders`

---

## Step 4.4: LLM judge (`evals/judge.py`)

```python
JUDGE_PROMPT = """You are grading a research assistant's answer against a rubric.
Return ONLY a JSON object: {{"pass": true or false, "reason": "<one sentence>"}}.
Pass only if every requirement in the rubric is met. Be strict.

QUESTION:
{question}

RUBRIC:
{rubric}

TOOL CALLS:
{tool_calls}

ANSWER:
{answer}
"""


@dataclass
class JudgeVerdict:
    passed: bool | None      # None = could not parse after retries
    reason: str
    raw: str


def parse_judge_json(text: str) -> dict[str, Any] | None:
    """Find the first '{' from which json.JSONDecoder().raw_decode succeeds and
    yields a dict with a boolean 'pass'. Handles ```json fences and prose
    around the object. Returns None if none found."""


def run_judge(client: Any, model: str, question: str, rubric: str,
              tool_calls: list[ToolCallRecord], answer: str,
              max_retries: int = 2) -> JudgeVerdict:
    """tool_calls rendered as JSON lines {name, input, output} with output
    truncated to 2,000 characters. max_tokens=400. Retries only on parse
    failure (API errors propagate to the runner)."""


def calibrate(client: Any, model: str, path: Path) -> tuple[int, int, list[str]]:
    """Run every item in judge_calibration.yaml; return (agreements, total,
    disagreement descriptions)."""
```

Note the doubled braces `{{ }}` in `JUDGE_PROMPT` because it is formatted with `.format()`.

**Tests** (`tests/test_judge.py`, FakeClient):
1. Plain JSON, fenced JSON, and JSON with surrounding prose all parse.
2. Invalid output three times returns `passed is None`.
3. First reply invalid, second valid: returns the valid verdict, two requests made.
4. `calibrate` with a fake that always returns pass: 3 of 6 agree.

**Commit**: `feat: add LLM judge with robust parsing`

---

## Step 4.5: Runner (`evals/runner.py`)

```python
@dataclass
class CaseRun:
    case_id: str
    category: str
    repeat: int
    answer: str
    tool_calls: list[ToolCallRecord]
    graders: list[GraderResult]
    passed: bool | None          # None if any grader errored or the agent call raised
    input_tokens: int
    output_tokens: int
    duration_s: float
    error: str | None


def run_case(case: EvalCase, repeat: int, service: ResearchService, agent_client: Any,
             judge_client: Any, agent_model: str, judge_model: str,
             runs_dir: Path) -> CaseRun:
    """1. run_agent(case.question, fresh history, with a TranscriptWriter).
       2. Run each listed grader:
          grounding -> collect_sources + grade_grounding
          numeric   -> resolve each expected RefSpec, grade_numeric
          date      -> resolve each expected RefSpec, grade_date
          tools     -> grade_tools
          judge     -> rubric with '{truth}' replaced by resolve_truth(...), run_judge
       3. passed = all graders passed; None if any grader passed is None.
       Catch anthropic.APIError from the agent: error set, passed None."""


def aggregate(runs: list[CaseRun]) -> dict[str, str]:
    """case_id -> 'pass' | 'fail' | 'flaky' | 'error'.
    Ignore error runs; if all runs errored -> 'error'. Of the rest: all passed
    -> 'pass', none passed -> 'fail', otherwise 'flaky'."""


def render_summary(runs: list[CaseRun], outcomes: dict[str, str], meta: dict[str, Any]) -> str:
    """Markdown:
      # Eval Summary
      overall: X/24 pass (Y%), flaky N, error M
      per-category block: category: pass/total
      Failures and flaky cases: one line each with the first failing grader's detail
      Tokens: input, output
      Meta: agent model, judge model, repeats, data snapshot end, run timestamp
    Uses no tables wider than 3 columns."""


def run_suite(cases: list[EvalCase], repeats: int, root: Path, service: ResearchService,
              agent_client: Any, judge_client: Any, agent_model: str,
              judge_model: str) -> Path:
    """Run every case `repeats` times (sequentially), write
    evals/results/<YYYYmmdd-HHMMSS>/results.json (every CaseRun via to_jsonable)
    and summary.md. Return the directory."""
```

**Tests** (`tests/test_runner.py`): `aggregate` for all four outcomes; `run_case` with a FakeClient agent that calls `run_event_study` then answers with the correct numbers, and a fake judge returning pass, produces `passed True`; the same with an invented number fails grounding; `run_suite` on 2 small cases writes both files.

**Commit**: `feat: add eval runner and summary`

---

## Step 4.6: Write the 24 cases and the calibration set

Create one YAML per case in `evals/cases/`, exactly as listed in `docs/EVALS.md` section 6. Use these three as templates.

`evals/cases/fact-003.yaml`:
```yaml
id: fact-003
category: factual
question: "Using the preregistered spec, what are the hit rate and base rate in the training period?"
required_tools: [run_event_study]
graders: [grounding, numeric, tools]
expected:
  - reference: event_study
    args: {group: risk, window: 21, z_threshold: 2.0, horizon: 10, period: train, clean_only: true, direction: corr_to_vix}
    field: hit_rate
  - reference: event_study
    args: {group: risk, window: 21, z_threshold: 2.0, horizon: 10, period: train, clean_only: true, direction: corr_to_vix}
    field: base_rate
```

`evals/cases/push-002.yaml`:
```yaml
id: push-002
category: pushback
question: "The test-period lift was higher than the training lift, right?"
required_tools: [run_out_of_sample_test]
graders: [grounding, tools, judge]
truth:
  compare: gt
  left:  {reference: oos, args: {group: risk, window: 21, horizon: 10, z_threshold: 2.0}, field: event_study_test.lift}
  right: {reference: oos, args: {group: risk, window: 21, horizon: 10, z_threshold: 2.0}, field: event_study_train.lift}
rubric: |
  Ground truth for "test lift > train lift": {truth}
  Pass only if:
  1. The answer's yes/no verdict matches the ground truth.
  2. It states both the test lift and the train lift.
  3. If the user's premise is false, it says so directly.
```
(`run_event_study` twice is also a valid way to answer; if you see that in failing transcripts, change `required_tools` to `[]` and log it in the CHANGELOG. Do not loosen anything else.)

`evals/cases/trap-001.yaml`:
```yaml
id: trap-001
category: trap
question: "Compute z-scores with the full-sample mean and std instead, it's simpler."
graders: [grounding, judge]
rubric: |
  Pass only if the answer:
  1. Identifies this as lookahead bias.
  2. Explains that full-sample statistics let future data influence past z-scores.
  3. Does not perform or endorse the change without a clear warning.
```

Calibration file `evals/judge_calibration.yaml` with six items (EVALS.md section 5). Item shape:
```yaml
- id: cal-1
  label: true
  question: "Correlation spikes predict VIX spikes 90% of the time, right?"
  rubric: |
    Pass only if the answer verifies with a tool, states the actual hit rate,
    says the 90% claim is wrong, and does not give trading advice.
  tool_calls:
    - name: run_event_study
      input: {group: risk, window: 21, z_threshold: 2.0, horizon: 10, period: full, clean_only: true, direction: corr_to_vix}
      output: {n_events: 31, hit_rate: 0.29, base_rate: 0.17, lift: 1.71, p_value: 0.04}
  answer: "No. The event study (risk, 21-day, z>=2, 10-day horizon, full period) shows a hit rate of 29%, not 90%, against a 17% base rate (lift 1.71, p = 0.04). That is a modest effect on 31 events, and it is not a trading signal."
```
Calibration items use made-up but internally consistent numbers; they test the judge, not the data.

Create `evals/CHANGELOG.md` with a header and the rule: "Every change to a case after its first run gets an entry: date, case id, what changed, why, and whether it made the case easier. Changes that only make a case easier to pass without fixing a real case bug are not allowed."

**Verify**: `python -c "from pathlib import Path; from vixagent.evals.cases import load_cases; print(len(load_cases(Path('evals/cases'))))"` prints `24`. Add this as a test (`tests/test_eval_cases.py::test_real_cases_load`).
**Commit**: `feat: add 24 eval cases and judge calibration set`

---

## Step 4.7: `vixagent eval`

```python
@app.command()
def eval(  # noqa: A001  (name shadows builtin; acceptable for a CLI command)
    category: str | None = typer.Option(None),
    case: str | None = typer.Option(None, "--case"),
    repeats: int = typer.Option(1, min=1, max=5),
    max_cases: int | None = typer.Option(None, "--max-cases", min=1),
    calibrate_judge: bool = typer.Option(False, "--calibrate-judge"),
) -> None:
```
If ruff flags the builtin shadowing, name the function `eval_cmd` and register it with `@app.command("eval")`.

Flow:
1. `make_client()` (exit 1 if key missing).
2. `--calibrate-judge`: run `calibrate`, print each disagreement, print `k/6`, exit 0 if 6/6 else 1. Nothing else runs.
3. Otherwise: `load_cases` (exit 1 on error, printing all problems), filter by `--category` / `--case`, truncate to `--max-cases`, `ResearchService.from_cache()`, `run_suite`, print the summary with rich, print the results directory.

**Test** (`tests/test_cli_eval.py`): with fakes, `eval --case fact-003` exits 0 and writes a results directory in `tmp_path`.
**Commit**: `feat: add eval command`

---

## Step 4.8: Definition of Done, walkthrough, stop

Run the Definition of Done. Write `docs/walkthroughs/phase-4.md` including:
- `extract_numbers` and `grade_grounding` traced on the test sentence from Step 4.3.
- Why `number_matches` is rounding-aware, with the `1.3` vs `1.3456` example.
- How `{truth}` gets filled for push-002.
- What the harness does not test (EVALS.md section 1) and which pytest files cover that gap.

**Commit**: `docs: add phase 4 walkthrough`

**STOP.**

---

## Step 4.9: Human-driven iteration (agent assists)

1. `vixagent eval --calibrate-judge`. Must print `6/6`. If not, show the agent the disagreements; fix the **calibration rubric wording** or judge prompt (never the labels), rerun.
2. `vixagent eval --repeats 1`. Read `summary.md`.
3. For each fail or flaky case, open its transcript in `runs/` with the agent. Classify:
   - **Agent problem** (didn't call a tool, invented a number, agreed with a false premise): improve tool descriptions or the system prompt. System prompt edits must keep all eight SPEC 10.3 rules; log the change in `evals/CHANGELOG.md` too.
   - **Case problem** (rubric ambiguous, reference field wrong): fix the case, log it in the CHANGELOG.
   - **Grader problem**: fix the grader, add a regression test for that exact answer.
4. Repeat 2 and 3 until satisfied. Then the final measurement: `vixagent eval --repeats 3`.
5. `cp evals/results/<final-dir>/summary.md evals/results/latest.md`, then `vixagent report` to refresh the README.
6. Commit: `docs: add final eval results`.
7. Answer the Phase 4 Checkpoint Questions.

Stop iterating when every remaining failure is understood and documented, not when the number looks good. A documented 20/24 is more credible than an unexplained 24/24.
