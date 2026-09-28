# AGENTS.md: Operating Rules for vix-research-agent

Read this file completely before doing anything. Then read `docs/SPEC.md`, `docs/PLAN.md`, `docs/EVALS.md`, and the step file for the current phase in `docs/steps/`.

The step file is the exact procedure: follow its steps in order, run every **Verify**, and use its **Commit** messages. If a step file and SPEC disagree, SPEC wins: stop and ask the human.

## What this repo is

A research pipeline, a tool-using LLM agent, and an eval harness that together test one question:
**do spikes in cross-asset correlation lead spikes in the VIX?**

`docs/SPEC.md` is the single source of truth. If anything here, in PLAN, or in a user message conflicts with SPEC, stop and ask the human. Never silently change the spec.

## How to work

1. Work on exactly one phase from `docs/PLAN.md` at a time, following `docs/steps/phase-N-*.md` step by step. Never start tasks from a later phase. Track one task-list item per step.
2. Your first message in a phase restates that phase's acceptance criteria.
3. Write tests alongside code, not after. Every function in `src/vixagent/features/`, `spikes.py`, `targets.py`, and `analysis/` gets unit tests.
4. At the end of a phase:
   1. Run every command in **Definition of Done** below. All must pass.
   2. Write `docs/walkthroughs/phase-N.md` (format below).
   3. **STOP** and wait for the human. Do not begin the next phase on your own.
5. If a requirement is ambiguous, contradictory, or looks statistically wrong, stop and ask. Do not guess.
6. Commit in small logical units with conventional messages: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`.

## Stack (do not change without asking)

- Python 3.11+, `src/` layout, package name `vixagent`
- Data: `yfinance`, `pandas`, `numpy`, `pyarrow` (parquet cache)
- Stats: `scipy`, `statsmodels`
- Plots: `matplotlib`, `seaborn`
- Agent: `anthropic` Python SDK (Messages API with tool use)
- Config/validation: `pydantic` v2, `pyyaml`
- CLI: `typer`, `rich`
- Dev: `pytest`, `pytest-cov`, `ruff`, `mypy`, `pandas-stubs`, `types-PyYAML`, `jsonschema`
- Optional (Phase 5 only): `streamlit`

Adding any dependency not listed here requires asking first.

## Commands

Setup:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # human fills in ANTHROPIC_API_KEY
```

**Definition of Done** (all must pass before a phase is complete):
```bash
ruff check .
ruff format --check .
mypy src
pytest -q
```

Network tests (manual only, never in CI):
```bash
pytest -q -m network
```

## Hard rules

**Statistics and data**
- **No lookahead.** Any feature value at date `t` may use only data dated `<= t`. Only `src/vixagent/targets.py` may use `shift(-h)` or any future-looking operation, and every target column name starts with `fwd_`. A test enforces this (see SPEC section 12).
- `config/preregistered.yaml` is **frozen at the end of Phase 1**. After that, never edit it. If you believe it is wrong, stop and ask.
- Never pick parameters by looking at test-period results.
- All randomness uses `settings.seed`. Results must be reproducible bit-for-bit given the same cached data.

**Testing**
- Unit tests never touch the network. Mock `yfinance.download` and the Anthropic client. Real-network tests are marked `@pytest.mark.network` and deselected by default.
- **Never weaken, skip, xfail, or delete a failing test to make it pass.** Fix the code. If you believe the test itself is wrong, stop and explain why to the human.
- Never lower a tolerance in a test without asking.

**Honesty of outputs**
- Never hardcode result numbers in `README.md` or docs. Numbers come only from generated files (`reports/results.json`, `evals/results/latest.md`) inserted by the report command.
- If the result is null (no lead effect), report it as null. Do not tweak definitions to find an effect.

**Code**
- Type hints on every public function. Docstrings on every public function, and for any statistical step the docstring explains *why*, not just what.
- `features/`, `spikes.py`, `targets.py`, `analysis/` are pure functions (DataFrame/Series in, DataFrame/Series/dataclass out). No file I/O, no network, no printing.
- I/O lives only in `data/`, `report/`, `agent/transcript.py`, `agent/client.py`, `evals/`, and `cli.py`.
- Every value returned by an agent tool must pass through `utils/jsonable.py::to_jsonable` (numpy types, `pd.Timestamp`, and `NaN` are not JSON-safe).

**Secrets and files**
- `ANTHROPIC_API_KEY` is read only from the environment or `.env`. Never print, log, or commit it.
- Gitignored: `.env`, `data/`, `runs/`, `evals/results/*/` (but commit `evals/results/latest.md`), `.venv/`, caches.
- Committed: `reports/figures/*.png`, `reports/results.md`, `reports/results.json`.

## Walkthrough format (`docs/walkthroughs/phase-N.md`)

Audience: a first-year CS student who knows Python and basic pandas but not statsmodels or the Anthropic SDK.

For each new module:
1. **Purpose** in 1 to 2 sentences.
2. **Key functions** with one line each.
3. **Worked trace**: one concrete input traced through the code with actual intermediate values.
4. **Non-obvious decisions**: why this approach, what the alternative was, what would break otherwise.

End the file with the phase's **Checkpoint Questions** copied from `docs/PLAN.md`, left unanswered for the human.
