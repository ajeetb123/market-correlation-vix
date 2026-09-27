# Docs Index

Reading order and what each file is for.

## For you (the human)
1. **`00-FOUNDATION.md`**: every concept the project uses, plus one-time Mac setup and how to drive the coding agent. Read before Phase 0.
2. **`PLAN.md`**: phase overview, acceptance criteria, and the Checkpoint Questions you answer after each phase.

## For the coding agent (and for you to review)
3. **`../AGENTS.md`** (repo root): operating rules. Read first, every session.
4. **`SPEC.md`**: the source of truth for what gets built and why. Wins every conflict.
5. **`steps/phase-N-*.md`**: the exact step-by-step procedure for each phase: files, signatures, algorithms, tests, verify commands, and commit messages.
6. **`EVALS.md`**: design of the eval harness (used in Phase 4).

## Step files
| Phase | File | Outcome |
|---|---|---|
| 0 | `steps/phase-0-scaffold.md` | Installable, CI-green skeleton with config loading |
| 1 | `steps/phase-1-data-features.md` | Data, features, spikes, no-lookahead proof, preregistration frozen |
| 2 | `steps/phase-2-analysis-report.md` | Event study, permutation, regression, OOS, grid, figures. Tag v0.1 |
| 3 | `steps/phase-3-agent.md` | Service, 9 tools, agent loop, `ask` and `chat` |
| 4 | `steps/phase-4-evals.md` | 24 cases, graders, calibrated judge, eval results |
| 5 | `steps/phase-5-polish.md` | Final README, diagram, optional Streamlit, tag v1.0 |

## Generated during the build
- `walkthroughs/phase-N.md`: the agent's explanation of what it built.
- `walkthroughs/phase-N-answers.md`: your answers to the Checkpoint Questions.

## Kickoff prompt for each phase
```
Read AGENTS.md, then docs/SPEC.md and docs/steps/phase-N-*.md.
Execute Phase N step by step. Use a task list with one item per step.
Do not skip any Verify. Stop at the end of the phase.
```
