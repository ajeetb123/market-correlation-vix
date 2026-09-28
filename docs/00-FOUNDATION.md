# 00 FOUNDATION: Read This Before Building Anything

This file is for **you** (the human), not the coding agent. It explains every concept the project uses, in the order you'll meet them, plus the one-time setup on your Mac. If you understand this file, you'll understand every line the agent writes.

Time to read carefully: about 60 to 90 minutes. Do it once, then come back to sections as the phases reach them.

---

## Part A: The Research Question

### A1. What the VIX is
The VIX is a number published by Cboe that measures how much volatility options traders expect in the S&P 500 over the next 30 days, expressed as an annualized percentage.

- Calm markets: VIX around 12 to 18.
- Stress: 25 to 40.
- Crises: 2008 peaked near 80; March 2020 peaked above 80 intraday.

Important: the VIX is a **level**, not a price you can buy. We never compute "VIX returns" for correlation. We only look at its level and how fast the level jumps.

### A2. What "correlation spike" means
Take daily returns of several assets (stocks, international stocks, high-yield bonds). In normal times they move somewhat independently. In a panic, investors sell everything risky at once, so the assets start moving together. Their pairwise correlations rise toward 1.

We summarize this with one number per day: the **average pairwise correlation** over the last `W` trading days. With 6 assets there are 15 pairs (6 choose 2), so the number is the average of 15 correlations.

### A3. The hypothesis
Your GMU research asked whether correlation behavior relates to VIX behavior. This project sharpens that into a testable claim:

> After average correlation spikes, a VIX spike becomes more likely within the next `h` trading days than it normally is.

If true, correlation would be a **leading indicator**. If false, correlation and the VIX just move together (**co-movement**), which is less interesting but still a legitimate finding.

### A4. Why a null result is fine
Most honest research tests find nothing. Interviewers are far more impressed by "I preregistered my test, it came back null, and here's how I know it's really null" than by a suspicious positive result. The project is scored on rigor.

---

## Part B: The Math, One Piece at a Time

### B1. Log returns
`r_t = ln(P_t / P_{t-1})`

Why logs instead of simple percent change: log returns add up over time (a +10% and -10% simple return don't cancel, but log returns do behave symmetrically), and they're the standard input for correlation work.

Example: price goes 100 to 102. Simple return 2%. Log return `ln(1.02) = 0.0198`.

### B2. Pearson correlation
Measures how linearly two series move together, from -1 (opposite) to +1 (lockstep). A **rolling** correlation recomputes it every day using only the most recent `W` days. That's what makes it a time series.

### B3. Z-score
`z = (x - mean) / std`

"How many standard deviations is today's value above normal?" A z of 2 means unusually high (about the top 2.5% if the data were normal).

The key detail: **which** mean and std. We use the previous 252 trading days (about one year), **excluding today**. Section C explains why that matters so much.

### B4. Median vs mean
For the VIX baseline we use the **median** of the prior 20 days. If one day in that window was a freak 60, the mean jumps a lot; the median barely moves. Robust baselines prevent one outlier from hiding the next spike.

### B5. Spike days vs events (declustering)
A crisis produces 10 or 20 consecutive "spike days". If you count each as a separate event, you'd claim 20 data points when you really observed **one** crisis. That fakes statistical significance.

So we **decluster**: a spike day only counts as a new **event** if there was no spike day in the previous 20 trading days.

### B6. Hit rate, base rate, lift
- **Hit rate**: of all correlation events, what fraction were followed by a VIX spike within `h` days?
- **Base rate**: of **all** days, what fraction were followed by a VIX spike within `h` days?
- **Lift** = hit rate / base rate.

Lift of 1 means correlation events tell you nothing. Lift of 2 means a VIX spike is twice as likely as usual after a correlation event.

Why the base rate matters: VIX spikes happen maybe 10% of the time within some window. If your hit rate is 12%, that sounds like a finding, but it's barely above chance.

### B7. Permutation test (how we get a p-value)
Question: "If correlation events were placed at random, how often would we see a hit rate this high?"

Procedure: slide the whole event pattern along the timeline by a random offset (wrapping around the end), recompute the hit rate, repeat 5,000 times. The p-value is the fraction of shuffled results at least as good as the real one.

Why slide (circular shift) instead of scattering events randomly: sliding keeps the events' spacing and clustering intact. Market events are clumpy; a test that ignores that is too optimistic.

### B8. Predictive regression and HAC errors
A second, continuous test: regress "how much the VIX changes over the next `h` days" on "today's correlation z-score".

Problem: if `h = 10`, today's target window overlaps 9 days with tomorrow's. Neighboring errors are correlated, so ordinary standard errors are too small and p-values look better than they are. **HAC** (Newey-West) standard errors correct for that. `maxlags = h` tells it how far the overlap reaches.

### B9. Train/test split and out-of-sample R squared
- **Train** (2007 to 2018): you're allowed to look at this.
- **Test** (2019 to mid-2026): the honest exam. Includes COVID and 2022.

`R2_oos` compares your model's predictions on the test period to the dumbest forecast possible: "the future VIX change equals the average change from the training period." Positive means you beat the dumb forecast. Zero or negative means you didn't. Many real-world predictors come out negative.

### B10. Preregistration and the overfitting check
If you try 24 parameter combinations, one will look good by luck. **Preregistration** means you write down your primary test **before** seeing results, commit it to git (the timestamp is proof), and report that as the headline no matter what.

The **overfitting check** then shows what happens if you cheat: pick the best-looking combination on training data, and watch how much worse it does on test data. That decay is a real, demonstrable lesson.

---

## Part C: Lookahead Bias (the #1 way research projects go wrong)

**Lookahead bias** = using information that wasn't available at the time.

Classic mistake: standardize the whole 2007 to 2026 series with its full-sample mean and std. Now the z-score for a day in 2009 is influenced by 2020's COVID spike. A trader in 2009 couldn't have known that. Results look great and are fake.

How this project prevents it, in three layers:
1. **Design**: every feature uses `shift(1)` (yesterday and earlier) for its baseline.
2. **Isolation**: only one file, `targets.py`, is allowed to look forward, and everything it produces is named `fwd_...`.
3. **Automated test**: compute features on the full data, then again on data chopped off at a random date. If any value before the chop point changes, the feature was peeking at the future, and the test fails.

In pandas:
- `x.shift(1)` means "yesterday's value on today's row". Safe.
- `x.shift(-10)` means "the value 10 days in the future on today's row". Only allowed in `targets.py`.

---

## Part D: The Agent

### D1. What an "agent" is here
A loop:
1. You ask a question.
2. Claude reads the question plus a list of **tools** (functions it may call, each described with a JSON Schema).
3. Claude either answers, or replies with a `tool_use` block: "call `run_event_study` with these arguments."
4. **Your Python code** runs that function and sends back a `tool_result`.
5. Repeat until Claude answers with plain text.

Claude never runs code itself. It only asks your code to run functions, and it only sees what your functions return. That's what makes the numbers trustworthy: they come from your tested pipeline, not from the model's memory.

### D2. Message shape (you'll be asked about this)
```
user:       "What's the headline result?"
assistant:  [text: "Let me check."] [tool_use id=A name=get_preregistered_spec]
user:       [tool_result for id=A]
assistant:  [tool_use id=B name=run_event_study ...]
user:       [tool_result for id=B]
assistant:  [text: final answer]
```
Rules the API enforces: the assistant message containing `tool_use` must be in the history before its `tool_result`, and **every** `tool_use` id must get a `tool_result` in the very next user message.

### D3. Why the system prompt is strict
The prompt tells Claude to never state a number it didn't get from a tool, to verify user claims, and to flag lookahead and overfitting. The eval harness then **checks** whether it actually behaves that way. Instructions alone prove nothing; measurement does.

---

## Part E: Evals

An eval is a test suite for model behavior. Each case is a question plus a way to grade the answer:

- **Grounding grader** (code): extracts every number from the answer and checks it appears in some tool output. Catches invented numbers.
- **Numeric grader** (code): checks the answer contains the correct value, computed fresh from the pipeline.
- **Tools grader** (code): checks the agent called the tools it needed.
- **Judge grader** (LLM): for judgment questions ("did it push back on the false premise?"), a second Claude call grades against a written rubric.

We **calibrate** the judge first: 6 answers where we already know the right grade. If the judge disagrees with any, we don't trust it.

We run each case 3 times because models are not perfectly consistent. A case that passes sometimes is marked **flaky**.

---

## Part F: One-Time Setup on Your Mac

Do this once before Phase 0.

1. **Python 3.11 or newer**
   ```bash
   brew install python@3.12
   python3.12 --version
   ```
   If you use 3.12, replace `python3.11` with `python3.12` in the setup commands. Anything 3.11+ works.

2. **Git identity** (if not already set)
   ```bash
   git config --global user.name "Ajeet Bondugula"
   git config --global user.email "ajeetb@vt.edu"
   ```

3. **GitHub CLI**: already installed and authenticated as `ajeetb123`. Check with `gh auth status`.

4. **Anthropic API key**
   - Create a key at console.anthropic.com, add a small amount of credit, and set a monthly spend limit (for example $20) in the console.
   - You'll paste it into `.env` in Phase 3. Never commit it, never paste it into chat.
   - Rough cost: Phases 3 and 4 use a few dollars total with a Sonnet-class model if you iterate with `--repeats 1`.

5. **Coding agent**: the operating rules live in `AGENTS.md` at the repo root. Point your coding agent at it at the start of every session.

---

## Part G: How to Drive the Agent

For each phase:
1. Start a fresh agent session in the repo.
2. Prompt: `Read AGENTS.md, then docs/steps/phase-N-*.md. Execute Phase N step by step. Stop at the end of the phase.`
3. Watch the task list. When a step's **Verify** fails, the agent must fix it before moving on.
4. When it stops, read `docs/walkthroughs/phase-N.md`, then answer the Checkpoint Questions **in your own words** in `docs/walkthroughs/phase-N-answers.md`.
5. If you can't answer one, ask the agent to explain that part again. Don't move on until you can.
6. Reply `Phase N approved. Begin Phase N+1.`

If the agent proposes changing SPEC, a test, or a tolerance, the answer is no unless you understand exactly why and agree.
