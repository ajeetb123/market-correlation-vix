# SPEC: VIX Research Agent

Source of truth for the project. Put this file at `docs/SPEC.md`.

---

## 1. Goal

Build three things that fit together:

1. **Research pipeline**: tests whether spikes in average cross-asset correlation lead spikes in the VIX, with pre-registered parameters, a train/test split, and permutation-based significance.
2. **Research agent**: an LLM (Claude via the Anthropic API) that answers questions about the research *only* by calling the pipeline's functions as tools, and pushes back on false premises, lookahead bias, and overfitting.
3. **Eval harness**: a scored test suite measuring whether the agent's answers are numerically grounded, correct, and appropriately skeptical.

### Hypotheses
- **H1**: After a correlation spike event, the probability of a VIX spike within the next `h` trading days is higher than the unconditional base rate.
- **H0**: It is not higher (lift <= 1).

A null result is an acceptable, publishable outcome. The project's value is the rigor, not a positive finding.

### Non-goals
- No trading strategy, backtest of P&L, or investment advice.
- No intraday data. Daily close only.
- No live deployment beyond an optional local Streamlit app.

---

## 2. Repository layout

```
vix-research-agent/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── .github/workflows/ci.yml
├── config/
│   ├── settings.yaml
│   └── preregistered.yaml
├── src/vixagent/
│   ├── __init__.py
│   ├── config.py              # load + validate YAML into pydantic models
│   ├── data/
│   │   ├── fetch.py           # yfinance download, retries, column normalization
│   │   ├── cache.py           # parquet cache + metadata JSON, network-free loader
│   │   ├── validate.py        # data sanity checks
│   │   └── align.py           # per-group inner alignment
│   ├── features/
│   │   ├── returns.py
│   │   ├── correlation.py
│   │   ├── zscore.py
│   │   ├── vix.py             # VIX ratio, momentum, log level
│   │   └── frame.py           # build_frame: all features for one group/window
│   ├── targets.py             # ONLY module allowed to look forward
│   ├── spikes.py
│   ├── periods.py             # period slicing helpers
│   ├── analysis/
│   │   ├── frames.py          # FrameStore: cached frames per group/window
│   │   ├── event_study.py
│   │   ├── permutation.py
│   │   ├── regression.py
│   │   ├── oos.py
│   │   └── grid.py
│   ├── report/
│   │   ├── figures.py
│   │   ├── results.py
│   │   └── readme.py          # inserts generated results into README markers
│   ├── agent/
│   │   ├── service.py         # cached facade over analysis; what tools call
│   │   ├── tools.py           # tool schemas + dispatch
│   │   ├── prompts.py         # system prompt
│   │   ├── methodology.py     # canonical methodology text per topic
│   │   ├── client.py          # Anthropic client factory, model IDs
│   │   ├── loop.py            # agent loop
│   │   └── transcript.py      # JSONL run logging
│   ├── evals/
│   │   ├── cases.py           # load + validate case YAML
│   │   ├── references.py      # reference-answer functions
│   │   ├── graders.py         # grounding, numeric, tools, judge
│   │   ├── judge.py
│   │   └── runner.py
│   ├── utils/jsonable.py
│   └── cli.py
├── evals/
│   ├── cases/*.yaml
│   ├── judge_calibration.yaml
│   ├── CHANGELOG.md
│   └── results/latest.md
├── reports/
│   ├── figures/
│   ├── results.md
│   └── results.json
├── docs/
│   ├── README.md              # docs index
│   ├── 00-FOUNDATION.md       # concepts primer for the human
│   ├── SPEC.md
│   ├── PLAN.md
│   ├── EVALS.md
│   ├── steps/phase-N-*.md     # step-by-step procedure per phase
│   └── walkthroughs/
├── tests/
│   ├── conftest.py
│   ├── fakes.py               # FakeClient for the Anthropic API
│   ├── fixtures/synthetic.py
│   └── test_*.py
├── app/streamlit_app.py       # Phase 5, optional
├── data/                      # gitignored
└── runs/                      # gitignored
```

---

## 3. Project configuration

### `pyproject.toml` essentials
- `requires-python = ">=3.11"`
- Dependencies with minimum versions: `yfinance>=0.2.54`, `pandas>=2.2`, `numpy>=1.26`, `scipy>=1.12`, `statsmodels>=0.14`, `matplotlib>=3.8`, `seaborn>=0.13`, `pyarrow>=15`, `anthropic>=0.40`, `pydantic>=2.6`, `pyyaml>=6`, `typer>=0.12`, `rich>=13`, `python-dotenv>=1.0`
- Extras: `dev = [pytest>=8, pytest-cov>=5, ruff>=0.5, mypy>=1.10, pandas-stubs, types-PyYAML, jsonschema>=4]`, `app = [streamlit>=1.35]`
- Script: `vixagent = "vixagent.cli:app"`
- pytest: `addopts = "-m 'not network'"`, register marker `network`
- ruff: `line-length = 100`, select `E, F, I, B, UP, SIM`
- mypy: `ignore_missing_imports = true`, not strict

### `.env.example`
```
ANTHROPIC_API_KEY=
VIX_AGENT_MODEL=claude-sonnet-5
VIX_JUDGE_MODEL=claude-sonnet-5
```
Model IDs are configurable. Verify current IDs at https://docs.claude.com before first run.

### `config/settings.yaml`
```yaml
seed: 42

data:
  start: "2007-05-01"          # HYG begins 2007-04-11; this lets warmup finish before the 2008 crisis
  snapshot_end: "2026-06-30"   # inclusive. yfinance `end` is exclusive, so pass snapshot_end + 1 day
  vix_ticker: "^VIX"
  cache_path: "data/raw/prices.parquet"

universe:
  risk: [SPY, QQQ, IWM, EFA, EEM, HYG]
  all:  [SPY, QQQ, IWM, EFA, EEM, HYG, TLT, IEF, LQD, GLD]

periods:
  train: ["2007-05-01", "2018-12-31"]
  test:  ["2019-01-01", "2026-06-30"]
  # "full" = data.start through data.snapshot_end

features:
  z_lookback: 252

vix_spike:
  median_lookback: 20
  ratio_threshold: 1.30
  cooldown: 20

corr_spike:
  cooldown: 20

event_study:
  clean_pre_window: 5
  n_permutations: 5000

grid:
  windows: [21, 63]
  horizons: [5, 10, 20]
  z_thresholds: [1.5, 2.0]
  groups: [risk, all]
```
If the original GMU study used a different asset list, the human may replace `universe` **before** Phase 1 ends. Each group must have at least 5 tickers.

### `config/preregistered.yaml` (frozen at end of Phase 1)
```yaml
primary:
  group: risk
  window: 21
  z_threshold: 2.0
  horizon: 10
  clean_only: true
  direction: corr_to_vix
  alpha: 0.05
note: >
  Fixed before any results were computed. The headline result reported in the README
  is this specification evaluated on the test period. All other grid results are exploratory.
```

`config.py` loads both files into pydantic models and validates: dates parse, train end < test start, groups non-empty, thresholds positive, `primary` values are members of the grid.

---

## 4. Data layer

### `data/fetch.py`
`fetch_prices(tickers: list[str], start: str, end_inclusive: str) -> pd.DataFrame`
- Call `yf.download(tickers, start=start, end=end_inclusive + 1 day, auto_adjust=True, progress=False, threads=False, group_by="column")`.
- Extract the `Close` field. Handle both MultiIndex columns `(field, ticker)` and flat columns. Result: columns = tickers, index = tz-naive `DatetimeIndex` named `date`.
- Retry up to 3 times with exponential backoff (2s, 4s, 8s) on exceptions, including rate-limit errors.
- Raise `DataError` if any ticker is missing or entirely NaN.

VIX is fetched in the same call. It is a level, not a price to compute returns from.

### `data/cache.py`
- `load_or_fetch(settings, refresh: bool = False) -> pd.DataFrame`
- Cache: `data/raw/prices.parquet` plus `data/raw/prices.meta.json` with `fetched_at`, `yfinance_version`, `tickers`, `start`, `end_inclusive`.
- Reuse cache only if metadata tickers and date range match settings exactly; otherwise refetch.

### Alignment
- For a group `g`, the working frame is the columns `universe[g] + [vix_ticker]`, keeping only dates where **all** of them are non-NaN (inner alignment). **No forward filling.**
- Record `rows_dropped_in_alignment`.

### `data/validate.py`
Errors:
- Index not monotonic increasing or has duplicates.
- Any price <= 0.
- VIX outside [5, 100].

Warnings (logged, not raised):
- Any ETF daily absolute log return > 0.5.
- Gaps > 5 calendar days between consecutive rows.

---

## 5. Features (all backward-looking)

Notation: `t` is a trading-day row index. `P_{i,t}` is the adjusted close of asset `i`. `V_t` is the VIX close.

### 5.1 Log returns (`features/returns.py`)
`r_{i,t} = ln(P_{i,t} / P_{i,t-1})`. First row is NaN and dropped.

### 5.2 Average pairwise correlation (`features/correlation.py`)
`avg_pairwise_corr(returns: pd.DataFrame, window: int) -> pd.Series`

`C_t = mean over pairs i<j of Pearson corr(r_i, r_j)` over rows `t-W+1 .. t` inclusive. NaN until `W` returns exist.

Implementation: average of `returns[a].rolling(W).corr(returns[b])` over all pairs `a < b` (equivalent to averaging the strict upper triangle of the rolling correlation matrix), clipped to [-1, 1]. NaN if any pair is undefined. Must be correct for any number of assets `n >= 2`.

Also provide `pairwise_corr_matrix(returns, end_date, window) -> pd.DataFrame` for the heatmap figure.

### 5.3 Correlation z-score (`features/zscore.py`)
`trailing_zscore(x: pd.Series, lookback: int) -> pd.Series`

`z_t = (x_t - mean(x_{t-L} .. x_{t-1})) / std(x_{t-L} .. x_{t-1})`, `L = 252`, sample std (`ddof=1`).

Implementation: `mu = x.shift(1).rolling(L).mean()`, `sd = x.shift(1).rolling(L).std()`. If `sd == 0`, result is NaN.

**Why the trailing window**: using full-sample mean/std would let 2020 data influence the 2009 z-score. That is lookahead.

### 5.4 VIX ratio
`R_t = V_t / median(V_{t-20} .. V_{t-1})`, implemented as `V / V.shift(1).rolling(20).median()`.

**Why median**: one extreme day does not distort the baseline the way a mean would.

---

## 6. Targets (`targets.py`, the only forward-looking module)

`fwd_log_vix_change(V: pd.Series, h: int) -> pd.Series`
`fwd_log_vix_change_h_t = ln(V_{t+h}) - ln(V_t)`. Last `h` rows are NaN.

Column names in any DataFrame produced here start with `fwd_`.

`fwd_any_within(flags: pd.Series, h: int) -> pd.Series`: True at `t` if any flag is True in rows `t+1 .. t+h`. Used for event-study hits. It lives here because it looks forward.

---

## 7. Spike detection (`spikes.py`)

### Spike days
- VIX spike day: `R_t >= 1.30`
- Correlation spike day: `z_t >= z*` (z* from params)

### Events (declustering)
`events_from_days(spike_days: pd.Series[bool], cooldown: int) -> pd.Series[bool]`

`t` is an event if `spike_day_t` is true **and** no spike day occurred in rows `t-cooldown .. t-1`. Cooldown = 20 for both kinds.

**Why declustering**: a crisis produces many consecutive spike days. Counting each as an independent event inflates the sample size and fakes significance.

Example: spike days at rows 10, 11, 12, 40, 45 with cooldown 20 give events at rows 10 and 40 only (45 is within 20 rows of spike day 40).

---

## 8. Analyses

### 8.1 Periods (`periods.py`)
Features are computed on the **full** aligned data, then sliced to a period. Using pre-period data as warmup is not lookahead because features are backward-looking.

For a period `[p_start, p_end]` and horizon `h`, the **eligible days** `D` are rows with `p_start <= date <= p_end`, a non-NaN feature value, **and** row index `t + h` still on or before `p_end`. This last condition acts as a natural embargo between train and test.

### 8.2 Event study (`analysis/event_study.py`)

`run_event_study(frame, params) -> EventStudyResult`

Direction `corr_to_vix` (the primary test):
- `E` = correlation events within `D`.
- `clean_only=True`: drop events with any VIX spike day in rows `t-5 .. t` (inclusive).
- `hit(t)` = any VIX spike day in rows `t+1 .. t+h`. **Strictly after `t`.**
- `hit_rate` = mean of `hit` over `E`.
- `base_rate` = mean of `hit` over all days in `D`. If `clean_only`, over days in `D` passing the same clean filter.
- `lift = hit_rate / base_rate`. NaN if `base_rate == 0`.

Direction `vix_to_corr` (reverse check): swap roles. Events are VIX events, hits are correlation spike days, and the clean filter excludes events with a correlation spike day in `t-5 .. t`.

**Why the clean filter**: in a crash, correlations and VIX jump on the same day. Without the filter, contemporaneous co-movement masquerades as prediction.

**Why the reverse check**: if VIX spikes predict correlation spikes as strongly, the relationship is co-movement, not a lead.

`EventStudyResult` fields: `n_events, n_hits, hit_rate, base_rate, lift, p_value, n_permutations, event_dates, hit_flags, warning`. `warning = "insufficient events (<5)"` when `n_events < 5`.

Hand-checkable example (must be a unit test): a period of 30 rows (0..29) with all features non-NaN, correlation events at rows 5 and 15, VIX spike days at rows 8 and 25, `h = 5`, `clean_only=False`. Event 5 covers rows 6..10 (includes 8, hit). Event 15 covers 16..20 (miss). `hit_rate = 0.5`. Base rate: days whose window `t+1..t+5` includes 8 are rows 3..7 (5 days), and rows 20..24 include 25 (5 days). Eligible days are rows 0..24 (row index + 5 <= 29), so `base_rate = 10 / 25 = 0.4`, `lift = 1.25`.

### 8.3 Permutation test (`analysis/permutation.py`)

One-sided p-value for `hit_rate`:
1. Take the event indicator over `D` **before** the clean filter.
2. For `n_perm` iterations, circularly shift it by an integer `k` drawn uniformly from `[h+1, |D|-h-1]` using `np.random.default_rng(seed)`.
3. Re-apply the clean filter, recompute `hit_rate`.
4. `p = (1 + count(perm_hit_rate >= observed)) / (1 + n_perm)`.

**Why circular shifts**: they preserve the clustering and autocorrelation of events. A standard t-test assumes independent events, which is false for market data.

### 8.4 Predictive regression (`analysis/regression.py`)

`y_t = fwd_log_vix_change_h_t`

- Base spec: `y_t = a + b * z_t + e_t`
- Controls spec: adds `ln(V_t / V_{t-5})` and `ln(V_t)`

OLS via `statsmodels` with `cov_type="HAC", cov_kwds={"maxlags": h}`. Rows restricted to eligible days `D`.

**Why HAC**: `h`-day forward windows overlap, so residuals are autocorrelated and ordinary standard errors are too small.

**Why controls**: tests whether correlation adds information beyond what VIX momentum and level already contain.

Output: `n`, per-coefficient `coef, t_hac, p_hac`, `r2`, `hac_maxlags`.

### 8.5 Out-of-sample (`analysis/oos.py`)
- Fit the **base spec** (z-score only) on train eligible days. Predict on test eligible days. The question is whether the correlation z-score alone forecasts VIX changes better than the historical average.
- `R2_oos = 1 - SUM((y - y_hat)^2) / SUM((y - mean_train_y)^2)`, where `mean_train_y` is the train-period mean of `y` (historical-mean benchmark).
- Also run the event study with the given params on train and on test separately.

`R2_oos <= 0` means the model does no better than guessing the historical average. Report it plainly.

### 8.6 Overfitting check (`analysis/grid.py`)
- Run the event study (`corr_to_vix`, `clean_only=True`) for every combination in `grid` on train.
- Among combinations with `n_events >= 5`, select the one with the highest train `lift`.
- Report that combination's train lift and **test** lift, next to the preregistered spec's train and test lift.
- Report the full grid (train and test) as exploratory.
- Report how many combinations were tested (`2 groups x 2 windows x 2 thresholds x 3 horizons = 24`).

**Why**: the best in-sample combination usually decays out of sample. Showing that decay honestly is one of the project's main points.

---

## 9. Report (`vixagent report`)

Writes `reports/figures/*.png` (dpi 150), `reports/results.json`, `reports/results.md`, and updates README between markers.

### Figures
1. `timeseries.png`: VIX (left axis) and `C_t` for risk group, W=21 (right axis). Vertical lines at VIX events. Test period shaded.
2. `corr_heatmaps.png`: 1x3 seaborn heatmaps of the `all`-group pairwise correlation matrix, window 63, ending on: (a) the date of minimum VIX in the full sample, (b) the date of maximum VIX in 2008, (c) the date of maximum VIX in 2020. `cmap="RdBu_r"`, `vmin=-1`, `vmax=1`, `annot=True`, `fmt=".2f"`. Dates selected programmatically and shown in subplot titles.
3. `event_study.png`: grouped bars of hit rate vs base rate for horizons 5, 10, 20 at the preregistered group/window/threshold. Two panels: train, test. Each bar pair labeled with `n_events`.
4. `grid_lift.png`: two heatmaps (train, test) of lift for group `risk`. Rows = (window, z) combinations, columns = horizons. Annotated, `center=1.0`.

### `results.json`
Keys: `dataset`, `preregistered` (train and test event study, reverse-direction test on test), `regression` (base and controls, full period), `oos`, `overfitting_check`, `grid`, `generated_at`, `data_snapshot_end`.

### `results.md`
Generated sections: Dataset, Headline (preregistered, test period), Reverse Direction Check, Regression, Out-of-Sample, Overfitting Check, Full Grid (exploratory), Limitations.

Limitations is static text, including: single data source (Yahoo), daily data only, the VIX spike threshold is a modeling choice, 24 grid combinations imply multiple testing, correlation spikes and VIX spikes can share a common cause, results are not a trading strategy.

### README markers
`report/readme.py` replaces content between `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->` with the headline and a short summary built from `results.json`. Same for `<!-- EVALS:START -->` / `<!-- EVALS:END -->` from `evals/results/latest.md`.

---

## 10. Agent

### 10.1 Service layer (`agent/service.py`)
- `ResearchService` loads the cached dataset once (raises a clear error telling the user to run `vixagent pull` if no cache).
- Memoizes results in a dict keyed by method name and primitive args, so repeated tool calls return the identical object.
- All outputs pass through `to_jsonable`.

### 10.2 Tools (`agent/tools.py`)
Each tool has a JSON Schema input, a pydantic model validating args, and returns a dict. Invalid args produce a `tool_result` with `is_error: true` and a message stating the allowed range. Tool outputs never include full time series; date lists are capped at 50 with `"truncated": true`.

Shared param bounds:
- `group`: `"risk" | "all"`
- `window`: integer 10 to 126
- `horizon`: integer 1 to 60
- `z_threshold`: number 0.5 to 4.0
- `period`: `"train" | "test" | "full"`

| # | Tool | Inputs | Output |
|---|------|--------|--------|
| 1 | `describe_dataset` | none | tickers by group, VIX ticker, start, end, n trading days, periods, rows dropped in alignment, data source statement |
| 2 | `get_correlation_summary` | group, window, period | mean, std, min + date, max + date, latest value + date, latest z |
| 3 | `get_spike_events` | kind (`vix`/`corr`), period; for `corr`: group, window, z_threshold | n_events, dates, truncated, definition string |
| 4 | `run_event_study` | group, window, z_threshold, horizon, period, clean_only, direction | EventStudyResult (event_dates capped) |
| 5 | `run_predictive_regression` | group, window, horizon, period, include_controls | `{n, r2, hac_maxlags, coefs: {const, z, vix_mom_5d, log_vix}}`, each coef `{coef, t_hac, p_hac}` (control keys only when included) |
| 6 | `run_out_of_sample_test` | group, window, horizon, z_threshold | `{r2_oos, n_train, n_test, event_study_train, event_study_test}` (event studies use `clean_only=true`, `corr_to_vix`) |
| 7 | `run_overfitting_check` | none | Section 8.6 output |
| 8 | `get_preregistered_spec` | none | file contents plus the freeze note |
| 9 | `get_methodology` | topic enum: `data, correlation, zscore, vix_spike, event_study, permutation_test, regression, out_of_sample, lookahead, limitations` | canonical text from `agent/methodology.py`, consistent with this SPEC |

Tool `description` strings must state what the tool computes, its key definitions, and when to use it.

### 10.3 System prompt (`agent/prompts.py`)
Use this text verbatim, filling `{start}`, `{end}`, `{groups}` from settings. In Phase 4 only, it may be amended with the human's approval to fix eval failures, provided all eight rules remain; every amendment is logged in `evals/CHANGELOG.md` and copied back into this section.

```
You are a skeptical quantitative research assistant for one project: testing whether
spikes in average cross-asset correlation lead spikes in the VIX. Data: daily closes
from {start} to {end}, asset groups {groups}, via Yahoo Finance.

Rules:
1. Every number you state must come from a tool result in this conversation. Never
   estimate, recall, or compute numbers yourself. If no tool can produce a number,
   say so.
2. When the user asserts a fact about the data or results, verify it with a tool
   before agreeing. If it is wrong, say so directly and give the correct value.
3. Flag lookahead bias, overfitting, multiple testing, and small sample sizes
   whenever they are relevant, including when the user proposes them.
4. Distinguish prediction from causation. A lead relationship is not a cause.
5. The preregistered specification is the headline result. Treat other parameter
   choices as exploratory and say so.
6. When a question is outside the dataset or tools (other assets, other dates,
   forecasts of future values, external papers), say it is untested or unavailable.
   Do not speculate.
7. Do not give trading or investment advice.
8. Be concise. Lead with the answer, then the key caveat. State which tool and
   parameters produced each number.
```

### 10.4 Loop (`agent/loop.py`)
`run_agent(question: str, history: list | None, client, settings) -> AgentResult`

1. `client.messages.create(model=VIX_AGENT_MODEL, max_tokens=4096, system=SYSTEM_PROMPT, tools=TOOL_SCHEMAS, messages=messages)`
2. Append the **full** assistant `content` (including `tool_use` blocks) to `messages`.
3. If `stop_reason == "tool_use"`: execute every `tool_use` block, then append **one** user message containing a `tool_result` for **each** `tool_use_id`, in the same order. Content is `json.dumps(result)`. Errors set `is_error: true`.
4. Repeat until `stop_reason` is `end_turn` or 12 iterations. On the limit, return a final answer stating the limit was reached.
5. `AgentResult`: `final_text`, `tool_calls` (name, input, output, is_error, duration_ms), `iterations`, `usage` (summed input/output tokens), `messages`.

The Anthropic client is injected so tests can pass a fake.

### 10.5 Transcripts (`agent/transcript.py`)
Each run writes `runs/<YYYYmmdd-HHMMSS>-<slug>.jsonl`, one event per line: `user`, `assistant`, `tool_call`, `tool_result`, `final`, `usage`. Serialize SDK blocks with `.model_dump()`. Never write the API key.

---

## 11. CLI (`cli.py`, Typer app `vixagent`)

| Command | Behavior |
|---|---|
| `vixagent pull [--refresh]` | fetch or load cache, validate, print summary table |
| `vixagent report` | run Section 9 |
| `vixagent ask "<question>" [--show-tools]` | single-turn agent answer |
| `vixagent chat [--show-tools]` | multi-turn REPL with rich output; `/tools` toggles tool display, `/reset` clears history, `/exit` quits |
| `vixagent eval [--category C] [--case ID] [--repeats N] [--max-cases N] [--calibrate-judge]` | see EVALS.md |

`ask`, `chat`, and `eval` exit with a clear message if `ANTHROPIC_API_KEY` is missing.

---

## 12. Testing requirements

Synthetic fixtures (`tests/fixtures/synthetic.py`): functions that generate price DataFrames from correlated random walks with specified correlation regimes, plus a VIX-like series, with controllable planted spikes. All use explicit seeds.

Required tests (minimum):

**Data**
- `fetch_prices` handles MultiIndex and flat columns (mock `yf.download`).
- Missing ticker raises `DataError`.
- Alignment drops non-overlapping dates and reports the count.
- Validation catches non-monotonic index, non-positive prices, out-of-range VIX.

**Features**
- `avg_pairwise_corr`: identical series gives 1.0; three series generated with true correlation 0.8 and W=252 gives mean within 0.05 of 0.8; works for n=2 and n=10.
- `trailing_zscore`: hand-computed small example; constant series gives NaN, not inf.
- **No-lookahead test** (critical): for every feature function (returns, avg corr, z-score, VIX ratio, spike days, events), compute on the full frame and on the frame truncated at 5 random dates `T` (fixed seed). Values on or before `T` must be equal (`atol=1e-12`, NaN-equal).
- A static check test: grep `src/vixagent` for `shift(-` and assert it appears only in `targets.py`.

**Targets**
- `fwd_log_vix_change` matches a manual computation; last `h` rows are NaN.

**Spikes**
- The declustering example from Section 7.

**Event study and permutation**
- The hand-checkable example from Section 8.2.
- Planted lead (correlation events placed 3 to 7 days before VIX spikes): lift > 2 and p < 0.05 at h=10.
- No relationship: across 20 seeds with `n_perm=500`, the fraction with p < 0.05 is at most 0.2.
- Clean filter removes an event with a VIX spike on the same day.
- Eligible days respect `t + h <= p_end`.

**Regression and OOS**
- Synthetic `y = 0.5x + noise`: coefficient within 0.05 of 0.5; `R2_oos > 0`.
- Pure noise: `R2_oos < 0.02`.
- HAC maxlags equals `h`.

**Agent**
- Every tool schema is valid JSON Schema; tool names match the dispatch registry.
- Every tool output is `json.dumps`-able with default settings (no numpy types, no NaN).
- Out-of-range args return `is_error: true`.
- Loop with a fake client: scripted `tool_use` then `end_turn`; asserts tool executed with correct args, `tool_result` ids match, message ordering is correct.
- Loop stops at 12 iterations.
- Multiple `tool_use` blocks in one response each get a `tool_result` in one user message.

**Evals**: see EVALS.md.

Coverage target: `pytest --cov=vixagent` at least 85% on `features/`, `spikes.py`, `targets.py`, `analysis/`.

---

## 13. CI (`.github/workflows/ci.yml`)
- Triggers: push, pull_request.
- Matrix: Python 3.11 and 3.12 on `ubuntu-latest`.
- Steps: checkout, setup-python with pip cache, `pip install -e ".[dev]"`, `ruff check .`, `ruff format --check .`, `mypy src`, `pytest -q`.
- No secrets and no network tests in CI.

---

## 14. README requirements (Phase 5)
1. Title and one-line summary.
2. `timeseries.png` as the hero image.
3. **Result** section (generated between markers).
4. **Why this is hard**: lookahead, clustering, multiple testing, in one short paragraph each.
5. Architecture diagram (Mermaid): data, features, analysis, service, tools, agent loop, eval harness.
6. Quickstart: install, `.env`, `vixagent pull`, `vixagent report`, `vixagent chat`.
7. Example transcript: one real, abridged `chat` session from `runs/` showing the agent pushing back on a false premise.
8. **Eval results** (generated between markers).
9. Limitations (from results.md).
10. **What I learned**: left as a placeholder for the human to write personally. The agent must not write this section.

---

## 15. Known pitfalls (read before coding)
- **yfinance** returns MultiIndex columns even for some single-ticker calls; `auto_adjust` defaults changed across versions, so always pass it explicitly; `end` is exclusive; rate limits happen, so cache aggressively.
- **Index timezone**: normalize to tz-naive dates.
- **`shift` direction**: `shift(1)` makes row `t` hold the value from `t-1` (safe). `shift(-h)` pulls the future (only in `targets.py`).
- **`rolling().std()`** uses `ddof=1` by default. Keep it and document it.
- **JSON**: numpy floats/ints, `pd.Timestamp`, and `NaN` break `json.dumps` or produce invalid JSON. Always use `to_jsonable` (NaN becomes `null`, timestamps become `YYYY-MM-DD`).
- **Anthropic tool use**: the assistant message with `tool_use` blocks must be appended before the `tool_result` message; every `tool_use_id` needs a matching `tool_result` in the very next user message.
- **Period slicing**: compute features on the full frame, then slice. Slicing first and recomputing would throw away warmup and change results.
- **Rolling correlation memory**: `rolling().corr()` on 10 assets over ~4,800 days is fine; do not optimize prematurely.
