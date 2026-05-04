# AI Startup Pitch Refinery with Task Decomposition

Multi-agent LangGraph system that converts a rough startup idea into:
- Refined concept
- Market analysis
- Business model
- Investor-ready `.pptx` pitch deck

## Architecture

`StateGraph` workflow with conditional control flow:

`START -> Supervisor -> Idea Agent -> Market Agent -> Source Validator -> (if low score and retries left -> Market Retry -> Source Validator) -> Business Agent -> Pitch Deck Agent -> END`

## Agents and Toolsets

1. Supervisor Agent
- Role: task decomposition + orchestration
- Tools: reasoning only

2. Idea Refinement Agent
- Role: problem, solution, value proposition
- Tools: LLM only

3. Market Research Agent
- Role: market size, trends, competitors
- Tools: Linkup web search + Google Trends + LLM

4. Business Model Agent
- Role: revenue, pricing, costs, projections
- Tools: Python REPL-style calculator + Scenario Analysis tool + LLM
- Notes: Financial assumptions are generated dynamically from idea + market context (bounded ranges), then calculated deterministically.

5. Source Validator Agent
- Role: claim-level evidence verification and reliability scoring
- Tools: LLM validator over Linkup evidence snippets/URLs

6. Pitch Deck Generator Agent
- Role: slide synthesis and `.pptx` generation
- Tools: `python-pptx` + LLM
- Notes: Applies styled layouts, backgrounds, accent elements, and optional per-slide imagery (with fallback placeholders).

Toolsets are intentionally non-identical.

## State + Checkpointing

- Graph state is a Pydantic model (`PitchState`).
- LangGraph uses `MemorySaver` checkpointer with `thread_id`.
- State evolves across nodes (`refined_idea`, `market_analysis`, `validation_report`, `business_model`, `pitch_content`, `tool_audit`, retry flags/counters).

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables:

```bash
cp .env.example .env
# Then edit .env and set your real keys:
# OPENAI_API_KEY=...
# LINKUP_API_KEY=...
# GOOGLE_TRENDS_HL=en-US                  # optional
# GOOGLE_TRENDS_TZ=360                    # optional
# GOOGLE_TRENDS_GEO=US                    # optional
# GOOGLE_TRENDS_TIMEFRAME=today 12-m      # optional
# GOOGLE_TRENDS_CONNECT_TIMEOUT=6          # optional
# GOOGLE_TRENDS_READ_TIMEOUT=12            # optional
# GOOGLE_TRENDS_RETRIES=2                  # optional
# GOOGLE_TRENDS_BACKOFF_FACTOR=0.4         # optional
# GOOGLE_TRENDS_MAX_KEYWORDS=5             # optional
# DISABLE_TRENDS=0                         # optional (set 1 to skip Google Trends)
# IMAGE_PROVIDER=auto                      # auto | unsplash | openai
# IMAGE_FETCH_ENABLED=1                    # set 0 to skip new image fetch/generation
# IMAGE_REUSE_CACHE=1                      # set 1 to reuse output/images/*.jpg when present
# OPENAI_IMAGE_ENABLED=0                   # set 1 to enable generated images (paid)
# OPENAI_IMAGE_MODEL=gpt-image-1           # optional
# OPENAI_IMAGE_SIZE=1536x1024              # optional
# OPENAI_IMAGE_QUALITY=low                 # optional (low/medium/high)
# Optional quality controls:
# LINKUP_ALLOWED_DOMAINS=imarcgroup.com,technavio.com,marketsandmarkets.com
# LINKUP_BLOCKED_DOMAINS=github.com,marketreportsworld.com
```

## Run

```bash
.venv/bin/python main.py --idea "An app that helps students study better"
```

Optional:

```bash
.venv/bin/python main.py --idea "AI coach for job interviews" --controller-policy adaptive --model "gpt-4.1-nano" --temperature 0 --seed 42 --thread-id run-001 --validation-threshold 75 --max-validation-retries 1 --max-tool-calls 12 --max-token-proxy 4500 --max-runtime-seconds 120 --save-json output/final_state.json
```

Methodology comparison mode (single-agent baseline vs fixed/adaptive multi-agent):

```bash
.venv/bin/python main.py \
  --mode compare \
  --idea "AI coach for job interviews" \
  --compare-strategies "single_agent,multi_agent,adaptive_controller" \
  --compare-runs 3 \
  --max-tool-calls 12 \
  --max-token-proxy 4500 \
  --max-total-tokens 4500 \
  --max-runtime-seconds 120 \
  --thread-id methodology-run \
  --compare-output output/methodology_comparison.json
```

Batch methodology comparison over multiple ideas with CSV exports:

```bash
.venv/bin/python main.py \
  --mode compare \
  --idea "placeholder" \
  --ideas-file prompts/methodology_ideas_v1.csv \
  --compare-strategies "single_agent,fixed_shallow,fixed_recursive,adaptive_controller" \
  --compare-runs 5 \
  --disable-image-fetch \
  --disable-trends \
  --max-tool-calls 12 \
  --max-total-tokens 18000 \
  --max-runtime-seconds 180 \
  --validation-threshold 75 \
  --max-validation-retries 2 \
  --compare-export-csv \
  --paper-mode \
  --thread-id methodology-final-v1
```

Notes:
- `--compare-runs` repeats each strategy to reduce variance.
- `--ideas-file` accepts either newline-delimited prompts or a CSV prompt suite with `idea_id`, `difficulty`, `domain`, and `idea` columns.
- `prompts/methodology_ideas_v1.csv` is the fixed benchmark-style prompt suite for methodology experiments.
- `--compare-generate-ppt` is optional and slower; compare mode defaults to metrics-first.
- Budget flags (`--max-tool-calls`, `--max-token-proxy`, `--max-total-tokens`, `--max-runtime-seconds`) are applied uniformly across strategies for matched-budget comparisons.
- Adaptive controller uses a persisted scorecard to choose `direct`, `shallow`, or `recursive`: `utility(mode)=expected_quality(mode)-lambda_cost*normalized_cost(mode)*100+llm_advisory_bonus`.
- The scorecard estimates expected quality from structural complexity, uncertainty, evidence need, and workflow coupling; it estimates cost from deterministic mode priors scaled by complexity and current budget pressure.
- For high-assurance experiments (`--validation-threshold >= 75`), the controller is quality-first but feedback-driven: `direct` is only eligible for very low-risk cases or tight budgets, most tasks start with `shallow`, and `shallow` can escalate to `recursive` only after validation fails and budget remains. These policy adjustments are exported in the compact CSV.
- The selected decomposition is saved as an inspectable graph `D=(V,E,tau,rho)`, where nodes are subtasks, edges are dependencies, `tau` is each subtask interface, and `rho` is the assigned executor.
- Search snippets passed to synthesis and validation are capped by mode and remaining budget so the dual-judge evaluator does not accidentally dominate token cost.
- Recursive retry is budget-gated: another market/validation pass only runs when enough total-token, tool-call, and runtime budget remains for the retry plus a reserve for business-model generation.
- Comparison report includes run-level metrics (reliability, judge agreement, supported-claim ratio, latency, actual token usage, retries, budget-hit indicators, reliability-per-cost metrics, controller complexity/uncertainty, utility margin, and decomposition graph metrics) and strategy-level aggregates (including mode distribution).
- Token accounting now includes both:
  - actual OpenAI token usage (`prompt_tokens_total`, `completion_tokens_total`, `actual_total_tokens`)
  - deterministic token proxy in saved state (`token_proxy_current`) for backward-compatible budget heuristics
- When `--compare-export-csv` is enabled, compact CSVs are also generated automatically:
  - `<runs_csv_output_stem>_compact.csv`
  - `<aggregate_csv_output_stem>_compact.csv`

Skip trends when blocked/rate-limited:

```bash
.venv/bin/python main.py --idea "AI coach for job interviews" --disable-trends
```

Fail run when validation stays low:

```bash
.venv/bin/python main.py --idea "AI coach for job interviews" --fail-on-low-validation --validation-threshold 75
```

Force fresh slide images (do not reuse cache):

```bash
.venv/bin/python main.py --idea "AI coach for job interviews" --disable-image-cache
```

## Output

- Structured state JSON at `output/final_state.json`
- Methodology comparison report at `output/methodology_comparison.json` (when `--mode compare`)
- Pitch deck at `output/pitch_<idea_slug>.pptx`
- Captured market source links in `market_sources`
- Captured trend API payload in `trend_signals`
- Tool execution log in `tool_audit`
- Claim verification output in `validation_report` and `validated_market_analysis`
- Dynamic bounded assumptions in `financial_assumptions`
- Deterministic scenario projections in `scenario_analysis`
- True OpenAI token accounting in `token_usage` and `*_tokens_*` metrics (separate from `token_proxy_*`)

## Project Structure

- `main.py`: CLI entrypoint
- `startup_pitch_refinery/state.py`: shared graph state
- `startup_pitch_refinery/tools.py`: web search, calculator, pptx generator tools
- `startup_pitch_refinery/agents.py`: specialized agents
- `startup_pitch_refinery/graph.py`: LangGraph orchestration
- `startup_pitch_refinery/methodology.py`: baseline/multi-agent comparison runner and metric aggregation
