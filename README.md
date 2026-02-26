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

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variables:

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
python main.py --idea "An app that helps students study better"
```

Optional:

```bash
python main.py --idea "AI coach for job interviews" --model "gpt-4.1-nano" --temperature 0 --seed 42 --thread-id run-001 --validation-threshold 75 --max-validation-retries 1 --save-json output/final_state.json
```

Skip trends when blocked/rate-limited:

```bash
python main.py --idea "AI coach for job interviews" --disable-trends
```

Fail run when validation stays low:

```bash
python main.py --idea "AI coach for job interviews" --fail-on-low-validation --validation-threshold 75
```

## Output

- Structured state JSON at `output/final_state.json`
- Pitch deck at `output/pitch_<idea_slug>.pptx`
- Captured market source links in `market_sources`
- Captured trend API payload in `trend_signals`
- Tool execution log in `tool_audit`
- Claim verification output in `validation_report` and `validated_market_analysis`
- Dynamic bounded assumptions in `financial_assumptions`
- Deterministic scenario projections in `scenario_analysis`

## Project Structure

- `main.py`: CLI entrypoint
- `startup_pitch_refinery/state.py`: shared graph state
- `startup_pitch_refinery/tools.py`: web search, calculator, pptx generator tools
- `startup_pitch_refinery/agents.py`: specialized agents
- `startup_pitch_refinery/graph.py`: LangGraph orchestration
