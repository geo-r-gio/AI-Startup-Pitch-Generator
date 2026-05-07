# AI Startup Pitch Refinery

AI Startup Pitch Refinery is a LangGraph-based research prototype for evidence-grounded startup analysis. It converts a rough startup idea into a refined concept, market analysis, source validation report, business model, and optional pitch deck. The current methodology branch focuses on comparing fixed and adaptive task decomposition strategies under reliability and cost constraints.

The core research question is whether adaptive decomposition can preserve or improve the reliability of a fixed shallow pipeline without blindly spending extra tokens. The current adaptive controller is score-monotonic: it starts from a validated shallow checkpoint, attempts bounded claim-level repair only when validation indicates a need, and accepts the repaired checkpoint only if revalidation improves the reliability score. Otherwise, it rolls back to the shallow checkpoint.

## What The System Produces

- Refined startup concept with problem, solution, user, and value proposition.
- Market analysis supported by retrieved sources.
- Claim-level source validation and reliability scoring.
- Business model with revenue assumptions, costs, and scenario analysis.
- Optional `.pptx` pitch deck.
- JSON and CSV experiment reports for methodology comparisons.

## Current Architecture

The implementation is organized as a controller-executor workflow. A shared idea-refinement step normalizes the input idea, then different execution strategies can be compared.

The main strategies are:

- `single_agent`: one structured LLM call produces the full analysis.
- `fixed_direct`: low-depth direct graph path with minimal decomposition.
- `fixed_shallow`: staged decomposition with market research, validation, and business modeling.
- `fixed_recursive`: shallow pipeline plus deeper retry/repair behavior.
- `adaptive_no_retry`: adaptive branch with repair disabled; useful as a checkpoint-sharing ablation.
- `adaptive_no_checkpoint`: adaptive repair without full checkpoint rollback protection.
- `adaptive_controller`: proposed score-monotonic adaptive controller.

The score-monotonic adaptive controller follows this logic:

```text
startup idea
-> shared idea refinement
-> shallow market research and validation
-> save shallow checkpoint C0
-> if C0 passes threshold: accept C0
-> if C0 fails threshold and evidence exists: attempt bounded claim-level repair
-> revalidate repaired output
-> if repaired score improves: accept repaired checkpoint
-> otherwise: roll back to C0
-> business model and optional pitch deck
```

This means the adaptive controller may spend extra tokens on failed repair attempts, but the selected final checkpoint should not score below the shallow checkpoint according to the validation score.

## Repository Structure

```text
main.py                         CLI entrypoint
startup_pitch_refinery/state.py Shared graph state schema
startup_pitch_refinery/tools.py Search, trends, calculator, scenario, and PPT tools
startup_pitch_refinery/agents.py LLM agents and structured output schemas
startup_pitch_refinery/graph.py LangGraph orchestration and adaptive repair flow
startup_pitch_refinery/methodology.py Baseline comparison, metrics, CSV/JSON exports
requirements.txt                Python dependencies
.env.example                    Environment variable template
README.md                       Project documentation
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create an environment file:

```bash
cp .env.example .env
```

At minimum, set:

```bash
OPENAI_API_KEY=...
LINKUP_API_KEY=...
```

Useful optional settings:

```bash
DISABLE_TRENDS=1
IMAGE_FETCH_ENABLED=0
IMAGE_REUSE_CACHE=1
LINKUP_ALLOWED_DOMAINS=imarcgroup.com,technavio.com,marketsandmarkets.com
LINKUP_BLOCKED_DOMAINS=github.com,marketreportsworld.com,verifiedmarketreports.com
```

For methodology experiments, `--disable-image-fetch` and `--disable-trends` are recommended to reduce runtime, cost, and external variability.

## Single Run

Run the default workflow on one idea:

```bash
.venv/bin/python main.py \
  --idea "AI scheduling assistant for dental clinics" \
  --model gpt-4.1-nano \
  --secondary-judge-model gpt-4.1-mini \
  --temperature 0 \
  --seed 42 \
  --validation-threshold 75 \
  --max-validation-retries 1 \
  --disable-image-fetch \
  --disable-trends \
  --thread-id single-run-demo
```

Save structured output to a chosen path:

```bash
.venv/bin/python main.py \
  --idea "AI coach for job interviews" \
  --save-json output/final_state.json \
  --disable-image-fetch \
  --disable-trends
```

## Methodology Comparison

The main research mode is `--mode compare`. It runs one or more strategies and exports run-level and aggregate metrics.

A compact smoke test:

```bash
.venv/bin/python main.py \
  --mode compare \
  --idea "AI scheduling assistant for dental clinics" \
  --compare-runs 1 \
  --compare-strategies fixed_shallow,adaptive_no_retry,adaptive_controller \
  --model gpt-4.1-nano \
  --secondary-judge-model gpt-4.1-mini \
  --temperature 0 \
  --seed 42 \
  --validation-threshold 76 \
  --max-validation-retries 1 \
  --disable-image-fetch \
  --disable-trends \
  --compare-export-csv \
  --paper-mode \
  --thread-id smoke-adaptive-test
```

A full main comparison over the balanced nine-idea benchmark:

```bash
.venv/bin/python main.py \
  --mode compare \
  --idea "placeholder" \
  --ideas-file experiments/methodology_balanced_9_ideas.csv \
  --compare-runs 1 \
  --compare-strategies single_agent,fixed_direct,fixed_shallow,fixed_recursive,adaptive_no_retry,adaptive_no_checkpoint,adaptive_controller \
  --model gpt-4.1-nano \
  --secondary-judge-model gpt-4.1-mini \
  --temperature 0 \
  --seed 42 \
  --validation-threshold 70 \
  --max-validation-retries 1 \
  --disable-image-fetch \
  --disable-trends \
  --no-strict-tools \
  --compare-export-csv \
  --paper-mode \
  --thread-id final-main-comparison-v1
```

A stricter adaptive stress test:

```bash
.venv/bin/python main.py \
  --mode compare \
  --idea "placeholder" \
  --ideas-file experiments/methodology_balanced_9_ideas.csv \
  --compare-runs 1 \
  --compare-strategies fixed_shallow,adaptive_no_retry,adaptive_no_checkpoint,adaptive_controller,fixed_recursive \
  --model gpt-4.1-nano \
  --secondary-judge-model gpt-4.1-mini \
  --temperature 0 \
  --seed 42 \
  --validation-threshold 76 \
  --max-validation-retries 1 \
  --disable-image-fetch \
  --disable-trends \
  --no-strict-tools \
  --compare-export-csv \
  --paper-mode \
  --thread-id adaptive-stress-threshold76-v1
```

## Key CLI Options

- `--mode run|compare`: run one workflow or compare strategies.
- `--idea`: input startup idea. Required even when using an ideas file.
- `--ideas-file`: newline or CSV file containing multiple ideas.
- `--compare-strategies`: comma-separated strategies to evaluate.
- `--compare-runs`: repeated runs per strategy.
- `--model`: primary LLM model.
- `--secondary-judge-model`: optional second judge model for validation.
- `--temperature`: sampling temperature; use `0` for reproducibility.
- `--seed`: model seed where supported.
- `--validation-threshold`: reliability threshold used to trigger adaptive repair.
- `--max-validation-retries`: maximum repair/retry rounds.
- `--disable-image-fetch`: skip slide image fetch/generation.
- `--disable-trends`: skip Google Trends calls.
- `--no-strict-tools`: continue when non-critical external tools fail.
- `--compare-export-csv`: export run and aggregate CSV files.
- `--paper-mode`: export compact paper-ready JSON and CSV reports.
- `--max-tool-calls`, `--max-total-tokens`, `--max-runtime-seconds`: optional budget controls.

## Metrics

The comparison runner records both quality and cost metrics.

Primary quality metrics:

- `reliability_score`: blended 0-100 validation score.
- `supported_ratio`: fraction of claims judged fully supported.
- `weak_or_better_ratio`: fraction of claims at least weakly supported.
- `validation_claim_coverage`: whether enough checkable claims were validated.
- `judge_agreement`: agreement between primary and secondary validation signals.

Adaptive and process metrics:

- `controller_mode`: selected or forced decomposition mode.
- `decomposition_depth_realized`: realized decomposition depth.
- `retry_count`: retry count in the graph.
- `repair_rounds`: number of repair rounds.
- `accepted_patch_count`: accepted repair patches.
- `rollback_count`: rollback or checkpoint-restoration behavior when available.
- `selected_action`: repair action selected by the policy.
- `retry_block_reasons`: reasons a repair was blocked.

Cost metrics:

- `prompt_tokens_total`
- `completion_tokens_total`
- `actual_total_tokens`
- `runtime_seconds`
- `reliability_per_1k_actual_token`

A useful secondary analysis metric is Coverage-Adjusted Claim Support (CACS):

```text
CACS = validation_claim_coverage * (0.7 * supported_ratio + 0.3 * weak_or_better_ratio) * 100
```

CACS is not a replacement for the reliability score. It is a robustness metric for checking whether a system achieves high support while still producing enough source-checkable claims.

## Output Files

By default, runs are written under:

```text
output/runs/<timestamp>_<mode>_<thread_id>_<idea_slug>/
```

Common outputs include:

```text
final_state.json                    Full graph state for one run
methodology_comparison.json          Full comparison report
methodology_runs.csv                 Run-level metrics
methodology_aggregate.csv            Strategy-level aggregate metrics
methodology_best_summary.json        Best-strategy summary
methodology_paper_report.json        Compact paper-ready report
methodology_paper_runs.csv           Compact paper-ready run table
*_compact.csv                        Smaller CSVs for quick inspection
```

The `output/runs/latest` symlink points to the most recent run folder when symlink creation is available.

## Reproducibility Notes

For research-style runs:

- Use `--temperature 0` and a fixed `--seed`.
- Keep the same `--ideas-file` across strategy comparisons.
- Disable non-essential external calls with `--disable-image-fetch` and `--disable-trends`.
- Use the same `--validation-threshold` across compared strategies.
- Export both JSON and CSV with `--compare-export-csv --paper-mode`.
- Report actual token usage rather than only proxy token estimates.

External search can still introduce variability because source availability and API responses can change. The run artifacts preserve retrieved sources, validation outputs, and token usage for auditability.

## Development Checks

Syntax-check the core files:

```bash
PYTHONPYCACHEPREFIX=.pycache_compile .venv/bin/python -m py_compile \
  main.py \
  startup_pitch_refinery/agents.py \
  startup_pitch_refinery/graph.py \
  startup_pitch_refinery/methodology.py \
  startup_pitch_refinery/state.py \
  startup_pitch_refinery/tools.py
```

Inspect the current branch and changed files:

```bash
git status --short
git branch --show-current
```

## Practical Interpretation

The current best framing is conservative. Fixed shallow decomposition is the strongest default baseline for reliability per cost. Fixed recursive can sometimes improve reliability but is less stable and more expensive. The adaptive controller is designed to be safer than broad recursion: it preserves the shallow checkpoint and only keeps repaired outputs when validation score improves.

The method should not be described as proving universal adaptive superiority. It is better described as a score-monotonic adaptive repair controller that can selectively improve difficult cases while making the cost of unsuccessful repair attempts visible.
