import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import sys

from dotenv import load_dotenv

from startup_pitch_refinery.graph import StartupPitchRefinery
from startup_pitch_refinery.methodology import (
    run_methodology_batch_comparison,
    run_methodology_comparison,
    save_methodology_csvs,
    save_methodology_report,
)


def _slugify(value: str, fallback: str = "run") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    if not cleaned:
        return fallback
    return cleaned[:64]


def _default_run_output_dir(mode: str, idea: str, thread_id: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_slug = _slugify(" ".join(idea.split()[:8]), fallback="idea")
    thread_slug = _slugify(thread_id, fallback="thread")
    return Path("output") / "runs" / f"{stamp}_{mode}_{thread_slug}_{idea_slug}"


def _update_latest_symlink(target_dir: Path) -> None:
    runs_root = Path("output") / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    latest_link = runs_root / "latest"

    # Safe replacement: only replace if existing path is a file/symlink.
    # If someone created a real directory at output/runs/latest, leave it untouched.
    if latest_link.exists() or latest_link.is_symlink():
        if latest_link.is_symlink() or latest_link.is_file():
            latest_link.unlink()
        else:
            return

    latest_link.symlink_to(target_dir.resolve(), target_is_directory=True)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="AI Startup Pitch Refinery with Task Decomposition"
    )
    parser.add_argument(
        "--idea",
        type=str,
        required=True,
        help='Raw startup idea. Example: "An app that helps students study better"',
    )
    parser.add_argument(
        "--mode",
        choices=["run", "compare"],
        default="run",
        help="`run`: execute standard multi-agent workflow. `compare`: run methodology strategy comparison.",
    )
    parser.add_argument(
        "--controller-policy",
        choices=["fixed", "adaptive"],
        default="fixed",
        help="Controller policy for run mode. `fixed` uses static workflow, `adaptive` chooses direct/shallow/recursive.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-nano",
        help="OpenAI chat model name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0.0 for maximum stability.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Model seed to reduce output variance across runs.",
    )
    parser.add_argument(
        "--no-strict-tools",
        action="store_true",
        help="Allow workflow to continue if a required tool fails.",
    )
    parser.add_argument(
        "--disable-trends",
        action="store_true",
        help="Skip Google Trends calls (useful if blocked/rate-limited).",
    )
    parser.add_argument(
        "--disable-image-fetch",
        action="store_true",
        help="Do not fetch/generate new slide images; reuse cached images if available.",
    )
    parser.add_argument(
        "--disable-image-cache",
        action="store_true",
        help="Do not reuse cached slide images; fetch/generate fresh images when possible.",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default="default-thread",
        help="Execution thread id used by LangGraph MemorySaver checkpoints.",
    )
    parser.add_argument(
        "--validation-threshold",
        type=int,
        default=70,
        help="If validation score is below this threshold, trigger one revision loop.",
    )
    parser.add_argument(
        "--max-validation-retries",
        type=int,
        default=1,
        help="Maximum market-research retries when validation is weak.",
    )
    parser.add_argument(
        "--fail-on-low-validation",
        action="store_true",
        help="Exit non-zero if final reliability score is below validation threshold.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="output/final_state.json",
        help="Where to save full structured state",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=-1,
        help="Hard budget: max allowed tool audit entries. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--max-token-proxy",
        type=int,
        default=-1,
        help="Hard budget: max allowed token proxy (chars/4 heuristic). Use -1 for unlimited.",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=-1,
        help="Hard budget: max allowed true OpenAI total tokens. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=-1.0,
        help="Hard budget: max wall-clock runtime seconds per run. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--compare-runs",
        type=int,
        default=1,
        help="Number of repeated runs per strategy in compare mode.",
    )
    parser.add_argument(
        "--compare-strategies",
        type=str,
        default="single_agent,multi_agent,adaptive_controller",
        help="Comma-separated strategies for compare mode. Allowed: single_agent,multi_agent,adaptive_controller",
    )
    parser.add_argument(
        "--compare-output",
        type=str,
        default="output/methodology_comparison.json",
        help="Where to save compare-mode report JSON.",
    )
    parser.add_argument(
        "--ideas-file",
        type=str,
        default="",
        help="Optional newline-delimited file of ideas/prompts for batch compare mode.",
    )
    parser.add_argument(
        "--compare-export-csv",
        action="store_true",
        help="Also export run-level CSV, aggregate CSV, and best-summary JSON in compare mode.",
    )
    parser.add_argument(
        "--runs-csv-output",
        type=str,
        default="output/methodology_runs.csv",
        help="Path for compare run-level CSV export.",
    )
    parser.add_argument(
        "--aggregate-csv-output",
        type=str,
        default="output/methodology_aggregate.csv",
        help="Path for compare aggregate CSV export.",
    )
    parser.add_argument(
        "--best-summary-output",
        type=str,
        default="output/methodology_best_summary.json",
        help="Path for compare best-strategy summary JSON export.",
    )
    parser.add_argument(
        "--compare-generate-ppt",
        action="store_true",
        help="Generate pptx files during compare mode (slower).",
    )
    parser.add_argument(
        "--run-output-dir",
        type=str,
        default="",
        help=(
            "Output folder for this run. If omitted, auto-creates "
            "output/runs/<timestamp>_<mode>_<thread>_<idea>/"
        ),
    )

    args = parser.parse_args()
    env_disable_trends = os.getenv("DISABLE_TRENDS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    env_disable_image_fetch = os.getenv("IMAGE_FETCH_ENABLED", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }
    env_disable_image_cache = os.getenv("IMAGE_REUSE_CACHE", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }
    if args.disable_image_fetch or env_disable_image_fetch:
        os.environ["IMAGE_FETCH_ENABLED"] = "0"
    if args.disable_image_cache or env_disable_image_cache:
        os.environ["IMAGE_REUSE_CACHE"] = "0"

    fetch_enabled = os.getenv("IMAGE_FETCH_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    reuse_cache = os.getenv("IMAGE_REUSE_CACHE", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    run_output_dir = (
        Path(args.run_output_dir).expanduser()
        if args.run_output_dir.strip()
        else _default_run_output_dir(args.mode, args.idea, args.thread_id)
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    _update_latest_symlink(run_output_dir)

    user_set_save_json = "--save-json" in sys.argv
    user_set_compare_output = "--compare-output" in sys.argv
    user_set_runs_csv = "--runs-csv-output" in sys.argv
    user_set_aggregate_csv = "--aggregate-csv-output" in sys.argv
    user_set_best_summary = "--best-summary-output" in sys.argv

    save_json_path = (
        Path(args.save_json)
        if user_set_save_json
        else run_output_dir / "final_state.json"
    )
    compare_output_path = (
        Path(args.compare_output)
        if user_set_compare_output
        else run_output_dir / "methodology_comparison.json"
    )
    runs_csv_path = (
        Path(args.runs_csv_output)
        if user_set_runs_csv
        else run_output_dir / "methodology_runs.csv"
    )
    aggregate_csv_path = (
        Path(args.aggregate_csv_output)
        if user_set_aggregate_csv
        else run_output_dir / "methodology_aggregate.csv"
    )
    best_summary_path = (
        Path(args.best_summary_output)
        if user_set_best_summary
        else run_output_dir / "methodology_best_summary.json"
    )

    if args.mode == "compare":
        max_tool_calls = None if args.max_tool_calls < 0 else args.max_tool_calls
        max_token_proxy = None if args.max_token_proxy < 0 else args.max_token_proxy
        max_total_tokens = None if args.max_total_tokens < 0 else args.max_total_tokens
        max_runtime_seconds = (
            None if args.max_runtime_seconds < 0 else args.max_runtime_seconds
        )
        strategy_list = [
            s.strip() for s in args.compare_strategies.split(",") if s.strip()
        ]
        ideas_file = args.ideas_file.strip()
        if ideas_file:
            ideas_path = Path(ideas_file)
            if not ideas_path.exists():
                raise FileNotFoundError(f"Ideas file not found: {ideas_path}")
            ideas = [
                line.strip()
                for line in ideas_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            report = run_methodology_batch_comparison(
                ideas=ideas,
                model=args.model,
                temperature=args.temperature,
                seed=args.seed,
                strict_tools=not args.no_strict_tools,
                enable_trends=not (args.disable_trends or env_disable_trends),
                compare_runs=max(1, args.compare_runs),
                strategies=strategy_list,
                validation_threshold=args.validation_threshold,
                max_validation_retries=args.max_validation_retries,
                max_tool_calls=max_tool_calls,
                max_token_proxy=max_token_proxy,
                max_total_tokens=max_total_tokens,
                max_runtime_seconds=max_runtime_seconds,
                generate_ppt=args.compare_generate_ppt,
                thread_prefix=args.thread_id,
                output_dir=str(run_output_dir),
            )
        else:
            report = run_methodology_comparison(
                idea=args.idea,
                model=args.model,
                temperature=args.temperature,
                seed=args.seed,
                strict_tools=not args.no_strict_tools,
                enable_trends=not (args.disable_trends or env_disable_trends),
                compare_runs=max(1, args.compare_runs),
                strategies=strategy_list,
                validation_threshold=args.validation_threshold,
                max_validation_retries=args.max_validation_retries,
                max_tool_calls=max_tool_calls,
                max_token_proxy=max_token_proxy,
                max_total_tokens=max_total_tokens,
                max_runtime_seconds=max_runtime_seconds,
                generate_ppt=args.compare_generate_ppt,
                thread_prefix=args.thread_id,
                output_dir=str(run_output_dir),
            )
        compare_saved = save_methodology_report(report, str(compare_output_path))
        csv_saved = None
        if args.compare_export_csv:
            csv_saved = save_methodology_csvs(
                report=report,
                run_csv_path=str(runs_csv_path),
                aggregate_csv_path=str(aggregate_csv_path),
                summary_json_path=str(best_summary_path),
            )

        print("\n=== Methodology Comparison ===")
        print(
            f"(Image config: fetch={'on' if fetch_enabled else 'off'}, "
            f"cache={'on' if reuse_cache else 'off'})"
        )
        if "idea" in report["metadata"]:
            print(f"Idea: {report['metadata']['idea']}")
        else:
            print(f"Ideas Count: {report['metadata'].get('ideas_count')}")
        print(f"Strategies: {', '.join(report['metadata']['strategies'])}")
        print(f"Runs per strategy: {report['metadata']['compare_runs']}")
        print(
            "Budgets: "
            f"max_tool_calls={report['metadata'].get('max_tool_calls')}, "
            f"max_token_proxy={report['metadata'].get('max_token_proxy')}, "
            f"max_total_tokens={report['metadata'].get('max_total_tokens')}, "
            f"max_runtime_seconds={report['metadata'].get('max_runtime_seconds')}"
        )
        print("\n=== Aggregate Metrics ===")
        aggregate = report.get("aggregate", {})
        if not aggregate:
            print("(no aggregate metrics)")
        for strategy, stats in aggregate.items():
            print(
                f"- {strategy}: "
                f"reliability_mean={stats.get('reliability_score_mean')}, "
                f"runtime_mean_s={stats.get('runtime_seconds_mean')}, "
                f"actual_tokens_mean={stats.get('actual_total_tokens_mean')}, "
                f"token_proxy_mean={stats.get('token_proxy_total_mean')}, "
                f"rel_per_1k_actual_tok_mean={stats.get('reliability_per_1k_actual_token_mean')}, "
                f"rel_per_1k_tok_mean={stats.get('reliability_per_1k_token_mean')}, "
                f"supported_ratio_mean={stats.get('supported_ratio_mean')}, "
                f"sources_mean={stats.get('market_sources_count_mean')}, "
                f"depth_mean={stats.get('decomposition_depth_realized_mean')}, "
                f"budget_override_mean={stats.get('controller_budget_override_mean')}, "
                f"budget_hits_mean={stats.get('budget_hit_mean')}"
            )
        print("\n=== Recommendation ===")
        rec = report.get("recommendation")
        if rec:
            print(f"Best Strategy: {rec.get('best_strategy')}")
            print(f"Rule: {rec.get('selection_rule')}")
        else:
            print("(no recommendation)")
        print(f"\nComparison report saved to: {compare_saved}")
        print(f"Run artifacts folder: {run_output_dir}")
        if csv_saved:
            print(f"Run-level CSV saved to: {csv_saved['run_csv']}")
            print(f"Aggregate CSV saved to: {csv_saved['aggregate_csv']}")
            print(f"Run-level compact CSV saved to: {csv_saved['run_csv_compact']}")
            print(
                "Aggregate compact CSV saved to: "
                f"{csv_saved['aggregate_csv_compact']}"
            )
            print(f"Best-summary JSON saved to: {csv_saved['summary_json']}")
        return

    app = StartupPitchRefinery(
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        strict_tools=not args.no_strict_tools,
        enable_trends=not (args.disable_trends or env_disable_trends),
        controller_policy=args.controller_policy,
        output_dir=str(run_output_dir),
    )
    state = app.run(
        idea=args.idea,
        thread_id=args.thread_id,
        max_validation_retries=args.max_validation_retries,
        validation_threshold=args.validation_threshold,
        max_tool_calls=(None if args.max_tool_calls < 0 else args.max_tool_calls),
        max_token_proxy=(None if args.max_token_proxy < 0 else args.max_token_proxy),
        max_total_tokens=(None if args.max_total_tokens < 0 else args.max_total_tokens),
        max_runtime_seconds=(
            None if args.max_runtime_seconds < 0 else args.max_runtime_seconds
        ),
    )

    save_path = save_json_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print("\n=== Task Plan ===")
    print(
        f"(Image config: fetch={'on' if fetch_enabled else 'off'}, "
        f"cache={'on' if reuse_cache else 'off'})"
    )
    print(f"(Controller policy: {args.controller_policy})")
    for step in state.get("task_plan", []):
        print(f"- {step}")

    print("\n=== Controller Decision ===")
    if state.get("controller_mode"):
        print(
            f"Mode: {state.get('controller_mode')} | "
            f"Confidence: {state.get('controller_confidence')} | "
            f"Target Depth: {state.get('decomposition_depth_target')} | "
            f"Realized Depth: {state.get('decomposition_depth_realized')}"
        )
        print(f"Rationale: {state.get('controller_rationale', '')}")
    else:
        print("(controller node not active in this run)")
    print(
        "Budget status: "
        f"hit={state.get('budget_hit', False)} | "
        f"reasons={state.get('budget_hit_reasons', [])} | "
        f"remaining={state.get('budget_remaining', {})}"
    )
    print(
        "Token usage: "
        f"actual={state.get('token_usage', {})} | "
        f"proxy={state.get('token_proxy_current', 0)}"
    )

    print("\n=== Refined Idea ===")
    print(state.get("refined_idea", ""))

    print("\n=== Market Analysis ===")
    print(state.get("market_analysis", ""))
    print("\n=== Validated Market Analysis ===")
    print(state.get("validated_market_analysis", "(not validated)"))
    print("\n=== Market Sources ===")
    sources = state.get("market_sources", [])
    if sources:
        for src in sources:
            print(f"- {src}")
    else:
        print("(no sources captured)")
    print("\n=== Trend Signals ===")
    print(json.dumps(state.get("trend_signals", {}), indent=2))

    print("\n=== Business Model ===")
    print(state.get("business_model", ""))
    print("\n=== Financial Assumptions ===")
    print(json.dumps(state.get("financial_assumptions", {}), indent=2))
    print("\n=== Scenario Analysis ===")
    print(json.dumps(state.get("scenario_analysis", {}), indent=2))

    print("\n=== Validation Report ===")
    validation = state.get("validation_report", {})
    if validation:
        print(
            f"Reliability Score: {validation.get('reliability_score', 'n/a')}/100 | "
            f"Evidence Gaps: {validation.get('evidence_gaps', '')}"
        )
        claims = validation.get("claims", [])
        for idx, claim in enumerate(claims, start=1):
            print(
                f"{idx}. [{claim.get('verdict')}] "
                f"confidence={claim.get('confidence')} claim={claim.get('claim')}"
            )
            for src in claim.get("supporting_sources", []):
                print(f"   - {src}")
    else:
        print("(no validation report)")
    print(
        f"Revision Needed: {state.get('needs_revision')} | "
        f"Retries Used: {state.get('retry_count', 0)}/{state.get('max_validation_retries', 1)}"
    )

    print("\n=== PPTX Output ===")
    print(state.get("ppt_path", "(not generated)"))

    print("\n=== Tool Audit ===")
    for entry in state.get("tool_audit", []):
        print(f"- {json.dumps(entry)}")
    print(f"\nFull state saved to: {save_path}")
    print(f"Run artifacts folder: {run_output_dir}")

    final_score = 0
    if validation:
        final_score = int(validation.get("reliability_score", 0))
    if args.fail_on_low_validation and final_score < args.validation_threshold:
        raise SystemExit(
            f"Validation gate failed: {final_score} < {args.validation_threshold} "
            "(use --no-strict-tools/threshold tuning or improve sources)."
        )


if __name__ == "__main__":
    main()
