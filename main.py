import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from startup_pitch_refinery.graph import StartupPitchRefinery
from startup_pitch_refinery.methodology import (
    run_methodology_comparison,
    save_methodology_report,
)


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
        "--compare-generate-ppt",
        action="store_true",
        help="Generate pptx files during compare mode (slower).",
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

    if args.mode == "compare":
        strategy_list = [
            s.strip() for s in args.compare_strategies.split(",") if s.strip()
        ]
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
            generate_ppt=args.compare_generate_ppt,
            thread_prefix=args.thread_id,
        )
        compare_saved = save_methodology_report(report, args.compare_output)

        print("\n=== Methodology Comparison ===")
        print(
            f"(Image config: fetch={'on' if fetch_enabled else 'off'}, "
            f"cache={'on' if reuse_cache else 'off'})"
        )
        print(f"Idea: {report['metadata']['idea']}")
        print(f"Strategies: {', '.join(report['metadata']['strategies'])}")
        print(f"Runs per strategy: {report['metadata']['compare_runs']}")
        print("\n=== Aggregate Metrics ===")
        aggregate = report.get("aggregate", {})
        if not aggregate:
            print("(no aggregate metrics)")
        for strategy, stats in aggregate.items():
            print(
                f"- {strategy}: "
                f"reliability_mean={stats.get('reliability_score_mean')}, "
                f"runtime_mean_s={stats.get('runtime_seconds_mean')}, "
                f"token_proxy_mean={stats.get('token_proxy_total_mean')}, "
                f"supported_ratio_mean={stats.get('supported_ratio_mean')}, "
                f"sources_mean={stats.get('market_sources_count_mean')}, "
                f"depth_mean={stats.get('decomposition_depth_realized_mean')}"
            )
        print("\n=== Recommendation ===")
        rec = report.get("recommendation")
        if rec:
            print(f"Best Strategy: {rec.get('best_strategy')}")
            print(f"Rule: {rec.get('selection_rule')}")
        else:
            print("(no recommendation)")
        print(f"\nComparison report saved to: {compare_saved}")
        return

    app = StartupPitchRefinery(
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        strict_tools=not args.no_strict_tools,
        enable_trends=not (args.disable_trends or env_disable_trends),
        controller_policy=args.controller_policy,
    )
    state = app.run(
        idea=args.idea,
        thread_id=args.thread_id,
        max_validation_retries=args.max_validation_retries,
        validation_threshold=args.validation_threshold,
    )

    save_path = Path(args.save_json)
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
