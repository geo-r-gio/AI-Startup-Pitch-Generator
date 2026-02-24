import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from startup_pitch_refinery.graph import StartupPitchRefinery


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

    args = parser.parse_args()
    env_disable_trends = os.getenv("DISABLE_TRENDS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    app = StartupPitchRefinery(
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        strict_tools=not args.no_strict_tools,
        enable_trends=not (args.disable_trends or env_disable_trends),
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
    for step in state.get("task_plan", []):
        print(f"- {step}")

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
