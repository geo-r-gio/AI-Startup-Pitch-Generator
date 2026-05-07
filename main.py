import argparse
import csv
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
    save_paper_mode_exports,
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

    if latest_link.exists() or latest_link.is_symlink():
        if latest_link.is_symlink() or latest_link.is_file():
            latest_link.unlink()
        else:
            return

    latest_link.symlink_to(target_dir.resolve(), target_is_directory=True)


def _load_ideas_file(path: Path):
    """Load either newline-delimited ideas or a CSV prompt suite."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
        ideas = []
        for idx, row in enumerate(rows):
            idea = (row.get("idea") or row.get("prompt") or "").strip()
            if not idea:
                continue
            ideas.append(
                {
                    "idea_index": idx,
                    "idea_id": (row.get("idea_id") or f"idea_{idx:03d}").strip(),
                    "difficulty": (row.get("difficulty") or "unspecified").strip(),
                    "domain": (row.get("domain") or "unspecified").strip(),
                    "idea": idea,
                }
            )
        return ideas

    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


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
        "--secondary-judge-model",
        type=str,
        default="",
        help="Optional model for Judge B in dual-judge validation. Defaults to --model.",
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
        help="Reliability target used by validation and the Bayesian repair gate.",
    )
    parser.add_argument(
        "--max-validation-retries",
        type=int,
        default=1,
        help="Maximum feedback-driven validation/repair attempts when expected utility is positive.",
    )
    parser.add_argument(
        "--policy-exploration",
        type=float,
        default=0.15,
        help="UCB exploration weight for Bayesian repair policy. Use 0.0 to disable UCB.",
    )
    parser.add_argument(
        "--disable-policy-ucb",
        action="store_true",
        help="Disable the UCB exploration bonus in the Bayesian repair policy.",
    )
    parser.add_argument(
        "--policy-memory-mode",
        choices=["carry_across_ideas", "reset_per_idea"],
        default="carry_across_ideas",
        help="Batch compare only: carry Bayesian policy_stats across ideas or reset for each idea.",
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
        default="single_agent,fixed_shallow,fixed_recursive,adaptive_controller",
        help=(
            "Comma-separated strategies for compare mode. "
            "Allowed: single_agent,multi_agent,fixed_direct,fixed_shallow,"
            "fixed_recursive,adaptive_no_retry,adaptive_no_checkpoint,adaptive_controller"
        ),
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
        "--paper-mode",
        action="store_true",
        help="In compare mode, also export a minimal paper-ready metrics JSON + CSV.",
    )
    parser.add_argument(
        "--paper-json-output",
        type=str,
        default="output/methodology_paper_report.json",
        help="Path for paper-mode JSON export.",
    )
    parser.add_argument(
        "--paper-csv-output",
        type=str,
        default="output/methodology_paper_runs.csv",
        help="Path for paper-mode run-level CSV export.",
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
    policy_exploration = 0.0 if args.disable_policy_ucb else max(0.0, float(args.policy_exploration))

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
    user_set_paper_json = "--paper-json-output" in sys.argv
    user_set_paper_csv = "--paper-csv-output" in sys.argv

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
    paper_json_path = (
        Path(args.paper_json_output)
        if user_set_paper_json
        else run_output_dir / "methodology_paper_report.json"
    )
    paper_csv_path = (
        Path(args.paper_csv_output)
        if user_set_paper_csv
        else run_output_dir / "methodology_paper_runs.csv"
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
            ideas = _load_ideas_file(ideas_path)
            report = run_methodology_batch_comparison(
                ideas=ideas,
                model=args.model,
                secondary_judge_model=args.secondary_judge_model.strip() or None,
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
                policy_exploration=policy_exploration,
                policy_memory_mode=args.policy_memory_mode,
            )
        else:
            report = run_methodology_comparison(
                idea=args.idea,
                model=args.model,
                secondary_judge_model=args.secondary_judge_model.strip() or None,
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
                policy_exploration=policy_exploration,
            )
        compare_saved = save_methodology_report(report, str(compare_output_path))
        csv_saved = None
        paper_saved = None
        if args.compare_export_csv:
            csv_saved = save_methodology_csvs(
                report=report,
                run_csv_path=str(runs_csv_path),
                aggregate_csv_path=str(aggregate_csv_path),
                summary_json_path=str(best_summary_path),
            )
        if args.paper_mode:
            paper_saved = save_paper_mode_exports(
                report=report,
                paper_json_path=str(paper_json_path),
                paper_csv_path=str(paper_csv_path),
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
            "Bayesian policy: "
            f"ucb_enabled={report['metadata'].get('policy_ucb_enabled')} | "
            f"exploration={report['metadata'].get('policy_exploration')} | "
            f"memory_scope={report['metadata'].get('policy_memory_scope')}"
        )
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
                f"judge_agreement_mean={stats.get('judge_agreement_mean')}, "
                f"rel_per_1k_actual_tok_mean={stats.get('reliability_per_1k_actual_token_mean')}, "
                f"supported_ratio_mean={stats.get('supported_ratio_mean')}, "
                f"sources_mean={stats.get('market_sources_count_mean')}, "
                f"depth_mean={stats.get('decomposition_depth_realized_mean')}, "
                f"budget_hits_mean={stats.get('budget_hit_mean')}"
            )
            # Extra Bayesian adaptive-controller diagnostics, shown only when present.
            bayes_keys = [
                "posterior_acceptance_last_mean",
                "retry_expected_utility_last_mean",
                "bayesian_retry_allowed_count_mean",
                "retrieval_diagnostic_mean_score_mean",
                "selected_claim_count_last_mean",
                "policy_repair_accept_rate_mean",
            ]
            bayes_parts = [
                f"{key}={stats.get(key)}" for key in bayes_keys if key in stats
            ]
            if bayes_parts:
                print("  Bayesian diagnostics: " + ", ".join(bayes_parts))
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
            print(
                "Policy calibration CSV saved to: "
                f"{csv_saved['policy_calibration_csv']}"
            )
            print(f"Best-summary JSON saved to: {csv_saved['summary_json']}")
        if paper_saved:
            print(f"Paper-mode JSON saved to: {paper_saved['paper_json']}")
            print(f"Paper-mode CSV saved to: {paper_saved['paper_csv']}")
            print(
                "Paper-mode policy calibration CSV saved to: "
                f"{paper_saved['policy_calibration_csv']}"
            )
        return

    app = StartupPitchRefinery(
        model=args.model,
        secondary_judge_model=args.secondary_judge_model.strip() or None,
        temperature=args.temperature,
        seed=args.seed,
        strict_tools=not args.no_strict_tools,
        enable_trends=not (args.disable_trends or env_disable_trends),
        controller_policy=args.controller_policy,
        output_dir=str(run_output_dir),
        utility_weights={"ucb_exploration": policy_exploration},
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
            f"Initial Mode: {state.get('controller_mode_initial', state.get('controller_mode'))} | "
            f"Realized Mode: {state.get('controller_mode')} | "
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

    retry_policy_decision = state.get("retry_policy_decision", {}) or {}
    if retry_policy_decision:
        print("\n=== Bayesian Retry Decision ===")
        print(
            "Retry allowed: "
            f"{retry_policy_decision.get('retry_allowed')} | "
            f"action={retry_policy_decision.get('action', retry_policy_decision.get('retry_type'))} | "
            f"posterior_acceptance={retry_policy_decision.get('posterior_acceptance', state.get('posterior_acceptance_probability'))} | "
            f"expected_gain={retry_policy_decision.get('expected_gain', state.get('retry_expected_gain'))} | "
            f"expected_utility={retry_policy_decision.get('expected_utility', state.get('retry_expected_utility'))} | "
            f"roi={retry_policy_decision.get('roi_per_1k', state.get('retry_roi_estimate'))}"
        )
        block_reasons = retry_policy_decision.get("block_reasons", [])
        if block_reasons:
            print(f"Block reasons: {block_reasons}")
        rejected_candidate_block_reasons = retry_policy_decision.get("rejected_candidate_block_reasons", [])
        if rejected_candidate_block_reasons:
            print(f"Rejected candidate block reasons: {rejected_candidate_block_reasons}")

    selected_action = state.get("selected_action", {}) or {}
    if selected_action:
        print("\n=== Selected Repair Action ===")
        print(json.dumps(selected_action, indent=2))

    retrieval_diagnostics = state.get("retrieval_diagnostics", []) or []
    if retrieval_diagnostics:
        print("\n=== Retrieval Diagnostics ===")
        for diag in retrieval_diagnostics[:5]:
            print(json.dumps(diag, indent=2))

    checkpoint_selection = state.get("adaptive_checkpoint_selection", {}) or {}
    if checkpoint_selection:
        print("\n=== Checkpoint Selection ===")
        print(json.dumps(checkpoint_selection, indent=2))

    policy_stats = state.get("policy_stats", {}) or {}
    if policy_stats:
        print("\n=== Bayesian Policy Stats ===")
        compact_policy = {
            "repair_observations": policy_stats.get("repair_observations"),
            "repair_accepted_total": policy_stats.get("repair_accepted_total"),
            "repair_rejected_total": policy_stats.get("repair_rejected_total"),
            "repair_accept_rate": policy_stats.get("repair_accept_rate"),
            "repair_mean_gain": policy_stats.get("repair_mean_gain"),
            "repair_mean_tokens": policy_stats.get("repair_mean_tokens"),
        }
        print(json.dumps(compact_policy, indent=2))

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
        agreement = validation.get("agreement_stats", {})
        if agreement:
            print(
                "Judge Agreement: "
                f"overall={agreement.get('overall_agreement')} | "
                f"score_delta_abs={agreement.get('score_delta_abs')} | "
                f"verdict_agreement={agreement.get('verdict_agreement_rate')}"
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

