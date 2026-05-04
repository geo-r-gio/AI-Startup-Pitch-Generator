from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from startup_pitch_refinery.agents import (
    IdeaRefinementAgent,
    PitchDeckGeneratorAgent,
    SourceValidatorAgent,
)
from startup_pitch_refinery.graph import StartupPitchRefinery
from startup_pitch_refinery.tools import GoogleTrendsTool, MarketSearchTool, ScenarioAnalysisTool


class SingleAgentOutput(BaseModel):
    refined_idea: str = Field(..., description="Investor-ready startup concept")
    market_analysis: str = Field(..., description="Market analysis text")
    business_model: str = Field(..., description="Business model text")
    users_year1: int = Field(..., ge=1000, le=500000)
    arpu_monthly: float = Field(..., ge=2.0, le=300.0)
    gross_margin: float = Field(..., ge=0.2, le=0.95)
    assumptions_rationale: str = Field(..., description="Rationale for assumptions")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _empty_token_usage() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _extract_token_usage(raw_message: Any) -> Dict[str, int]:
    usage = _empty_token_usage()
    if raw_message is None:
        return usage

    usage_meta = getattr(raw_message, "usage_metadata", None) or {}
    response_meta = getattr(raw_message, "response_metadata", None) or {}
    token_usage = response_meta.get("token_usage", {}) if isinstance(response_meta, dict) else {}

    prompt = (
        usage_meta.get("input_tokens")
        or usage_meta.get("prompt_tokens")
        or token_usage.get("prompt_tokens")
        or 0
    )
    completion = (
        usage_meta.get("output_tokens")
        or usage_meta.get("completion_tokens")
        or token_usage.get("completion_tokens")
        or 0
    )
    total = (
        usage_meta.get("total_tokens")
        or token_usage.get("total_tokens")
        or (int(prompt) + int(completion))
    )
    usage["prompt_tokens"] = max(0, int(prompt))
    usage["completion_tokens"] = max(0, int(completion))
    usage["total_tokens"] = max(0, int(total))
    return usage


def _invoke_structured_with_usage(runnable: Any, messages: Any) -> tuple[Any, Dict[str, int]]:
    payload = runnable.invoke(messages)
    if isinstance(payload, dict) and "parsed" in payload:
        parsed = payload.get("parsed")
        if parsed is None:
            raise ValueError(f"Structured output parsing failed: {payload.get('parsing_error')}")
        usage = _extract_token_usage(payload.get("raw"))
        return parsed, usage
    return payload, _empty_token_usage()


def _estimate_tokens_proxy(state: Dict[str, Any]) -> int:
    """Deterministic token proxy for cost comparisons (chars / 4 heuristic)."""
    chunks: List[str] = [
        str(state.get("idea", "")),
        str(state.get("refined_idea", "")),
        str(state.get("market_analysis", "")),
        str(state.get("business_model", "")),
        str(state.get("validated_market_analysis", "")),
        json.dumps(state.get("pitch_content", {}), ensure_ascii=True),
        json.dumps(state.get("trend_signals", {}), ensure_ascii=True),
        json.dumps(state.get("validation_report", {}), ensure_ascii=True),
    ]
    total_chars = sum(len(c) for c in chunks)
    return max(1, total_chars // 4)


def _claim_ratios(validation_report: Dict[str, Any]) -> Dict[str, float]:
    claims = validation_report.get("claims", []) if validation_report else []
    if not claims:
        return {
            "supported_ratio": 0.0,
            "weak_or_better_ratio": 0.0,
        }
    supported = 0
    weak_or_better = 0
    for claim in claims:
        verdict = str(claim.get("verdict", "")).strip().lower()
        if verdict == "supported":
            supported += 1
        if verdict in {"supported", "weakly_supported"}:
            weak_or_better += 1
    total = len(claims)
    return {
        "supported_ratio": round(supported / total, 4),
        "weak_or_better_ratio": round(weak_or_better / total, 4),
    }


def _extract_metrics(state: Dict[str, Any], strategy: str, runtime_seconds: float) -> Dict[str, Any]:
    validation = state.get("validation_report") or {}
    ratios = _claim_ratios(validation)
    budget_hit = bool(state.get("budget_hit", False))
    budget_violation_count = len(state.get("budget_hit_reasons", []) or [])
    reliability_score = _safe_int(validation.get("reliability_score", 0), 0)
    token_usage = state.get("token_usage", {}) or {}
    prompt_tokens_total = _safe_int(token_usage.get("prompt_tokens", 0), 0)
    completion_tokens_total = _safe_int(token_usage.get("completion_tokens", 0), 0)
    actual_total_tokens = _safe_int(token_usage.get("total_tokens", 0), 0)
    reliability_per_1k_actual_token = round(
        (reliability_score * 1000.0) / max(1, actual_total_tokens), 4
    )
    reliability_per_second = round(
        reliability_score / max(0.001, float(runtime_seconds)), 4
    )
    agreement_stats = validation.get("agreement_stats", {}) if isinstance(validation, dict) else {}
    scorecard = state.get("controller_scorecard", {}) or {}
    forced_mode_applied = (
        bool(scorecard.get("forced_mode_applied", False))
        if isinstance(scorecard, dict)
        else False
    )
    score_features = scorecard.get("features", {}) if isinstance(scorecard, dict) else {}
    mode_scores = scorecard.get("mode_scores", {}) if isinstance(scorecard, dict) else {}
    controller_mode_realized = str(state.get("controller_mode", "n/a") or "n/a")
    controller_mode_initial = str(
        state.get("controller_mode_initial")
        or scorecard.get("selected_mode", "")
        or controller_mode_realized
    )
    selected_score = (
        mode_scores.get(controller_mode_initial, {}) if isinstance(mode_scores, dict) else {}
    )
    realized_score = (
        mode_scores.get(controller_mode_realized, {}) if isinstance(mode_scores, dict) else {}
    )
    has_controller_estimate = bool(realized_score)
    realized_expected_quality = (
        float(realized_score.get("expected_quality", 0.0) or 0.0)
        if has_controller_estimate
        else 0.0
    )
    controller_calibration_error = (
        round(realized_expected_quality - float(reliability_score), 4)
        if has_controller_estimate
        else 0.0
    )
    decomposition_graph = state.get("decomposition_graph", {}) or {}
    graph_metrics = (
        decomposition_graph.get("metrics", {}) if isinstance(decomposition_graph, dict) else {}
    )
    validator_audits = [
        audit
        for audit in (state.get("tool_audit", []) or [])
        if isinstance(audit, dict)
        and audit.get("tool")
        in {"llm_claim_verifier_dual_judge", "llm_claim_repair_validator"}
    ]
    validator_audit = validator_audits[-1] if validator_audits else {}
    repair_validator_audits = [
        audit
        for audit in (state.get("tool_audit", []) or [])
        if isinstance(audit, dict) and audit.get("tool") == "llm_claim_repair_validator"
    ]
    lightweight_repair_validation_count = len(repair_validator_audits)
    repair_validation_tokens = sum(
        _safe_int(audit.get("total_tokens", 0), 0) for audit in repair_validator_audits
    )
    repair_patch_accepted_count = sum(
        1 for audit in repair_validator_audits if bool(audit.get("repair_patch_accepted", False))
    )
    repair_cascade_audits = [
        audit
        for audit in (state.get("tool_audit", []) or [])
        if isinstance(audit, dict) and audit.get("tool") == "repair_validator_cascade"
    ]
    repair_validator_escalation_count = len(repair_cascade_audits)
    repair_cascade_score_delta = sum(
        _safe_int(audit.get("full_validation_score", 0), 0)
        - _safe_int(audit.get("previous_reliability_score", 0), 0)
        for audit in repair_cascade_audits
    )
    secondary_judge_fallback = (
        1 if bool(validator_audit.get("secondary_judge_fallback", False)) else 0
    )
    minimum_claims_required = _safe_int(
        validator_audit.get("minimum_claims_required", 3), 3
    )
    claims_total = len(validation.get("claims", []) if validation else [])
    low_claim_count_flag = (
        1
        if bool(validator_audit.get("low_claim_count_flag", claims_total < minimum_claims_required))
        else 0
    )
    claim_count_penalty = _safe_int(validator_audit.get("claim_count_penalty", 0), 0)
    validation_claim_coverage = round(
        min(1.0, claims_total / max(1, minimum_claims_required)),
        4,
    )
    judge_a_model = str(validator_audit.get("judge_a_model", "") or "")
    judge_b_model = str(validator_audit.get("judge_b_model", "") or "")
    cross_model_judging = (
        1 if bool(validator_audit.get("cross_model_judging", False)) else 0
    )
    retry_decisions = [
        decision
        for decision in (state.get("retry_budget_decisions", []) or [])
        if isinstance(decision, dict)
    ]
    retry_allowed_count = sum(1 for d in retry_decisions if bool(d.get("retry_allowed", False)))
    retry_blocked_count = sum(
        1
        for d in retry_decisions
        if not bool(d.get("retry_allowed", False)) and bool(d.get("needs_revision", False))
    )
    checkpoint_selection = state.get("adaptive_checkpoint_selection", {}) or {}
    last_retry_decision = retry_decisions[-1] if retry_decisions else {}
    adaptive_retry_enabled = 1 if bool(state.get("adaptive_retry_enabled", True)) else 0
    adaptive_checkpoint_enabled = (
        1 if bool(state.get("adaptive_checkpoint_enabled", True)) else 0
    )
    validation_snapshots = [
        item
        for item in (state.get("validation_snapshots", []) or [])
        if isinstance(item, dict)
    ]
    initial_validation_score = (
        _safe_int(validation_snapshots[0].get("reliability_score", reliability_score), reliability_score)
        if validation_snapshots
        else reliability_score
    )
    latest_validation_score = (
        _safe_int(validation_snapshots[-1].get("reliability_score", reliability_score), reliability_score)
        if validation_snapshots
        else reliability_score
    )
    retry_count_value = _safe_int(state.get("retry_count", 0), 0)
    retry_effectiveness = (
        round(float(reliability_score - initial_validation_score), 4)
        if retry_count_value > 0
        else 0.0
    )
    raw_retry_score_delta = (
        round(float(latest_validation_score - initial_validation_score), 4)
        if len(validation_snapshots) > 1
        else 0.0
    )
    checkpoint_saved_score = round(float(reliability_score - latest_validation_score), 4)
    focused_repair_audits = [
        audit
        for audit in (state.get("tool_audit", []) or [])
        if isinstance(audit, dict)
        and audit.get("tool") == "focused_repair_search"
        and audit.get("status") == "ok"
        and str(audit.get("query", "") or "").strip()
    ]
    focused_repair_search_count = len(focused_repair_audits)
    focused_repair_source_count = sum(
        _safe_int(audit.get("source_count", 0), 0) for audit in focused_repair_audits
    )
    micro_repair_audits = [
        audit
        for audit in (state.get("tool_audit", []) or [])
        if isinstance(audit, dict) and audit.get("tool") == "claim_micro_repair"
    ]
    micro_repair_count = len(micro_repair_audits)
    micro_repair_source_count = sum(
        _safe_int(audit.get("source_count", 0), 0) for audit in micro_repair_audits
    )
    micro_repair_tokens = sum(
        _safe_int(audit.get("total_tokens", 0), 0) for audit in micro_repair_audits
    )
    micro_repair_search_replace_count = sum(
        _safe_int(
            (audit.get("repair_action_counts", {}) or {}).get("search_and_replace", 0),
            0,
        )
        for audit in micro_repair_audits
        if isinstance(audit.get("repair_action_counts", {}), dict)
    )
    micro_repair_qualify_remove_count = sum(
        _safe_int(
            (audit.get("repair_action_counts", {}) or {}).get("qualify_or_remove", 0),
            0,
        )
        + _safe_int((audit.get("repair_action_counts", {}) or {}).get("remove", 0), 0)
        for audit in micro_repair_audits
        if isinstance(audit.get("repair_action_counts", {}), dict)
    )
    repair_gain_per_1k_tokens = (
        round((raw_retry_score_delta * 1000.0) / max(1, micro_repair_tokens), 4)
        if micro_repair_count
        else 0.0
    )
    metrics = {
        "strategy": strategy,
        "controller_mode": controller_mode_realized,
        "controller_mode_initial": controller_mode_initial,
        "controller_mode_realized": controller_mode_realized,
        "controller_escalated": 1
        if controller_mode_initial != controller_mode_realized
        and controller_mode_initial != "n/a"
        and controller_mode_realized != "n/a"
        else 0,
        "runtime_seconds": round(runtime_seconds, 3),
        "reliability_score": reliability_score,
        "claims_total": claims_total,
        "minimum_claims_required": minimum_claims_required,
        "validation_claim_coverage": validation_claim_coverage,
        "low_claim_count_flag": low_claim_count_flag,
        "claim_count_penalty": claim_count_penalty,
        "supported_ratio": ratios["supported_ratio"],
        "market_sources_count": len(state.get("market_sources", [])),
        "tool_calls": len(state.get("tool_audit", [])),
        "retry_count": retry_count_value,
        "decomposition_depth_target": _safe_int(
            state.get("decomposition_depth_target", 0), 0
        ),
        "decomposition_depth_realized": _safe_int(state.get("decomposition_depth_realized", 0), 0),
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "actual_total_tokens": actual_total_tokens,
        "reliability_per_1k_actual_token": reliability_per_1k_actual_token,
        "reliability_per_second": reliability_per_second,
        "judge_agreement": float(agreement_stats.get("overall_agreement", 0.0) or 0.0),
        "judge_score_delta_abs": float(agreement_stats.get("score_delta_abs", 0.0) or 0.0),
        "judge_a_model": judge_a_model,
        "judge_b_model": judge_b_model,
        "cross_model_judging": cross_model_judging,
        "secondary_judge_fallback": secondary_judge_fallback,
        "budget_hit": 1 if budget_hit else 0,
        "budget_violation_count": budget_violation_count,
        "finished_under_budget": 0 if budget_hit else 1,
        "controller_structural_complexity": float(
            score_features.get("structural_complexity", 0.0) or 0.0
        ),
        "controller_uncertainty_need": float(
            score_features.get("uncertainty_need", 0.0) or 0.0
        ),
        "controller_budget_pressure": float(scorecard.get("budget_pressure", 0.0) or 0.0)
        if isinstance(scorecard, dict)
        else 0.0,
        "controller_selected_utility": float(scorecard.get("selected_utility", 0.0) or 0.0)
        if isinstance(scorecard, dict)
        else 0.0,
        "controller_utility_margin": float(scorecard.get("utility_margin", 0.0) or 0.0)
        if isinstance(scorecard, dict)
        else 0.0,
        "controller_expected_quality": float(
            selected_score.get("expected_quality", 0.0) or 0.0
        ),
        "controller_realized_expected_quality": float(
            realized_expected_quality
        ),
        "controller_calibration_error": controller_calibration_error,
        "controller_calibration_error_abs": round(abs(controller_calibration_error), 4),
        "controller_direct_eligible": (
            1 if bool(scorecard.get("direct_eligible", False)) else 0
        )
        if isinstance(scorecard, dict)
        else 0,
        "controller_recursive_upfront_allowed": (
            1 if bool(scorecard.get("recursive_upfront_allowed", False)) else 0
        )
        if isinstance(scorecard, dict)
        else 0,
        "controller_evidence_sensitive_medium": (
            1 if bool(scorecard.get("evidence_sensitive_medium", False)) else 0
        )
        if isinstance(scorecard, dict)
        else 0,
        "controller_recursive_cost_efficient": (
            1 if bool(scorecard.get("recursive_cost_efficient", False)) else 0
        )
        if isinstance(scorecard, dict)
        else 0,
        "controller_recursive_quality_advantage": float(
            scorecard.get("recursive_quality_advantage", 0.0) or 0.0
        )
        if isinstance(scorecard, dict)
        else 0.0,
        "controller_recursive_marginal_quality_per_1k_token": float(
            scorecard.get("recursive_marginal_quality_per_1k_token", 0.0) or 0.0
        )
        if isinstance(scorecard, dict)
        else 0.0,
        "controller_policy_adjustments": (
            "|".join(scorecard.get("policy_adjustments", []) or [])
        )
        if isinstance(scorecard, dict) and not forced_mode_applied
        else "",
        "controller_policy_adjustment_count": len(
            scorecard.get("policy_adjustments", []) or []
        )
        if isinstance(scorecard, dict) and not forced_mode_applied
        else 0,
        "adaptive_retry_enabled": adaptive_retry_enabled,
        "adaptive_checkpoint_enabled": adaptive_checkpoint_enabled,
        "decomposition_node_count": _safe_int(graph_metrics.get("node_count", 0), 0),
        "decomposition_edge_count": _safe_int(graph_metrics.get("edge_count", 0), 0),
        "decomposition_atomicity_ratio": float(
            graph_metrics.get("atomicity_ratio", 0.0) or 0.0
        ),
        "validation_evidence_items_raw": _safe_int(
            validator_audit.get("evidence_items_raw", 0), 0
        ),
        "validation_evidence_items_used": _safe_int(
            validator_audit.get("evidence_items_used", 0), 0
        ),
        "retry_allowed_count": retry_allowed_count,
        "retry_blocked_count": retry_blocked_count,
        "retry_expected_gain_last": float(
            last_retry_decision.get("retry_expected_gain", 0.0) or 0.0
        ),
        "retry_roi_last": float(last_retry_decision.get("retry_roi", 0.0) or 0.0),
        "initial_validation_score": initial_validation_score,
        "latest_validation_score": latest_validation_score,
        "retry_effectiveness": retry_effectiveness,
        "raw_retry_score_delta": raw_retry_score_delta,
        "checkpoint_saved_score": checkpoint_saved_score,
        "focused_repair_search_count": focused_repair_search_count,
        "focused_repair_source_count": focused_repair_source_count,
        "micro_repair_count": micro_repair_count,
        "micro_repair_source_count": micro_repair_source_count,
        "micro_repair_search_replace_count": micro_repair_search_replace_count,
        "micro_repair_qualify_remove_count": micro_repair_qualify_remove_count,
        "micro_repair_tokens": micro_repair_tokens,
        "repair_gain_per_1k_tokens": repair_gain_per_1k_tokens,
        "lightweight_repair_validation_count": lightweight_repair_validation_count,
        "repair_validation_tokens": repair_validation_tokens,
        "repair_patch_accepted_count": repair_patch_accepted_count,
        "repair_validator_escalation_count": repair_validator_escalation_count,
        "repair_cascade_score_delta": repair_cascade_score_delta,
        "validation_checkpoint_count": len(validation_snapshots),
        "best_validation_score": _safe_int(
            checkpoint_selection.get("best_validation_score", reliability_score),
            reliability_score,
        ),
        "selected_previous_checkpoint": (
            1 if bool(checkpoint_selection.get("selected_previous_checkpoint", False)) else 0
        ),
        "checkpoint_score_delta_vs_current": float(
            checkpoint_selection.get("score_delta_vs_current", 0.0) or 0.0
        ),
    }
    return metrics


def _attach_relative_metrics(run_rows: List[Dict[str, Any]]) -> None:
    """Attach per-run comparative metrics against the single-agent baseline."""
    baseline_by_key: Dict[tuple[int, int], Dict[str, Any]] = {}
    for row in run_rows:
        if row.get("strategy") != "single_agent":
            continue
        key = (_safe_int(row.get("idea_index", 0), 0), _safe_int(row.get("run_index", 0), 0))
        baseline_by_key[key] = row.get("metrics", {})

    for row in run_rows:
        metrics = row.get("metrics", {})
        key = (_safe_int(row.get("idea_index", 0), 0), _safe_int(row.get("run_index", 0), 0))
        baseline = baseline_by_key.get(key)
        if baseline is None:
            metrics["reliability_gain_vs_single"] = 0.0
            metrics["depth_delta_vs_single"] = 0.0
            metrics["depth_vs_reliability_gain"] = 0.0
            continue

        rel_gain = float(metrics.get("reliability_score", 0.0)) - float(
            baseline.get("reliability_score", 0.0)
        )
        depth_delta = float(metrics.get("decomposition_depth_realized", 0.0)) - float(
            baseline.get("decomposition_depth_realized", 0.0)
        )
        if abs(depth_delta) < 1e-9:
            depth_vs_rel_gain = rel_gain
        else:
            depth_vs_rel_gain = rel_gain / depth_delta
        metrics["reliability_gain_vs_single"] = round(rel_gain, 4)
        metrics["depth_delta_vs_single"] = round(depth_delta, 4)
        metrics["depth_vs_reliability_gain"] = round(depth_vs_rel_gain, 4)


def _aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(rec["strategy"], []).append(rec)

    numeric_fields = [
        "runtime_seconds",
        "reliability_score",
        "claims_total",
        "minimum_claims_required",
        "validation_claim_coverage",
        "low_claim_count_flag",
        "claim_count_penalty",
        "supported_ratio",
        "market_sources_count",
        "tool_calls",
        "retry_count",
        "decomposition_depth_target",
        "decomposition_depth_realized",
        "prompt_tokens_total",
        "completion_tokens_total",
        "actual_total_tokens",
        "reliability_per_1k_actual_token",
        "reliability_per_second",
        "judge_agreement",
        "judge_score_delta_abs",
        "cross_model_judging",
        "secondary_judge_fallback",
        "reliability_gain_vs_single",
        "depth_delta_vs_single",
        "depth_vs_reliability_gain",
        "budget_hit",
        "budget_violation_count",
        "finished_under_budget",
        "controller_escalated",
        "controller_structural_complexity",
        "controller_uncertainty_need",
        "controller_budget_pressure",
        "controller_selected_utility",
        "controller_utility_margin",
        "controller_expected_quality",
        "controller_realized_expected_quality",
        "controller_calibration_error",
        "controller_calibration_error_abs",
        "controller_evidence_sensitive_medium",
        "adaptive_retry_enabled",
        "adaptive_checkpoint_enabled",
        "controller_recursive_cost_efficient",
        "controller_recursive_quality_advantage",
        "controller_recursive_marginal_quality_per_1k_token",
        "decomposition_node_count",
        "decomposition_edge_count",
        "decomposition_atomicity_ratio",
        "validation_evidence_items_raw",
        "validation_evidence_items_used",
        "retry_allowed_count",
        "retry_blocked_count",
        "retry_expected_gain_last",
        "retry_roi_last",
        "initial_validation_score",
        "latest_validation_score",
        "retry_effectiveness",
        "raw_retry_score_delta",
        "checkpoint_saved_score",
        "focused_repair_search_count",
        "focused_repair_source_count",
        "micro_repair_count",
        "micro_repair_source_count",
        "micro_repair_search_replace_count",
        "micro_repair_qualify_remove_count",
        "micro_repair_tokens",
        "repair_gain_per_1k_tokens",
        "lightweight_repair_validation_count",
        "repair_validation_tokens",
        "repair_patch_accepted_count",
        "repair_validator_escalation_count",
        "repair_cascade_score_delta",
        "validation_checkpoint_count",
        "best_validation_score",
        "selected_previous_checkpoint",
        "checkpoint_score_delta_vs_current",
    ]
    aggregate: Dict[str, Dict[str, float]] = {}
    for strategy, rows in grouped.items():
        strat_summary: Dict[str, float] = {"runs": float(len(rows))}
        for field in numeric_fields:
            values = [float(r[field]) for r in rows]
            strat_summary[f"{field}_mean"] = round(statistics.fmean(values), 4)
            strat_summary[f"{field}_min"] = round(min(values), 4)
            strat_summary[f"{field}_max"] = round(max(values), 4)
            strat_summary[f"{field}_std"] = (
                round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
            )

        mode_counts: Dict[str, int] = {}
        initial_mode_counts: Dict[str, int] = {}
        realized_mode_counts: Dict[str, int] = {}
        for row in rows:
            mode = str(row.get("controller_mode", "n/a")).strip().lower() or "n/a"
            initial_mode = (
                str(row.get("controller_mode_initial", mode)).strip().lower() or "n/a"
            )
            realized_mode = (
                str(row.get("controller_mode_realized", mode)).strip().lower() or "n/a"
            )
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            initial_mode_counts[initial_mode] = initial_mode_counts.get(initial_mode, 0) + 1
            realized_mode_counts[realized_mode] = realized_mode_counts.get(realized_mode, 0) + 1
        strat_summary["mode_distribution"] = mode_counts
        strat_summary["initial_mode_distribution"] = initial_mode_counts
        strat_summary["realized_mode_distribution"] = realized_mode_counts
        if rows:
            strat_summary["mode_distribution_pct"] = {
                mode: round(count / len(rows), 4) for mode, count in mode_counts.items()
            }
            strat_summary["initial_mode_distribution_pct"] = {
                mode: round(count / len(rows), 4)
                for mode, count in initial_mode_counts.items()
            }
            strat_summary["realized_mode_distribution_pct"] = {
                mode: round(count / len(rows), 4)
                for mode, count in realized_mode_counts.items()
            }
        aggregate[strategy] = strat_summary
    return aggregate


def _build_recommendation(aggregate: Dict[str, Dict[str, Any]]) -> Dict[str, Any] | None:
    if not aggregate:
        return None

    quality_ranked = sorted(
        aggregate.items(),
        key=lambda kv: (
            kv[1].get("budget_violation_count_mean", float("inf")),
            -kv[1].get("reliability_score_mean", 0.0),
            kv[1].get("runtime_seconds_mean", float("inf")),
            kv[1].get("actual_total_tokens_mean", float("inf")),
        ),
    )
    efficiency_ranked = sorted(
        aggregate.items(),
        key=lambda kv: (
            kv[1].get("budget_violation_count_mean", float("inf")),
            -kv[1].get("reliability_per_1k_actual_token_mean", 0.0),
            -kv[1].get("reliability_score_mean", 0.0),
            kv[1].get("actual_total_tokens_mean", float("inf")),
        ),
    )
    balanced_ranked = sorted(
        aggregate.items(),
        key=lambda kv: (
            kv[1].get("budget_violation_count_mean", float("inf")),
            -(
                0.65 * kv[1].get("reliability_score_mean", 0.0)
                + 0.35 * kv[1].get("reliability_per_1k_actual_token_mean", 0.0)
            ),
            kv[1].get("actual_total_tokens_mean", float("inf")),
        ),
    )
    best_name, best_stats = quality_ranked[0]
    efficiency_name, efficiency_stats = efficiency_ranked[0]
    balanced_name, balanced_stats = balanced_ranked[0]
    return {
        "best_strategy": best_name,
        "selection_rule": (
            "quality-first: lowest budget violations, then highest mean reliability "
            "score, then lower runtime and actual total tokens"
        ),
        "stats": best_stats,
        "best_quality_strategy": best_name,
        "best_quality_stats": best_stats,
        "best_efficiency_strategy": efficiency_name,
        "best_efficiency_rule": (
            "efficiency-first: lowest budget violations, then highest reliability "
            "per 1K actual tokens"
        ),
        "best_efficiency_stats": efficiency_stats,
        "best_balanced_strategy": balanced_name,
        "best_balanced_rule": (
            "balanced score: 0.65*mean reliability + 0.35*mean reliability per "
            "1K actual tokens after budget-violation filtering"
        ),
        "best_balanced_stats": balanced_stats,
    }


def _apply_budget_posthoc(
    state: Dict[str, Any],
    runtime_seconds: float,
    max_tool_calls: int | None,
    max_token_proxy: int | None,
    max_total_tokens: int | None,
    max_runtime_seconds: float | None,
) -> Dict[str, Any]:
    merged = dict(state)
    reasons = list(merged.get("budget_hit_reasons", []) or [])
    budget_hit = bool(merged.get("budget_hit", False))
    tool_calls = len(merged.get("tool_audit", []) or [])
    token_proxy = _estimate_tokens_proxy(merged)
    token_usage = merged.get("token_usage", {}) or {}
    prompt_tokens = _safe_int(token_usage.get("prompt_tokens", 0), 0)
    completion_tokens = _safe_int(token_usage.get("completion_tokens", 0), 0)
    total_tokens = _safe_int(token_usage.get("total_tokens", 0), 0)

    if max_tool_calls is not None and max_tool_calls >= 0 and tool_calls > max_tool_calls:
        budget_hit = True
        reasons.append(f"max_tool_calls_exceeded:{tool_calls}>{max_tool_calls}")
    if max_token_proxy is not None and max_token_proxy >= 0 and token_proxy > max_token_proxy:
        budget_hit = True
        reasons.append(f"max_token_proxy_exceeded:{token_proxy}>{max_token_proxy}")
    if max_total_tokens is not None and max_total_tokens >= 0 and total_tokens > max_total_tokens:
        budget_hit = True
        reasons.append(f"max_total_tokens_exceeded:{total_tokens}>{max_total_tokens}")
    if (
        max_runtime_seconds is not None
        and max_runtime_seconds >= 0
        and runtime_seconds > max_runtime_seconds
    ):
        budget_hit = True
        reasons.append(
            f"max_runtime_seconds_exceeded:{runtime_seconds:.3f}>{float(max_runtime_seconds):.3f}"
        )

    # Deduplicate by category prefix so runtime formatting jitter doesn't duplicate reasons.
    dedup = []
    seen_prefix = set()
    for reason in reasons:
        prefix = str(reason).split(":", 1)[0]
        if prefix not in seen_prefix:
            seen_prefix.add(prefix)
            dedup.append(reason)

    merged["max_tool_calls"] = max_tool_calls
    merged["max_token_proxy"] = max_token_proxy
    merged["max_total_tokens"] = max_total_tokens
    merged["max_runtime_seconds"] = max_runtime_seconds
    merged["runtime_elapsed_seconds"] = round(runtime_seconds, 3)
    merged["tool_calls_current"] = tool_calls
    merged["token_proxy_current"] = token_proxy
    merged["prompt_tokens_current"] = prompt_tokens
    merged["completion_tokens_current"] = completion_tokens
    merged["total_tokens_current"] = total_tokens
    merged["budget_hit"] = budget_hit
    merged["budget_hit_reasons"] = dedup
    merged["budget_remaining"] = {
        "tool_calls": None if max_tool_calls is None else max_tool_calls - tool_calls,
        "token_proxy": None if max_token_proxy is None else max_token_proxy - token_proxy,
        "total_tokens": (
            None if max_total_tokens is None else max_total_tokens - total_tokens
        ),
        "runtime_seconds": (
            None
            if max_runtime_seconds is None
            else round(float(max_runtime_seconds) - runtime_seconds, 3)
        ),
    }
    return merged


class SingleAgentPitchRunner:
    """Single-pass baseline for methodology comparisons."""

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        secondary_judge_model: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        enable_trends: bool = True,
        output_dir: str = "output",
    ) -> None:
        llm = ChatOpenAI(model=model, temperature=temperature, seed=seed)
        secondary_judge_llm = (
            ChatOpenAI(model=secondary_judge_model, temperature=temperature, seed=seed + 101)
            if secondary_judge_model
            else None
        )
        self.llm = llm.with_structured_output(SingleAgentOutput, include_raw=True)
        self.validator = SourceValidatorAgent(
            ChatOpenAI(model=model, temperature=temperature, seed=seed),
            secondary_judge_llm=secondary_judge_llm,
        )
        self.pitch = PitchDeckGeneratorAgent(
            ChatOpenAI(model=model, temperature=temperature, seed=seed),
            output_dir=output_dir,
        )
        self.search_tool = MarketSearchTool()
        self.trends_tool = GoogleTrendsTool() if enable_trends else None
        self.scenario_tool = ScenarioAnalysisTool()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "# Role\n"
                    "You are a single-pass startup analyst baseline.\n\n"
                    "# Instructions\n"
                    "- Produce refined idea, market analysis, and business model in one pass.\n"
                    "- Also propose bounded Year-1 assumptions (users/arpu/margin).\n"
                    "- Keep outputs concise and investor-oriented.\n\n"
                    "# Output Format\n"
                    "- Return content that strictly matches the structured schema fields.",
                ),
                (
                    "human",
                    "Raw startup idea: {idea}\n\n"
                    "Generate refined idea, market analysis, business model, and bounded assumptions.",
                ),
            ]
        )

    def run(
        self,
        idea: str,
        validation_threshold: int = 70,
        max_tool_calls: int | None = None,
        max_token_proxy: int | None = None,
        max_total_tokens: int | None = None,
        max_runtime_seconds: float | None = None,
        generate_ppt: bool = False,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        tool_audit: List[Dict[str, Any]] = []

        def budget_view(state_like: Dict[str, Any]) -> Dict[str, Any]:
            elapsed = max(0.0, time.perf_counter() - started)
            calls = len(state_like.get("tool_audit", []))
            token_proxy = _estimate_tokens_proxy(state_like)
            token_usage = state_like.get("token_usage", {}) or {}
            prompt_tokens = _safe_int(token_usage.get("prompt_tokens", 0), 0)
            completion_tokens = _safe_int(token_usage.get("completion_tokens", 0), 0)
            total_tokens = _safe_int(token_usage.get("total_tokens", 0), 0)
            reasons: List[str] = []
            if max_tool_calls is not None and max_tool_calls >= 0 and calls > max_tool_calls:
                reasons.append(f"max_tool_calls_exceeded:{calls}>{max_tool_calls}")
            if (
                max_token_proxy is not None
                and max_token_proxy >= 0
                and token_proxy > max_token_proxy
            ):
                reasons.append(f"max_token_proxy_exceeded:{token_proxy}>{max_token_proxy}")
            if (
                max_total_tokens is not None
                and max_total_tokens >= 0
                and total_tokens > max_total_tokens
            ):
                reasons.append(f"max_total_tokens_exceeded:{total_tokens}>{max_total_tokens}")
            if (
                max_runtime_seconds is not None
                and max_runtime_seconds >= 0
                and elapsed > max_runtime_seconds
            ):
                reasons.append(
                    f"max_runtime_seconds_exceeded:{elapsed:.3f}>{float(max_runtime_seconds):.3f}"
                )
            return {
                "runtime_elapsed_seconds": round(elapsed, 3),
                "tool_calls_current": calls,
                "token_proxy_current": token_proxy,
                "prompt_tokens_current": prompt_tokens,
                "completion_tokens_current": completion_tokens,
                "total_tokens_current": total_tokens,
                "budget_hit": bool(reasons),
                "budget_hit_reasons": reasons,
                "budget_remaining": {
                    "tool_calls": None if max_tool_calls is None else max_tool_calls - calls,
                    "token_proxy": (
                        None if max_token_proxy is None else max_token_proxy - token_proxy
                    ),
                    "total_tokens": (
                        None
                        if max_total_tokens is None
                        else max_total_tokens - total_tokens
                    ),
                    "runtime_seconds": (
                        None
                        if max_runtime_seconds is None
                        else round(float(max_runtime_seconds) - elapsed, 3)
                    ),
                },
            }

        def apply_budget(state_like: Dict[str, Any], node_name: str) -> Dict[str, Any]:
            view = budget_view(state_like)
            if view["budget_hit"]:
                audit = list(state_like.get("tool_audit", []))
                audit.append(
                    {
                        "agent": "budget_guard",
                        "tool": "runtime_budget_check",
                        "status": "halted",
                        "node": node_name,
                        "reasons": view["budget_hit_reasons"],
                    }
                )
                state_like["tool_audit"] = audit
            state_like.update(view)
            return state_like

        result, usage = _invoke_structured_with_usage(
            self.llm, self.prompt.format_messages(idea=idea)
        )
        tool_audit.append(
            {
                "agent": "single_agent",
                "tool": "llm_single_pass",
                "status": "ok",
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        )

        query = f"startup market size competitors trends for: {result.refined_idea}"
        search_payload = self.search_tool.search(query)
        tool_audit.append(
            {
                "agent": "single_agent",
                "tool": "linkup_search",
                "status": search_payload.get("status", "unknown"),
                "source_count": len(search_payload.get("sources", [])),
                "error": search_payload.get("error", ""),
            }
        )

        keywords = [
            "startup market",
            "industry trends",
            "competitor landscape",
            "customer adoption",
            "automation demand",
        ]
        if self.trends_tool is not None:
            trends_payload = self.trends_tool.fetch(keywords)
        else:
            trends_payload = {
                "status": "skipped",
                "keywords": keywords,
                "data": {},
                "error": "Google Trends disabled by configuration.",
            }
        tool_audit.append(
            {
                "agent": "single_agent",
                "tool": "google_trends",
                "status": trends_payload.get("status", "unknown"),
                "keyword_count": len(trends_payload.get("keywords", [])),
                "error": trends_payload.get("error", ""),
            }
        )

        assumptions = {
            "users_year1": int(result.users_year1),
            "arpu_monthly": round(float(result.arpu_monthly), 2),
            "gross_margin": round(float(result.gross_margin), 3),
            "rationale": result.assumptions_rationale,
        }
        scenario = self.scenario_tool.run(
            users_year1=assumptions["users_year1"],
            arpu_monthly=assumptions["arpu_monthly"],
            gross_margin=assumptions["gross_margin"],
        )

        baseline_state: Dict[str, Any] = {
            "idea": idea,
            "refined_idea": result.refined_idea,
            "market_analysis": result.market_analysis,
            "market_sources": search_payload.get("sources", []),
            "market_evidence": search_payload.get("results", []),
            "trend_signals": trends_payload,
            "business_model": (
                f"{result.business_model}\n"
                f"Financial Assumptions:\n{json.dumps(assumptions, indent=2)}\n"
                f"Scenario Analysis:\n{json.dumps(scenario, indent=2)}"
            ),
            "financial_assumptions": assumptions,
            "scenario_analysis": scenario,
            "retry_count": 0,
            "max_validation_retries": 0,
            "validation_threshold": validation_threshold,
            "tool_audit": tool_audit,
            "token_usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "max_tool_calls": max_tool_calls,
            "max_token_proxy": max_token_proxy,
            "max_total_tokens": max_total_tokens,
            "max_runtime_seconds": max_runtime_seconds,
        }
        baseline_state = apply_budget(baseline_state, "single_agent_synthesis")
        if baseline_state.get("budget_hit"):
            return baseline_state

        # Reuse the same validator logic for fair quality scoring across strategies.
        validated_update = self.validator.run(baseline_state)
        baseline_state.update(validated_update)
        baseline_state = apply_budget(baseline_state, "single_agent_validator")
        if baseline_state.get("budget_hit"):
            return baseline_state

        if generate_ppt:
            pitch_update = self.pitch.run(baseline_state)
            baseline_state.update(pitch_update)
            baseline_state = apply_budget(baseline_state, "single_agent_pitch")

        return baseline_state


def _shared_refinement_for_run(
    *,
    idea: str,
    model: str,
    temperature: float,
    seed: int,
    run_idx: int,
) -> Dict[str, Any]:
    """Generate one shared refined idea for graph-based strategy branches."""
    llm = ChatOpenAI(model=model, temperature=temperature, seed=seed + run_idx)
    agent = IdeaRefinementAgent(llm)
    update = agent.run({"idea": idea, "token_usage": {}})
    refined = str(update.get("refined_idea", "") or "")
    usage = dict(update.get("token_usage", {}) or {})
    return {
        "refined_idea": refined,
        "token_usage": usage,
        "tool_audit": [
            {
                "agent": "shared_refinement",
                "tool": "llm_idea_refinement",
                "status": "ok",
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        ],
    }


def run_methodology_comparison(
    idea: str,
    model: str = "gpt-4.1-nano",
    secondary_judge_model: str | None = None,
    temperature: float = 0.0,
    seed: int = 42,
    strict_tools: bool = True,
    enable_trends: bool = True,
    compare_runs: int = 1,
    strategies: List[str] | None = None,
    validation_threshold: int = 70,
    max_validation_retries: int = 1,
    max_tool_calls: int | None = None,
    max_token_proxy: int | None = None,
    max_total_tokens: int | None = None,
    max_runtime_seconds: float | None = None,
    generate_ppt: bool = False,
    thread_prefix: str = "methodology",
    output_dir: str = "output",
) -> Dict[str, Any]:
    strategies = strategies or [
        "single_agent",
        "fixed_shallow",
        "fixed_recursive",
        "adaptive_controller",
    ]
    normalized = [s.strip().lower() for s in strategies if s.strip()]

    allowed = {
        "single_agent",
        "multi_agent",
        "fixed_direct",
        "fixed_shallow",
        "fixed_recursive",
        "adaptive_no_retry",
        "adaptive_no_checkpoint",
        "adaptive_controller",
    }
    invalid = [s for s in normalized if s not in allowed]
    if invalid:
        raise ValueError(f"Unsupported strategies: {invalid}. Allowed: {sorted(allowed)}")

    single_runner = None
    if "single_agent" in normalized:
        single_runner = SingleAgentPitchRunner(
            model=model,
            secondary_judge_model=secondary_judge_model,
            temperature=temperature,
            seed=seed,
            enable_trends=enable_trends,
            output_dir=output_dir,
        )

    multi_runner = None
    if "multi_agent" in normalized:
        multi_runner = StartupPitchRefinery(
            model=model,
            secondary_judge_model=secondary_judge_model,
            temperature=temperature,
            seed=seed,
            strict_tools=strict_tools,
            enable_trends=enable_trends,
            generate_pitch=generate_ppt,
            output_dir=output_dir,
        )

    adaptive_runner = None
    adaptive_strategy_names = {
        "adaptive_controller",
        "adaptive_no_retry",
        "adaptive_no_checkpoint",
    }
    if any(s in normalized for s in adaptive_strategy_names) or any(
        s in normalized for s in {"fixed_direct", "fixed_shallow", "fixed_recursive"}
    ):
        adaptive_runner = StartupPitchRefinery(
            model=model,
            secondary_judge_model=secondary_judge_model,
            temperature=temperature,
            seed=seed,
            strict_tools=strict_tools,
            enable_trends=enable_trends,
            generate_pitch=generate_ppt,
            controller_policy="adaptive",
            output_dir=output_dir,
        )

    run_rows: List[Dict[str, Any]] = []
    states: Dict[str, Dict[str, Any]] = {}
    graph_based_strategies = {
        "multi_agent",
        "fixed_direct",
        "fixed_shallow",
        "fixed_recursive",
        "adaptive_no_retry",
        "adaptive_no_checkpoint",
        "adaptive_controller",
    }
    use_shared_refinement = any(strategy in graph_based_strategies for strategy in normalized)

    for run_idx in range(compare_runs):
        shared_refinement = (
            _shared_refinement_for_run(
                idea=idea,
                model=model,
                temperature=temperature,
                seed=seed,
                run_idx=run_idx,
            )
            if use_shared_refinement
            else {}
        )
        paired_adaptive_no_retry_state: Dict[str, Any] | None = None
        paired_adaptive_no_retry_runtime = 0.0
        for strategy in normalized:
            start = time.perf_counter()
            if strategy == "single_agent":
                assert single_runner is not None
                state = single_runner.run(
                    idea=idea,
                    validation_threshold=validation_threshold,
                    max_tool_calls=max_tool_calls,
                    max_token_proxy=max_token_proxy,
                    max_total_tokens=max_total_tokens,
                    max_runtime_seconds=max_runtime_seconds,
                    generate_ppt=generate_ppt,
                )
            elif strategy == "multi_agent":
                assert multi_runner is not None
                state = multi_runner.run(
                    idea=idea,
                    thread_id=f"{thread_prefix}-{strategy}-{run_idx}",
                    max_validation_retries=max_validation_retries,
                    validation_threshold=validation_threshold,
                    max_tool_calls=max_tool_calls,
                    max_token_proxy=max_token_proxy,
                    max_total_tokens=max_total_tokens,
                    max_runtime_seconds=max_runtime_seconds,
                    shared_refined_idea=shared_refinement.get("refined_idea"),
                    shared_refinement_token_usage=shared_refinement.get("token_usage"),
                    shared_refinement_tool_audit=shared_refinement.get("tool_audit"),
                )
            elif strategy in {"fixed_direct", "fixed_shallow", "fixed_recursive"}:
                assert adaptive_runner is not None
                forced = strategy.replace("fixed_", "")
                state = adaptive_runner.run(
                    idea=idea,
                    thread_id=f"{thread_prefix}-{strategy}-{run_idx}",
                    max_validation_retries=max_validation_retries,
                    validation_threshold=validation_threshold,
                    max_tool_calls=max_tool_calls,
                    max_token_proxy=max_token_proxy,
                    max_total_tokens=max_total_tokens,
                    max_runtime_seconds=max_runtime_seconds,
                    forced_controller_mode=forced,
                    shared_refined_idea=shared_refinement.get("refined_idea"),
                    shared_refinement_token_usage=shared_refinement.get("token_usage"),
                    shared_refinement_tool_audit=shared_refinement.get("tool_audit"),
                )
            else:
                assert adaptive_runner is not None
                adaptive_retry_enabled = strategy != "adaptive_no_retry"
                adaptive_checkpoint_enabled = strategy != "adaptive_no_checkpoint"
                if (
                    strategy == "adaptive_controller"
                    and paired_adaptive_no_retry_state is not None
                ):
                    state = adaptive_runner.continue_adaptive_from_validated_state(
                        paired_adaptive_no_retry_state,
                        adaptive_retry_enabled=True,
                        adaptive_checkpoint_enabled=True,
                    )
                    continuation_runtime = time.perf_counter() - start
                    start = time.perf_counter() - (
                        paired_adaptive_no_retry_runtime + continuation_runtime
                    )
                else:
                    state = adaptive_runner.run(
                        idea=idea,
                        thread_id=f"{thread_prefix}-{strategy}-{run_idx}",
                        max_validation_retries=max_validation_retries,
                        validation_threshold=validation_threshold,
                        max_tool_calls=max_tool_calls,
                        max_token_proxy=max_token_proxy,
                        max_total_tokens=max_total_tokens,
                        max_runtime_seconds=max_runtime_seconds,
                        adaptive_retry_enabled=adaptive_retry_enabled,
                        adaptive_checkpoint_enabled=adaptive_checkpoint_enabled,
                        shared_refined_idea=shared_refinement.get("refined_idea"),
                        shared_refinement_token_usage=shared_refinement.get("token_usage"),
                        shared_refinement_tool_audit=shared_refinement.get("tool_audit"),
                    )

            runtime = time.perf_counter() - start
            state = _apply_budget_posthoc(
                state=state,
                runtime_seconds=runtime,
                max_tool_calls=max_tool_calls,
                max_token_proxy=max_token_proxy,
                max_total_tokens=max_total_tokens,
                max_runtime_seconds=max_runtime_seconds,
            )
            if strategy == "adaptive_no_retry":
                paired_adaptive_no_retry_state = dict(state)
                paired_adaptive_no_retry_runtime = runtime
            metrics = _extract_metrics(state, strategy, runtime)
            run_key = f"{strategy}_run_{run_idx}"
            states[run_key] = state
            run_rows.append(
                {
                    "idea_index": 0,
                    "idea": idea,
                    "run_index": run_idx,
                    "strategy": strategy,
                    "metrics": metrics,
                }
            )

    _attach_relative_metrics(run_rows)
    aggregate = _aggregate_metrics([row["metrics"] for row in run_rows])

    recommendation = _build_recommendation(aggregate)

    return {
        "metadata": {
            "idea": idea,
            "model": model,
            "secondary_judge_model": secondary_judge_model or model,
            "temperature": temperature,
            "seed": seed,
            "strategies": normalized,
            "compare_runs": compare_runs,
            "validation_threshold": validation_threshold,
            "max_validation_retries": max_validation_retries,
            "max_tool_calls": max_tool_calls,
            "max_token_proxy": max_token_proxy,
            "max_total_tokens": max_total_tokens,
            "max_runtime_seconds": max_runtime_seconds,
            "generated_at_unix": int(time.time()),
        },
        "runs": run_rows,
        "aggregate": aggregate,
        "recommendation": recommendation,
        "states": states,
    }


def save_methodology_report(report: Dict[str, Any], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(path)


def save_methodology_csvs(
    report: Dict[str, Any],
    run_csv_path: str,
    aggregate_csv_path: str,
    summary_json_path: str,
) -> Dict[str, str]:
    def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    run_path = Path(run_csv_path)
    agg_path = Path(aggregate_csv_path)
    summary_path = Path(summary_json_path)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    run_rows = report.get("runs", [])
    run_flat_rows: List[Dict[str, Any]] = []
    for row in run_rows:
        flat = {
            "idea_index": row.get("idea_index", 0),
            "idea_id": row.get("idea_id", ""),
            "difficulty": row.get("difficulty", ""),
            "domain": row.get("domain", ""),
            "idea": row.get("idea", report.get("metadata", {}).get("idea", "")),
            "run_index": row.get("run_index"),
            "strategy": row.get("strategy"),
        }
        metrics = row.get("metrics", {})
        for k, v in metrics.items():
            flat[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
        run_flat_rows.append(flat)

    run_fieldnames = sorted({k for row in run_flat_rows for k in row.keys()})
    _write_csv(run_path, run_flat_rows, run_fieldnames)

    aggregate = report.get("aggregate", {})
    agg_flat_rows: List[Dict[str, Any]] = []
    for strategy, stats in aggregate.items():
        flat = {"strategy": strategy}
        for k, v in stats.items():
            flat[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
        agg_flat_rows.append(flat)

    agg_fieldnames = sorted({k for row in agg_flat_rows for k in row.keys()})
    _write_csv(agg_path, agg_flat_rows, agg_fieldnames)

    # Compact CSV exports for reporting (lower-column views).
    run_compact_path = run_path.with_name(f"{run_path.stem}_compact{run_path.suffix}")
    agg_compact_path = agg_path.with_name(f"{agg_path.stem}_compact{agg_path.suffix}")

    run_compact_columns = [
        "idea_index",
        "idea_id",
        "difficulty",
        "domain",
        "idea",
        "run_index",
        "strategy",
        "controller_mode_initial",
        "controller_mode_realized",
        "controller_mode",
        "controller_escalated",
        "controller_structural_complexity",
        "controller_uncertainty_need",
        "controller_utility_margin",
        "controller_expected_quality",
        "controller_realized_expected_quality",
        "controller_calibration_error",
        "controller_calibration_error_abs",
        "controller_direct_eligible",
        "controller_recursive_upfront_allowed",
        "controller_evidence_sensitive_medium",
        "controller_recursive_cost_efficient",
        "controller_recursive_quality_advantage",
        "controller_recursive_marginal_quality_per_1k_token",
        "controller_policy_adjustments",
        "adaptive_retry_enabled",
        "adaptive_checkpoint_enabled",
        "decomposition_node_count",
        "decomposition_edge_count",
        "decomposition_atomicity_ratio",
        "validation_evidence_items_used",
        "retry_allowed_count",
        "retry_blocked_count",
        "retry_expected_gain_last",
        "retry_roi_last",
        "retry_effectiveness",
        "raw_retry_score_delta",
        "checkpoint_saved_score",
        "focused_repair_search_count",
        "focused_repair_source_count",
        "micro_repair_count",
        "micro_repair_source_count",
        "micro_repair_search_replace_count",
        "micro_repair_qualify_remove_count",
        "micro_repair_tokens",
        "repair_gain_per_1k_tokens",
        "lightweight_repair_validation_count",
        "repair_validation_tokens",
        "repair_patch_accepted_count",
        "repair_validator_escalation_count",
        "repair_cascade_score_delta",
        "validation_checkpoint_count",
        "best_validation_score",
        "selected_previous_checkpoint",
        "checkpoint_score_delta_vs_current",
        "reliability_score",
        "claims_total",
        "validation_claim_coverage",
        "low_claim_count_flag",
        "claim_count_penalty",
        "judge_agreement",
        "judge_a_model",
        "judge_b_model",
        "cross_model_judging",
        "secondary_judge_fallback",
        "supported_ratio",
        "runtime_seconds",
        "actual_total_tokens",
        "reliability_per_1k_actual_token",
        "tool_calls",
        "decomposition_depth_realized",
        "reliability_gain_vs_single",
        "budget_hit",
    ]
    run_compact_fieldnames = [c for c in run_compact_columns if c in run_fieldnames]
    run_compact_rows = [{c: row.get(c, "") for c in run_compact_fieldnames} for row in run_flat_rows]
    _write_csv(run_compact_path, run_compact_rows, run_compact_fieldnames)

    agg_compact_columns = [
        "strategy",
        "runs",
        "reliability_score_mean",
        "claims_total_mean",
        "validation_claim_coverage_mean",
        "low_claim_count_flag_mean",
        "claim_count_penalty_mean",
        "runtime_seconds_mean",
        "actual_total_tokens_mean",
        "judge_agreement_mean",
        "cross_model_judging_mean",
        "secondary_judge_fallback_mean",
        "reliability_per_1k_actual_token_mean",
        "supported_ratio_mean",
        "reliability_score_std",
        "actual_total_tokens_std",
        "reliability_per_1k_actual_token_std",
        "decomposition_depth_realized_mean",
        "reliability_gain_vs_single_mean",
        "budget_hit_mean",
        "controller_escalated_mean",
        "controller_structural_complexity_mean",
        "controller_uncertainty_need_mean",
        "controller_utility_margin_mean",
        "controller_expected_quality_mean",
        "controller_realized_expected_quality_mean",
        "controller_calibration_error_mean",
        "controller_calibration_error_abs_mean",
        "controller_direct_eligible_mean",
        "controller_recursive_upfront_allowed_mean",
        "controller_evidence_sensitive_medium_mean",
        "controller_recursive_cost_efficient_mean",
        "controller_recursive_quality_advantage_mean",
        "controller_recursive_marginal_quality_per_1k_token_mean",
        "controller_policy_adjustment_count_mean",
        "adaptive_retry_enabled_mean",
        "adaptive_checkpoint_enabled_mean",
        "decomposition_node_count_mean",
        "decomposition_edge_count_mean",
        "decomposition_atomicity_ratio_mean",
        "validation_evidence_items_used_mean",
        "retry_allowed_count_mean",
        "retry_blocked_count_mean",
        "retry_expected_gain_last_mean",
        "retry_roi_last_mean",
        "retry_effectiveness_mean",
        "raw_retry_score_delta_mean",
        "checkpoint_saved_score_mean",
        "focused_repair_search_count_mean",
        "focused_repair_source_count_mean",
        "micro_repair_count_mean",
        "micro_repair_source_count_mean",
        "micro_repair_search_replace_count_mean",
        "micro_repair_qualify_remove_count_mean",
        "micro_repair_tokens_mean",
        "repair_gain_per_1k_tokens_mean",
        "lightweight_repair_validation_count_mean",
        "repair_validation_tokens_mean",
        "repair_patch_accepted_count_mean",
        "repair_validator_escalation_count_mean",
        "repair_cascade_score_delta_mean",
        "validation_checkpoint_count_mean",
        "best_validation_score_mean",
        "selected_previous_checkpoint_mean",
        "checkpoint_score_delta_vs_current_mean",
        "initial_mode_distribution",
        "realized_mode_distribution",
        "mode_distribution",
    ]
    agg_compact_fieldnames = [c for c in agg_compact_columns if c in agg_fieldnames]
    agg_compact_rows = [{c: row.get(c, "") for c in agg_compact_fieldnames} for row in agg_flat_rows]
    _write_csv(agg_compact_path, agg_compact_rows, agg_compact_fieldnames)

    summary = {
        "metadata": report.get("metadata", {}),
        "recommendation": report.get("recommendation"),
        "aggregate": report.get("aggregate", {}),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "run_csv": str(run_path),
        "aggregate_csv": str(agg_path),
        "run_csv_compact": str(run_compact_path),
        "aggregate_csv_compact": str(agg_compact_path),
        "summary_json": str(summary_path),
    }


def save_paper_mode_exports(
    report: Dict[str, Any],
    paper_json_path: str,
    paper_csv_path: str,
) -> Dict[str, str]:
    """
    Save a minimal paper-ready view:
    - run-level CSV with only core evaluation metrics
    - JSON with compact aggregate + recommendation
    """
    json_path = Path(paper_json_path)
    csv_path = Path(paper_csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    run_rows = report.get("runs", [])
    paper_columns = [
        "idea_index",
        "idea_id",
        "difficulty",
        "domain",
        "idea",
        "run_index",
        "strategy",
        "controller_mode_initial",
        "controller_mode_realized",
        "controller_mode",
        "controller_escalated",
        "controller_structural_complexity",
        "controller_uncertainty_need",
        "controller_utility_margin",
        "controller_expected_quality",
        "controller_realized_expected_quality",
        "controller_calibration_error",
        "controller_calibration_error_abs",
        "controller_evidence_sensitive_medium",
        "adaptive_retry_enabled",
        "adaptive_checkpoint_enabled",
        "controller_recursive_cost_efficient",
        "controller_recursive_quality_advantage",
        "controller_recursive_marginal_quality_per_1k_token",
        "decomposition_node_count",
        "decomposition_edge_count",
        "decomposition_atomicity_ratio",
        "validation_evidence_items_used",
        "retry_allowed_count",
        "retry_blocked_count",
        "retry_expected_gain_last",
        "retry_roi_last",
        "retry_effectiveness",
        "raw_retry_score_delta",
        "checkpoint_saved_score",
        "focused_repair_search_count",
        "focused_repair_source_count",
        "micro_repair_count",
        "micro_repair_source_count",
        "micro_repair_search_replace_count",
        "micro_repair_qualify_remove_count",
        "micro_repair_tokens",
        "repair_gain_per_1k_tokens",
        "lightweight_repair_validation_count",
        "repair_validation_tokens",
        "repair_patch_accepted_count",
        "repair_validator_escalation_count",
        "repair_cascade_score_delta",
        "validation_checkpoint_count",
        "best_validation_score",
        "selected_previous_checkpoint",
        "checkpoint_score_delta_vs_current",
        "reliability_score",
        "claims_total",
        "validation_claim_coverage",
        "low_claim_count_flag",
        "claim_count_penalty",
        "judge_agreement",
        "judge_a_model",
        "judge_b_model",
        "cross_model_judging",
        "secondary_judge_fallback",
        "supported_ratio",
        "runtime_seconds",
        "actual_total_tokens",
        "reliability_per_1k_actual_token",
        "decomposition_depth_realized",
        "budget_hit",
    ]

    paper_run_rows: List[Dict[str, Any]] = []
    for row in run_rows:
        metrics = row.get("metrics", {})
        flat = {
            "idea_index": row.get("idea_index", 0),
            "idea_id": row.get("idea_id", ""),
            "difficulty": row.get("difficulty", ""),
            "domain": row.get("domain", ""),
            "idea": row.get("idea", report.get("metadata", {}).get("idea", "")),
            "run_index": row.get("run_index"),
            "strategy": row.get("strategy"),
            "controller_mode_initial": metrics.get("controller_mode_initial", "n/a"),
            "controller_mode_realized": metrics.get("controller_mode_realized", "n/a"),
            "controller_mode": metrics.get("controller_mode", "n/a"),
            "controller_escalated": metrics.get("controller_escalated", 0),
            "controller_structural_complexity": metrics.get(
                "controller_structural_complexity", 0.0
            ),
            "controller_uncertainty_need": metrics.get("controller_uncertainty_need", 0.0),
            "controller_utility_margin": metrics.get("controller_utility_margin", 0.0),
            "controller_expected_quality": metrics.get("controller_expected_quality", 0.0),
            "controller_realized_expected_quality": metrics.get(
                "controller_realized_expected_quality", 0.0
            ),
            "controller_calibration_error": metrics.get(
                "controller_calibration_error", 0.0
            ),
            "controller_calibration_error_abs": metrics.get(
                "controller_calibration_error_abs", 0.0
            ),
            "controller_evidence_sensitive_medium": metrics.get(
                "controller_evidence_sensitive_medium", 0
            ),
            "adaptive_retry_enabled": metrics.get("adaptive_retry_enabled", 1),
            "adaptive_checkpoint_enabled": metrics.get("adaptive_checkpoint_enabled", 1),
            "controller_recursive_cost_efficient": metrics.get(
                "controller_recursive_cost_efficient", 0
            ),
            "controller_recursive_quality_advantage": metrics.get(
                "controller_recursive_quality_advantage", 0.0
            ),
            "controller_recursive_marginal_quality_per_1k_token": metrics.get(
                "controller_recursive_marginal_quality_per_1k_token", 0.0
            ),
            "decomposition_node_count": metrics.get("decomposition_node_count", 0),
            "decomposition_edge_count": metrics.get("decomposition_edge_count", 0),
            "decomposition_atomicity_ratio": metrics.get(
                "decomposition_atomicity_ratio", 0.0
            ),
            "validation_evidence_items_used": metrics.get(
                "validation_evidence_items_used", 0
            ),
            "retry_allowed_count": metrics.get("retry_allowed_count", 0),
            "retry_blocked_count": metrics.get("retry_blocked_count", 0),
            "retry_expected_gain_last": metrics.get("retry_expected_gain_last", 0.0),
            "retry_roi_last": metrics.get("retry_roi_last", 0.0),
            "retry_effectiveness": metrics.get("retry_effectiveness", 0.0),
            "raw_retry_score_delta": metrics.get("raw_retry_score_delta", 0.0),
            "checkpoint_saved_score": metrics.get("checkpoint_saved_score", 0.0),
            "focused_repair_search_count": metrics.get("focused_repair_search_count", 0),
            "focused_repair_source_count": metrics.get("focused_repair_source_count", 0),
            "micro_repair_count": metrics.get("micro_repair_count", 0),
            "micro_repair_source_count": metrics.get("micro_repair_source_count", 0),
            "micro_repair_search_replace_count": metrics.get(
                "micro_repair_search_replace_count", 0
            ),
            "micro_repair_qualify_remove_count": metrics.get(
                "micro_repair_qualify_remove_count", 0
            ),
            "micro_repair_tokens": metrics.get("micro_repair_tokens", 0),
            "repair_gain_per_1k_tokens": metrics.get("repair_gain_per_1k_tokens", 0.0),
            "lightweight_repair_validation_count": metrics.get(
                "lightweight_repair_validation_count", 0
            ),
            "repair_validation_tokens": metrics.get("repair_validation_tokens", 0),
            "repair_patch_accepted_count": metrics.get("repair_patch_accepted_count", 0),
            "repair_validator_escalation_count": metrics.get(
                "repair_validator_escalation_count", 0
            ),
            "repair_cascade_score_delta": metrics.get("repair_cascade_score_delta", 0),
            "validation_checkpoint_count": metrics.get("validation_checkpoint_count", 0),
            "best_validation_score": metrics.get("best_validation_score", 0),
            "selected_previous_checkpoint": metrics.get("selected_previous_checkpoint", 0),
            "checkpoint_score_delta_vs_current": metrics.get(
                "checkpoint_score_delta_vs_current", 0.0
            ),
            "reliability_score": metrics.get("reliability_score", 0),
            "claims_total": metrics.get("claims_total", 0),
            "validation_claim_coverage": metrics.get("validation_claim_coverage", 0.0),
            "low_claim_count_flag": metrics.get("low_claim_count_flag", 0),
            "claim_count_penalty": metrics.get("claim_count_penalty", 0),
            "judge_agreement": metrics.get("judge_agreement", 0.0),
            "judge_a_model": metrics.get("judge_a_model", ""),
            "judge_b_model": metrics.get("judge_b_model", ""),
            "cross_model_judging": metrics.get("cross_model_judging", 0),
            "secondary_judge_fallback": metrics.get("secondary_judge_fallback", 0),
            "supported_ratio": metrics.get("supported_ratio", 0.0),
            "runtime_seconds": metrics.get("runtime_seconds", 0.0),
            "actual_total_tokens": metrics.get("actual_total_tokens", 0),
            "reliability_per_1k_actual_token": metrics.get(
                "reliability_per_1k_actual_token", 0.0
            ),
            "decomposition_depth_realized": metrics.get(
                "decomposition_depth_realized", 0
            ),
            "budget_hit": metrics.get("budget_hit", 0),
        }
        paper_run_rows.append(flat)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=paper_columns)
        writer.writeheader()
        for row in paper_run_rows:
            writer.writerow(row)

    agg = report.get("aggregate", {})
    compact_aggregate: Dict[str, Dict[str, Any]] = {}
    for strategy, stats in agg.items():
        compact_aggregate[strategy] = {
            "runs": stats.get("runs"),
            "reliability_score_mean": stats.get("reliability_score_mean"),
            "claims_total_mean": stats.get("claims_total_mean"),
            "validation_claim_coverage_mean": stats.get(
                "validation_claim_coverage_mean"
            ),
            "low_claim_count_flag_mean": stats.get("low_claim_count_flag_mean"),
            "claim_count_penalty_mean": stats.get("claim_count_penalty_mean"),
            "judge_agreement_mean": stats.get("judge_agreement_mean"),
            "cross_model_judging_mean": stats.get("cross_model_judging_mean"),
            "secondary_judge_fallback_mean": stats.get("secondary_judge_fallback_mean"),
            "supported_ratio_mean": stats.get("supported_ratio_mean"),
            "runtime_seconds_mean": stats.get("runtime_seconds_mean"),
            "actual_total_tokens_mean": stats.get("actual_total_tokens_mean"),
            "reliability_per_1k_actual_token_mean": stats.get(
                "reliability_per_1k_actual_token_mean"
            ),
            "decomposition_depth_realized_mean": stats.get(
                "decomposition_depth_realized_mean"
            ),
            "budget_hit_mean": stats.get("budget_hit_mean"),
            "controller_escalated_mean": stats.get("controller_escalated_mean"),
            "controller_structural_complexity_mean": stats.get(
                "controller_structural_complexity_mean"
            ),
            "controller_uncertainty_need_mean": stats.get(
                "controller_uncertainty_need_mean"
            ),
            "controller_utility_margin_mean": stats.get("controller_utility_margin_mean"),
            "controller_expected_quality_mean": stats.get(
                "controller_expected_quality_mean"
            ),
            "controller_realized_expected_quality_mean": stats.get(
                "controller_realized_expected_quality_mean"
            ),
            "controller_calibration_error_mean": stats.get(
                "controller_calibration_error_mean"
            ),
            "controller_calibration_error_abs_mean": stats.get(
                "controller_calibration_error_abs_mean"
            ),
            "controller_evidence_sensitive_medium_mean": stats.get(
                "controller_evidence_sensitive_medium_mean"
            ),
            "adaptive_retry_enabled_mean": stats.get("adaptive_retry_enabled_mean"),
            "adaptive_checkpoint_enabled_mean": stats.get(
                "adaptive_checkpoint_enabled_mean"
            ),
            "controller_recursive_cost_efficient_mean": stats.get(
                "controller_recursive_cost_efficient_mean"
            ),
            "controller_recursive_quality_advantage_mean": stats.get(
                "controller_recursive_quality_advantage_mean"
            ),
            "controller_recursive_marginal_quality_per_1k_token_mean": stats.get(
                "controller_recursive_marginal_quality_per_1k_token_mean"
            ),
            "decomposition_node_count_mean": stats.get("decomposition_node_count_mean"),
            "decomposition_edge_count_mean": stats.get("decomposition_edge_count_mean"),
            "decomposition_atomicity_ratio_mean": stats.get(
                "decomposition_atomicity_ratio_mean"
            ),
            "validation_evidence_items_used_mean": stats.get(
                "validation_evidence_items_used_mean"
            ),
            "retry_allowed_count_mean": stats.get("retry_allowed_count_mean"),
            "retry_blocked_count_mean": stats.get("retry_blocked_count_mean"),
            "retry_expected_gain_last_mean": stats.get("retry_expected_gain_last_mean"),
            "retry_roi_last_mean": stats.get("retry_roi_last_mean"),
            "retry_effectiveness_mean": stats.get("retry_effectiveness_mean"),
            "raw_retry_score_delta_mean": stats.get("raw_retry_score_delta_mean"),
            "checkpoint_saved_score_mean": stats.get("checkpoint_saved_score_mean"),
            "focused_repair_search_count_mean": stats.get("focused_repair_search_count_mean"),
            "focused_repair_source_count_mean": stats.get("focused_repair_source_count_mean"),
            "micro_repair_count_mean": stats.get("micro_repair_count_mean"),
            "micro_repair_source_count_mean": stats.get("micro_repair_source_count_mean"),
            "micro_repair_search_replace_count_mean": stats.get(
                "micro_repair_search_replace_count_mean"
            ),
            "micro_repair_qualify_remove_count_mean": stats.get(
                "micro_repair_qualify_remove_count_mean"
            ),
            "micro_repair_tokens_mean": stats.get("micro_repair_tokens_mean"),
            "repair_gain_per_1k_tokens_mean": stats.get("repair_gain_per_1k_tokens_mean"),
            "lightweight_repair_validation_count_mean": stats.get(
                "lightweight_repair_validation_count_mean"
            ),
            "repair_validation_tokens_mean": stats.get("repair_validation_tokens_mean"),
            "repair_patch_accepted_count_mean": stats.get(
                "repair_patch_accepted_count_mean"
            ),
            "repair_validator_escalation_count_mean": stats.get(
                "repair_validator_escalation_count_mean"
            ),
            "repair_cascade_score_delta_mean": stats.get(
                "repair_cascade_score_delta_mean"
            ),
            "validation_checkpoint_count_mean": stats.get(
                "validation_checkpoint_count_mean"
            ),
            "best_validation_score_mean": stats.get("best_validation_score_mean"),
            "selected_previous_checkpoint_mean": stats.get(
                "selected_previous_checkpoint_mean"
            ),
            "checkpoint_score_delta_vs_current_mean": stats.get(
                "checkpoint_score_delta_vs_current_mean"
            ),
            "initial_mode_distribution": stats.get("initial_mode_distribution"),
            "realized_mode_distribution": stats.get("realized_mode_distribution"),
            "mode_distribution": stats.get("mode_distribution"),
        }

    paper_json = {
        "metadata": report.get("metadata", {}),
        "evaluation_primary_metrics": [
            "reliability_score",
            "claims_total",
            "validation_claim_coverage",
            "low_claim_count_flag",
            "claim_count_penalty",
            "judge_agreement",
            "judge_a_model",
            "judge_b_model",
            "cross_model_judging",
            "secondary_judge_fallback",
            "supported_ratio",
            "runtime_seconds",
            "actual_total_tokens",
            "reliability_per_1k_actual_token",
            "decomposition_depth_realized",
            "budget_hit",
            "controller_mode_initial",
            "controller_mode_realized",
            "controller_escalated",
            "controller_structural_complexity",
            "controller_uncertainty_need",
            "controller_utility_margin",
            "controller_expected_quality",
            "controller_realized_expected_quality",
            "controller_calibration_error",
            "controller_calibration_error_abs",
            "controller_evidence_sensitive_medium",
            "adaptive_retry_enabled",
            "adaptive_checkpoint_enabled",
            "controller_recursive_cost_efficient",
            "controller_recursive_quality_advantage",
            "controller_recursive_marginal_quality_per_1k_token",
            "decomposition_node_count",
            "decomposition_edge_count",
            "decomposition_atomicity_ratio",
            "validation_evidence_items_used",
            "retry_allowed_count",
            "retry_blocked_count",
            "retry_expected_gain_last",
            "retry_roi_last",
            "retry_effectiveness",
            "raw_retry_score_delta",
            "checkpoint_saved_score",
            "focused_repair_search_count",
        "focused_repair_source_count",
        "micro_repair_count",
        "micro_repair_source_count",
        "micro_repair_search_replace_count",
        "micro_repair_qualify_remove_count",
        "micro_repair_tokens",
        "repair_gain_per_1k_tokens",
        "lightweight_repair_validation_count",
        "repair_validation_tokens",
        "repair_patch_accepted_count",
        "repair_validator_escalation_count",
        "repair_cascade_score_delta",
        "validation_checkpoint_count",
        "best_validation_score",
        "selected_previous_checkpoint",
        "checkpoint_score_delta_vs_current",
        ],
        "aggregate": compact_aggregate,
        "recommendation": report.get("recommendation"),
    }
    json_path.write_text(json.dumps(paper_json, indent=2), encoding="utf-8")

    return {
        "paper_json": str(json_path),
        "paper_csv": str(csv_path),
    }


def _normalize_idea_records(ideas: List[Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(ideas):
        if isinstance(item, dict):
            idea = str(item.get("idea") or item.get("prompt") or "").strip()
            if not idea:
                continue
            records.append(
                {
                    "idea_index": idx,
                    "idea_id": str(item.get("idea_id") or f"idea_{idx:03d}").strip(),
                    "difficulty": str(item.get("difficulty") or "unspecified").strip(),
                    "domain": str(item.get("domain") or "unspecified").strip(),
                    "idea": idea,
                }
            )
            continue

        idea = str(item).strip()
        if idea:
            records.append(
                {
                    "idea_index": idx,
                    "idea_id": f"idea_{idx:03d}",
                    "difficulty": "unspecified",
                    "domain": "unspecified",
                    "idea": idea,
                }
            )
    return records


def run_methodology_batch_comparison(
    ideas: List[Any],
    model: str = "gpt-4.1-nano",
    secondary_judge_model: str | None = None,
    temperature: float = 0.0,
    seed: int = 42,
    strict_tools: bool = True,
    enable_trends: bool = True,
    compare_runs: int = 1,
    strategies: List[str] | None = None,
    validation_threshold: int = 70,
    max_validation_retries: int = 1,
    max_tool_calls: int | None = None,
    max_token_proxy: int | None = None,
    max_total_tokens: int | None = None,
    max_runtime_seconds: float | None = None,
    generate_ppt: bool = False,
    thread_prefix: str = "methodology",
    output_dir: str = "output",
) -> Dict[str, Any]:
    idea_records = _normalize_idea_records(ideas)
    if not idea_records:
        raise ValueError("No ideas provided for batch comparison.")

    per_idea_reports: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    all_states: Dict[str, Dict[str, Any]] = {}

    for idx, idea_record in enumerate(idea_records):
        idea = idea_record["idea"]
        report = run_methodology_comparison(
            idea=idea,
            model=model,
            secondary_judge_model=secondary_judge_model,
            temperature=temperature,
            seed=seed,
            strict_tools=strict_tools,
            enable_trends=enable_trends,
            compare_runs=compare_runs,
            strategies=strategies,
            validation_threshold=validation_threshold,
            max_validation_retries=max_validation_retries,
            max_tool_calls=max_tool_calls,
            max_token_proxy=max_token_proxy,
            max_total_tokens=max_total_tokens,
            max_runtime_seconds=max_runtime_seconds,
            generate_ppt=generate_ppt,
            thread_prefix=f"{thread_prefix}-idea-{idx}",
            output_dir=output_dir,
        )
        per_idea_reports.append(
            {
                "idea_index": idx,
                "idea_id": idea_record["idea_id"],
                "difficulty": idea_record["difficulty"],
                "domain": idea_record["domain"],
                "idea": idea,
                "aggregate": report.get("aggregate", {}),
                "recommendation": report.get("recommendation"),
            }
        )
        for row in report.get("runs", []):
            copied = dict(row)
            copied["idea_index"] = idx
            copied["idea_id"] = idea_record["idea_id"]
            copied["difficulty"] = idea_record["difficulty"]
            copied["domain"] = idea_record["domain"]
            copied["idea"] = idea
            all_rows.append(copied)
        for key, state in report.get("states", {}).items():
            all_states[f"idea_{idx}:{key}"] = state

    _attach_relative_metrics(all_rows)
    aggregate = _aggregate_metrics([row["metrics"] for row in all_rows])
    recommendation = _build_recommendation(aggregate)

    return {
        "metadata": {
            "ideas_count": len(idea_records),
            "ideas": [record["idea"] for record in idea_records],
            "idea_records": idea_records,
            "model": model,
            "secondary_judge_model": secondary_judge_model or model,
            "temperature": temperature,
            "seed": seed,
            "strategies": (
                strategies
                or [
                    "single_agent",
                    "fixed_shallow",
                    "fixed_recursive",
                    "adaptive_controller",
                ]
            ),
            "compare_runs": compare_runs,
            "validation_threshold": validation_threshold,
            "max_validation_retries": max_validation_retries,
            "max_tool_calls": max_tool_calls,
            "max_token_proxy": max_token_proxy,
            "max_total_tokens": max_total_tokens,
            "max_runtime_seconds": max_runtime_seconds,
            "generated_at_unix": int(time.time()),
        },
        "ideas": per_idea_reports,
        "runs": all_rows,
        "aggregate": aggregate,
        "recommendation": recommendation,
        "states": all_states,
    }
