from __future__ import annotations

import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from startup_pitch_refinery.agents import (
    AdaptiveControllerAgent,
    BayesianRetryPolicy,
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


DEFAULT_STRATEGIES = [
    "single_agent",
    "fixed_shallow",
    "fixed_recursive",
    "adaptive_no_retry",
    "adaptive_no_checkpoint",
    "adaptive_controller",
]

ALLOWED_STRATEGIES = {
    "single_agent",
    "multi_agent",
    "fixed_direct",
    "fixed_shallow",
    "fixed_recursive",
    "adaptive_no_retry",
    "adaptive_no_checkpoint",
    "adaptive_controller",
}

GRAPH_BASED_STRATEGIES = {
    "multi_agent",
    "fixed_direct",
    "fixed_shallow",
    "fixed_recursive",
    "adaptive_no_retry",
    "adaptive_no_checkpoint",
    "adaptive_controller",
}

ADAPTIVE_STRATEGIES = {
    "adaptive_controller",
    "adaptive_no_retry",
    "adaptive_no_checkpoint",
}

POLICY_CALIBRATION_COLUMNS = [
    "strategy",
    "bucket_key",
    "base_bucket_key",
    "prior_alpha",
    "prior_beta",
    "prior_acceptance",
    "observations",
    "accepted",
    "rejected",
    "patch_accepted",
    "patch_rejected",
    "observed_accept_rate",
    "observed_patch_accept_rate",
    "posterior_alpha",
    "posterior_beta",
    "posterior_mean",
    "ucb_bonus",
    "posterior_acceptance_ucb",
    "prior_to_posterior_delta",
    "expected_gain",
    "expected_tokens",
    "expected_seconds",
    "gain_kappa",
    "token_kappa",
    "seconds_kappa",
    "mean_observed_gain",
    "mean_observed_tokens",
    "mean_observed_roi_per_1k",
    "mean_terminal_utility_delta",
    "mean_terminal_utility_per_1k_tokens",
]

DIRECT_ENTRY_PRIORS = {
    # Priors are deliberately conservative: a direct precheck only saves tokens
    # when its probability of passing validation is high enough to offset the
    # failed-direct escalation cost.
    "simple_plain": {"alpha": 3.0, "beta": 4.0, "direct_tokens": 5600.0, "shallow_tokens": 9300.0},
    "simple_workflow": {"alpha": 2.0, "beta": 4.0, "direct_tokens": 5600.0, "shallow_tokens": 9300.0},
    "moderate": {"alpha": 2.0, "beta": 5.0, "direct_tokens": 5600.0, "shallow_tokens": 9300.0},
    "high_complexity": {"alpha": 2.0, "beta": 6.0, "direct_tokens": 5600.0, "shallow_tokens": 9300.0},
}

DIRECT_ENTRY_DEFAULT_PRIOR = DIRECT_ENTRY_PRIORS["moderate"]
DIRECT_ENTRY_MIN_ACCEPTANCE = 0.75
DIRECT_ENTRY_MIN_NET_TOKENS = 1000.0


CORE_NUMERIC_FIELDS = [
    "runtime_seconds",
    "reliability_score",
    "claims_total",
    "claim_units_total",
    "failing_claim_count",
    "unsupported_material_claim_count",
    "minimum_claims_required",
    "missing_claim_count",
    "validation_claim_coverage",
    "low_claim_count_flag",
    "claim_count_penalty",
    "supported_ratio",
    "weak_or_better_ratio",
    "market_sources_count",
    "tool_calls",
    "retry_count",
    "repair_rounds",
    "decomposition_depth_target",
    "decomposition_depth_realized",
    "prompt_tokens_total",
    "completion_tokens_total",
    "actual_total_tokens",
    "token_proxy",
    "reliability_per_1k_actual_token",
    "reliability_per_second",
    "judge_agreement",
    "judge_score_delta_abs",
    "cross_model_judging",
    "secondary_judge_fallback",
    "reliability_gain_vs_single",
    "depth_delta_vs_single",
    "depth_vs_reliability_gain",
    "reliability_gain_vs_fixed_shallow",
    "token_delta_vs_fixed_shallow",
    "efficiency_gain_vs_fixed_shallow",
    "adaptive_quality_win_vs_fixed_shallow",
    "adaptive_efficiency_win_vs_fixed_shallow",
    "budget_hit",
    "budget_violation_count",
    "finished_under_budget",
    "controller_escalated",
    "direct_precheck_allowed",
    "direct_precheck_skipped",
    "direct_precheck_escalated",
    "direct_precheck_accepted",
    "direct_precheck_score",
    "direct_precheck_incremental_tokens",
    "direct_precheck_gate_probability",
    "direct_precheck_gate_threshold",
    "direct_precheck_gate_expected_net_tokens",
    "controller_structural_complexity",
    "controller_uncertainty_need",
    "controller_budget_pressure",
    "controller_selected_utility",
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
    "controller_policy_adjustment_count",
    "adaptive_retry_enabled",
    "adaptive_checkpoint_enabled",
    "decomposition_node_count",
    "decomposition_edge_count",
    "decomposition_atomicity_ratio",
    "validation_evidence_items_raw",
    "validation_evidence_items_used",
    "retry_allowed_count",
    "retry_blocked_count",
    "claim_micro_repair_allowed_count",
    "fixed_recursive_market_retry_count",
    "shallow_escalation_count",
    "recursive_retry_count",
    "retry_expected_gain_last",
    "retry_roi_last",
    "retry_roi_estimate",
    "retry_expected_utility",
    "posterior_acceptance_probability",
    "retry_policy_candidate_count_last",
    "retry_policy_evaluated_candidate_count_last",
    "retry_policy_viable_candidate_count_last",
    "retry_policy_rejected_candidate_count_last",
    "retry_policy_blocked_by_retrieval_count_last",
    "retry_policy_blocked_by_roi_count_last",
    "retry_policy_blocked_by_expected_utility_count_last",
    "retry_policy_blocked_by_posterior_count_last",
    "retry_policy_blocked_by_token_reserve_count_last",
    "retry_policy_blocked_by_threshold_crossing_count_last",
    "baseline_shallow_reliability",
    "baseline_shallow_tokens",
    "delta_vs_shared_shallow",
    "token_delta_vs_shared_shallow",
    "posterior_acceptance_last",
    "retry_expected_utility_last",
    "retry_expected_seconds_last",
    "retry_incremental_tokens_last",
    "bayesian_retry_gate_count",
    "bayesian_retry_allowed_count",
    "positive_eu_retry_count",
    "skipped_due_to_retrieval_gate_count",
    "retrieval_diagnostic_count",
    "retrieval_diagnostic_mean_score",
    "retrieval_diagnostic_max_score",
    "coverage_enhancement_target_count",
    "retrieval_strong_evidence_count",
    "retrieval_medium_or_better_count",
    "evidence_probe_count",
    "evidence_probe_upgrade_count",
    "selected_claim_count_last",
    "selected_action_type",
    "selected_bucket_key",
    "selected_base_bucket_key",
    "selected_action_expected_gain",
    "selected_action_expected_utility",
    "selected_action_expected_tokens",
    "selected_action_expected_seconds",
    "selected_action_posterior_acceptance",
    "selected_action_posterior_mean",
    "selected_action_ucb_bonus",
    "selected_action_n_empirical",
    "selected_action_roi_per_1k",
    "selected_action_large_deficit_incremental_exception",
    "selected_action_threshold_crossing_repair_exception",
    "selected_action_source_backed_material_repair_exception",
    "selected_action_coverage_addition_exception",
    "selected_action_token_reserve_exception",
    "selected_action_effective_tail_token_reserve",
    "selected_action_reserve_after_expected_repair",
    "policy_repair_observations",
    "policy_repair_accepted_total",
    "policy_repair_rejected_total",
    "policy_repair_accept_rate",
    "policy_last_accepted",
    "policy_last_patch_accepted",
    "policy_last_success",
    "policy_last_gain",
    "policy_last_tokens",
    "policy_last_generation_tokens",
    "policy_last_validation_tokens",
    "policy_last_observed_roi_per_1k",
    "policy_last_terminal_utility_before",
    "policy_last_terminal_utility_after",
    "policy_last_terminal_utility_delta",
    "policy_last_terminal_utility_per_1k_tokens",
    "policy_last_repair_success_margin",
    "checkpoint_terminal_utility",
    "checkpoint_lcb_utility",
    "cross_model_judging_any",
    "material_failing_claim_count_last",
    "specific_repair_claim_count_last",
    "accepted_patch_count",
    "rejected_patch_count",
    "repair_outcome_accepted_count",
    "repair_outcome_rejected_count",
    "repair_outcome_policy_success_count",
    "repair_outcome_policy_failed_count",
    "repair_outcome_terminal_utility_before_last",
    "repair_outcome_terminal_utility_after_last",
    "repair_outcome_terminal_utility_delta_last",
    "repair_outcome_observed_roi_per_1k_last",
    "repair_outcome_terminal_utility_per_1k_tokens_last",
    "repair_outcome_repair_success_margin_last",
    "over_decomposition_flag",
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
    "micro_repair_coverage_addition_count",
    "micro_repair_qualify_remove_count",
    "micro_repair_tokens",
    "repair_generation_tokens",
    "observed_repair_roundtrip_tokens_last",
    "observed_repair_roundtrip_tokens_total",
    "repair_gain_per_1k_tokens",
    "repair_gain_per_1k_roundtrip_tokens",
    "lightweight_repair_validation_count",
    "repair_validation_tokens",
    "repair_full_validation_tokens",
    "post_repair_validation_tokens",
    "repair_patch_accepted_count",
    "repair_validator_escalation_count",
    "repair_cascade_score_delta",
    "validation_checkpoint_count",
    "best_validation_score",
    "selected_previous_checkpoint",
    "checkpoint_quality_gain_override",
    "checkpoint_score_delta_vs_current",
]

PAPER_COLUMNS = [
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
    "direct_precheck_allowed",
    "direct_precheck_skipped",
    "direct_precheck_escalated",
    "direct_precheck_accepted",
    "direct_precheck_score",
    "direct_precheck_incremental_tokens",
    "direct_precheck_gate_bucket",
    "direct_precheck_gate_reason",
    "direct_precheck_gate_block_reasons",
    "direct_precheck_gate_probability",
    "direct_precheck_gate_threshold",
    "direct_precheck_gate_expected_net_tokens",
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
    "adaptive_retry_enabled",
    "adaptive_checkpoint_enabled",
    "controller_recursive_cost_efficient",
    "controller_recursive_quality_advantage",
    "controller_recursive_marginal_quality_per_1k_token",
    "decomposition_node_count",
    "decomposition_edge_count",
    "decomposition_atomicity_ratio",
    "validation_evidence_items_used",
    "minimum_claims_required",
    "missing_claim_count",
    "retry_allowed_count",
    "retry_blocked_count",
    "repair_rounds",
    "claim_micro_repair_allowed_count",
    "fixed_recursive_market_retry_count",
    "shallow_escalation_count",
    "recursive_retry_count",
    "retry_expected_gain_last",
    "retry_roi_last",
    "retry_roi_estimate",
    "retry_expected_utility",
    "posterior_acceptance_probability",
    "retry_policy_candidate_count_last",
    "retry_policy_evaluated_candidate_count_last",
    "retry_policy_viable_candidate_count_last",
    "retry_policy_rejected_candidate_count_last",
    "retry_policy_top_block_reason_last",
    "retry_policy_rejected_candidate_top_block_reason_last",
    "retry_policy_blocked_by_retrieval_count_last",
    "retry_policy_blocked_by_roi_count_last",
    "retry_policy_blocked_by_expected_utility_count_last",
    "retry_policy_blocked_by_posterior_count_last",
    "retry_policy_blocked_by_token_reserve_count_last",
    "retry_policy_blocked_by_threshold_crossing_count_last",
    "retry_policy_action_last",
    "retry_policy_reason_last",
    "retry_policy_retry_type_last",
    "retry_block_reason_last",
    "retry_rejected_candidate_block_reason_last",
    "baseline_shallow_reliability",
    "baseline_shallow_tokens",
    "delta_vs_shared_shallow",
    "token_delta_vs_shared_shallow",
    "posterior_acceptance_last",
    "retry_expected_utility_last",
    "retry_expected_seconds_last",
    "retry_incremental_tokens_last",
    "bayesian_retry_gate_count",
    "bayesian_retry_allowed_count",
    "positive_eu_retry_count",
    "skipped_due_to_retrieval_gate_count",
    "retrieval_diagnostic_count",
    "retrieval_diagnostic_mean_score",
    "retrieval_diagnostic_max_score",
    "coverage_enhancement_target_count",
    "retrieval_strong_evidence_count",
    "retrieval_medium_or_better_count",
    "evidence_probe_count",
    "evidence_probe_upgrade_count",
    "selected_claim_count_last",
    "selected_action_type",
    "selected_bucket_key",
    "selected_base_bucket_key",
    "selected_action_expected_gain",
    "selected_action_expected_utility",
    "selected_action_expected_tokens",
    "selected_action_expected_seconds",
    "selected_action_posterior_acceptance",
    "selected_action_posterior_mean",
    "selected_action_ucb_bonus",
    "selected_action_n_empirical",
    "selected_action_roi_per_1k",
    "selected_action_large_deficit_incremental_exception",
    "selected_action_threshold_crossing_repair_exception",
    "selected_action_source_backed_material_repair_exception",
    "selected_action_coverage_addition_exception",
    "selected_action_token_reserve_exception",
    "selected_action_token_reserve_exception_reason",
    "selected_action_effective_tail_token_reserve",
    "selected_action_reserve_after_expected_repair",
    "policy_repair_observations",
    "policy_repair_accepted_total",
    "policy_repair_rejected_total",
    "policy_repair_accept_rate",
    "policy_last_accepted",
    "policy_last_patch_accepted",
    "policy_last_success",
    "policy_last_gain",
    "policy_last_tokens",
    "policy_last_generation_tokens",
    "policy_last_validation_tokens",
    "policy_last_observed_roi_per_1k",
    "policy_last_terminal_utility_before",
    "policy_last_terminal_utility_after",
    "policy_last_terminal_utility_delta",
    "policy_last_terminal_utility_per_1k_tokens",
    "policy_last_repair_success_margin",
    "checkpoint_terminal_utility",
    "checkpoint_lcb_utility",
    "cross_model_judging_any",
    "material_failing_claim_count_last",
    "specific_repair_claim_count_last",
    "accepted_patch_count",
    "rejected_patch_count",
    "repair_outcome_accepted_count",
    "repair_outcome_rejected_count",
    "repair_outcome_policy_success_count",
    "repair_outcome_policy_failed_count",
    "repair_outcome_policy_outcome_last",
    "repair_outcome_terminal_utility_before_last",
    "repair_outcome_terminal_utility_after_last",
    "repair_outcome_terminal_utility_delta_last",
    "repair_outcome_observed_roi_per_1k_last",
    "repair_outcome_terminal_utility_per_1k_tokens_last",
    "repair_outcome_repair_success_margin_last",
    "over_decomposition_flag",
    "retry_effectiveness",
    "raw_retry_score_delta",
    "checkpoint_saved_score",
    "focused_repair_search_count",
    "focused_repair_source_count",
    "micro_repair_count",
    "micro_repair_source_count",
    "micro_repair_search_replace_count",
    "micro_repair_coverage_addition_count",
    "micro_repair_qualify_remove_count",
    "micro_repair_tokens",
    "repair_generation_tokens",
    "observed_repair_roundtrip_tokens_last",
    "observed_repair_roundtrip_tokens_total",
    "repair_gain_per_1k_tokens",
    "repair_gain_per_1k_roundtrip_tokens",
    "lightweight_repair_validation_count",
    "repair_validation_tokens",
    "repair_full_validation_tokens",
    "post_repair_validation_tokens",
    "repair_patch_accepted_count",
    "repair_validator_escalation_count",
    "repair_cascade_score_delta",
    "validation_checkpoint_count",
    "best_validation_score",
    "selected_previous_checkpoint",
    "checkpoint_quality_gain_override",
    "checkpoint_score_delta_vs_current",
    "reliability_score",
    "claims_total",
    "claim_units_total",
    "failing_claim_count",
    "unsupported_material_claim_count",
    "validation_claim_coverage",
    "low_claim_count_flag",
    "claim_count_penalty",
    "judge_agreement",
    "judge_a_model",
    "judge_b_model",
    "cross_model_judging",
    "secondary_judge_fallback",
    "supported_ratio",
    "weak_or_better_ratio",
    "runtime_seconds",
    "actual_total_tokens",
    "token_proxy",
    "reliability_per_1k_actual_token",
    "decomposition_depth_realized",
    "reliability_gain_vs_single",
    "reliability_gain_vs_fixed_shallow",
    "token_delta_vs_fixed_shallow",
    "efficiency_gain_vs_fixed_shallow",
    "budget_hit",
]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _empty_token_usage() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


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
    """Deterministic token proxy for cost comparisons using a chars/4 heuristic."""
    chunks: List[str] = [
        str(state.get("idea", "")),
        str(state.get("refined_idea", "")),
        str(state.get("market_analysis", "")),
        str(state.get("business_model", "")),
        str(state.get("validated_market_analysis", "")),
        json.dumps(state.get("pitch_content", {}), ensure_ascii=True),
        json.dumps(state.get("trend_signals", {}), ensure_ascii=True),
        json.dumps(state.get("validation_report", {}), ensure_ascii=True),
        json.dumps(state.get("repair_plan", {}), ensure_ascii=True),
        json.dumps(state.get("repair_patches", []), ensure_ascii=True),
        json.dumps(state.get("micro_validation", {}), ensure_ascii=True),
    ]
    total_chars = sum(len(c) for c in chunks)
    return max(1, total_chars // 4)


def _claim_ratios(validation_report: Dict[str, Any]) -> Dict[str, float]:
    claims = validation_report.get("claims", []) if validation_report else []
    if not claims:
        return {"supported_ratio": 0.0, "weak_or_better_ratio": 0.0}
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


def _latest_validator_audit(state: Dict[str, Any]) -> Dict[str, Any]:
    audits = [
        audit
        for audit in (state.get("tool_audit", []) or [])
        if isinstance(audit, dict)
        and audit.get("tool") in {"llm_claim_verifier_dual_judge", "llm_claim_repair_validator"}
    ]
    return audits[-1] if audits else {}


def _retry_type_counts(retry_decisions: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for decision in retry_decisions:
        retry_type = str(decision.get("retry_type", "none") or "none").strip().lower()
        counts[retry_type] = counts.get(retry_type, 0) + 1
    return counts


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
    token_proxy = _safe_int(state.get("token_proxy_current", _estimate_tokens_proxy(state)), 0)
    reliability_per_1k_actual_token = round(
        (reliability_score * 1000.0) / max(1, actual_total_tokens), 4
    )
    reliability_per_second = round(reliability_score / max(0.001, runtime_seconds), 4)

    agreement_stats = validation.get("agreement_stats", {}) if isinstance(validation, dict) else {}
    scorecard = state.get("controller_scorecard", {}) or {}
    forced_mode_applied = bool(scorecard.get("forced_mode_applied", False)) if isinstance(scorecard, dict) else False
    score_features = scorecard.get("features", {}) if isinstance(scorecard, dict) else {}
    mode_scores = scorecard.get("mode_scores", {}) if isinstance(scorecard, dict) else {}
    controller_mode_realized = str(
        state.get("controller_mode_realized")
        or state.get("controller_mode", "n/a")
        or "n/a"
    )
    controller_mode_initial = str(
        state.get("controller_mode_initial")
        or (scorecard.get("selected_mode", "") if isinstance(scorecard, dict) else "")
        or controller_mode_realized
    )
    selected_score = mode_scores.get(controller_mode_initial, {}) if isinstance(mode_scores, dict) else {}
    realized_score = mode_scores.get(controller_mode_realized, {}) if isinstance(mode_scores, dict) else {}
    has_controller_estimate = bool(realized_score)
    realized_expected_quality = _safe_float(realized_score.get("expected_quality", 0.0), 0.0) if has_controller_estimate else 0.0
    controller_calibration_error = (
        round(realized_expected_quality - float(reliability_score), 4)
        if has_controller_estimate
        else 0.0
    )

    decomposition_graph = state.get("decomposition_graph", {}) or {}
    graph_metrics = decomposition_graph.get("metrics", {}) if isinstance(decomposition_graph, dict) else {}
    validator_audit = _latest_validator_audit(state)
    repair_validator_audits = [
        audit
        for audit in (state.get("tool_audit", []) or [])
        if isinstance(audit, dict) and audit.get("tool") == "llm_claim_repair_validator"
    ]
    lightweight_repair_validation_count = len(repair_validator_audits)
    repair_validation_tokens = sum(_safe_int(audit.get("total_tokens", 0), 0) for audit in repair_validator_audits)
    repair_patch_accepted_count = sum(1 for audit in repair_validator_audits if bool(audit.get("repair_patch_accepted", False)))

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

    secondary_judge_fallback = 1 if bool(validator_audit.get("secondary_judge_fallback", False)) else 0
    minimum_claims_required = _safe_int(validator_audit.get("minimum_claims_required", 3), 3)
    missing_claim_count = _safe_int(validator_audit.get("missing_claim_count", 0), 0)
    claims_total = len(validation.get("claims", []) if validation else [])
    claim_units_total = len(validation.get("claim_units", []) if validation else [])
    failing_claim_count = len(validation.get("failing_claims", []) if validation else [])
    unsupported_material_claim_count = _safe_int(validation.get("unsupported_material_claim_count", 0) if validation else 0, 0)
    low_claim_count_flag = 1 if bool(validator_audit.get("low_claim_count_flag", claims_total < minimum_claims_required)) else 0
    claim_count_penalty = _safe_int(validator_audit.get("claim_count_penalty", 0), 0)
    validation_claim_coverage = round(min(1.0, claims_total / max(1, minimum_claims_required)), 4)
    judge_a_model = str(validator_audit.get("judge_a_model", "") or "")
    judge_b_model = str(validator_audit.get("judge_b_model", "") or "")
    cross_model_judging = 1 if bool(validator_audit.get("cross_model_judging", False)) else 0

    retry_decisions = [
        decision
        for decision in (state.get("retry_budget_decisions", []) or [])
        if isinstance(decision, dict)
    ]
    retry_allowed_count = sum(1 for d in retry_decisions if bool(d.get("retry_allowed", False)))
    retry_blocked_count = sum(
        1 for d in retry_decisions if not bool(d.get("retry_allowed", False)) and bool(d.get("needs_revision", False))
    )
    retry_counts = _retry_type_counts(retry_decisions)
    claim_micro_repair_allowed_count = retry_counts.get("claim_micro_repair", 0)
    fixed_recursive_market_retry_count = retry_counts.get("fixed_recursive_market_retry", 0)
    shallow_escalation_count = retry_counts.get("shallow_escalation", 0)
    recursive_retry_count = retry_counts.get("recursive_retry", 0)
    last_retry_decision = retry_decisions[-1] if retry_decisions else {}

    baseline_checkpoint = state.get("baseline_checkpoint", {}) or {}
    baseline_validation = baseline_checkpoint.get("validation_report", {}) if isinstance(baseline_checkpoint, dict) else {}
    baseline_shallow_reliability = _safe_int(
        baseline_checkpoint.get("reliability_score", baseline_validation.get("reliability_score", reliability_score))
        if isinstance(baseline_checkpoint, dict) else reliability_score,
        reliability_score,
    )
    baseline_shallow_tokens = _safe_int(
        baseline_checkpoint.get("total_tokens_at_checkpoint", actual_total_tokens)
        if isinstance(baseline_checkpoint, dict) else actual_total_tokens,
        actual_total_tokens,
    )

    retry_policy_decision = state.get("retry_policy_decision", {}) or last_retry_decision or {}
    selected_action = state.get("selected_action", {}) or {}
    retrieval_diagnostics = [
        item for item in (state.get("retrieval_diagnostics", []) or []) if isinstance(item, dict)
    ]
    retrieval_scores = [_safe_float(item.get("retrieval_score", 0.0), 0.0) for item in retrieval_diagnostics]
    retrieval_diagnostic_count = len(retrieval_diagnostics)
    retrieval_diagnostic_mean_score = round(statistics.fmean(retrieval_scores), 4) if retrieval_scores else 0.0
    retrieval_diagnostic_max_score = round(max(retrieval_scores), 4) if retrieval_scores else 0.0
    retrieval_strong_evidence_count = sum(1 for item in retrieval_diagnostics if str(item.get("evidence_strength", "")).lower() == "strong")
    retrieval_medium_or_better_count = sum(1 for item in retrieval_diagnostics if str(item.get("evidence_strength", "")).lower() in {"medium", "strong"})
    evidence_probe_count = sum(
        1
        for item in retrieval_diagnostics
        if any("evidence_probe_" in str(reason) for reason in (item.get("reasons", []) or []))
    )
    evidence_probe_upgrade_count = sum(
        1
        for item in retrieval_diagnostics
        if "evidence_probe_upgraded_to_search_and_replace" in (item.get("reasons", []) or [])
    )
    coverage_enhancement_target_count = sum(
        1
        for claim in (state.get("failing_claims", []) or [])
        if isinstance(claim, dict) and bool(claim.get("coverage_enhancement", False))
    )
    if coverage_enhancement_target_count <= 0:
        coverage_enhancement_target_count = sum(
            _safe_int(audit.get("coverage_enhancement_target_count", 0), 0)
            for audit in (state.get("tool_audit", []) or [])
            if isinstance(audit, dict) and audit.get("tool") == "retrieval_diagnostics"
        )

    policy_audits = [
        audit for audit in (state.get("tool_audit", []) or [])
        if isinstance(audit, dict) and audit.get("tool") == "bayesian_retry_policy"
    ]
    bayesian_retry_gate_count = len(policy_audits)
    bayesian_retry_allowed_count = sum(1 for audit in policy_audits if bool(audit.get("retry_allowed", False)))
    positive_eu_retry_count = sum(1 for audit in policy_audits if _safe_float(audit.get("expected_utility", 0.0), 0.0) > 0.0 and bool(audit.get("retry_allowed", False)))
    skipped_due_to_retrieval_gate_count = sum(1 for d in retry_decisions if str(d.get("reason", "")) == "no_positive_expected_utility_repair" and retrieval_diagnostic_count > 0 and bayesian_retry_allowed_count == 0)

    policy_stats = state.get("policy_stats", {}) or {}
    selected_claim_ids = selected_action.get("selected_claim_ids", []) if isinstance(selected_action, dict) else []
    retry_policy_action_last = str(retry_policy_decision.get("action", "") or "")
    retry_policy_reason_last = str(retry_policy_decision.get("reason", "") or "")
    retry_policy_retry_type_last = str(retry_policy_decision.get("retry_type", "") or "")
    retry_allowed_last = bool(retry_policy_decision.get("retry_allowed", False)) or retry_policy_action_last in {"repair_bundle", "broad_retry"}
    retry_block_reasons = retry_policy_decision.get("block_reasons", []) if isinstance(retry_policy_decision, dict) else []
    if not isinstance(retry_block_reasons, list):
        retry_block_reasons = [str(retry_block_reasons)]
    if retry_allowed_last:
        retry_block_reasons = []
    if not retry_block_reasons and retry_policy_reason_last and not retry_allowed_last:
        retry_block_reasons = [retry_policy_reason_last]
    retry_block_reason_last = "|".join(str(reason) for reason in retry_block_reasons if str(reason).strip())
    retry_policy_top_block_reason_last = (
        str(retry_block_reasons[0])
        if retry_block_reasons and str(retry_block_reasons[0]).strip()
        else ""
    )
    rejected_candidates = (
        retry_policy_decision.get("rejected_candidates", [])
        if isinstance(retry_policy_decision, dict)
        else []
    )
    if not isinstance(rejected_candidates, list):
        rejected_candidates = []
    rejected_candidate_block_reasons = (
        retry_policy_decision.get("rejected_candidate_block_reasons", [])
        if isinstance(retry_policy_decision, dict)
        else []
    )
    if not isinstance(rejected_candidate_block_reasons, list):
        rejected_candidate_block_reasons = [str(rejected_candidate_block_reasons)]
    if not rejected_candidate_block_reasons:
        seen_rejected_reasons: List[str] = []
        for candidate in rejected_candidates:
            if not isinstance(candidate, dict):
                continue
            gates = candidate.get("blocking_gates", []) or []
            if not isinstance(gates, list):
                gates = [str(gates)]
            for gate in gates:
                reason = str(gate).strip()
                if reason and reason not in seen_rejected_reasons:
                    seen_rejected_reasons.append(reason)
        rejected_candidate_block_reasons = seen_rejected_reasons
    retry_rejected_candidate_block_reason_last = "|".join(
        str(reason) for reason in rejected_candidate_block_reasons if str(reason).strip()
    )
    retry_policy_rejected_candidate_top_block_reason_last = (
        str(rejected_candidate_block_reasons[0])
        if rejected_candidate_block_reasons and str(rejected_candidate_block_reasons[0]).strip()
        else ""
    )
    def _block_count(marker: str) -> int:
        total = 0
        for candidate in rejected_candidates:
            if not isinstance(candidate, dict):
                continue
            gates = candidate.get("blocking_gates", []) or []
            if not isinstance(gates, list):
                gates = [str(gates)]
            if any(marker in str(gate) for gate in gates):
                total += 1
        return total
    selected_action_json = json.dumps(selected_action, sort_keys=True, ensure_ascii=True) if isinstance(selected_action, dict) else "{}"
    retry_policy_decision_json = json.dumps(retry_policy_decision, sort_keys=True, ensure_ascii=True) if isinstance(retry_policy_decision, dict) else "{}"
    retry_block_reasons_json = json.dumps(retry_block_reasons, sort_keys=True, ensure_ascii=True)
    rejected_candidate_block_reasons_json = json.dumps(rejected_candidate_block_reasons, sort_keys=True, ensure_ascii=True)
    rejected_candidates_json = json.dumps(rejected_candidates, sort_keys=True, ensure_ascii=True)
    repair_outcomes = [
        item for item in (state.get("repair_validation_history", []) or []) if isinstance(item, dict)
    ]
    last_repair_outcome = repair_outcomes[-1] if repair_outcomes else {}
    repair_outcome_accepted_count = sum(1 for item in repair_outcomes if bool(item.get("accepted", False)))
    repair_outcome_rejected_count = sum(1 for item in repair_outcomes if not bool(item.get("accepted", False)))
    repair_outcome_policy_success_count = sum(1 for item in repair_outcomes if bool(item.get("policy_success", False)))
    repair_outcome_policy_failed_count = len(repair_outcomes) - repair_outcome_policy_success_count
    repair_outcome_terminal_utility_before_last = _safe_float(
        last_repair_outcome.get("terminal_utility_before", 0.0), 0.0
    )
    repair_outcome_terminal_utility_after_last = _safe_float(
        last_repair_outcome.get("terminal_utility_after", 0.0), 0.0
    )
    repair_outcome_terminal_utility_delta_last = _safe_float(
        last_repair_outcome.get("terminal_utility_delta", 0.0), 0.0
    )
    repair_outcome_observed_roi_per_1k_last = _safe_float(
        last_repair_outcome.get("observed_roi_per_1k", 0.0), 0.0
    )
    repair_outcome_terminal_utility_per_1k_tokens_last = _safe_float(
        last_repair_outcome.get("terminal_utility_per_1k_tokens", 0.0), 0.0
    )
    repair_outcome_repair_success_margin_last = _safe_float(
        last_repair_outcome.get("repair_success_margin", 0.0), 0.0
    )
    repair_outcome_policy_outcome_last = str(last_repair_outcome.get("policy_outcome", "") or "")
    observed_repair_roundtrip_tokens_last = _safe_int(last_repair_outcome.get("observed_tokens", 0), 0)
    observed_repair_roundtrip_tokens_total = sum(_safe_int(item.get("observed_tokens", 0), 0) for item in repair_outcomes)

    checkpoint_selection = state.get("adaptive_checkpoint_selection", {}) or {}
    adaptive_retry_enabled = 1 if bool(state.get("adaptive_retry_enabled", True)) else 0
    adaptive_checkpoint_enabled = 1 if bool(state.get("adaptive_checkpoint_enabled", True)) else 0
    validation_snapshots = [
        item for item in (state.get("validation_snapshots", []) or []) if isinstance(item, dict)
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
    retry_effectiveness = round(float(reliability_score - initial_validation_score), 4) if retry_count_value > 0 else 0.0
    raw_retry_score_delta = round(float(latest_validation_score - initial_validation_score), 4) if len(validation_snapshots) > 1 else 0.0
    checkpoint_saved_score = round(float(reliability_score - latest_validation_score), 4)

    focused_repair_audits = [
        audit
        for audit in (state.get("tool_audit", []) or [])
        if isinstance(audit, dict)
        and audit.get("tool") == "focused_repair_search"
        and str(audit.get("query", "") or "").strip()
    ]
    focused_repair_search_count = len(focused_repair_audits)
    focused_repair_source_count = sum(_safe_int(audit.get("source_count", 0), 0) for audit in focused_repair_audits)

    micro_repair_audits = [
        audit
        for audit in (state.get("tool_audit", []) or [])
        if isinstance(audit, dict) and audit.get("tool") == "claim_micro_repair"
    ]
    micro_repair_count = len(micro_repair_audits)
    micro_repair_source_count = sum(_safe_int(audit.get("source_count", 0), 0) for audit in micro_repair_audits)
    micro_repair_tokens = sum(_safe_int(audit.get("total_tokens", 0), 0) for audit in micro_repair_audits)
    micro_repair_search_replace_count = sum(
        _safe_int((audit.get("repair_action_counts", {}) or {}).get("search_and_replace", 0), 0)
        for audit in micro_repair_audits
        if isinstance(audit.get("repair_action_counts", {}), dict)
    )
    micro_repair_coverage_addition_count = sum(
        _safe_int((audit.get("repair_action_counts", {}) or {}).get("coverage_addition", 0), 0)
        for audit in micro_repair_audits
        if isinstance(audit.get("repair_action_counts", {}), dict)
    )
    micro_repair_qualify_remove_count = sum(
        _safe_int((audit.get("repair_action_counts", {}) or {}).get("qualify_or_remove", 0), 0)
        + _safe_int((audit.get("repair_action_counts", {}) or {}).get("remove", 0), 0)
        for audit in micro_repair_audits
        if isinstance(audit.get("repair_action_counts", {}), dict)
    )
    repair_generation_tokens = micro_repair_tokens
    repair_full_validation_tokens = (
        _safe_int(validator_audit.get("total_tokens", 0), 0)
        if retry_count_value > 0 and str(validator_audit.get("tool", "")) == "llm_claim_verifier_dual_judge"
        else 0
    )
    post_repair_validation_tokens = repair_validation_tokens + repair_full_validation_tokens
    if repair_outcomes and observed_repair_roundtrip_tokens_last <= 0:
        observed_repair_roundtrip_tokens_last = repair_generation_tokens + post_repair_validation_tokens
    if repair_outcomes and observed_repair_roundtrip_tokens_total <= 0:
        observed_repair_roundtrip_tokens_total = repair_generation_tokens + post_repair_validation_tokens
    repair_gain_per_1k_tokens = (
        round((raw_retry_score_delta * 1000.0) / max(1, micro_repair_tokens), 4)
        if micro_repair_count
        else 0.0
    )
    repair_gain_per_1k_roundtrip_tokens = (
        round((raw_retry_score_delta * 1000.0) / max(1, observed_repair_roundtrip_tokens_last), 4)
        if observed_repair_roundtrip_tokens_last > 0
        else 0.0
    )

    metrics = {
        "strategy": strategy,
        "controller_mode": controller_mode_realized,
        "controller_mode_initial": controller_mode_initial,
        "controller_mode_realized": controller_mode_realized,
        "controller_escalated": 1
        if bool(state.get("controller_escalated", False))
        or (
            controller_mode_initial != controller_mode_realized
            and controller_mode_initial != "n/a"
            and controller_mode_realized != "n/a"
        )
        else 0,
        "direct_precheck_allowed": 1 if bool(state.get("direct_precheck_allowed", False)) else 0,
        "direct_precheck_skipped": 1 if bool(state.get("direct_precheck_skipped", False)) else 0,
        "direct_precheck_escalated": 1 if bool(state.get("direct_precheck_escalated", False)) else 0,
        "direct_precheck_accepted": 1 if bool(state.get("direct_precheck_accepted", False)) else 0,
        "direct_precheck_score": _safe_int(state.get("direct_precheck_score", 0), 0),
        "direct_precheck_incremental_tokens": _safe_int(
            state.get("direct_precheck_incremental_tokens", 0),
            0,
        ),
        "direct_precheck_gate_bucket": str(state.get("direct_precheck_gate_bucket", "") or ""),
        "direct_precheck_gate_reason": str(state.get("direct_precheck_gate_reason", "") or ""),
        "direct_precheck_gate_block_reasons": "|".join(state.get("direct_precheck_gate_block_reasons", []) or [])
        if isinstance(state.get("direct_precheck_gate_block_reasons", []), list)
        else str(state.get("direct_precheck_gate_block_reasons", "") or ""),
        "direct_precheck_gate_probability": _safe_float(
            state.get("direct_precheck_gate_probability", 0.0),
            0.0,
        ),
        "direct_precheck_gate_threshold": _safe_float(
            state.get("direct_precheck_gate_threshold", 0.0),
            0.0,
        ),
        "direct_precheck_gate_expected_net_tokens": _safe_float(
            state.get("direct_precheck_gate_expected_net_tokens", 0.0),
            0.0,
        ),
        "runtime_seconds": round(runtime_seconds, 3),
        "reliability_score": reliability_score,
        "claims_total": claims_total,
        "claim_units_total": claim_units_total,
        "failing_claim_count": failing_claim_count,
        "unsupported_material_claim_count": unsupported_material_claim_count,
        "minimum_claims_required": minimum_claims_required,
        "missing_claim_count": missing_claim_count,
        "validation_claim_coverage": validation_claim_coverage,
        "low_claim_count_flag": low_claim_count_flag,
        "claim_count_penalty": claim_count_penalty,
        "supported_ratio": ratios["supported_ratio"],
        "weak_or_better_ratio": ratios["weak_or_better_ratio"],
        "market_sources_count": len(state.get("market_sources", [])),
        "tool_calls": len(state.get("tool_audit", [])),
        "retry_count": retry_count_value,
        "repair_rounds": _safe_int(state.get("repair_rounds", 0), 0),
        "decomposition_depth_target": _safe_int(state.get("decomposition_depth_target", 0), 0),
        "decomposition_depth_realized": _safe_int(state.get("decomposition_depth_realized", 0), 0),
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "actual_total_tokens": actual_total_tokens,
        "token_proxy": token_proxy,
        "reliability_per_1k_actual_token": reliability_per_1k_actual_token,
        "reliability_per_second": reliability_per_second,
        "judge_agreement": _safe_float(agreement_stats.get("overall_agreement", 0.0), 0.0),
        "judge_score_delta_abs": _safe_float(agreement_stats.get("score_delta_abs", 0.0), 0.0),
        "judge_a_model": judge_a_model,
        "judge_b_model": judge_b_model,
        "cross_model_judging": cross_model_judging,
        "secondary_judge_fallback": secondary_judge_fallback,
        "budget_hit": 1 if budget_hit else 0,
        "budget_violation_count": budget_violation_count,
        "finished_under_budget": 0 if budget_hit else 1,
        "controller_structural_complexity": _safe_float(score_features.get("structural_complexity", 0.0), 0.0),
        "controller_uncertainty_need": _safe_float(score_features.get("uncertainty_need", 0.0), 0.0),
        "controller_budget_pressure": _safe_float(scorecard.get("budget_pressure", 0.0), 0.0) if isinstance(scorecard, dict) else 0.0,
        "controller_selected_utility": _safe_float(scorecard.get("selected_utility", 0.0), 0.0) if isinstance(scorecard, dict) else 0.0,
        "controller_utility_margin": _safe_float(scorecard.get("utility_margin", 0.0), 0.0) if isinstance(scorecard, dict) else 0.0,
        "controller_expected_quality": _safe_float(selected_score.get("expected_quality", 0.0), 0.0),
        "controller_realized_expected_quality": _safe_float(realized_expected_quality, 0.0),
        "controller_calibration_error": controller_calibration_error,
        "controller_calibration_error_abs": round(abs(controller_calibration_error), 4),
        "controller_direct_eligible": 1 if isinstance(scorecard, dict) and bool(scorecard.get("direct_eligible", False)) else 0,
        "controller_recursive_upfront_allowed": 1 if isinstance(scorecard, dict) and bool(scorecard.get("recursive_upfront_allowed", False)) else 0,
        "controller_evidence_sensitive_medium": 1 if isinstance(scorecard, dict) and bool(scorecard.get("evidence_sensitive_medium", False)) else 0,
        "controller_recursive_cost_efficient": 1 if isinstance(scorecard, dict) and bool(scorecard.get("recursive_cost_efficient", False)) else 0,
        "controller_recursive_quality_advantage": _safe_float(scorecard.get("recursive_quality_advantage", 0.0), 0.0) if isinstance(scorecard, dict) else 0.0,
        "controller_recursive_marginal_quality_per_1k_token": _safe_float(scorecard.get("recursive_marginal_quality_per_1k_token", 0.0), 0.0) if isinstance(scorecard, dict) else 0.0,
        "controller_policy_adjustments": "|".join(scorecard.get("policy_adjustments", []) or []) if isinstance(scorecard, dict) and not forced_mode_applied else "",
        "controller_policy_adjustment_count": len(scorecard.get("policy_adjustments", []) or []) if isinstance(scorecard, dict) and not forced_mode_applied else 0,
        "adaptive_retry_enabled": adaptive_retry_enabled,
        "adaptive_checkpoint_enabled": adaptive_checkpoint_enabled,
        "decomposition_node_count": _safe_int(graph_metrics.get("node_count", 0), 0),
        "decomposition_edge_count": _safe_int(graph_metrics.get("edge_count", 0), 0),
        "decomposition_atomicity_ratio": _safe_float(graph_metrics.get("atomicity_ratio", 0.0), 0.0),
        "validation_evidence_items_raw": _safe_int(validator_audit.get("evidence_items_raw", 0), 0),
        "validation_evidence_items_used": _safe_int(validator_audit.get("evidence_items_used", 0), 0),
        "retry_allowed_count": retry_allowed_count,
        "retry_blocked_count": retry_blocked_count,
        "claim_micro_repair_allowed_count": claim_micro_repair_allowed_count,
        "fixed_recursive_market_retry_count": fixed_recursive_market_retry_count,
        "shallow_escalation_count": shallow_escalation_count,
        "recursive_retry_count": recursive_retry_count,
        "retry_expected_gain_last": _safe_float(last_retry_decision.get("retry_expected_gain", 0.0), 0.0),
        "retry_roi_last": _safe_float(last_retry_decision.get("retry_roi", 0.0), 0.0),
        "retry_roi_estimate": _safe_float(state.get("retry_roi_estimate", 0.0), 0.0),
        "retry_expected_utility": _safe_float(state.get("retry_expected_utility", 0.0), 0.0),
        "posterior_acceptance_probability": _safe_float(state.get("posterior_acceptance_probability", 0.0), 0.0),
        "retry_policy_candidate_count_last": _safe_int(retry_policy_decision.get("candidate_count", len(retry_policy_decision.get("candidates", []) or [])), 0),
        "retry_policy_evaluated_candidate_count_last": _safe_int(retry_policy_decision.get("evaluated_candidate_count", 0), 0),
        "retry_policy_viable_candidate_count_last": _safe_int(retry_policy_decision.get("viable_candidate_count", 0), 0),
        "retry_policy_rejected_candidate_count_last": _safe_int(retry_policy_decision.get("rejected_candidate_count", len(rejected_candidates)), 0),
        "retry_policy_action_last": retry_policy_action_last,
        "retry_policy_reason_last": retry_policy_reason_last,
        "retry_policy_retry_type_last": retry_policy_retry_type_last,
        "retry_policy_top_block_reason_last": retry_policy_top_block_reason_last,
        "retry_policy_rejected_candidate_top_block_reason_last": retry_policy_rejected_candidate_top_block_reason_last,
        "retry_policy_blocked_by_retrieval_count_last": _block_count("retrieval"),
        "retry_policy_blocked_by_roi_count_last": _block_count("roi"),
        "retry_policy_blocked_by_expected_utility_count_last": _block_count("expected_utility"),
        "retry_policy_blocked_by_posterior_count_last": _block_count("posterior"),
        "retry_policy_blocked_by_token_reserve_count_last": _block_count("token_reserve"),
        "retry_policy_blocked_by_threshold_crossing_count_last": _block_count("threshold_crossing"),
        "retry_block_reason_last": retry_block_reason_last,
        "retry_rejected_candidate_block_reason_last": retry_rejected_candidate_block_reason_last,
        "retry_block_reasons": retry_block_reasons_json,
        "retry_block_reasons_json": retry_block_reasons_json,
        "retry_rejected_candidate_block_reasons": rejected_candidate_block_reasons_json,
        "retry_rejected_candidate_block_reasons_json": rejected_candidate_block_reasons_json,
        "retry_policy_decision": retry_policy_decision_json,
        "retry_policy_decision_json": retry_policy_decision_json,
        "retry_policy_rejected_candidates": rejected_candidates_json,
        "baseline_shallow_reliability": baseline_shallow_reliability,
        "baseline_shallow_tokens": baseline_shallow_tokens,
        "delta_vs_shared_shallow": round(float(reliability_score - baseline_shallow_reliability), 4),
        "token_delta_vs_shared_shallow": round(float(actual_total_tokens - baseline_shallow_tokens), 4),
        "posterior_acceptance_last": _safe_float(retry_policy_decision.get("posterior_acceptance", state.get("posterior_acceptance_probability", 0.0)), 0.0),
        "retry_expected_utility_last": _safe_float(retry_policy_decision.get("expected_utility", state.get("retry_expected_utility", 0.0)), 0.0),
        "retry_expected_seconds_last": _safe_float(retry_policy_decision.get("expected_seconds", state.get("retry_expected_cost_seconds", 0.0)), 0.0),
        "retry_incremental_tokens_last": _safe_int(retry_policy_decision.get("expected_tokens", state.get("retry_expected_cost_tokens", 0)), 0),
        "bayesian_retry_gate_count": bayesian_retry_gate_count,
        "bayesian_retry_allowed_count": bayesian_retry_allowed_count,
        "positive_eu_retry_count": positive_eu_retry_count,
        "skipped_due_to_retrieval_gate_count": skipped_due_to_retrieval_gate_count,
        "retrieval_diagnostic_count": retrieval_diagnostic_count,
        "retrieval_diagnostic_mean_score": retrieval_diagnostic_mean_score,
        "retrieval_diagnostic_max_score": retrieval_diagnostic_max_score,
        "coverage_enhancement_target_count": coverage_enhancement_target_count,
        "retrieval_strong_evidence_count": retrieval_strong_evidence_count,
        "retrieval_medium_or_better_count": retrieval_medium_or_better_count,
        "evidence_probe_count": evidence_probe_count,
        "evidence_probe_upgrade_count": evidence_probe_upgrade_count,
        "selected_claim_count_last": len(selected_claim_ids or []),
        "selected_action_type": str(selected_action.get("action_type", "") or ""),
        "selected_bucket_key": str(selected_action.get("bucket_key", "") or ""),
        "selected_base_bucket_key": str(selected_action.get("base_bucket_key", "") or ""),
        "selected_action": selected_action_json,
        "selected_action_json": selected_action_json,
        "selected_action_expected_gain": _safe_float(selected_action.get("expected_gain", 0.0), 0.0),
        "selected_action_expected_utility": _safe_float(selected_action.get("expected_utility", 0.0), 0.0),
        "selected_action_expected_tokens": _safe_int(selected_action.get("expected_tokens", 0), 0),
        "selected_action_expected_seconds": _safe_float(selected_action.get("expected_seconds", 0.0), 0.0),
        "selected_action_posterior_acceptance": _safe_float(selected_action.get("posterior_acceptance", 0.0), 0.0),
        "selected_action_posterior_mean": _safe_float(selected_action.get("posterior_mean", 0.0), 0.0),
        "selected_action_ucb_bonus": _safe_float(selected_action.get("ucb_bonus", 0.0), 0.0),
        "selected_action_n_empirical": _safe_float(selected_action.get("n_empirical", 0.0), 0.0),
        "selected_action_roi_per_1k": _safe_float(selected_action.get("roi_per_1k", 0.0), 0.0),
        "selected_action_large_deficit_incremental_exception": _safe_int(
            selected_action.get("large_deficit_incremental_exception", 0),
            0,
        ),
        "selected_action_threshold_crossing_repair_exception": _safe_int(
            selected_action.get("threshold_crossing_repair_exception", 0),
            0,
        ),
        "selected_action_source_backed_material_repair_exception": _safe_int(
            selected_action.get("source_backed_material_repair_exception", 0),
            0,
        ),
        "selected_action_coverage_addition_exception": _safe_int(
            selected_action.get("coverage_addition_exception", 0),
            0,
        ),
        "selected_action_token_reserve_exception": _safe_int(
            selected_action.get("token_reserve_exception", 0),
            0,
        ),
        "selected_action_token_reserve_exception_reason": str(
            selected_action.get("token_reserve_exception_reason", "") or ""
        ),
        "selected_action_effective_tail_token_reserve": _safe_int(
            selected_action.get("effective_tail_token_reserve", 0),
            0,
        ),
        "selected_action_reserve_after_expected_repair": _safe_int(
            selected_action.get("reserve_after_expected_repair", 0),
            0,
        ),
        "policy_repair_observations": _safe_int(policy_stats.get("repair_observations", 0), 0),
        "policy_repair_accepted_total": _safe_int(policy_stats.get("repair_accepted_total", 0), 0),
        "policy_repair_rejected_total": _safe_int(policy_stats.get("repair_rejected_total", 0), 0),
        "policy_repair_accept_rate": _safe_float(policy_stats.get("repair_accept_rate", 0.0), 0.0),
        "policy_last_accepted": 1 if bool(policy_stats.get("last_accepted", False)) else 0,
        "policy_last_patch_accepted": 1 if bool(policy_stats.get("last_patch_accepted", False)) else 0,
        "policy_last_success": 1 if bool(policy_stats.get("last_policy_success", policy_stats.get("last_accepted", False))) else 0,
        "policy_last_gain": _safe_float(policy_stats.get("last_gain", 0.0), 0.0),
        "policy_last_tokens": _safe_int(policy_stats.get("last_tokens", 0), 0),
        "policy_last_generation_tokens": _safe_int(policy_stats.get("last_generation_tokens", 0), 0),
        "policy_last_validation_tokens": _safe_int(policy_stats.get("last_validation_tokens", 0), 0),
        "policy_last_observed_roi_per_1k": _safe_float(policy_stats.get("last_observed_roi_per_1k", 0.0), 0.0),
        "policy_last_terminal_utility_before": _safe_float(policy_stats.get("last_terminal_utility_before", 0.0), 0.0),
        "policy_last_terminal_utility_after": _safe_float(policy_stats.get("last_terminal_utility_after", 0.0), 0.0),
        "policy_last_terminal_utility_delta": _safe_float(policy_stats.get("last_terminal_utility_delta", 0.0), 0.0),
        "policy_last_terminal_utility_per_1k_tokens": _safe_float(
            policy_stats.get("last_terminal_utility_per_1k_tokens", 0.0),
            0.0,
        ),
        "policy_last_repair_success_margin": _safe_float(policy_stats.get("last_repair_success_margin", 0.0), 0.0),
        "checkpoint_terminal_utility": _safe_float(checkpoint_selection.get("terminal_utility", 0.0), 0.0),
        "checkpoint_lcb_utility": _safe_float(checkpoint_selection.get("lcb_utility", 0.0), 0.0),
        "cross_model_judging_any": 1 if any(bool(audit.get("cross_model_judging", False)) for audit in (state.get("tool_audit", []) or []) if isinstance(audit, dict)) else 0,
        "material_failing_claim_count_last": _safe_int(last_retry_decision.get("material_failing_claim_count", 0), 0),
        "specific_repair_claim_count_last": _safe_int(last_retry_decision.get("specific_repair_claim_count", 0), 0),
        "accepted_patch_count": _safe_int(state.get("accepted_patch_count", 0), 0),
        "rejected_patch_count": _safe_int(state.get("rejected_patch_count", 0), 0),
        "repair_outcome_accepted_count": repair_outcome_accepted_count,
        "repair_outcome_rejected_count": repair_outcome_rejected_count,
        "repair_outcome_policy_success_count": repair_outcome_policy_success_count,
        "repair_outcome_policy_failed_count": repair_outcome_policy_failed_count,
        "repair_outcome_policy_outcome_last": repair_outcome_policy_outcome_last,
        "repair_outcome_terminal_utility_before_last": repair_outcome_terminal_utility_before_last,
        "repair_outcome_terminal_utility_after_last": repair_outcome_terminal_utility_after_last,
        "repair_outcome_terminal_utility_delta_last": repair_outcome_terminal_utility_delta_last,
        "repair_outcome_observed_roi_per_1k_last": repair_outcome_observed_roi_per_1k_last,
        "repair_outcome_terminal_utility_per_1k_tokens_last": repair_outcome_terminal_utility_per_1k_tokens_last,
        "repair_outcome_repair_success_margin_last": repair_outcome_repair_success_margin_last,
        "over_decomposition_flag": 1 if bool(state.get("over_decomposition_flag", False)) else 0,
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
        "micro_repair_coverage_addition_count": micro_repair_coverage_addition_count,
        "micro_repair_qualify_remove_count": micro_repair_qualify_remove_count,
        "micro_repair_tokens": micro_repair_tokens,
        "repair_generation_tokens": repair_generation_tokens,
        "observed_repair_roundtrip_tokens_last": observed_repair_roundtrip_tokens_last,
        "observed_repair_roundtrip_tokens_total": observed_repair_roundtrip_tokens_total,
        "repair_gain_per_1k_tokens": repair_gain_per_1k_tokens,
        "repair_gain_per_1k_roundtrip_tokens": repair_gain_per_1k_roundtrip_tokens,
        "lightweight_repair_validation_count": lightweight_repair_validation_count,
        "repair_validation_tokens": repair_validation_tokens,
        "repair_full_validation_tokens": repair_full_validation_tokens,
        "post_repair_validation_tokens": post_repair_validation_tokens,
        "repair_patch_accepted_count": repair_patch_accepted_count,
        "repair_validator_escalation_count": repair_validator_escalation_count,
        "repair_cascade_score_delta": repair_cascade_score_delta,
        "validation_checkpoint_count": len(validation_snapshots),
        "best_validation_score": _safe_int(checkpoint_selection.get("best_validation_score", reliability_score), reliability_score),
        "selected_previous_checkpoint": 1 if bool(checkpoint_selection.get("selected_previous_checkpoint", False)) else 0,
        "checkpoint_quality_gain_override": 1 if bool(checkpoint_selection.get("quality_gain_override", False)) else 0,
        "checkpoint_score_delta_vs_current": _safe_float(checkpoint_selection.get("score_delta_vs_current", 0.0), 0.0),
        # Filled by _attach_relative_metrics and _attach_relative_metrics_vs_shallow.
        "reliability_gain_vs_single": 0.0,
        "depth_delta_vs_single": 0.0,
        "depth_vs_reliability_gain": 0.0,
        "reliability_gain_vs_fixed_shallow": 0.0,
        "token_delta_vs_fixed_shallow": 0.0,
        "efficiency_gain_vs_fixed_shallow": 0.0,
        "adaptive_quality_win_vs_fixed_shallow": 0,
        "adaptive_efficiency_win_vs_fixed_shallow": 0,
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

        rel_gain = _safe_float(metrics.get("reliability_score", 0.0)) - _safe_float(baseline.get("reliability_score", 0.0))
        depth_delta = _safe_float(metrics.get("decomposition_depth_realized", 0.0)) - _safe_float(baseline.get("decomposition_depth_realized", 0.0))
        depth_vs_rel_gain = rel_gain if abs(depth_delta) < 1e-9 else rel_gain / depth_delta
        metrics["reliability_gain_vs_single"] = round(rel_gain, 4)
        metrics["depth_delta_vs_single"] = round(depth_delta, 4)
        metrics["depth_vs_reliability_gain"] = round(depth_vs_rel_gain, 4)


def _attach_relative_metrics_vs_shallow(run_rows: List[Dict[str, Any]]) -> None:
    """Attach per-run comparative metrics against the fixed shallow baseline."""
    shallow_by_key: Dict[tuple[int, int], Dict[str, Any]] = {}
    for row in run_rows:
        if row.get("strategy") != "fixed_shallow":
            continue
        key = (_safe_int(row.get("idea_index", 0), 0), _safe_int(row.get("run_index", 0), 0))
        shallow_by_key[key] = row.get("metrics", {})

    for row in run_rows:
        metrics = row.get("metrics", {})
        key = (_safe_int(row.get("idea_index", 0), 0), _safe_int(row.get("run_index", 0), 0))
        shallow = shallow_by_key.get(key)
        if shallow is None:
            metrics["reliability_gain_vs_fixed_shallow"] = 0.0
            metrics["token_delta_vs_fixed_shallow"] = 0.0
            metrics["efficiency_gain_vs_fixed_shallow"] = 0.0
            metrics["adaptive_quality_win_vs_fixed_shallow"] = 0
            metrics["adaptive_efficiency_win_vs_fixed_shallow"] = 0
            continue

        reliability_gain = _safe_float(metrics.get("reliability_score", 0.0)) - _safe_float(shallow.get("reliability_score", 0.0))
        token_delta = _safe_float(metrics.get("actual_total_tokens", 0.0)) - _safe_float(shallow.get("actual_total_tokens", 0.0))
        efficiency_gain = _safe_float(metrics.get("reliability_per_1k_actual_token", 0.0)) - _safe_float(shallow.get("reliability_per_1k_actual_token", 0.0))
        metrics["reliability_gain_vs_fixed_shallow"] = round(reliability_gain, 4)
        metrics["token_delta_vs_fixed_shallow"] = round(token_delta, 4)
        metrics["efficiency_gain_vs_fixed_shallow"] = round(efficiency_gain, 4)
        metrics["adaptive_quality_win_vs_fixed_shallow"] = 1 if reliability_gain > 0 else 0
        metrics["adaptive_efficiency_win_vs_fixed_shallow"] = 1 if efficiency_gain > 0 else 0


def _aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(rec["strategy"], []).append(rec)

    aggregate: Dict[str, Dict[str, Any]] = {}
    for strategy, rows in grouped.items():
        strat_summary: Dict[str, Any] = {"runs": float(len(rows))}
        for field in CORE_NUMERIC_FIELDS:
            values = [_safe_float(r.get(field, 0.0), 0.0) for r in rows]
            if not values:
                continue
            strat_summary[f"{field}_mean"] = round(statistics.fmean(values), 4)
            strat_summary[f"{field}_min"] = round(min(values), 4)
            strat_summary[f"{field}_max"] = round(max(values), 4)
            strat_summary[f"{field}_std"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0

        for mode_field, output_name in [
            ("controller_mode", "mode_distribution"),
            ("controller_mode_initial", "initial_mode_distribution"),
            ("controller_mode_realized", "realized_mode_distribution"),
        ]:
            counts: Dict[str, int] = {}
            for row in rows:
                mode = str(row.get(mode_field, "n/a")).strip().lower() or "n/a"
                counts[mode] = counts.get(mode, 0) + 1
            strat_summary[output_name] = counts
            strat_summary[f"{output_name}_pct"] = {
                mode: round(count / max(1, len(rows)), 4) for mode, count in counts.items()
            }

        retry_type_distribution: Dict[str, int] = {}
        for row in rows:
            for field, name in [
                ("claim_micro_repair_allowed_count", "claim_micro_repair"),
                ("fixed_recursive_market_retry_count", "fixed_recursive_market_retry"),
                ("shallow_escalation_count", "shallow_escalation"),
                ("recursive_retry_count", "recursive_retry"),
            ]:
                retry_type_distribution[name] = retry_type_distribution.get(name, 0) + _safe_int(row.get(field, 0), 0)
        strat_summary["retry_type_distribution"] = retry_type_distribution
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
            "efficiency-first: lowest budget violations, then highest reliability per 1K actual tokens"
        ),
        "best_efficiency_stats": efficiency_stats,
        "best_balanced_strategy": balanced_name,
        "best_balanced_rule": (
            "balanced score: 0.65*mean reliability + 0.35*mean reliability per 1K actual tokens"
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
    if max_runtime_seconds is not None and max_runtime_seconds >= 0 and runtime_seconds > max_runtime_seconds:
        budget_hit = True
        reasons.append(f"max_runtime_seconds_exceeded:{runtime_seconds:.3f}>{float(max_runtime_seconds):.3f}")

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
        "total_tokens": None if max_total_tokens is None else max_total_tokens - total_tokens,
        "runtime_seconds": None if max_runtime_seconds is None else round(float(max_runtime_seconds) - runtime_seconds, 3),
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
                    "- Keep outputs concise, evidence-aware, and investor-oriented.\n\n"
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
            if max_token_proxy is not None and max_token_proxy >= 0 and token_proxy > max_token_proxy:
                reasons.append(f"max_token_proxy_exceeded:{token_proxy}>{max_token_proxy}")
            if max_total_tokens is not None and max_total_tokens >= 0 and total_tokens > max_total_tokens:
                reasons.append(f"max_total_tokens_exceeded:{total_tokens}>{max_total_tokens}")
            if max_runtime_seconds is not None and max_runtime_seconds >= 0 and elapsed > max_runtime_seconds:
                reasons.append(f"max_runtime_seconds_exceeded:{elapsed:.3f}>{float(max_runtime_seconds):.3f}")
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
                    "token_proxy": None if max_token_proxy is None else max_token_proxy - token_proxy,
                    "total_tokens": None if max_total_tokens is None else max_total_tokens - total_tokens,
                    "runtime_seconds": None if max_runtime_seconds is None else round(float(max_runtime_seconds) - elapsed, 3),
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

        result, usage = _invoke_structured_with_usage(self.llm, self.prompt.format_messages(idea=idea))
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
                "query": query,
                "status": search_payload.get("status", "unknown"),
                "source_count": len(search_payload.get("sources", [])),
                "error": search_payload.get("error", ""),
            }
        )

        keywords = ["startup market", "industry trends", "competitor landscape", "customer adoption", "automation demand"]
        if self.trends_tool is not None:
            trends_payload = self.trends_tool.fetch(keywords)
        else:
            trends_payload = {"status": "skipped", "keywords": keywords, "data": {}, "error": "Google Trends disabled by configuration."}
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
            "controller_policy": "single_agent",
            "controller_mode": "single_agent",
            "controller_mode_initial": "single_agent",
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
            "decomposition_depth_target": 0,
            "decomposition_depth_realized": 0,
            "adaptive_retry_enabled": False,
            "adaptive_checkpoint_enabled": False,
        }
        baseline_state = apply_budget(baseline_state, "single_agent_synthesis")
        if baseline_state.get("budget_hit"):
            return baseline_state

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


def _make_single_runner(
    *,
    normalized: List[str],
    model: str,
    secondary_judge_model: str | None,
    temperature: float,
    seed: int,
    enable_trends: bool,
    output_dir: str,
) -> SingleAgentPitchRunner | None:
    if "single_agent" not in normalized:
        return None
    return SingleAgentPitchRunner(
        model=model,
        secondary_judge_model=secondary_judge_model,
        temperature=temperature,
        seed=seed,
        enable_trends=enable_trends,
        output_dir=output_dir,
    )


def _make_graph_runner(
    *,
    needed: bool,
    model: str,
    secondary_judge_model: str | None,
    temperature: float,
    seed: int,
    strict_tools: bool,
    enable_trends: bool,
    generate_ppt: bool,
    controller_policy: str,
    output_dir: str,
    utility_weights: Optional[Dict[str, float]] = None,
) -> StartupPitchRefinery | None:
    if not needed:
        return None
    return StartupPitchRefinery(
        model=model,
        secondary_judge_model=secondary_judge_model,
        temperature=temperature,
        seed=seed,
        strict_tools=strict_tools,
        enable_trends=enable_trends,
        generate_pitch=generate_ppt,
        controller_policy=controller_policy,
        output_dir=output_dir,
        utility_weights=utility_weights,
    )


def _clone_jsonable(value: Any) -> Any:
    """Deep-copy JSON-like experiment state without sharing mutable memory."""
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return dict(value) if isinstance(value, dict) else value


def _policy_stats_summary(policy_stats_by_strategy: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for strategy, stats in (policy_stats_by_strategy or {}).items():
        if not isinstance(stats, dict):
            continue
        buckets = stats.get("repair_buckets", {}) if isinstance(stats.get("repair_buckets", {}), dict) else {}
        direct_buckets = (
            stats.get("direct_entry_buckets", {})
            if isinstance(stats.get("direct_entry_buckets", {}), dict)
            else {}
        )
        summary[strategy] = {
            "repair_observations": int(stats.get("repair_observations", 0) or 0),
            "repair_accepted_total": int(stats.get("repair_accepted_total", 0) or 0),
            "repair_rejected_total": int(stats.get("repair_rejected_total", 0) or 0),
            "repair_accept_rate": float(stats.get("repair_accept_rate", 0.0) or 0.0),
            "patch_accepted_total": int(stats.get("patch_accepted_total", 0) or 0),
            "patch_rejected_total": int(stats.get("patch_rejected_total", 0) or 0),
            "patch_accept_rate": float(stats.get("patch_accept_rate", 0.0) or 0.0),
            "bucket_count": len(buckets),
            "buckets": {
                str(bucket): {
                    "observations": int(bucket_stats.get("observations", 0) or 0),
                    "accepted": int(bucket_stats.get("accepted", 0) or 0),
                    "rejected": int(bucket_stats.get("rejected", 0) or 0),
                    "patch_accepted": int(bucket_stats.get("patch_accepted", 0) or 0),
                    "patch_rejected": int(bucket_stats.get("patch_rejected", 0) or 0),
                    "accept_rate": float(bucket_stats.get("accept_rate", 0.0) or 0.0),
                    "patch_accept_rate": float(bucket_stats.get("patch_accept_rate", 0.0) or 0.0),
                    "mean_gain": float(bucket_stats.get("mean_gain", 0.0) or 0.0),
                    "mean_tokens": float(bucket_stats.get("mean_tokens", 0.0) or 0.0),
                    "mean_terminal_utility_delta": float(bucket_stats.get("mean_terminal_utility_delta", 0.0) or 0.0),
                }
                for bucket, bucket_stats in buckets.items()
                if isinstance(bucket_stats, dict)
            },
            "direct_precheck_attempts": int(stats.get("direct_precheck_attempts", 0) or 0),
            "direct_precheck_accepted_total": int(stats.get("direct_precheck_accepted_total", 0) or 0),
            "direct_precheck_rejected_total": int(stats.get("direct_precheck_rejected_total", 0) or 0),
            "direct_precheck_skipped_total": int(stats.get("direct_precheck_skipped_total", 0) or 0),
            "direct_precheck_accept_rate": float(stats.get("direct_precheck_accept_rate", 0.0) or 0.0),
            "direct_entry_bucket_count": len(direct_buckets),
            "direct_entry_buckets": {
                str(bucket): {
                    "observations": int(bucket_stats.get("observations", 0) or 0),
                    "accepted": int(bucket_stats.get("accepted", 0) or 0),
                    "rejected": int(bucket_stats.get("rejected", 0) or 0),
                    "skipped": int(bucket_stats.get("skipped", 0) or 0),
                    "accept_rate": float(bucket_stats.get("accept_rate", 0.0) or 0.0),
                    "mean_direct_tokens": float(bucket_stats.get("mean_direct_tokens", 0.0) or 0.0),
                    "mean_shallow_tokens": float(bucket_stats.get("mean_shallow_tokens", 0.0) or 0.0),
                    "mean_score": float(bucket_stats.get("mean_score", 0.0) or 0.0),
                }
                for bucket, bucket_stats in direct_buckets.items()
                if isinstance(bucket_stats, dict)
            },
        }
    return summary


def _policy_ucb_bonus(*, total_attempts: float, n_empirical: float, exploration: float) -> float:
    if exploration <= 0.0:
        return 0.0
    exploration_weight = 1.0 / max(1.0, math.sqrt(1.0 + n_empirical))
    raw_bonus = exploration_weight * math.sqrt(
        2.0 * math.log(max(2.0, total_attempts + 1.0)) / max(1.0, n_empirical + 1.0)
    )
    return min(0.20, exploration * raw_bonus)


def _direct_entry_bucket(features: Dict[str, Any]) -> str:
    structural = _safe_float(features.get("structural_complexity", 0.0), 0.0)
    uncertainty = _safe_float(features.get("uncertainty_need", 0.0), 0.0)
    evidence_count = _safe_int(features.get("evidence_count", 0), 0)
    marker_count = _safe_int(features.get("marker_count", 0), 0)
    workflow_count = _safe_int(features.get("workflow_count", 0), 0)

    if structural >= 50.0 or uncertainty >= 50.0 or evidence_count >= 3 or marker_count >= 4:
        return "high_complexity"
    if structural >= 23.0 or evidence_count >= 1 or marker_count >= 1 or workflow_count >= 2:
        return "moderate"
    if workflow_count >= 1:
        return "simple_workflow"
    return "simple_plain"


def _direct_entry_gate_decision(
    *,
    idea: str,
    policy_stats: Dict[str, Any],
    policy_exploration: float,
) -> Dict[str, Any]:
    """Decide whether adaptive should pay for a direct validation precheck."""
    features = AdaptiveControllerAgent._complexity_features(idea)
    bucket = _direct_entry_bucket(features)
    prior = DIRECT_ENTRY_PRIORS.get(bucket, DIRECT_ENTRY_DEFAULT_PRIOR)
    stats = policy_stats if isinstance(policy_stats, dict) else {}
    buckets = stats.get("direct_entry_buckets", {})
    observed = buckets.get(bucket, {}) if isinstance(buckets, dict) and isinstance(buckets.get(bucket), dict) else {}

    accepted = _safe_float(observed.get("accepted", 0.0), 0.0)
    rejected = _safe_float(observed.get("rejected", 0.0), 0.0)
    observations = max(_safe_float(observed.get("observations", 0.0), 0.0), accepted + rejected)
    alpha = _safe_float(prior.get("alpha", DIRECT_ENTRY_DEFAULT_PRIOR["alpha"]), DIRECT_ENTRY_DEFAULT_PRIOR["alpha"])
    beta = _safe_float(prior.get("beta", DIRECT_ENTRY_DEFAULT_PRIOR["beta"]), DIRECT_ENTRY_DEFAULT_PRIOR["beta"])
    posterior_alpha = alpha + accepted
    posterior_beta = beta + rejected
    posterior_mean = posterior_alpha / max(1e-6, posterior_alpha + posterior_beta)
    total_attempts = _safe_float(stats.get("direct_precheck_attempts", 0.0), 0.0)
    ucb_bonus = _policy_ucb_bonus(
        total_attempts=total_attempts,
        n_empirical=observations,
        exploration=max(0.0, float(policy_exploration)),
    )
    posterior_acceptance = min(1.0, posterior_mean + ucb_bonus)

    obs_direct_tokens = _safe_float(
        observed.get("mean_direct_tokens", prior.get("direct_tokens", 0.0)),
        _safe_float(prior.get("direct_tokens", 0.0), 0.0),
    )
    obs_shallow_tokens = _safe_float(
        observed.get("mean_shallow_tokens", prior.get("shallow_tokens", 0.0)),
        _safe_float(prior.get("shallow_tokens", 0.0), 0.0),
    )
    token_kappa = 3.0
    expected_direct_tokens = (
        token_kappa * _safe_float(prior.get("direct_tokens", 0.0), 0.0)
        + observations * obs_direct_tokens
    ) / max(1e-6, token_kappa + observations)
    expected_shallow_tokens = (
        token_kappa * _safe_float(prior.get("shallow_tokens", 0.0), 0.0)
        + observations * obs_shallow_tokens
    ) / max(1e-6, token_kappa + observations)

    acceptance_threshold = min(
        0.85,
        max(
            DIRECT_ENTRY_MIN_ACCEPTANCE,
            (expected_direct_tokens + DIRECT_ENTRY_MIN_NET_TOKENS) / max(1.0, expected_shallow_tokens),
        ),
    )
    expected_net_tokens = posterior_acceptance * expected_shallow_tokens - expected_direct_tokens

    block_reasons: List[str] = []
    if posterior_acceptance < acceptance_threshold:
        block_reasons.append(
            f"posterior_acceptance {posterior_acceptance:.3f} < threshold {acceptance_threshold:.3f}"
        )
    if expected_net_tokens < DIRECT_ENTRY_MIN_NET_TOKENS:
        block_reasons.append(
            f"expected_net_tokens {expected_net_tokens:.0f} < {DIRECT_ENTRY_MIN_NET_TOKENS:.0f}"
        )

    allowed = not block_reasons
    return {
        "allowed": allowed,
        "bucket_key": bucket,
        "features": features,
        "prior_alpha": round(alpha, 4),
        "prior_beta": round(beta, 4),
        "observations": int(observations),
        "accepted": int(accepted),
        "rejected": int(rejected),
        "posterior_mean": round(posterior_mean, 4),
        "ucb_bonus": round(ucb_bonus, 4),
        "posterior_acceptance": round(posterior_acceptance, 4),
        "acceptance_threshold": round(acceptance_threshold, 4),
        "expected_direct_tokens": round(expected_direct_tokens, 2),
        "expected_shallow_tokens": round(expected_shallow_tokens, 2),
        "expected_net_tokens": round(expected_net_tokens, 2),
        "min_expected_net_tokens": DIRECT_ENTRY_MIN_NET_TOKENS,
        "block_reasons": block_reasons,
        "reason": "direct_precheck_expected_token_savings_positive" if allowed else "direct_precheck_gate_blocked",
    }


def _direct_entry_policy_stats_update(
    policy_stats: Dict[str, Any],
    *,
    gate_decision: Dict[str, Any],
    attempted: bool,
    accepted: bool = False,
    direct_tokens: int = 0,
    shallow_tokens: int = 0,
    direct_score: int = 0,
) -> Dict[str, Any]:
    updated = _clone_jsonable(policy_stats or {})
    bucket = str(gate_decision.get("bucket_key", "moderate") or "moderate")
    buckets = updated.setdefault("direct_entry_buckets", {})
    bucket_stats = buckets.setdefault(bucket, {})

    if not attempted:
        updated["direct_precheck_skipped_total"] = _safe_int(updated.get("direct_precheck_skipped_total", 0), 0) + 1
        bucket_stats["skipped"] = _safe_int(bucket_stats.get("skipped", 0), 0) + 1
        return updated

    observations = _safe_int(bucket_stats.get("observations", 0), 0) + 1
    old_obs = max(0, observations - 1)
    direct_tokens = max(0, _safe_int(direct_tokens, 0))
    shallow_tokens = max(0, _safe_int(shallow_tokens, 0))

    bucket_stats["observations"] = observations
    bucket_stats["accepted"] = _safe_int(bucket_stats.get("accepted", 0), 0) + (1 if accepted else 0)
    bucket_stats["rejected"] = _safe_int(bucket_stats.get("rejected", 0), 0) + (0 if accepted else 1)
    bucket_stats["accept_rate"] = round(
        _safe_int(bucket_stats.get("accepted", 0), 0) / max(1, observations),
        4,
    )
    bucket_stats["mean_direct_tokens"] = round(
        (
            _safe_float(bucket_stats.get("mean_direct_tokens", 0.0), 0.0) * old_obs
            + direct_tokens
        )
        / max(1, observations),
        4,
    )
    if shallow_tokens > 0:
        bucket_stats["mean_shallow_tokens"] = round(
            (
                _safe_float(bucket_stats.get("mean_shallow_tokens", 0.0), 0.0) * old_obs
                + shallow_tokens
            )
            / max(1, observations),
            4,
        )
    bucket_stats["mean_score"] = round(
        (
            _safe_float(bucket_stats.get("mean_score", 0.0), 0.0) * old_obs
            + _safe_float(direct_score, 0.0)
        )
        / max(1, observations),
        4,
    )
    bucket_stats["mean_expected_net_tokens"] = round(
        (
            _safe_float(bucket_stats.get("mean_expected_net_tokens", 0.0), 0.0) * old_obs
            + _safe_float(gate_decision.get("expected_net_tokens", 0.0), 0.0)
        )
        / max(1, observations),
        4,
    )

    updated["direct_precheck_attempts"] = _safe_int(updated.get("direct_precheck_attempts", 0), 0) + 1
    updated["direct_precheck_accepted_total"] = _safe_int(updated.get("direct_precheck_accepted_total", 0), 0) + (1 if accepted else 0)
    updated["direct_precheck_rejected_total"] = _safe_int(updated.get("direct_precheck_rejected_total", 0), 0) + (0 if accepted else 1)
    updated["direct_precheck_accept_rate"] = round(
        _safe_int(updated.get("direct_precheck_accepted_total", 0), 0)
        / max(1, _safe_int(updated.get("direct_precheck_attempts", 0), 0)),
        4,
    )
    return updated


def _policy_calibration_rows(
    policy_stats_by_strategy: Dict[str, Dict[str, Any]],
    *,
    policy_exploration: float = 0.15,
) -> List[Dict[str, Any]]:
    """Return auditable prior-to-posterior rows for Bayesian repair buckets."""
    rows: List[Dict[str, Any]] = []
    exploration = max(0.0, float(policy_exploration))
    default_prior = BayesianRetryPolicy.DEFAULT_PRIORS["qualify_or_remove"]
    for strategy, stats in (policy_stats_by_strategy or {}).items():
        if not isinstance(stats, dict):
            continue
        buckets = stats.get("repair_buckets", {}) if isinstance(stats.get("repair_buckets", {}), dict) else {}
        bucket_names = sorted(set(BayesianRetryPolicy.DEFAULT_PRIORS.keys()) | set(str(k) for k in buckets.keys()))
        total_attempts = _safe_float(stats.get("repair_observations", 0.0), 0.0)
        for bucket in bucket_names:
            observed = buckets.get(bucket, {}) if isinstance(buckets.get(bucket, {}), dict) else {}
            base_bucket = BayesianRetryPolicy.base_bucket_key(bucket)
            prior = BayesianRetryPolicy.DEFAULT_PRIORS.get(base_bucket, default_prior)
            accepted = _safe_float(observed.get("accepted", 0.0), 0.0)
            rejected = _safe_float(observed.get("rejected", 0.0), 0.0)
            patch_accepted = _safe_float(observed.get("patch_accepted", 0.0), 0.0)
            patch_rejected = _safe_float(observed.get("patch_rejected", 0.0), 0.0)
            observations = max(_safe_float(observed.get("observations", 0.0), 0.0), accepted + rejected)
            alpha = _safe_float(prior.get("alpha", default_prior["alpha"]), default_prior["alpha"])
            beta = _safe_float(prior.get("beta", default_prior["beta"]), default_prior["beta"])
            posterior_alpha = alpha + accepted
            posterior_beta = beta + rejected
            prior_acceptance = alpha / max(1e-6, alpha + beta)
            posterior_mean = posterior_alpha / max(1e-6, posterior_alpha + posterior_beta)
            ucb_bonus = _policy_ucb_bonus(
                total_attempts=total_attempts,
                n_empirical=observations,
                exploration=exploration,
            )
            posterior_acceptance_ucb = min(1.0, posterior_mean + ucb_bonus)
            default_weights = BayesianRetryPolicy.DEFAULT_WEIGHTS
            gain_kappa = max(0.0, _safe_float(default_weights.get("gain_kappa", 2.0), 2.0))
            token_kappa = max(0.0, _safe_float(default_weights.get("token_kappa", 3.0), 3.0))
            seconds_kappa = max(0.0, _safe_float(default_weights.get("seconds_kappa", 5.0), 5.0))
            obs_gain = _safe_float(observed.get("mean_gain", prior.get("gain", 0.0)), _safe_float(prior.get("gain", 0.0), 0.0))
            obs_tokens = _safe_float(observed.get("mean_tokens", prior.get("tokens", 0.0)), _safe_float(prior.get("tokens", 0.0), 0.0))
            obs_seconds = _safe_float(observed.get("mean_seconds", prior.get("seconds", 0.0)), _safe_float(prior.get("seconds", 0.0), 0.0))
            expected_gain = (
                gain_kappa * _safe_float(prior.get("gain", 0.0), 0.0) + observations * obs_gain
            ) / max(1e-6, gain_kappa + observations)
            expected_tokens = (
                token_kappa * _safe_float(prior.get("tokens", 0.0), 0.0) + observations * obs_tokens
            ) / max(1e-6, token_kappa + observations)
            expected_seconds = (
                seconds_kappa * _safe_float(prior.get("seconds", 0.0), 0.0) + observations * obs_seconds
            ) / max(1e-6, seconds_kappa + observations)
            rows.append(
                {
                    "strategy": strategy,
                    "bucket_key": bucket,
                    "base_bucket_key": base_bucket,
                    "prior_alpha": round(alpha, 4),
                    "prior_beta": round(beta, 4),
                    "prior_acceptance": round(prior_acceptance, 4),
                    "observations": int(observations),
                    "accepted": int(accepted),
                    "rejected": int(rejected),
                    "patch_accepted": int(patch_accepted),
                    "patch_rejected": int(patch_rejected),
                    "observed_accept_rate": round(accepted / max(1.0, observations), 4) if observations else 0.0,
                    "observed_patch_accept_rate": round(patch_accepted / max(1.0, observations), 4) if observations else 0.0,
                    "posterior_alpha": round(posterior_alpha, 4),
                    "posterior_beta": round(posterior_beta, 4),
                    "posterior_mean": round(posterior_mean, 4),
                    "ucb_bonus": round(ucb_bonus, 4),
                    "posterior_acceptance_ucb": round(posterior_acceptance_ucb, 4),
                    "prior_to_posterior_delta": round(posterior_mean - prior_acceptance, 4),
                    "expected_gain": round(expected_gain, 4),
                    "expected_tokens": round(expected_tokens, 4),
                    "expected_seconds": round(expected_seconds, 4),
                    "gain_kappa": round(gain_kappa, 4),
                    "token_kappa": round(token_kappa, 4),
                    "seconds_kappa": round(seconds_kappa, 4),
                    "mean_observed_gain": round(obs_gain, 4),
                    "mean_observed_tokens": round(obs_tokens, 4),
                    "mean_observed_roi_per_1k": round(
                        _safe_float(observed.get("mean_observed_roi_per_1k", 0.0), 0.0),
                        4,
                    ),
                    "mean_terminal_utility_delta": round(
                        _safe_float(observed.get("mean_terminal_utility_delta", 0.0), 0.0),
                        4,
                    ),
                    "mean_terminal_utility_per_1k_tokens": round(
                        _safe_float(observed.get("mean_terminal_utility_per_1k_tokens", 0.0), 0.0),
                        4,
                    ),
                }
            )
    return rows


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
    initial_policy_stats_by_strategy: Optional[Dict[str, Dict[str, Any]]] = None,
    policy_exploration: float = 0.15,
) -> Dict[str, Any]:
    """Run strategy comparison with adaptive as a paired validated cascade.

    Adaptive strategies use canonical direct and shallow checkpoints to isolate
    the controller effect. Direct can stop early if it validates; otherwise the
    adaptive branch continues from the exact fixed-shallow checkpoint, and only
    differs from it when the recursive repair gate executes.
    """
    strategies = strategies or list(DEFAULT_STRATEGIES)
    normalized = [s.strip().lower() for s in strategies if s and s.strip()]
    invalid = [s for s in normalized if s not in ALLOWED_STRATEGIES]
    if invalid:
        raise ValueError(f"Unsupported strategies: {invalid}. Allowed: {sorted(ALLOWED_STRATEGIES)}")
    active_adaptive_strategies = [s for s in normalized if s in ADAPTIVE_STRATEGIES]
    utility_weights = {"ucb_exploration": max(0.0, float(policy_exploration))}

    single_runner = _make_single_runner(
        normalized=normalized,
        model=model,
        secondary_judge_model=secondary_judge_model,
        temperature=temperature,
        seed=seed,
        enable_trends=enable_trends,
        output_dir=output_dir,
    )
    multi_runner = _make_graph_runner(
        needed="multi_agent" in normalized,
        model=model,
        secondary_judge_model=secondary_judge_model,
        temperature=temperature,
        seed=seed,
        strict_tools=strict_tools,
        enable_trends=enable_trends,
        generate_ppt=generate_ppt,
        controller_policy="fixed",
        output_dir=output_dir,
        utility_weights=utility_weights,
    )
    adaptive_runner = _make_graph_runner(
        needed=any(s in normalized for s in ADAPTIVE_STRATEGIES | {"fixed_direct", "fixed_shallow", "fixed_recursive"}),
        model=model,
        secondary_judge_model=secondary_judge_model,
        temperature=temperature,
        seed=seed,
        strict_tools=strict_tools,
        enable_trends=enable_trends,
        generate_ppt=generate_ppt,
        controller_policy="adaptive",
        output_dir=output_dir,
        utility_weights=utility_weights,
    )

    run_rows: List[Dict[str, Any]] = []
    states: Dict[str, Dict[str, Any]] = {}
    use_shared_refinement = any(strategy in GRAPH_BASED_STRATEGIES for strategy in normalized)
    initial_policy_stats_by_strategy = initial_policy_stats_by_strategy or {}
    policy_stats_by_strategy: Dict[str, Dict[str, Any]] = {
        s: _clone_jsonable(initial_policy_stats_by_strategy.get(s, {}))
        for s in ADAPTIVE_STRATEGIES
    }

    def run_fixed_shallow_once(run_idx: int, shared_refinement: Dict[str, Any]) -> tuple[Dict[str, Any], float]:
        assert adaptive_runner is not None
        start = time.perf_counter()
        state = adaptive_runner.run(
            idea=idea,
            thread_id=f"{thread_prefix}-fixed_shallow-{run_idx}",
            max_validation_retries=max_validation_retries,
            validation_threshold=validation_threshold,
            max_tool_calls=max_tool_calls,
            max_token_proxy=max_token_proxy,
            max_total_tokens=max_total_tokens,
            max_runtime_seconds=max_runtime_seconds,
            forced_controller_mode="shallow",
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
        return state, runtime

    def run_fixed_direct_once(run_idx: int, shared_refinement: Dict[str, Any]) -> tuple[Dict[str, Any], float]:
        assert adaptive_runner is not None
        start = time.perf_counter()
        state = adaptive_runner.run(
            idea=idea,
            thread_id=f"{thread_prefix}-fixed_direct-{run_idx}",
            max_validation_retries=max_validation_retries,
            validation_threshold=validation_threshold,
            max_tool_calls=max_tool_calls,
            max_token_proxy=max_token_proxy,
            max_total_tokens=max_total_tokens,
            max_runtime_seconds=max_runtime_seconds,
            forced_controller_mode="direct",
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
        return state, runtime

    def direct_precheck_passes(state: Dict[str, Any]) -> bool:
        validation = state.get("validation_report", {}) or {}
        if not isinstance(validation, dict):
            return False
        score = _safe_int(validation.get("reliability_score", 0), 0)
        validator_audit = _latest_validator_audit(state)
        minimum_claims = _safe_int(validator_audit.get("minimum_claims_required", 3), 3)
        claims_total = len(validation.get("claims", []) or [])
        low_claim_count = bool(
            validator_audit.get("low_claim_count_flag", claims_total < minimum_claims)
        )
        return (
            score >= validation_threshold
            and not bool(state.get("needs_revision", False))
            and not low_claim_count
            and not bool(state.get("budget_hit", False))
        )

    def incremental_usage_after_shared(state: Dict[str, Any], shared_refinement: Dict[str, Any]) -> Dict[str, int]:
        usage = state.get("token_usage", {}) or {}
        shared_usage = shared_refinement.get("token_usage", {}) or {}
        return {
            "prompt_tokens": max(
                0,
                _safe_int(usage.get("prompt_tokens", 0), 0)
                - _safe_int(shared_usage.get("prompt_tokens", 0), 0),
            ),
            "completion_tokens": max(
                0,
                _safe_int(usage.get("completion_tokens", 0), 0)
                - _safe_int(shared_usage.get("completion_tokens", 0), 0),
            ),
            "total_tokens": max(
                0,
                _safe_int(usage.get("total_tokens", 0), 0)
                - _safe_int(shared_usage.get("total_tokens", 0), 0),
            ),
        }

    def merge_usage(left: Dict[str, Any], right: Dict[str, int]) -> Dict[str, int]:
        return {
            "prompt_tokens": _safe_int(left.get("prompt_tokens", 0), 0) + _safe_int(right.get("prompt_tokens", 0), 0),
            "completion_tokens": _safe_int(left.get("completion_tokens", 0), 0) + _safe_int(right.get("completion_tokens", 0), 0),
            "total_tokens": _safe_int(left.get("total_tokens", 0), 0) + _safe_int(right.get("total_tokens", 0), 0),
        }

    def direct_extra_audits(state: Dict[str, Any], shared_refinement: Dict[str, Any]) -> List[Dict[str, Any]]:
        audits = list(state.get("tool_audit", []) or [])
        shared_count = len(shared_refinement.get("tool_audit", []) or [])
        return _clone_jsonable(audits[shared_count:]) if len(audits) >= shared_count else _clone_jsonable(audits)

    def annotate_direct_acceptance(
        state: Dict[str, Any],
        *,
        policy_stats: Dict[str, Any],
        adaptive_retry_enabled: bool,
        adaptive_checkpoint_enabled: bool,
        shared_refinement: Dict[str, Any],
        gate_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        accepted = _clone_jsonable(state)
        validation = accepted.get("validation_report", {}) or {}
        score = _safe_int(validation.get("reliability_score", 0), 0) if isinstance(validation, dict) else 0
        incremental_usage = incremental_usage_after_shared(accepted, shared_refinement)
        accepted.update(
            {
                "controller_policy": "adaptive",
                "forced_controller_mode": None,
                "controller_mode": "direct",
                "controller_mode_initial": "direct",
                "controller_mode_realized": "direct",
                "controller_escalated": False,
                "controller_escalation_reason": None,
                "direct_precheck_allowed": True,
                "direct_precheck_skipped": False,
                "direct_precheck_accepted": True,
                "direct_precheck_escalated": False,
                "direct_precheck_score": score,
                "direct_precheck_incremental_tokens": incremental_usage["total_tokens"],
                "direct_precheck_gate_bucket": gate_decision.get("bucket_key"),
                "direct_precheck_gate_reason": gate_decision.get("reason"),
                "direct_precheck_gate_block_reasons": gate_decision.get("block_reasons", []),
                "direct_precheck_gate_probability": gate_decision.get("posterior_acceptance", 0.0),
                "direct_precheck_gate_threshold": gate_decision.get("acceptance_threshold", 0.0),
                "direct_precheck_gate_expected_net_tokens": gate_decision.get("expected_net_tokens", 0.0),
                "adaptive_retry_enabled": adaptive_retry_enabled,
                "adaptive_checkpoint_enabled": adaptive_checkpoint_enabled,
                "policy_stats": dict(policy_stats or accepted.get("policy_stats", {}) or {}),
                "adaptive_cascade_source": "validated_direct_precheck_accepted",
            }
        )
        audit = list(accepted.get("tool_audit", []) or [])
        audit.append(
            {
                "agent": "adaptive_controller",
                "tool": "validated_direct_precheck",
                "status": "accepted",
                "score": score,
                "threshold": validation_threshold,
                "incremental_tokens": incremental_usage["total_tokens"],
                "gate_bucket": gate_decision.get("bucket_key"),
                "gate_probability": gate_decision.get("posterior_acceptance", 0.0),
                "gate_threshold": gate_decision.get("acceptance_threshold", 0.0),
                "gate_expected_net_tokens": gate_decision.get("expected_net_tokens", 0.0),
            }
        )
        accepted["tool_audit"] = audit
        return accepted

    def annotate_direct_skip_to_shallow(
        state: Dict[str, Any],
        *,
        gate_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        skipped = _clone_jsonable(state)
        realized = (
            "recursive"
            if _safe_int(skipped.get("repair_rounds", 0), 0) > 0
            or _safe_int(skipped.get("retry_count", 0), 0) > 0
            else "shallow"
        )
        audit = list(skipped.get("tool_audit", []) or [])
        audit.append(
            {
                "agent": "adaptive_controller",
                "tool": "direct_entry_gate",
                "status": "skipped",
                "gate_bucket": gate_decision.get("bucket_key"),
                "gate_probability": gate_decision.get("posterior_acceptance", 0.0),
                "gate_threshold": gate_decision.get("acceptance_threshold", 0.0),
                "gate_expected_net_tokens": gate_decision.get("expected_net_tokens", 0.0),
                "reason": gate_decision.get("reason"),
                "block_reasons": gate_decision.get("block_reasons", []),
            }
        )
        skipped.update(
            {
                "tool_audit": audit,
                "controller_policy": "adaptive",
                "forced_controller_mode": None,
                "controller_mode_initial": "shallow",
                "controller_mode_realized": realized,
                "controller_mode": realized,
                "controller_escalated": realized != "shallow",
                "controller_escalation_reason": (
                    "direct_entry_gate_skipped_to_recursive_repair"
                    if realized == "recursive"
                    else "direct_entry_gate_skipped_to_shallow"
                ),
                "direct_precheck_allowed": False,
                "direct_precheck_skipped": True,
                "direct_precheck_accepted": False,
                "direct_precheck_escalated": False,
                "direct_precheck_score": 0,
                "direct_precheck_incremental_tokens": 0,
                "direct_precheck_gate_bucket": gate_decision.get("bucket_key"),
                "direct_precheck_gate_reason": gate_decision.get("reason"),
                "direct_precheck_gate_block_reasons": gate_decision.get("block_reasons", []),
                "direct_precheck_gate_probability": gate_decision.get("posterior_acceptance", 0.0),
                "direct_precheck_gate_threshold": gate_decision.get("acceptance_threshold", 0.0),
                "direct_precheck_gate_expected_net_tokens": gate_decision.get("expected_net_tokens", 0.0),
                "adaptive_cascade_source": "direct_entry_gate_skipped_to_shallow_checkpoint",
            }
        )
        return skipped

    def add_direct_precheck_overhead(
        state: Dict[str, Any],
        *,
        direct_state: Dict[str, Any],
        shared_refinement: Dict[str, Any],
        gate_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = _clone_jsonable(state)
        direct_validation = direct_state.get("validation_report", {}) or {}
        direct_score = (
            _safe_int(direct_validation.get("reliability_score", 0), 0)
            if isinstance(direct_validation, dict)
            else 0
        )
        incremental_usage = incremental_usage_after_shared(direct_state, shared_refinement)
        merged["token_usage"] = merge_usage(merged.get("token_usage", {}) or {}, incremental_usage)
        audit = list(merged.get("tool_audit", []) or [])
        audit.extend(direct_extra_audits(direct_state, shared_refinement))
        audit.append(
            {
                "agent": "adaptive_controller",
                "tool": "validated_direct_precheck",
                "status": "escalated",
                "score": direct_score,
                "threshold": validation_threshold,
                "incremental_tokens": incremental_usage["total_tokens"],
                "reason": "direct_validation_failed_or_low_claim_coverage",
                "gate_bucket": gate_decision.get("bucket_key"),
                "gate_probability": gate_decision.get("posterior_acceptance", 0.0),
                "gate_threshold": gate_decision.get("acceptance_threshold", 0.0),
                "gate_expected_net_tokens": gate_decision.get("expected_net_tokens", 0.0),
            }
        )
        merged.update(
            {
                "tool_audit": audit,
                "controller_policy": "adaptive",
                "forced_controller_mode": None,
                "controller_mode_initial": "direct",
                "controller_mode_realized": (
                    "recursive"
                    if _safe_int(merged.get("repair_rounds", 0), 0) > 0
                    or _safe_int(merged.get("retry_count", 0), 0) > 0
                    else "shallow"
                ),
                "controller_mode": (
                    "recursive"
                    if _safe_int(merged.get("repair_rounds", 0), 0) > 0
                    or _safe_int(merged.get("retry_count", 0), 0) > 0
                    else "shallow"
                ),
                "controller_escalated": True,
                "controller_escalation_reason": "direct_precheck_failed_escalated_to_shallow",
                "direct_precheck_allowed": True,
                "direct_precheck_skipped": False,
                "direct_precheck_accepted": False,
                "direct_precheck_escalated": True,
                "direct_precheck_score": direct_score,
                "direct_precheck_incremental_tokens": incremental_usage["total_tokens"],
                "direct_precheck_gate_bucket": gate_decision.get("bucket_key"),
                "direct_precheck_gate_reason": gate_decision.get("reason"),
                "direct_precheck_gate_block_reasons": gate_decision.get("block_reasons", []),
                "direct_precheck_gate_probability": gate_decision.get("posterior_acceptance", 0.0),
                "direct_precheck_gate_threshold": gate_decision.get("acceptance_threshold", 0.0),
                "direct_precheck_gate_expected_net_tokens": gate_decision.get("expected_net_tokens", 0.0),
                "adaptive_cascade_source": "validated_direct_precheck_then_shallow_checkpoint",
            }
        )
        return merged

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
        paired_fixed_shallow_state: Dict[str, Any] | None = None
        paired_fixed_shallow_runtime = 0.0
        paired_fixed_direct_state: Dict[str, Any] | None = None
        paired_fixed_direct_runtime = 0.0

        # Materialize canonical baselines lazily. Adaptive strategies use these
        # paired checkpoints instead of rerunning shallow independently: direct
        # can be accepted as a validated low-cost exit, otherwise the adaptive
        # branch continues from the exact fixed-shallow checkpoint.
        needs_shared_shallow = "fixed_shallow" in normalized
        if needs_shared_shallow:
            paired_fixed_shallow_state, paired_fixed_shallow_runtime = run_fixed_shallow_once(run_idx, shared_refinement)

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
                runtime = time.perf_counter() - start
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
                runtime = time.perf_counter() - start
            elif strategy == "fixed_shallow":
                if paired_fixed_shallow_state is None:
                    paired_fixed_shallow_state, paired_fixed_shallow_runtime = run_fixed_shallow_once(run_idx, shared_refinement)
                state = dict(paired_fixed_shallow_state)
                runtime = paired_fixed_shallow_runtime
            elif strategy == "fixed_direct":
                if paired_fixed_direct_state is None:
                    paired_fixed_direct_state, paired_fixed_direct_runtime = run_fixed_direct_once(run_idx, shared_refinement)
                state = dict(paired_fixed_direct_state)
                runtime = paired_fixed_direct_runtime
            elif strategy == "fixed_recursive":
                assert adaptive_runner is not None
                state = adaptive_runner.run(
                    idea=idea,
                    thread_id=f"{thread_prefix}-{strategy}-{run_idx}",
                    max_validation_retries=max_validation_retries,
                    validation_threshold=validation_threshold,
                    max_tool_calls=max_tool_calls,
                    max_token_proxy=max_token_proxy,
                    max_total_tokens=max_total_tokens,
                    max_runtime_seconds=max_runtime_seconds,
                    forced_controller_mode="recursive",
                    shared_refined_idea=shared_refinement.get("refined_idea"),
                    shared_refinement_token_usage=shared_refinement.get("token_usage"),
                    shared_refinement_tool_audit=shared_refinement.get("tool_audit"),
                )
                runtime = time.perf_counter() - start
            else:
                assert adaptive_runner is not None
                adaptive_retry_enabled = strategy != "adaptive_no_retry"
                adaptive_checkpoint_enabled = strategy != "adaptive_no_checkpoint"
                policy_stats = policy_stats_by_strategy.get(strategy, {})
                gate_decision = _direct_entry_gate_decision(
                    idea=idea,
                    policy_stats=policy_stats,
                    policy_exploration=policy_exploration,
                )
                if not bool(gate_decision.get("allowed", False)):
                    if paired_fixed_shallow_state is None:
                        paired_fixed_shallow_state, paired_fixed_shallow_runtime = run_fixed_shallow_once(run_idx, shared_refinement)
                    policy_stats = _direct_entry_policy_stats_update(
                        policy_stats,
                        gate_decision=gate_decision,
                        attempted=False,
                    )
                    continuation_start = time.perf_counter()
                    state = adaptive_runner.continue_from_shallow_checkpoint(
                        paired_fixed_shallow_state,
                        adaptive_retry_enabled=adaptive_retry_enabled,
                        adaptive_checkpoint_enabled=adaptive_checkpoint_enabled,
                        policy_stats=policy_stats,
                    )
                    continuation_runtime = time.perf_counter() - continuation_start
                    state = annotate_direct_skip_to_shallow(
                        state,
                        gate_decision=gate_decision,
                    )
                    runtime = paired_fixed_shallow_runtime + continuation_runtime
                else:
                    if paired_fixed_direct_state is None:
                        paired_fixed_direct_state, paired_fixed_direct_runtime = run_fixed_direct_once(run_idx, shared_refinement)
                    direct_validation = paired_fixed_direct_state.get("validation_report", {}) or {}
                    direct_score = (
                        _safe_int(direct_validation.get("reliability_score", 0), 0)
                        if isinstance(direct_validation, dict)
                        else 0
                    )
                    direct_usage = incremental_usage_after_shared(paired_fixed_direct_state, shared_refinement)
                    shallow_tokens = 0
                    if paired_fixed_shallow_state is not None:
                        shallow_tokens = incremental_usage_after_shared(paired_fixed_shallow_state, shared_refinement)["total_tokens"]
                    direct_accepted = direct_precheck_passes(paired_fixed_direct_state)
                    policy_stats = _direct_entry_policy_stats_update(
                        policy_stats,
                        gate_decision=gate_decision,
                        attempted=True,
                        accepted=direct_accepted,
                        direct_tokens=direct_usage["total_tokens"],
                        shallow_tokens=shallow_tokens,
                        direct_score=direct_score,
                    )
                    if direct_accepted:
                        state = annotate_direct_acceptance(
                            paired_fixed_direct_state,
                            policy_stats=policy_stats,
                            adaptive_retry_enabled=adaptive_retry_enabled,
                            adaptive_checkpoint_enabled=adaptive_checkpoint_enabled,
                            shared_refinement=shared_refinement,
                            gate_decision=gate_decision,
                        )
                        runtime = paired_fixed_direct_runtime
                    else:
                        if paired_fixed_shallow_state is None:
                            paired_fixed_shallow_state, paired_fixed_shallow_runtime = run_fixed_shallow_once(run_idx, shared_refinement)
                            shallow_tokens = incremental_usage_after_shared(
                                paired_fixed_shallow_state,
                                shared_refinement,
                            )["total_tokens"]
                            direct_entry_bucket = (
                                policy_stats.get("direct_entry_buckets", {})
                                if isinstance(policy_stats.get("direct_entry_buckets", {}), dict)
                                else {}
                            ).get(str(gate_decision.get("bucket_key", "")), {})
                            if isinstance(direct_entry_bucket, dict) and shallow_tokens > 0:
                                observations = max(1, _safe_int(direct_entry_bucket.get("observations", 1), 1))
                                old_obs = max(0, observations - 1)
                                direct_entry_bucket["mean_shallow_tokens"] = round(
                                    (
                                        _safe_float(direct_entry_bucket.get("mean_shallow_tokens", 0.0), 0.0) * old_obs
                                        + shallow_tokens
                                    )
                                    / max(1, observations),
                                    4,
                                )
                        continuation_start = time.perf_counter()
                        state = adaptive_runner.continue_from_shallow_checkpoint(
                            paired_fixed_shallow_state,
                            adaptive_retry_enabled=adaptive_retry_enabled,
                            adaptive_checkpoint_enabled=adaptive_checkpoint_enabled,
                            policy_stats=policy_stats,
                        )
                        continuation_runtime = time.perf_counter() - continuation_start
                        state = add_direct_precheck_overhead(
                            state,
                            direct_state=paired_fixed_direct_state,
                            shared_refinement=shared_refinement,
                            gate_decision=gate_decision,
                        )
                        runtime = paired_fixed_direct_runtime + paired_fixed_shallow_runtime + continuation_runtime
                policy_stats_by_strategy[strategy] = dict(state.get("policy_stats", {}) or {})

            state = _apply_budget_posthoc(
                state=state,
                runtime_seconds=runtime,
                max_tool_calls=max_tool_calls,
                max_token_proxy=max_token_proxy,
                max_total_tokens=max_total_tokens,
                max_runtime_seconds=max_runtime_seconds,
            )
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
    _attach_relative_metrics_vs_shallow(run_rows)
    aggregate = _aggregate_metrics([row["metrics"] for row in run_rows])
    recommendation = _build_recommendation(aggregate)
    active_policy_stats = {
        strategy: policy_stats_by_strategy.get(strategy, {})
        for strategy in active_adaptive_strategies
    }
    policy_stats_summary = _policy_stats_summary(active_policy_stats)
    policy_calibration_table = _policy_calibration_rows(
        active_policy_stats,
        policy_exploration=utility_weights["ucb_exploration"],
    )

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
            "adaptive_extension_source": "paired_validated_direct_shallow_recursive_cascade",
            "adaptive_policy": "bayes",
            "policy_exploration": utility_weights["ucb_exploration"],
            "policy_ucb_enabled": utility_weights["ucb_exploration"] > 0.0,
            "policy_memory_scope": "comparison_runs",
            "policy_calibration_row_count": len(policy_calibration_table),
            "policy_memory_initialized": any(
                bool(initial_policy_stats_by_strategy.get(strategy))
                for strategy in active_adaptive_strategies
            ),
            "generated_at_unix": int(time.time()),
        },
        "runs": run_rows,
        "aggregate": aggregate,
        "recommendation": recommendation,
        "states": states,
        "policy_stats_by_strategy": active_policy_stats,
        "policy_stats_summary": policy_stats_summary,
        "policy_calibration_table": policy_calibration_table,
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
        for key, value in metrics.items():
            flat[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
        run_flat_rows.append(flat)

    run_fieldnames = sorted({k for row in run_flat_rows for k in row.keys()})
    _write_csv(run_path, run_flat_rows, run_fieldnames)

    aggregate = report.get("aggregate", {})
    agg_flat_rows: List[Dict[str, Any]] = []
    for strategy, stats in aggregate.items():
        flat = {"strategy": strategy}
        for key, value in stats.items():
            flat[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
        agg_flat_rows.append(flat)

    agg_fieldnames = sorted({k for row in agg_flat_rows for k in row.keys()})
    _write_csv(agg_path, agg_flat_rows, agg_fieldnames)

    run_compact_path = run_path.with_name(f"{run_path.stem}_compact{run_path.suffix}")
    agg_compact_path = agg_path.with_name(f"{agg_path.stem}_compact{agg_path.suffix}")
    calibration_path = run_path.with_name(f"{run_path.stem}_policy_calibration{run_path.suffix}")

    run_compact_fieldnames = [c for c in PAPER_COLUMNS if c in run_fieldnames]
    run_compact_rows = [{c: row.get(c, "") for c in run_compact_fieldnames} for row in run_flat_rows]
    _write_csv(run_compact_path, run_compact_rows, run_compact_fieldnames)

    agg_compact_columns = ["strategy", "runs"] + [f"{field}_mean" for field in CORE_NUMERIC_FIELDS] + [
        "reliability_score_std",
        "actual_total_tokens_std",
        "reliability_per_1k_actual_token_std",
        "initial_mode_distribution",
        "realized_mode_distribution",
        "mode_distribution",
        "retry_type_distribution",
    ]
    agg_compact_fieldnames = [c for c in agg_compact_columns if c in agg_fieldnames]
    agg_compact_rows = [{c: row.get(c, "") for c in agg_compact_fieldnames} for row in agg_flat_rows]
    _write_csv(agg_compact_path, agg_compact_rows, agg_compact_fieldnames)

    calibration_rows = [
        {c: row.get(c, "") for c in POLICY_CALIBRATION_COLUMNS}
        for row in report.get("policy_calibration_table", [])
        if isinstance(row, dict)
    ]
    _write_csv(calibration_path, calibration_rows, POLICY_CALIBRATION_COLUMNS)

    summary = {
        "metadata": report.get("metadata", {}),
        "recommendation": report.get("recommendation"),
        "aggregate": report.get("aggregate", {}),
        "policy_stats_summary": report.get("policy_stats_summary", {}),
        "policy_calibration_table": report.get("policy_calibration_table", []),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "run_csv": str(run_path),
        "aggregate_csv": str(agg_path),
        "run_csv_compact": str(run_compact_path),
        "aggregate_csv_compact": str(agg_compact_path),
        "policy_calibration_csv": str(calibration_path),
        "summary_json": str(summary_path),
    }


def save_paper_mode_exports(
    report: Dict[str, Any],
    paper_json_path: str,
    paper_csv_path: str,
) -> Dict[str, str]:
    """Save a minimal paper-ready JSON and CSV view."""
    json_path = Path(paper_json_path)
    csv_path = Path(paper_csv_path)
    calibration_csv_path = csv_path.with_name(f"{csv_path.stem}_policy_calibration{csv_path.suffix}")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    paper_run_rows: List[Dict[str, Any]] = []
    for row in report.get("runs", []):
        metrics = row.get("metrics", {})
        flat = {
            "idea_index": row.get("idea_index", 0),
            "idea_id": row.get("idea_id", ""),
            "difficulty": row.get("difficulty", ""),
            "domain": row.get("domain", ""),
            "idea": row.get("idea", report.get("metadata", {}).get("idea", "")),
            "run_index": row.get("run_index"),
            "strategy": row.get("strategy"),
        }
        for col in PAPER_COLUMNS:
            if col not in flat:
                flat[col] = metrics.get(col, "")
        paper_run_rows.append(flat)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_COLUMNS)
        writer.writeheader()
        for row in paper_run_rows:
            writer.writerow(row)

    calibration_rows = [
        {c: row.get(c, "") for c in POLICY_CALIBRATION_COLUMNS}
        for row in report.get("policy_calibration_table", [])
        if isinstance(row, dict)
    ]
    with calibration_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=POLICY_CALIBRATION_COLUMNS)
        writer.writeheader()
        for row in calibration_rows:
            writer.writerow(row)

    agg = report.get("aggregate", {})
    compact_aggregate: Dict[str, Dict[str, Any]] = {}
    compact_fields = [
        "runs",
        "reliability_score_mean",
        "reliability_score_std",
        "claims_total_mean",
        "validation_claim_coverage_mean",
        "low_claim_count_flag_mean",
        "claim_count_penalty_mean",
        "judge_agreement_mean",
        "cross_model_judging_mean",
        "secondary_judge_fallback_mean",
        "supported_ratio_mean",
        "weak_or_better_ratio_mean",
        "runtime_seconds_mean",
        "actual_total_tokens_mean",
        "token_proxy_mean",
        "reliability_per_1k_actual_token_mean",
        "decomposition_depth_realized_mean",
        "budget_hit_mean",
        "controller_escalated_mean",
        "controller_calibration_error_abs_mean",
        "reliability_gain_vs_single_mean",
        "reliability_gain_vs_fixed_shallow_mean",
        "token_delta_vs_fixed_shallow_mean",
        "efficiency_gain_vs_fixed_shallow_mean",
        "adaptive_quality_win_vs_fixed_shallow_mean",
        "adaptive_efficiency_win_vs_fixed_shallow_mean",
        "retry_allowed_count_mean",
        "retry_blocked_count_mean",
        "claim_micro_repair_allowed_count_mean",
        "fixed_recursive_market_retry_count_mean",
        "shallow_escalation_count_mean",
        "recursive_retry_count_mean",
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
        "micro_repair_coverage_addition_count_mean",
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
        "retry_type_distribution",
    ]
    for strategy, stats in agg.items():
        compact_aggregate[strategy] = {field: stats.get(field) for field in compact_fields if field in stats}

    paper_json = {
        "metadata": report.get("metadata", {}),
        "evaluation_primary_metrics": [c for c in PAPER_COLUMNS if c not in {"idea", "idea_id", "difficulty", "domain"}],
        "aggregate": compact_aggregate,
        "policy_stats_summary": report.get("policy_stats_summary", {}),
        "policy_calibration_table": report.get("policy_calibration_table", []),
        "recommendation": report.get("recommendation"),
    }
    json_path.write_text(json.dumps(paper_json, indent=2), encoding="utf-8")
    return {
        "paper_json": str(json_path),
        "paper_csv": str(csv_path),
        "policy_calibration_csv": str(calibration_csv_path),
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
    policy_exploration: float = 0.15,
    policy_memory_mode: str = "carry_across_ideas",
) -> Dict[str, Any]:
    idea_records = _normalize_idea_records(ideas)
    if not idea_records:
        raise ValueError("No ideas provided for batch comparison.")

    normalized_strategies = strategies or list(DEFAULT_STRATEGIES)
    normalized_strategy_names = [s.strip().lower() for s in normalized_strategies if s and s.strip()]
    adaptive_strategy_names = [s for s in normalized_strategy_names if s in ADAPTIVE_STRATEGIES]
    normalized_memory_mode = str(policy_memory_mode or "carry_across_ideas").strip().lower()
    if normalized_memory_mode not in {"carry_across_ideas", "reset_per_idea"}:
        raise ValueError("policy_memory_mode must be one of: carry_across_ideas, reset_per_idea")
    carry_policy_memory = normalized_memory_mode == "carry_across_ideas"
    per_idea_reports: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    all_states: Dict[str, Dict[str, Any]] = {}
    policy_stats_by_strategy: Dict[str, Dict[str, Any]] = {s: {} for s in ADAPTIVE_STRATEGIES}

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
            strategies=normalized_strategies,
            validation_threshold=validation_threshold,
            max_validation_retries=max_validation_retries,
            max_tool_calls=max_tool_calls,
            max_token_proxy=max_token_proxy,
            max_total_tokens=max_total_tokens,
            max_runtime_seconds=max_runtime_seconds,
            generate_ppt=generate_ppt,
            thread_prefix=f"{thread_prefix}-idea-{idx}",
            output_dir=output_dir,
            initial_policy_stats_by_strategy=policy_stats_by_strategy if carry_policy_memory else {},
            policy_exploration=policy_exploration,
        )
        updated_policy_stats = report.get("policy_stats_by_strategy", {})
        if carry_policy_memory and isinstance(updated_policy_stats, dict):
            for strategy in adaptive_strategy_names:
                if isinstance(updated_policy_stats.get(strategy), dict):
                    policy_stats_by_strategy[strategy] = _clone_jsonable(updated_policy_stats[strategy])
        idea_policy_stats_for_summary = (
            policy_stats_by_strategy
            if carry_policy_memory
            else updated_policy_stats if isinstance(updated_policy_stats, dict) else {}
        )
        idea_policy_stats_active = {
            strategy: idea_policy_stats_for_summary.get(strategy, {})
            for strategy in adaptive_strategy_names
        }
        per_idea_reports.append(
            {
                "idea_index": idx,
                "idea_id": idea_record["idea_id"],
                "difficulty": idea_record["difficulty"],
                "domain": idea_record["domain"],
                "idea": idea,
                "aggregate": report.get("aggregate", {}),
                "recommendation": report.get("recommendation"),
                "policy_stats_summary_after_idea": _policy_stats_summary(idea_policy_stats_active),
                "policy_calibration_after_idea": _policy_calibration_rows(
                    idea_policy_stats_active,
                    policy_exploration=policy_exploration,
                ),
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
    _attach_relative_metrics_vs_shallow(all_rows)
    aggregate = _aggregate_metrics([row["metrics"] for row in all_rows])
    recommendation = _build_recommendation(aggregate)
    final_policy_stats = {
        strategy: policy_stats_by_strategy.get(strategy, {})
        for strategy in adaptive_strategy_names
    } if carry_policy_memory else {}
    final_policy_stats_summary = _policy_stats_summary(final_policy_stats)
    final_policy_calibration_table = _policy_calibration_rows(
        final_policy_stats,
        policy_exploration=policy_exploration,
    )

    return {
        "metadata": {
            "ideas_count": len(idea_records),
            "ideas": [record["idea"] for record in idea_records],
            "idea_records": idea_records,
            "model": model,
            "secondary_judge_model": secondary_judge_model or model,
            "temperature": temperature,
            "seed": seed,
            "strategies": normalized_strategies,
            "compare_runs": compare_runs,
            "validation_threshold": validation_threshold,
            "max_validation_retries": max_validation_retries,
            "max_tool_calls": max_tool_calls,
            "max_token_proxy": max_token_proxy,
            "max_total_tokens": max_total_tokens,
            "max_runtime_seconds": max_runtime_seconds,
            "adaptive_extension_source": "paired_validated_direct_shallow_recursive_cascade",
            "adaptive_policy": "bayes",
            "policy_exploration": max(0.0, float(policy_exploration)),
            "policy_ucb_enabled": max(0.0, float(policy_exploration)) > 0.0,
            "policy_memory_scope": normalized_memory_mode,
            "policy_memory_strategies": adaptive_strategy_names,
            "policy_calibration_row_count": len(final_policy_calibration_table),
            "policy_stats_summary": final_policy_stats_summary,
            "generated_at_unix": int(time.time()),
        },
        "ideas": per_idea_reports,
        "runs": all_rows,
        "aggregate": aggregate,
        "recommendation": recommendation,
        "states": all_states,
        "policy_stats_by_strategy": final_policy_stats,
        "policy_stats_summary": final_policy_stats_summary,
        "policy_calibration_table": final_policy_calibration_table,
    }
















