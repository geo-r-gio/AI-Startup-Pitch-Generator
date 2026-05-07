from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from startup_pitch_refinery.agents import (
    AdaptiveControllerAgent,
    BayesianRetryPolicy,
    BusinessModelAgent,
    ClaimRepairAgent,
    DirectStrategyAgent,
    IdeaRefinementAgent,
    MarketResearchAgent,
    PitchDeckGeneratorAgent,
    SourceValidatorAgent,
    SupervisorAgent,
    build_claim_repair_plan,
)
from startup_pitch_refinery.state import PitchState


class StartupPitchRefinery:
    """LangGraph workflow for validated adaptive decomposition.

    Adaptive execution is a validated cascade: use direct only when the
    controller predicts that low decomposition can satisfy validation, escalate
    failed direct attempts to shallow when adaptive repair is enabled, and use
    recursive claim repair only for source-bound factual gaps.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        secondary_judge_model: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        strict_tools: bool = True,
        enable_trends: bool = True,
        generate_pitch: bool = True,
        controller_policy: str = "fixed",
        output_dir: str = "output",
        adaptive_policy: str = "bayes",
        policy_thresholds: Optional[Dict[str, float]] = None,
        utility_weights: Optional[Dict[str, float]] = None,
    ):
        llm = ChatOpenAI(model=model, temperature=temperature, seed=seed)
        secondary_judge_llm = (
            ChatOpenAI(model=secondary_judge_model, temperature=temperature, seed=seed + 101)
            if secondary_judge_model
            else None
        )

        self.generate_pitch = generate_pitch
        self.controller_policy = controller_policy.strip().lower()
        if self.controller_policy not in {"fixed", "adaptive"}:
            raise ValueError(
                f"Unsupported controller_policy `{controller_policy}`. Allowed: fixed, adaptive."
            )
        self.adaptive_policy = adaptive_policy.strip().lower() or "bayes"
        self.policy_thresholds = dict(policy_thresholds or {})
        self.utility_weights = dict(utility_weights or {})

        self.supervisor = SupervisorAgent()
        self.idea_agent = IdeaRefinementAgent(llm)
        self.controller_agent = (
            AdaptiveControllerAgent(llm) if self.controller_policy == "adaptive" else None
        )
        self.market_agent = MarketResearchAgent(
            llm,
            strict_tools=strict_tools,
            enable_trends=enable_trends,
        )
        self.claim_repair_agent = ClaimRepairAgent(llm, strict_tools=strict_tools)
        self.retry_policy = BayesianRetryPolicy(
            thresholds=self.policy_thresholds,
            utility_weights=self.utility_weights,
        )
        self.validator_agent = SourceValidatorAgent(
            llm,
            secondary_judge_llm=secondary_judge_llm,
        )
        self.direct_agent = (
            DirectStrategyAgent(
                llm,
                strict_tools=strict_tools,
                enable_trends=enable_trends,
                secondary_judge_llm=secondary_judge_llm,
            )
            if self.controller_policy == "adaptive"
            else None
        )
        self.business_agent = BusinessModelAgent(llm, strict_tools=strict_tools)
        self.pitch_agent = (
            PitchDeckGeneratorAgent(llm, output_dir=output_dir)
            if self.generate_pitch
            else None
        )
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    @staticmethod
    def _sget(state: PitchState | Dict[str, Any], key: str, default: Any = None) -> Any:
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)

    @staticmethod
    def _merge_token_usage(old: Dict[str, int], new: Dict[str, int]) -> Dict[str, int]:
        return {
            "prompt_tokens": int(old.get("prompt_tokens", 0) or 0) + int(new.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(old.get("completion_tokens", 0) or 0) + int(new.get("completion_tokens", 0) or 0),
            "total_tokens": int(old.get("total_tokens", 0) or 0) + int(new.get("total_tokens", 0) or 0),
        }

    @staticmethod
    def _estimate_tokens_proxy(state: Dict[str, Any]) -> int:
        chunks = [
            str(state.get("idea", "")),
            str(state.get("refined_idea", "")),
            str(state.get("market_analysis", "")),
            str(state.get("business_model", "")),
            str(state.get("validated_market_analysis", "")),
            json.dumps(state.get("pitch_content", {}), ensure_ascii=True),
            json.dumps(state.get("trend_signals", {}), ensure_ascii=True),
            json.dumps(state.get("validation_report", {}), ensure_ascii=True),
            json.dumps(state.get("repair_plan", []), ensure_ascii=True),
            json.dumps(state.get("repair_patches", []), ensure_ascii=True),
            json.dumps(state.get("micro_validation", {}), ensure_ascii=True),
            json.dumps(state.get("retrieval_diagnostics", []), ensure_ascii=True),
            json.dumps(state.get("retry_policy_decision", {}), ensure_ascii=True),
            json.dumps(state.get("selected_action", {}), ensure_ascii=True),
            json.dumps(state.get("policy_stats", {}), ensure_ascii=True),
            json.dumps(state.get("repair_validation_history", []), ensure_ascii=True),
        ]
        total_chars = sum(len(c) for c in chunks)
        return max(1, total_chars // 4)

    def _compute_budget_updates(self, merged: Dict[str, Any]) -> Dict[str, Any]:
        max_tool_calls = self._sget(merged, "max_tool_calls")
        max_token_proxy = self._sget(merged, "max_token_proxy")
        max_total_tokens = self._sget(merged, "max_total_tokens")
        max_runtime_seconds = self._sget(merged, "max_runtime_seconds")

        tool_calls_current = len(self._sget(merged, "tool_audit", []))
        token_proxy_current = self._estimate_tokens_proxy(merged)
        token_usage = self._sget(merged, "token_usage", {}) or {}
        prompt_tokens_current = int(token_usage.get("prompt_tokens", 0) or 0)
        completion_tokens_current = int(token_usage.get("completion_tokens", 0) or 0)
        total_tokens_current = int(token_usage.get("total_tokens", 0) or 0)
        started_at = self._sget(merged, "runtime_started_at")
        runtime_elapsed_seconds = (
            max(0.0, time.time() - float(started_at))
            if isinstance(started_at, (int, float)) and started_at > 0
            else 0.0
        )

        reasons = list(self._sget(merged, "budget_hit_reasons", []) or [])
        budget_hit = bool(self._sget(merged, "budget_hit", False))
        if isinstance(max_tool_calls, int) and max_tool_calls >= 0 and tool_calls_current > max_tool_calls:
            budget_hit = True
            reasons.append(f"max_tool_calls_exceeded:{tool_calls_current}>{max_tool_calls}")
        if isinstance(max_token_proxy, int) and max_token_proxy >= 0 and token_proxy_current > max_token_proxy:
            budget_hit = True
            reasons.append(f"max_token_proxy_exceeded:{token_proxy_current}>{max_token_proxy}")
        if isinstance(max_total_tokens, int) and max_total_tokens >= 0 and total_tokens_current > max_total_tokens:
            budget_hit = True
            reasons.append(f"max_total_tokens_exceeded:{total_tokens_current}>{max_total_tokens}")
        if isinstance(max_runtime_seconds, (int, float)) and max_runtime_seconds >= 0 and runtime_elapsed_seconds > float(max_runtime_seconds):
            budget_hit = True
            reasons.append(
                f"max_runtime_seconds_exceeded:{runtime_elapsed_seconds:.3f}>{float(max_runtime_seconds):.3f}"
            )

        dedup_reasons: List[str] = []
        seen = set()
        for reason in reasons:
            if reason not in seen:
                dedup_reasons.append(reason)
                seen.add(reason)

        return {
            "runtime_elapsed_seconds": round(runtime_elapsed_seconds, 3),
            "tool_calls_current": tool_calls_current,
            "token_proxy_current": token_proxy_current,
            "prompt_tokens_current": prompt_tokens_current,
            "completion_tokens_current": completion_tokens_current,
            "total_tokens_current": total_tokens_current,
            "budget_hit": budget_hit,
            "budget_hit_reasons": dedup_reasons,
            "budget_remaining": {
                "tool_calls": None if not isinstance(max_tool_calls, int) or max_tool_calls < 0 else max_tool_calls - tool_calls_current,
                "token_proxy": None if not isinstance(max_token_proxy, int) or max_token_proxy < 0 else max_token_proxy - token_proxy_current,
                "total_tokens": None if not isinstance(max_total_tokens, int) or max_total_tokens < 0 else max_total_tokens - total_tokens_current,
                "runtime_seconds": None if not isinstance(max_runtime_seconds, (int, float)) or max_runtime_seconds < 0 else float(max_runtime_seconds) - runtime_elapsed_seconds,
            },
        }

    def _run_node_with_budget(self, state: PitchState | Dict[str, Any], runner, node_name: str) -> Dict[str, Any]:
        base_state = state if isinstance(state, dict) else state.model_dump()
        precheck = self._compute_budget_updates(base_state)
        if precheck["budget_hit"]:
            return {**precheck, "controller_rationale": self._sget(base_state, "controller_rationale")}
        enriched_state = {**base_state, **precheck}
        updates = runner(enriched_state)
        merged = {**enriched_state, **updates}
        postcheck = self._compute_budget_updates(merged)
        if postcheck["budget_hit"]:
            audit = list(self._sget(merged, "tool_audit", []) or [])
            audit.append(
                {
                    "agent": "budget_guard",
                    "tool": "runtime_budget_check",
                    "status": "halted",
                    "node": node_name,
                    "reasons": postcheck["budget_hit_reasons"],
                }
            )
            postcheck["tool_audit"] = audit
        return {**updates, **postcheck}

    @staticmethod
    def _validation_score(validation: Dict[str, Any]) -> int:
        if not isinstance(validation, dict):
            return 0
        try:
            return int(validation.get("reliability_score", 0) or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _validation_supported_ratio(validation: Dict[str, Any]) -> float:
        claims = validation.get("claims", []) if isinstance(validation, dict) else []
        if not claims:
            return 0.0
        supported = sum(1 for c in claims if isinstance(c, dict) and str(c.get("verdict", "")).lower() == "supported")
        return supported / max(1, len(claims))

    @staticmethod
    def _material_failing_count(validation: Dict[str, Any]) -> int:
        if not isinstance(validation, dict):
            return 0
        count = 0
        claims = validation.get("claim_units") or validation.get("claims", []) or []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            verdict = str(claim.get("verdict") or claim.get("primary_verdict") or claim.get("status") or "").lower()
            materiality = int(claim.get("materiality", 1) or 1)
            if verdict in {"weakly_supported", "unsupported", "needs_review"} and materiality >= 4:
                count += 1
        return count

    @staticmethod
    def _low_claim_count(validation: Dict[str, Any]) -> bool:
        if not isinstance(validation, dict):
            return False
        agg = validation.get("judge_scores", {}).get("aggregated", {}) if isinstance(validation.get("judge_scores", {}), dict) else {}
        return bool(agg.get("low_claim_count_flag", False))

    @staticmethod
    def _coverage_enhancement_target(
        base_state: Dict[str, Any],
        validation: Dict[str, Any],
        *,
        missing_claim_count: int = 0,
        claims_total: int = 0,
        minimum_claims_required: int = 0,
    ) -> Dict[str, Any]:
        """Create one bounded target for missing source-backed claim coverage."""
        idea = str(base_state.get("refined_idea") or base_state.get("idea") or "").strip()
        text = idea.lower()
        evidence_gaps = str(validation.get("evidence_gaps", "") or "").strip()
        regulated = any(
            marker in text
            for marker in [
                "hipaa",
                "healthcare",
                "clinic",
                "clinical",
                "patient",
                "ehr",
                "compliance",
                "regulatory",
                "authorization",
            ]
        )
        ecommerce = any(marker in text for marker in ["shopify", "cart", "ecommerce", "commerce", "conversion"])
        if regulated:
            category = "regulatory_or_compliance"
            claim = (
                "Coverage enhancement target: add source-backed regulatory or workflow-specific "
                "market detail for HIPAA-compliant prior-authorization automation in specialty clinics."
            )
            queries = [
                "HIPAA prior authorization automation specialty clinics EHR integration market evidence",
                "AI prior authorization healthcare workflow automation compliance specialty clinics evidence",
            ]
        elif ecommerce:
            category = "market_size_or_numeric"
            claim = (
                "Coverage enhancement target: add source-backed numeric market or conversion benchmark "
                "for Shopify abandoned-cart recovery and AI personalization."
            )
            queries = [
                "Shopify abandoned cart recovery conversion benchmark AI personalization evidence",
                "ecommerce abandoned cart rate personalized discount campaign market evidence",
            ]
        else:
            category = "market_size_or_numeric"
            claim = (
                "Coverage enhancement target: add source-backed market sizing, adoption, or usage "
                "benchmark that makes the startup market analysis more specific."
            )
            queries = [
                f"{idea} market size adoption benchmark evidence",
                f"{idea} user adoption market evidence source",
            ]
        coverage_note = ""
        if missing_claim_count > 0:
            required = int(minimum_claims_required or (claims_total + missing_claim_count))
            coverage_note = (
                f" The validator found {claims_total} source-checkable claims and requires "
                f"{required}, so {missing_claim_count} source-backed "
                "claim(s) are missing."
            )
        rationale = (
            "The validation report has incomplete atomic-claim coverage, so this target repairs "
            "rubric-level coverage and specificity rather than only rewriting an existing failed claim."
            f"{coverage_note}"
        )
        if evidence_gaps:
            rationale = f"{rationale} Validator evidence gaps: {evidence_gaps[:240]}"
        return {
            "claim_id": "coverage_gap_1",
            "claim": claim,
            "claim_text": claim,
            "verdict": "needs_review",
            "confidence": 0.5,
            "rationale": rationale,
            "supporting_sources": [],
            "category": category,
            "materiality": 5,
            "failure_type": "coverage_gap",
            "action": "coverage_addition",
            "coverage_enhancement": True,
            "missing_claim_count": max(0, int(missing_claim_count or 0)),
            "claims_total_before_repair": max(0, int(claims_total or 0)),
            "target_add_count": min(max(1, int(missing_claim_count or 1)), 2),
            "candidate_queries": queries,
        }

    @staticmethod
    def _claim_coverage_counts(validation: Dict[str, Any]) -> Dict[str, int]:
        if not isinstance(validation, dict):
            return {"claims_total": 0, "minimum_claims_required": 0, "missing_claim_count": 0}
        claims = validation.get("claim_units", []) or validation.get("claims", []) or []
        claims_total = sum(1 for claim in claims if isinstance(claim, dict))
        judge_scores = validation.get("judge_scores", {})
        aggregated = judge_scores.get("aggregated", {}) if isinstance(judge_scores, dict) else {}
        minimum_claims = int(aggregated.get("minimum_claims_required", 0) or 0)
        if minimum_claims <= 0:
            minimum_claims = 5 if claims_total > 0 else 0
        reported_missing = int(aggregated.get("missing_claim_count", 0) or 0)
        computed_missing = max(0, minimum_claims - claims_total)
        missing = max(0, reported_missing, computed_missing)
        return {
            "claims_total": claims_total,
            "minimum_claims_required": minimum_claims,
            "missing_claim_count": missing,
        }

    def _validation_quality_fields(self, validation: Dict[str, Any]) -> Dict[str, Any]:
        coverage = self._claim_coverage_counts(validation)
        claims_total = int(coverage.get("claims_total", 0) or 0)
        minimum_claims = int(coverage.get("minimum_claims_required", 0) or 0)
        coverage_ratio = min(1.0, claims_total / max(1, minimum_claims)) if minimum_claims > 0 else 0.0
        return {
            "claims_total": claims_total,
            "minimum_claims_required": minimum_claims,
            "missing_claim_count": int(coverage.get("missing_claim_count", 0) or 0),
            "claim_coverage_ratio": round(coverage_ratio, 4),
            "supported_ratio": round(self._validation_supported_ratio(validation), 4),
            "material_failing_claim_count": self._material_failing_count(validation),
            "low_claim_count_flag": self._low_claim_count(validation),
        }

    def _quality_safe_repair_checkpoint(
        self,
        candidate: Dict[str, Any],
        baseline: Dict[str, Any],
    ) -> tuple[bool, List[str], Dict[str, Any]]:
        """Reject score gains that are produced by collapsing validation coverage."""
        candidate_quality = self._validation_quality_fields(candidate.get("validation_report", {}) or {})
        baseline_quality = self._validation_quality_fields(baseline.get("validation_report", {}) or {})

        for key in (
            "claims_total",
            "minimum_claims_required",
            "missing_claim_count",
            "claim_coverage_ratio",
            "supported_ratio",
            "material_failing_claim_count",
            "low_claim_count_flag",
        ):
            if key in candidate:
                candidate_quality[key] = candidate.get(key)
            if key in baseline:
                baseline_quality[key] = baseline.get(key)

        candidate_claims = int(candidate_quality.get("claims_total", 0) or 0)
        candidate_minimum = int(candidate_quality.get("minimum_claims_required", 0) or 0)
        candidate_missing = int(candidate_quality.get("missing_claim_count", 0) or 0)
        baseline_missing = int(baseline_quality.get("missing_claim_count", 0) or 0)
        candidate_coverage = float(candidate_quality.get("claim_coverage_ratio", 0.0) or 0.0)
        baseline_coverage = float(baseline_quality.get("claim_coverage_ratio", 0.0) or 0.0)
        candidate_material = int(candidate_quality.get("material_failing_claim_count", 0) or 0)
        baseline_material = int(baseline_quality.get("material_failing_claim_count", 0) or 0)
        candidate_low_claim = bool(candidate_quality.get("low_claim_count_flag", False))

        reasons: List[str] = []
        if candidate_minimum > 0 and candidate_claims < candidate_minimum:
            reasons.append("repaired_claim_count_below_required")
        if candidate_missing > baseline_missing:
            reasons.append("missing_claim_count_increased")
        if candidate_coverage + 1e-9 < baseline_coverage:
            reasons.append("claim_coverage_decreased")
        if candidate_low_claim:
            reasons.append("low_claim_count_after_repair")
        if candidate_material > baseline_material:
            reasons.append("material_failing_claim_count_increased")

        summary = {
            "candidate_claims_total": candidate_claims,
            "candidate_minimum_claims_required": candidate_minimum,
            "candidate_missing_claim_count": candidate_missing,
            "candidate_claim_coverage_ratio": round(candidate_coverage, 4),
            "candidate_material_failing_claim_count": candidate_material,
            "candidate_low_claim_count_flag": candidate_low_claim,
            "baseline_claims_total": int(baseline_quality.get("claims_total", 0) or 0),
            "baseline_minimum_claims_required": int(baseline_quality.get("minimum_claims_required", 0) or 0),
            "baseline_missing_claim_count": baseline_missing,
            "baseline_claim_coverage_ratio": round(baseline_coverage, 4),
            "baseline_material_failing_claim_count": baseline_material,
            "quality_block_reasons": reasons,
        }
        return not reasons, reasons, summary

    def _build_validation_snapshot(self, state: Dict[str, Any]) -> Dict[str, Any]:
        validation = self._sget(state, "validation_report", {}) or {}
        agreement = validation.get("agreement_stats", {}).get("overall_agreement", 0.0) if isinstance(validation, dict) else 0.0
        token_usage = self._sget(state, "token_usage", {}) or {}
        return {
            "checkpoint_index": len(self._sget(state, "validation_snapshots", []) or []),
            "retry_count": int(self._sget(state, "retry_count", 0) or 0),
            "repair_rounds": int(self._sget(state, "repair_rounds", 0) or 0),
            "controller_mode": self._sget(state, "controller_mode"),
            "reliability_score": self._validation_score(validation),
            "judge_agreement": float(agreement or 0.0),
            "supported_ratio": round(self._validation_supported_ratio(validation), 4),
            "material_failing_claim_count": self._material_failing_count(validation),
            "low_claim_count_flag": self._low_claim_count(validation),
            **self._validation_quality_fields(validation),
            "total_tokens_at_checkpoint": int(token_usage.get("total_tokens", 0) or 0),
            "market_analysis": self._sget(state, "market_analysis"),
            "validated_market_analysis": self._sget(state, "validated_market_analysis"),
            "validation_report": validation,
            "market_sources": list(self._sget(state, "market_sources", []) or []),
            "market_evidence": list(self._sget(state, "market_evidence", []) or []),
            "trend_signals": dict(self._sget(state, "trend_signals", {}) or {}),
        }

    def _ensure_baseline_checkpoint(self, state: Dict[str, Any]) -> Dict[str, Any]:
        baseline = self._sget(state, "baseline_checkpoint", {}) or {}
        if isinstance(baseline, dict) and baseline:
            return baseline
        snapshots = [s for s in (self._sget(state, "validation_snapshots", []) or []) if isinstance(s, dict)]
        return snapshots[0] if snapshots else self._build_validation_snapshot(state)

    def _checkpoint_utility(self, snapshot: Dict[str, Any], baseline: Dict[str, Any], state: Dict[str, Any]) -> float:
        weights = {
            "eta_a": 0.02,
            "eta_s": 0.03,
            "lambda_crit": 4.0,
            "lambda_low": 6.0,
            **(self._sget(state, "utility_weights", {}) or {}),
            **self.utility_weights,
        }
        q = float(snapshot.get("reliability_score", 0.0) or 0.0)
        agreement = float(snapshot.get("judge_agreement", 0.0) or 0.0)
        support = float(snapshot.get("supported_ratio", 0.0) or 0.0)
        material_failures = int(snapshot.get("material_failing_claim_count", 0) or 0)
        low_claim = 1 if snapshot.get("low_claim_count_flag") else 0
        return (
            q
            + weights["eta_a"] * 100.0 * agreement
            + weights["eta_s"] * 100.0 * support
            - weights["lambda_crit"] * material_failures
            - weights["lambda_low"] * low_claim
        )

    def _checkpoint_uncertainty(self, snapshot: Dict[str, Any], state: Dict[str, Any]) -> float:
        weights = {"z_uncertainty": 1.0, **(self._sget(state, "utility_weights", {}) or {}), **self.utility_weights}
        agreement = float(snapshot.get("judge_agreement", 0.0) or 0.0)
        support = float(snapshot.get("supported_ratio", 0.0) or 0.0)
        low_claim = 1.0 if snapshot.get("low_claim_count_flag") else 0.0
        sigma = 8.0 * (1.0 - agreement) + 5.0 * (1.0 - support) + 4.0 * low_claim
        return float(weights.get("z_uncertainty", 1.0)) * sigma

    def _run_supervisor(self, state: PitchState):
        return self._run_node_with_budget(state, self.supervisor.run, "supervisor")

    def _run_idea(self, state: PitchState):
        if bool(self._sget(state, "shared_refinement_locked", False)) and self._sget(state, "refined_idea"):
            return {}
        return self._run_node_with_budget(state, self.idea_agent.run, "idea")

    def _run_controller(self, state: PitchState):
        if self.controller_agent is None:
            return {}
        return self._run_node_with_budget(state, self.controller_agent.run, "controller")

    def _run_market(self, state: PitchState):
        updates = self._run_node_with_budget(state, self.market_agent.run, "market")
        base_state = state if isinstance(state, dict) else state.model_dump()
        if (
            str(self._sget(base_state, "controller_mode", "") or "").strip().lower() == "direct"
            and bool(self._sget(base_state, "needs_revision", False))
            and self._sget(base_state, "validation_report")
        ):
            updates.update(
                {
                    "direct_precheck_escalated": True,
                    "controller_escalated": True,
                    "controller_escalation_reason": "direct_validation_failed_escalated_to_shallow",
                    "controller_mode_realized": "shallow",
                }
            )
        return updates

    def _run_direct(self, state: PitchState):
        if self.direct_agent is None:
            return {}
        return self._run_node_with_budget(state, self.direct_agent.run, "direct")

    def _run_validator(self, state: PitchState):
        base_state = state if isinstance(state, dict) else state.model_dump()
        updates = self._run_node_with_budget(base_state, self.validator_agent.run, "validator")
        updates = self._record_repair_outcome(base_state, updates, validation_mode="full_dual_judge")
        merged = {**base_state, **updates}
        snapshot = self._build_validation_snapshot(merged)
        snapshot["validation_mode"] = "full_dual_judge"
        snapshots = list(self._sget(base_state, "validation_snapshots", []) or [])
        snapshots.append(snapshot)
        updates["validation_snapshots"] = snapshots
        updates["checkpoint_history"] = snapshots
        if not self._sget(base_state, "baseline_checkpoint", {}) and not self._sget(base_state, "forced_controller_mode"):
            updates.setdefault("baseline_checkpoint", snapshot)
        return updates

    def _run_lightweight_repair_validator(self, state: PitchState):
        base_state = state if isinstance(state, dict) else state.model_dump()
        updates = self._run_node_with_budget(base_state, self.validator_agent.run_repair_only, "repair_validator")
        updates = self._record_repair_outcome(base_state, updates, validation_mode="lightweight_repair")
        merged = {**base_state, **updates}
        snapshot = self._build_validation_snapshot(merged)
        snapshot["validation_mode"] = "lightweight_repair"
        snapshots = list(self._sget(base_state, "validation_snapshots", []) or [])
        snapshots.append(snapshot)
        updates["validation_snapshots"] = snapshots
        updates["checkpoint_history"] = snapshots
        return updates

    def _record_repair_outcome(self, base_state: Dict[str, Any], updates: Dict[str, Any], validation_mode: str) -> Dict[str, Any]:
        selected_action = self._sget(base_state, "selected_action", {}) or {}
        if selected_action.get("action") != "repair_bundle":
            return updates
        validation = updates.get("validation_report", {}) or {}
        previous_score = int(selected_action.get("score_before", self._validation_score(self._sget(base_state, "validation_report", {}) or {})) or 0)
        current_score = self._validation_score(validation)
        micro = updates.get("micro_validation", {}) or validation.get("repair_validation", {}) if isinstance(validation, dict) else {}
        patch_accepted = bool(micro.get("accepted", False) or micro.get("can_accept_patch", False))
        if validation_mode == "full_dual_judge":
            patch_accepted = patch_accepted or current_score > previous_score
        gain = current_score - previous_score
        old_tokens = int(selected_action.get("tokens_before", 0) or 0)
        new_tokens = int((updates.get("token_usage") or self._sget(base_state, "token_usage", {}) or {}).get("total_tokens", 0) or 0)
        observed_tokens = max(0, new_tokens - old_tokens) or int(selected_action.get("expected_tokens", 0) or 0)
        base_audits = list(self._sget(base_state, "tool_audit", []) or [])
        base_audit_count = len(base_audits)
        returned_audits = list(updates.get("tool_audit", []) or [])
        new_audits = returned_audits[base_audit_count:] if len(returned_audits) > base_audit_count else returned_audits
        repair_generation_tokens = int(
            sum(
                int(audit.get("total_tokens", 0) or 0)
                for audit in new_audits
                if isinstance(audit, dict) and audit.get("tool") == "claim_micro_repair"
            )
        )
        if repair_generation_tokens <= 0:
            for audit in reversed(base_audits):
                if isinstance(audit, dict) and audit.get("tool") == "claim_micro_repair":
                    repair_generation_tokens = int(audit.get("total_tokens", 0) or 0)
                    break
        repair_validation_tokens = int(
            sum(
                int(audit.get("total_tokens", 0) or 0)
                for audit in new_audits
                if isinstance(audit, dict)
                and audit.get("tool") in {"llm_claim_verifier_dual_judge", "llm_claim_repair_validator"}
            )
        )
        bucket = str(selected_action.get("bucket_key") or selected_action.get("action_type") or "unknown")
        previous_validation = self._sget(base_state, "validation_report", {}) or {}
        previous_token_usage = self._sget(base_state, "token_usage", {}) or {}
        previous_tokens = old_tokens or int(previous_token_usage.get("total_tokens", 0) or 0)
        current_tokens = new_tokens or previous_tokens + observed_tokens
        previous_snapshot = {
            "reliability_score": previous_score,
            "judge_agreement": float((previous_validation.get("agreement_stats", {}) or {}).get("overall_agreement", 0.0) or 0.0)
            if isinstance(previous_validation, dict)
            else 0.0,
            **self._validation_quality_fields(previous_validation),
            "total_tokens_at_checkpoint": previous_tokens,
            "validation_report": previous_validation,
        }
        current_snapshot = {
            "reliability_score": current_score,
            "judge_agreement": float((validation.get("agreement_stats", {}) or {}).get("overall_agreement", 0.0) or 0.0)
            if isinstance(validation, dict)
            else 0.0,
            **self._validation_quality_fields(validation),
            "total_tokens_at_checkpoint": current_tokens,
            "validation_report": validation,
        }
        terminal_utility_before = self._checkpoint_utility(previous_snapshot, previous_snapshot, base_state)
        terminal_utility_after = self._checkpoint_utility(current_snapshot, previous_snapshot, {**base_state, **updates})
        terminal_utility_delta = terminal_utility_after - terminal_utility_before
        observed_score_roi_per_1k = (gain * 1000.0 / observed_tokens) if observed_tokens > 0 else 0.0
        observed_terminal_utility_per_1k = (
            terminal_utility_delta * 1000.0 / observed_tokens
            if observed_tokens > 0
            else 0.0
        )
        repair_success_margin = 0.0
        quality_safe, quality_block_reasons, quality_summary = self._quality_safe_repair_checkpoint(
            current_snapshot,
            previous_snapshot,
        )
        policy_success = bool(current_score > previous_score and quality_safe)
        if policy_success:
            policy_outcome = "score_improved_quality_safe"
        elif current_score > previous_score:
            policy_outcome = "score_improved_but_quality_unsafe"
        elif patch_accepted:
            policy_outcome = "patch_accepted_score_not_improved"
        else:
            policy_outcome = "patch_rejected_score_not_improved"

        policy_stats = json.loads(json.dumps(self._sget(base_state, "policy_stats", {}) or {}))
        buckets = policy_stats.setdefault("repair_buckets", {})
        stat = buckets.setdefault(bucket, {})
        prior_count = int(stat.get("observations", 0) or 0)
        new_count = prior_count + 1
        stat["observations"] = new_count
        stat["accepted"] = int(stat.get("accepted", 0) or 0) + (1 if policy_success else 0)
        stat["rejected"] = int(stat.get("rejected", 0) or 0) + (0 if policy_success else 1)
        stat["patch_accepted"] = int(stat.get("patch_accepted", 0) or 0) + (1 if patch_accepted else 0)
        stat["patch_rejected"] = int(stat.get("patch_rejected", 0) or 0) + (0 if patch_accepted else 1)
        stat["mean_gain"] = round(((float(stat.get("mean_gain", 0.0) or 0.0) * prior_count) + gain) / new_count, 4)
        stat["mean_tokens"] = round(((float(stat.get("mean_tokens", 0.0) or 0.0) * prior_count) + observed_tokens) / new_count, 4)
        stat["mean_seconds"] = round(((float(stat.get("mean_seconds", 0.0) or 0.0) * prior_count) + float(selected_action.get("expected_seconds", 0.0) or 0.0)) / new_count, 4)
        stat["mean_observed_roi_per_1k"] = round(
            ((float(stat.get("mean_observed_roi_per_1k", 0.0) or 0.0) * prior_count) + observed_score_roi_per_1k)
            / new_count,
            4,
        )
        stat["accept_rate"] = round(stat["accepted"] / max(1, new_count), 4)
        stat["patch_accept_rate"] = round(stat["patch_accepted"] / max(1, new_count), 4)
        stat["mean_terminal_utility_delta"] = round(
            ((float(stat.get("mean_terminal_utility_delta", 0.0) or 0.0) * prior_count) + terminal_utility_delta)
            / new_count,
            4,
        )
        stat["mean_terminal_utility_per_1k_tokens"] = round(
            (
                (float(stat.get("mean_terminal_utility_per_1k_tokens", 0.0) or 0.0) * prior_count)
                + observed_terminal_utility_per_1k
            )
            / new_count,
            4,
        )
        buckets[bucket] = stat
        policy_stats["repair_observations"] = int(policy_stats.get("repair_observations", 0) or 0) + 1
        policy_stats["repair_accepted_total"] = int(policy_stats.get("repair_accepted_total", 0) or 0) + (1 if policy_success else 0)
        policy_stats["repair_rejected_total"] = int(policy_stats.get("repair_rejected_total", 0) or 0) + (0 if policy_success else 1)
        policy_stats["patch_accepted_total"] = int(policy_stats.get("patch_accepted_total", 0) or 0) + (1 if patch_accepted else 0)
        policy_stats["patch_rejected_total"] = int(policy_stats.get("patch_rejected_total", 0) or 0) + (0 if patch_accepted else 1)
        policy_stats["repair_accept_rate"] = round(policy_stats["repair_accepted_total"] / max(1, policy_stats["repair_observations"]), 4)
        policy_stats["patch_accept_rate"] = round(policy_stats["patch_accepted_total"] / max(1, policy_stats["repair_observations"]), 4)
        policy_stats["last_accepted"] = bool(policy_success)
        policy_stats["last_policy_success"] = bool(policy_success)
        policy_stats["last_patch_accepted"] = bool(patch_accepted)
        policy_stats["last_policy_outcome"] = policy_outcome
        policy_stats["last_gain"] = gain
        policy_stats["last_tokens"] = observed_tokens
        policy_stats["last_generation_tokens"] = repair_generation_tokens
        policy_stats["last_validation_tokens"] = repair_validation_tokens
        policy_stats["last_bucket_key"] = bucket
        policy_stats["last_observed_roi_per_1k"] = round(observed_score_roi_per_1k, 4)
        policy_stats["last_terminal_utility_before"] = round(terminal_utility_before, 4)
        policy_stats["last_terminal_utility_after"] = round(terminal_utility_after, 4)
        policy_stats["last_terminal_utility_delta"] = round(terminal_utility_delta, 4)
        policy_stats["last_terminal_utility_per_1k_tokens"] = round(observed_terminal_utility_per_1k, 4)
        policy_stats["last_repair_success_margin"] = round(repair_success_margin, 4)
        policy_stats["last_repair_quality_safe"] = bool(quality_safe)
        policy_stats["last_repair_quality_block_reasons"] = quality_block_reasons

        history = list(self._sget(base_state, "repair_validation_history", []) or [])
        history.append(
            {
                "bucket_key": bucket,
                "validation_mode": validation_mode,
                "accepted": bool(patch_accepted),
                "patch_accepted": bool(patch_accepted),
                "policy_success": bool(policy_success),
                "policy_outcome": policy_outcome,
                "score_before": previous_score,
                "score_after": current_score,
                "gain": gain,
                "observed_tokens": observed_tokens,
                "observed_roi_per_1k": round(observed_score_roi_per_1k, 4),
                "repair_generation_tokens": repair_generation_tokens,
                "repair_validation_tokens": repair_validation_tokens,
                "terminal_utility_before": round(terminal_utility_before, 4),
                "terminal_utility_after": round(terminal_utility_after, 4),
                "terminal_utility_delta": round(terminal_utility_delta, 4),
                "terminal_utility_per_1k_tokens": round(observed_terminal_utility_per_1k, 4),
                "repair_success_margin": round(repair_success_margin, 4),
                "quality_safe": bool(quality_safe),
                "quality_block_reasons": quality_block_reasons,
                **quality_summary,
                "selected_action": selected_action,
            }
        )
        updates["policy_stats"] = policy_stats
        updates["repair_validation_history"] = history
        updates["policy_success_count"] = int(self._sget(base_state, "policy_success_count", 0) or 0) + (1 if policy_success else 0)
        updates["policy_failure_count"] = int(self._sget(base_state, "policy_failure_count", 0) or 0) + (0 if policy_success else 1)
        if "accepted_patch_count" not in updates and "rejected_patch_count" not in updates:
            updates["accepted_patch_count"] = int(self._sget(base_state, "accepted_patch_count", 0) or 0) + (1 if patch_accepted else 0)
            updates["rejected_patch_count"] = int(self._sget(base_state, "rejected_patch_count", 0) or 0) + (0 if patch_accepted else 1)
        return updates

    def _run_repair_diagnostics(self, state: PitchState) -> Dict[str, Any]:
        base_state = state if isinstance(state, dict) else state.model_dump()
        validation = self._sget(base_state, "validation_report", {}) or {}
        claims = validation.get("claim_units", []) or validation.get("claims", []) if isinstance(validation, dict) else []
        failing_claims = []
        for claim in claims or []:
            if not isinstance(claim, dict):
                continue
            verdict = str(claim.get("verdict", "") or "").strip().lower()
            if verdict in {"weakly_supported", "unsupported", "needs_review"}:
                failing_claims.append(claim)
        baseline = self._ensure_baseline_checkpoint(base_state)
        score = self._validation_score(validation)
        threshold = int(self._sget(base_state, "validation_threshold", 70) or 70)
        coverage_counts = self._claim_coverage_counts(validation)
        missing_claim_count = int(coverage_counts.get("missing_claim_count", 0) or 0)
        claims_total = int(coverage_counts.get("claims_total", 0) or 0)
        minimum_claims_required = int(coverage_counts.get("minimum_claims_required", 0) or 0)
        has_coverage_target = any(
            isinstance(claim, dict) and bool(claim.get("coverage_enhancement"))
            for claim in failing_claims
        )
        if missing_claim_count > 0 and (score < threshold or bool(self._sget(base_state, "needs_revision", False))):
            if not has_coverage_target:
                failing_claims.append(
                    self._coverage_enhancement_target(
                        base_state,
                        validation,
                        missing_claim_count=missing_claim_count,
                        claims_total=claims_total,
                        minimum_claims_required=minimum_claims_required,
                    )
                )
        if not failing_claims:
            if score < threshold:
                failing_claims = [
                    self._coverage_enhancement_target(
                        base_state,
                        validation,
                        missing_claim_count=missing_claim_count,
                        claims_total=claims_total,
                        minimum_claims_required=minimum_claims_required,
                    )
                ]
            else:
                return {
                    "baseline_checkpoint": baseline,
                    "best_checkpoint": baseline,
                    "retrieval_diagnostics": [],
                    "failing_claims": [],
                }
        diagnostics = self.claim_repair_agent.diagnose_repairability(
            refined_idea=str(self._sget(base_state, "refined_idea", "") or ""),
            claims=failing_claims,
            max_results_per_query=2,
        )
        diag_dicts = [d.model_dump() if hasattr(d, "model_dump") else dict(d) for d in diagnostics]
        audit = list(self._sget(base_state, "tool_audit", []) or [])
        audit.append(
            {
                "agent": "adaptive_controller",
                "tool": "retrieval_diagnostics",
                "status": "ok",
                "claim_count": len(failing_claims),
                "diagnostic_count": len(diag_dicts),
                "coverage_enhancement_target_count": sum(
                    1 for claim in failing_claims if isinstance(claim, dict) and claim.get("coverage_enhancement")
                ),
                "mean_retrieval_score": round(sum(float(d.get("retrieval_score", 0.0) or 0.0) for d in diag_dicts) / max(1, len(diag_dicts)), 4),
                "strong_evidence_count": sum(1 for d in diag_dicts if d.get("evidence_strength") == "strong"),
                "evidence_probe_count": sum(
                    1
                    for d in diag_dicts
                    if any("evidence_probe_" in str(reason) for reason in (d.get("reasons", []) or []))
                ),
                "evidence_probe_upgrade_count": sum(
                    1
                    for d in diag_dicts
                    if "evidence_probe_upgraded_to_search_and_replace" in (d.get("reasons", []) or [])
                ),
            }
        )
        return {
            "baseline_checkpoint": baseline,
            "best_checkpoint": baseline,
            "failing_claims": failing_claims,
            "retrieval_diagnostics": diag_dicts,
            "tool_audit": audit,
        }

    def _run_policy(self, state: PitchState) -> Dict[str, Any]:
        base_state = state if isinstance(state, dict) else state.model_dump()
        forced_mode = str(self._sget(base_state, "forced_controller_mode", "") or "").strip().lower()
        retry_enabled = bool(self._sget(base_state, "adaptive_retry_enabled", True))
        needs_revision = bool(self._sget(base_state, "needs_revision", False))
        retry_count = int(self._sget(base_state, "retry_count", 0) or 0)
        max_retries = int(self._sget(base_state, "max_validation_retries", 1) or 1)
        validation = self._sget(base_state, "validation_report", {}) or {}
        score = self._validation_score(validation)
        threshold = int(self._sget(base_state, "validation_threshold", 70) or 70)
        token_usage = self._sget(base_state, "token_usage", {}) or {}
        tokens_before = int(token_usage.get("total_tokens", 0) or 0)

        if forced_mode in {"direct", "shallow"} or self.controller_policy != "adaptive":
            decision = {
                "action": "stop",
                "retry_allowed": False,
                "retry_type": "none",
                "reason": "fixed_baseline_no_adaptive_retry" if forced_mode else "fixed_policy_no_adaptive_retry",
                "needs_revision": needs_revision,
                "validation_score": score,
                "validation_threshold": threshold,
            }
        elif forced_mode == "recursive":
            allowed = needs_revision and retry_count < max_retries
            decision = {
                "action": "broad_retry" if allowed else "stop",
                "retry_allowed": bool(allowed),
                "retry_type": "fixed_recursive_market_retry" if allowed else "none",
                "reason": "fixed_recursive_retry" if allowed else "fixed_recursive_stop",
                "needs_revision": needs_revision,
                "validation_score": score,
                "validation_threshold": threshold,
                "retry_expected_gain": max(0, threshold - score),
                "retry_roi": 0.0,
            }
        elif not retry_enabled:
            decision = {
                "action": "stop",
                "retry_allowed": False,
                "retry_type": "none",
                "reason": "adaptive_retry_disabled_by_ablation",
                "needs_revision": needs_revision,
                "validation_score": score,
                "validation_threshold": threshold,
            }
        elif retry_count >= max_retries:
            decision = {
                "action": "stop",
                "retry_allowed": False,
                "retry_type": "none",
                "reason": "max_validation_retries_reached",
                "needs_revision": needs_revision,
                "validation_score": score,
                "validation_threshold": threshold,
            }
        elif not needs_revision:
            decision = {
                "action": "stop",
                "retry_allowed": False,
                "retry_type": "none",
                "reason": "validation_passed_stop",
                "needs_revision": needs_revision,
                "validation_score": score,
                "validation_threshold": threshold,
            }
        else:
            policy_decision = self.retry_policy.decide(
                state=base_state,
                claim_diagnostics=self._sget(base_state, "retrieval_diagnostics", []) or [],
            )
            decision = policy_decision.model_dump()
            decision["retry_allowed"] = decision.get("action") == "repair_bundle"
            decision["retry_type"] = "claim_micro_repair" if decision["retry_allowed"] else "none"
            decision["validation_score"] = score
            decision["validation_threshold"] = threshold
            decision["retry_expected_gain"] = decision.get("expected_gain", 0.0)
            decision["retry_roi"] = decision.get("roi_per_1k", 0.0)

        selected_action = dict(decision)
        selected_action.setdefault("score_before", score)
        selected_action.setdefault("tokens_before", tokens_before)
        if decision.get("candidates") and decision.get("selected_claim_ids"):
            selected_id = str(decision["selected_claim_ids"][0])
            candidate = next((c for c in decision.get("candidates", []) if str(c.get("claim_id")) == selected_id), {})
            selected_action.update({k: v for k, v in candidate.items() if k not in selected_action or selected_action.get(k) in (None, "")})

        decisions = list(self._sget(base_state, "retry_budget_decisions", []) or [])
        decisions.append(decision)
        audit = list(self._sget(base_state, "tool_audit", []) or [])
        audit.append(
            {
                "agent": "adaptive_controller",
                "tool": "bayesian_retry_policy",
                "status": "ok",
                "retry_allowed": bool(decision.get("retry_allowed", False)),
                "action": decision.get("action"),
                "reason": decision.get("reason"),
                "posterior_acceptance": decision.get("posterior_acceptance", 0.0),
                "expected_utility": decision.get("expected_utility", 0.0),
                "roi_per_1k": decision.get("roi_per_1k", 0.0),
                "evaluated_candidate_count": decision.get("evaluated_candidate_count", 0),
                "viable_candidate_count": decision.get("viable_candidate_count", 0),
                "rejected_candidate_count": decision.get("rejected_candidate_count", 0),
                "block_reasons": decision.get("block_reasons", []),
                "rejected_candidate_block_reasons": decision.get("rejected_candidate_block_reasons", []),
            }
        )
        return {
            "retry_budget_decisions": decisions,
            "retry_policy_decision": decision,
            "selected_action": selected_action,
            "selected_action_history": list(self._sget(base_state, "selected_action_history", []) or []) + [selected_action],
            "retry_roi_estimate": float(decision.get("roi_per_1k", decision.get("retry_roi", 0.0)) or 0.0),
            "retry_expected_utility": float(decision.get("expected_utility", 0.0) or 0.0),
            "retry_expected_gain": float(decision.get("expected_gain", decision.get("retry_expected_gain", 0.0)) or 0.0),
            "retry_expected_cost_tokens": int(decision.get("expected_tokens", 0) or 0),
            "retry_expected_cost_seconds": float(decision.get("expected_seconds", 0.0) or 0.0),
            "posterior_acceptance_probability": float(decision.get("posterior_acceptance", 0.0) or 0.0),
            "tool_audit": audit,
            "needs_revision": False if not bool(decision.get("retry_allowed", False)) else needs_revision,
        }

    @staticmethod
    def _last_retry_decision(state: PitchState | Dict[str, Any]) -> Dict[str, Any]:
        decisions = StartupPitchRefinery._sget(state, "retry_budget_decisions", []) or []
        last = decisions[-1] if decisions else {}
        return last if isinstance(last, dict) else {}

    def _build_market_repair_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        validation = self._sget(state, "validation_report", {}) or {}
        claims = validation.get("claim_units", []) or validation.get("claims", []) if isinstance(validation, dict) else []
        weak_claims: List[Dict[str, Any]] = []
        supported_claims: List[Dict[str, Any]] = []
        for idx, claim in enumerate(claims or [], start=1):
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("claim") or claim.get("claim_text") or "").strip()
            if not text:
                continue
            verdict = str(claim.get("verdict", "") or "").strip().lower()
            item = dict(claim)
            item["claim_id"] = str(claim.get("claim_id") or f"c{idx}")
            item["claim"] = text
            item["claim_text"] = text
            if verdict == "supported":
                supported_claims.append(item)
            elif verdict in {"weakly_supported", "unsupported", "needs_review"}:
                weak_claims.append(item)
        selected_action = self._sget(state, "selected_action", {}) or {}
        selected_ids = {str(x) for x in selected_action.get("selected_claim_ids", []) or []}
        state_failing_claims = [
            c
            for c in (self._sget(state, "failing_claims", []) or [])
            if isinstance(c, dict)
        ]
        existing_ids = {str(c.get("claim_id")) for c in weak_claims if isinstance(c, dict)}
        for claim in state_failing_claims:
            claim_id = str(claim.get("claim_id") or "")
            if claim_id and claim_id not in existing_ids:
                weak_claims.append(dict(claim))
                existing_ids.add(claim_id)
        if selected_ids:
            weak_claims = [c for c in weak_claims if str(c.get("claim_id")) in selected_ids]
        repair_plan = build_claim_repair_plan(weak_claims)
        selected_action_type = str(selected_action.get("action_type") or "").strip()
        guard_adjusted_action = str(selected_action.get("guard_adjusted_action") or "").strip()
        requested_action_type = str(selected_action.get("requested_action_type") or selected_action_type).strip()
        guard_reasons = selected_action.get("source_bound_guard_reasons", []) or []
        if selected_action_type in {"search_and_replace", "coverage_addition", "qualify_or_remove", "remove"}:
            for item in repair_plan:
                item["policy_selected_action"] = requested_action_type or selected_action_type
                item["policy_guard_adjusted_action"] = guard_adjusted_action or selected_action_type
                if guard_reasons:
                    item["policy_guard_reasons"] = list(guard_reasons)
                    item.setdefault("guard_reason", "|".join(str(x) for x in guard_reasons))
                forced_action = guard_adjusted_action or selected_action_type
                if forced_action in {"search_and_replace", "coverage_addition"}:
                    item["action"] = forced_action
                elif forced_action in {"qualify_or_remove", "remove"}:
                    item["policy_conservative_action_not_forced"] = True
        return {
            "repair_strategy": "baseline_dominating_bayesian_claim_repair",
            "previous_validation_score": self._validation_score(validation),
            "previous_supported_ratio": round(self._validation_supported_ratio(validation), 4),
            "previous_market_analysis": self._sget(state, "market_analysis", "") or "",
            "previous_market_sources": list(self._sget(state, "market_sources", []) or []),
            "weak_or_unsupported_claims": weak_claims[:8],
            "supported_claims": supported_claims[:8],
            "repair_plan": repair_plan[:8],
            "retrieval_diagnostics": list(self._sget(state, "retrieval_diagnostics", []) or []),
            "selected_action": selected_action,
            "repair_action_counts": {
                "search_and_replace": sum(1 for x in repair_plan if x.get("action") == "search_and_replace"),
                "coverage_addition": sum(1 for x in repair_plan if x.get("action") == "coverage_addition"),
                "qualify_or_remove": sum(1 for x in repair_plan if x.get("action") == "qualify_or_remove"),
                "remove": sum(1 for x in repair_plan if x.get("action") == "remove"),
            },
        }

    def _market_retry(self, state: PitchState):
        base_state = state if isinstance(state, dict) else state.model_dump()
        current_retry = int(self._sget(base_state, "retry_count", 0) or 0)
        last_decision = self._last_retry_decision(base_state)
        retry_type = str(last_decision.get("retry_type", "") or "").strip().lower()
        forced_mode = str(self._sget(base_state, "forced_controller_mode", "") or "").strip().lower()
        repair_context = self._build_market_repair_context(base_state)
        retry_state = {
            **base_state,
            "retry_count": current_retry + 1,
            "market_repair_context": repair_context,
        }
        use_claim_repair = retry_type == "claim_micro_repair" and forced_mode != "recursive"
        runner = self.claim_repair_agent.run if use_claim_repair else self.market_agent.run
        node_name = "claim_micro_repair" if use_claim_repair else "market_retry"
        if use_claim_repair:
            retry_state["repair_rounds"] = int(self._sget(base_state, "repair_rounds", 0) or 0) + 1
        updates = self._run_node_with_budget(retry_state, runner, node_name)
        updates["retry_count"] = current_retry + 1
        if use_claim_repair:
            updates["repair_rounds"] = int(self._sget(base_state, "repair_rounds", 0) or 0) + 1
        updates["market_repair_context"] = repair_context
        history = list(self._sget(base_state, "market_repair_history", []) or [])
        actual_plan = updates.get("repair_plan", []) or repair_context.get("repair_plan", []) or []
        actual_action_counts = {
            "search_and_replace": sum(1 for x in actual_plan if isinstance(x, dict) and x.get("action") == "search_and_replace"),
            "coverage_addition": sum(1 for x in actual_plan if isinstance(x, dict) and x.get("action") == "coverage_addition"),
            "qualify_or_remove": sum(1 for x in actual_plan if isinstance(x, dict) and x.get("action") == "qualify_or_remove"),
            "remove": sum(1 for x in actual_plan if isinstance(x, dict) and x.get("action") == "remove"),
        }
        guard_reasons = [
            str(x.get("guard_reason"))
            for x in actual_plan
            if isinstance(x, dict)
            and str(x.get("guard_reason", "")).strip()
            and str(x.get("guard_reason")) not in {
                "source_bound_search_allowed",
                "coverage_addition_source_search_allowed",
                "already_conservative_action",
            }
        ]
        history.append(
            {
                "retry_count": current_retry + 1,
                "retry_type": retry_type or "unknown",
                "repair_node": node_name,
                "forced_mode": forced_mode or None,
                "previous_validation_score": repair_context.get("previous_validation_score"),
                "previous_supported_ratio": repair_context.get("previous_supported_ratio"),
                "repair_action_counts": repair_context.get("repair_action_counts", {}),
                "actual_repair_action_counts": actual_action_counts,
                "guarded_repair_count": len(guard_reasons),
                "guard_reasons": guard_reasons[:5],
                "selected_action": self._sget(base_state, "selected_action", {}) or {},
            }
        )
        updates["market_repair_history"] = history
        return updates

    def _route_after_validation(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        forced_mode = str(self._sget(state, "forced_controller_mode", "") or "").strip().lower()
        if self.controller_policy != "adaptive" or forced_mode in {"direct", "shallow"}:
            return "select"
        if forced_mode == "recursive":
            return "policy"
        if not bool(self._sget(state, "needs_revision", False)):
            return "select"
        return "diagnose"

    def _route_after_policy(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        decision = self._sget(state, "retry_policy_decision", {}) or self._last_retry_decision(state)
        if isinstance(decision, dict) and bool(decision.get("retry_allowed", False)):
            return "repair"
        return "select"

    def _should_use_lightweight_repair_validation(self, state: PitchState | Dict[str, Any]) -> bool:
        selected_action = self._sget(state, "selected_action", {}) or {}
        if selected_action.get("action") != "repair_bundle":
            return False
        repair_plan = self._sget(state, "repair_plan", []) or []
        actual_actions = {
            str(item.get("action", "")).strip()
            for item in repair_plan
            if isinstance(item, dict) and str(item.get("action", "")).strip()
        }
        if actual_actions:
            return actual_actions.issubset({"qualify_or_remove", "remove"})
        return str(selected_action.get("action_type", "") or "").strip() in {"qualify_or_remove", "remove"}

    def _route_after_market_retry(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        if self._should_use_lightweight_repair_validation(state):
            return "repair_validator"
        return "validator"

    def _route_after_repair_validation(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        return "select"

    def _run_select_best_checkpoint(self, state: PitchState) -> Dict[str, Any]:
        base_state = state if isinstance(state, dict) else state.model_dump()
        forced_mode = str(self._sget(base_state, "forced_controller_mode", "") or "").strip().lower()
        if forced_mode in {"direct", "shallow", "recursive"}:
            return {}
        checkpoint_enabled = bool(self._sget(base_state, "adaptive_checkpoint_enabled", True))
        snapshots = [item for item in (self._sget(base_state, "validation_snapshots", []) or []) if isinstance(item, dict)]
        if not snapshots:
            return {}
        baseline = self._ensure_baseline_checkpoint(base_state)
        utilities: List[Dict[str, Any]] = []
        for snapshot in snapshots:
            utility = self._checkpoint_utility(snapshot, baseline, base_state)
            uncertainty = self._checkpoint_uncertainty(snapshot, base_state)
            utilities.append(
                {
                    "checkpoint_index": snapshot.get("checkpoint_index"),
                    "reliability_score": snapshot.get("reliability_score"),
                    "terminal_utility": round(utility, 4),
                    "uncertainty_penalty": round(uncertainty, 4),
                    "lcb_utility": round(utility - uncertainty, 4),
                    "tokens": snapshot.get("total_tokens_at_checkpoint", 0),
                }
            )
        current = snapshots[-1]
        if checkpoint_enabled:
            baseline_index = baseline.get("checkpoint_index")

            def _score_monotonic_key(snapshot: Dict[str, Any]) -> tuple:
                is_baseline = snapshot.get("checkpoint_index") == baseline_index
                quality_safe = True if is_baseline else self._quality_safe_repair_checkpoint(snapshot, baseline)[0]
                return (
                    1 if quality_safe else 0,
                    int(snapshot.get("reliability_score", 0) or 0),
                    -int(snapshot.get("material_failing_claim_count", 0) or 0),
                    0 if snapshot.get("low_claim_count_flag") else 1,
                    float(snapshot.get("claim_coverage_ratio", 0.0) or 0.0),
                    float(snapshot.get("supported_ratio", 0.0) or 0.0),
                    float(snapshot.get("judge_agreement", 0.0) or 0.0),
                    -int(snapshot.get("total_tokens_at_checkpoint", 0) or 0),
                )

            best = max(snapshots, key=_score_monotonic_key)
            best_meta = next(
                (u for u in utilities if u.get("checkpoint_index") == best.get("checkpoint_index")),
                utilities[-1],
            )
            baseline_score = int(baseline.get("reliability_score", 0) or 0)
            current_score = int(current.get("reliability_score", 0) or 0)
            repair_history = [
                item for item in (self._sget(base_state, "repair_validation_history", []) or [])
                if isinstance(item, dict)
            ]
            last_repair = repair_history[-1] if repair_history else {}
            latest_repair_accepted = bool(last_repair.get("policy_success", last_repair.get("accepted", False)))
            quality_gain_override = (
                latest_repair_accepted
                and current_score > baseline_score
            )
            if quality_gain_override:
                best = current
                best_meta = utilities[-1]
        else:
            best = current
            best_meta = utilities[-1]
            quality_gain_override = False
        selected_previous = best.get("checkpoint_index") != current.get("checkpoint_index")
        best_quality_safe = True
        best_quality_block_reasons: List[str] = []
        if checkpoint_enabled and best.get("checkpoint_index") != baseline.get("checkpoint_index"):
            best_quality_safe, best_quality_block_reasons, _ = self._quality_safe_repair_checkpoint(best, baseline)
        selection = {
            "selected_checkpoint_index": best.get("checkpoint_index"),
            "selected_retry_count": best.get("retry_count"),
            "selected_previous_checkpoint": bool(selected_previous),
            "best_validation_score": int(best.get("reliability_score", 0) or 0),
            "current_validation_score_before_selection": int(current.get("reliability_score", 0) or 0),
            "score_delta_vs_current": int(best.get("reliability_score", 0) or 0) - int(current.get("reliability_score", 0) or 0),
            "checkpoint_enabled": checkpoint_enabled,
            "terminal_utility": best_meta.get("terminal_utility"),
            "lcb_utility": best_meta.get("lcb_utility"),
            "quality_gain_override": bool(quality_gain_override),
            "selected_checkpoint_quality_safe": bool(best_quality_safe),
            "selected_checkpoint_quality_block_reasons": best_quality_block_reasons,
            "selection_rule": (
                "score-monotonic quality-safe repair improvement retained"
                if quality_gain_override
                else "highest quality-safe validator score checkpoint"
                if checkpoint_enabled
                else "checkpoint rollback disabled; latest validation state retained"
            ),
        }
        audit = list(self._sget(base_state, "tool_audit", []) or [])
        audit.append({"agent": "adaptive_controller", "tool": "checkpoint_selector", "status": "ok", **selection})
        updates: Dict[str, Any] = {
            "selected_validation_checkpoint": best,
            "selected_checkpoint": best,
            "best_checkpoint": best,
            "baseline_checkpoint": baseline,
            "checkpoint_utilities": utilities,
            "adaptive_checkpoint_selection": selection,
            "tool_audit": audit,
            "needs_revision": False,
        }
        if selected_previous:
            updates.update(
                {
                    "market_analysis": best.get("market_analysis"),
                    "validated_market_analysis": best.get("validated_market_analysis"),
                    "validation_report": best.get("validation_report"),
                    "market_sources": best.get("market_sources", []),
                    "market_evidence": best.get("market_evidence", []),
                    "trend_signals": best.get("trend_signals", {}),
                }
            )
        return updates

    def _run_business_with_depth(self, state: PitchState):
        updates = self._run_node_with_budget(state, self.business_agent.run, "business")
        mode = str(
            self._sget(state, "controller_mode_realized")
            or self._sget(state, "controller_mode", "")
            or ""
        ).strip().lower()
        retry_count = int(self._sget(state, "retry_count", 0) or 0)
        if mode == "direct" and not bool(self._sget(state, "direct_precheck_escalated", False)):
            depth = 0
        else:
            depth = 2 if retry_count > 0 else 1
        updates["decomposition_depth_realized"] = depth
        return updates

    def _run_pitch(self, state: PitchState):
        if self.pitch_agent is None:
            return {}
        return self._run_node_with_budget(state, self.pitch_agent.run, "pitch")

    def _route_after_controller(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        mode = str(self._sget(state, "controller_mode", "shallow") or "shallow").strip().lower()
        if mode not in {"direct", "shallow", "recursive"}:
            return "shallow"
        return mode

    def _route_by_budget(self, state: PitchState) -> str:
        return "stop" if bool(self._sget(state, "budget_hit", False)) else "continue"

    def _route_after_business(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        return "pitch" if self.generate_pitch and self.pitch_agent is not None else "end"

    def _route_after_direct(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        if (
            self.controller_policy == "adaptive"
            and not str(self._sget(state, "forced_controller_mode", "") or "").strip().lower()
            and bool(self._sget(state, "adaptive_retry_enabled", True))
            and bool(self._sget(state, "needs_revision", False))
        ):
            return "market"
        return "pitch" if self.generate_pitch and self.pitch_agent is not None else "end"

    def _build_graph(self):
        workflow = StateGraph(PitchState)
        workflow.add_node("supervisor", self._run_supervisor)
        workflow.add_node("idea", self._run_idea)
        if self.controller_policy == "adaptive" and self.controller_agent is not None:
            workflow.add_node("controller", self._run_controller)
        workflow.add_node("market", self._run_market)
        workflow.add_node("validator", self._run_validator)
        workflow.add_node("repair_diagnostics", self._run_repair_diagnostics)
        workflow.add_node("policy", self._run_policy)
        workflow.add_node("market_retry", self._market_retry)
        workflow.add_node("repair_validator", self._run_lightweight_repair_validator)
        workflow.add_node("select_best_checkpoint", self._run_select_best_checkpoint)
        workflow.add_node("business", self._run_business_with_depth)
        if self.controller_policy == "adaptive" and self.direct_agent is not None:
            workflow.add_node("direct", self._run_direct)
        if self.generate_pitch and self.pitch_agent is not None:
            workflow.add_node("pitch", self._run_pitch)

        workflow.add_edge(START, "supervisor")
        workflow.add_conditional_edges("supervisor", self._route_by_budget, {"stop": END, "continue": "idea"})
        if self.controller_policy == "adaptive" and self.controller_agent is not None:
            workflow.add_conditional_edges("idea", self._route_by_budget, {"stop": END, "continue": "controller"})
            workflow.add_conditional_edges(
                "controller",
                self._route_after_controller,
                {"stop": END, "direct": "direct", "shallow": "market", "recursive": "market"},
            )
        else:
            workflow.add_conditional_edges("idea", self._route_by_budget, {"stop": END, "continue": "market"})
        workflow.add_conditional_edges("market", self._route_by_budget, {"stop": END, "continue": "validator"})
        workflow.add_conditional_edges(
            "validator",
            self._route_after_validation,
            {"stop": END, "select": "select_best_checkpoint", "diagnose": "repair_diagnostics", "policy": "policy"},
        )
        workflow.add_edge("repair_diagnostics", "policy")
        workflow.add_conditional_edges("policy", self._route_after_policy, {"stop": END, "select": "select_best_checkpoint", "repair": "market_retry"})
        workflow.add_conditional_edges("market_retry", self._route_after_market_retry, {"stop": END, "repair_validator": "repair_validator", "validator": "validator"})
        workflow.add_conditional_edges("repair_validator", self._route_after_repair_validation, {"stop": END, "select": "select_best_checkpoint"})
        workflow.add_conditional_edges("select_best_checkpoint", self._route_by_budget, {"stop": END, "continue": "business"})
        if self.controller_policy == "adaptive" and self.direct_agent is not None:
            direct_route = {"stop": END, "end": END}
            direct_route["market"] = "market"
            if self.generate_pitch and self.pitch_agent is not None:
                direct_route["pitch"] = "pitch"
            workflow.add_conditional_edges("direct", self._route_after_direct, direct_route)
        business_route = {"stop": END, "end": END}
        if self.generate_pitch and self.pitch_agent is not None:
            business_route["pitch"] = "pitch"
        workflow.add_conditional_edges("business", self._route_after_business, business_route)
        if self.generate_pitch and self.pitch_agent is not None:
            workflow.add_edge("pitch", END)
        return workflow.compile(checkpointer=self.checkpointer)

    def run(
        self,
        idea: str,
        thread_id: str = "default-thread",
        max_validation_retries: int = 1,
        validation_threshold: int = 70,
        max_tool_calls: int | None = None,
        max_token_proxy: int | None = None,
        max_total_tokens: int | None = None,
        max_runtime_seconds: float | None = None,
        forced_controller_mode: str | None = None,
        adaptive_retry_enabled: bool = True,
        adaptive_checkpoint_enabled: bool = True,
        shared_refined_idea: str | None = None,
        shared_refinement_token_usage: Dict[str, int] | None = None,
        shared_refinement_tool_audit: list[Dict[str, Any]] | None = None,
        policy_stats: Dict[str, Any] | None = None,
    ) -> PitchState:
        initial_state = PitchState(
            idea=idea,
            controller_policy=self.controller_policy,
            forced_controller_mode=forced_controller_mode,
            max_validation_retries=max_validation_retries,
            validation_threshold=validation_threshold,
            max_tool_calls=None if max_tool_calls is None or max_tool_calls < 0 else int(max_tool_calls),
            max_token_proxy=None if max_token_proxy is None or max_token_proxy < 0 else int(max_token_proxy),
            max_total_tokens=None if max_total_tokens is None or max_total_tokens < 0 else int(max_total_tokens),
            max_runtime_seconds=None if max_runtime_seconds is None or max_runtime_seconds < 0 else float(max_runtime_seconds),
            runtime_started_at=time.time(),
            adaptive_retry_enabled=adaptive_retry_enabled,
            adaptive_checkpoint_enabled=adaptive_checkpoint_enabled,
            policy_thresholds={**BayesianRetryPolicy.DEFAULT_THRESHOLDS, **self.policy_thresholds},
            utility_weights={**BayesianRetryPolicy.DEFAULT_WEIGHTS, **self.utility_weights},
            policy_stats=dict(policy_stats or {}),
        ).model_dump()
        if shared_refined_idea:
            initial_state["refined_idea"] = shared_refined_idea
            initial_state["shared_refinement_locked"] = True
            initial_state["token_usage"] = dict(shared_refinement_token_usage or {})
            initial_state["tool_audit"] = list(shared_refinement_tool_audit or [])
        return self.graph.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})

    def continue_from_shallow_checkpoint(
        self,
        state: PitchState | Dict[str, Any],
        *,
        adaptive_retry_enabled: bool = True,
        adaptive_checkpoint_enabled: bool = True,
        policy_mode: str = "bayes",
        policy_stats: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        current = dict(state if isinstance(state, dict) else state.model_dump())
        validation = current.get("validation_report", {}) or {}
        threshold = int(current.get("validation_threshold", 70) or 70)
        score = self._validation_score(validation)
        snapshots = [s for s in (current.get("validation_snapshots", []) or []) if isinstance(s, dict)]
        if not snapshots:
            snapshots = [self._build_validation_snapshot(current)]
            current["validation_snapshots"] = snapshots
            current["checkpoint_history"] = snapshots
        baseline = snapshots[0]
        current.update(
            {
                "controller_policy": "adaptive",
                "forced_controller_mode": None,
                "controller_mode": current.get("controller_mode") or "shallow",
                "controller_mode_initial": current.get("controller_mode_initial") or "shallow",
                "adaptive_retry_enabled": adaptive_retry_enabled,
                "adaptive_checkpoint_enabled": adaptive_checkpoint_enabled,
                "needs_revision": score < threshold,
                "baseline_checkpoint": baseline,
                "best_checkpoint": baseline,
                "retry_budget_decisions": [],
                "retrieval_diagnostics": [],
                "retry_policy_decision": {},
                "selected_action": {},
                "selected_action_history": [],
                "policy_stats": dict(policy_stats or current.get("policy_stats", {}) or {}),
                "policy_thresholds": {**BayesianRetryPolicy.DEFAULT_THRESHOLDS, **self.policy_thresholds, **(current.get("policy_thresholds", {}) or {})},
                "utility_weights": {**BayesianRetryPolicy.DEFAULT_WEIGHTS, **self.utility_weights, **(current.get("utility_weights", {}) or {})},
                "repair_rounds": 0,
            }
        )

        retried = False
        if adaptive_retry_enabled and current.get("needs_revision", False):
            diag_updates = self._run_repair_diagnostics(current)
            current.update(diag_updates)
            policy_updates = self._run_policy(current)
            current.update(policy_updates)
            decision = current.get("retry_policy_decision", {}) or {}
            if bool(decision.get("retry_allowed", False)):
                retried = True
                retry_updates = self._market_retry(current)
                current.update(retry_updates)
                if bool(current.get("budget_hit", False)):
                    return current
                if self._should_use_lightweight_repair_validation(current):
                    val_updates = self._run_lightweight_repair_validator(current)
                else:
                    val_updates = self._run_validator(current)
                current.update(val_updates)
                if bool(current.get("budget_hit", False)):
                    return current
        else:
            policy_updates = self._run_policy(current)
            current.update(policy_updates)

        checkpoint_updates = self._run_select_best_checkpoint(current)
        current.update(checkpoint_updates)
        if bool(current.get("budget_hit", False)):
            return current

        if retried or not current.get("business_model"):
            business_updates = self._run_business_with_depth(current)
            current.update(business_updates)
        else:
            current["decomposition_depth_realized"] = 1

        if retried and self.generate_pitch and self.pitch_agent is not None and not bool(current.get("budget_hit", False)):
            current.update(self._run_pitch(current))
        audit = list(current.get("tool_audit", []) or [])
        audit.append(
            {
                "agent": "adaptive_controller",
                "tool": "shared_shallow_continuation",
                "status": "ok",
                "policy_mode": policy_mode,
                "retried": retried,
                "baseline_checkpoint_score": baseline.get("reliability_score"),
                "checkpoint_enabled": adaptive_checkpoint_enabled,
                "retry_enabled": adaptive_retry_enabled,
            }
        )
        current["tool_audit"] = audit
        return current

    def continue_adaptive_from_validated_state(
        self,
        state: PitchState | Dict[str, Any],
        *,
        adaptive_retry_enabled: bool = True,
        adaptive_checkpoint_enabled: bool = True,
    ) -> Dict[str, Any]:
        return self.continue_from_shallow_checkpoint(
            state,
            adaptive_retry_enabled=adaptive_retry_enabled,
            adaptive_checkpoint_enabled=adaptive_checkpoint_enabled,
            policy_mode="bayes",
        )
