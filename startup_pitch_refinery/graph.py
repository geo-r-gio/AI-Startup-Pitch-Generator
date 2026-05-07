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

        # Snapshot-level values are authoritative when present because older
        # snapshots may be built from merged state rather than raw validation only.
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
            # Still write an explicit stop decision for paper-mode analysis.
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
        # Backwards-compatible alias. New methodology should call
        # continue_from_shallow_checkpoint() with the exact fixed_shallow state.
        return self.continue_from_shallow_checkpoint(
            state,
            adaptive_retry_enabled=adaptive_retry_enabled,
            adaptive_checkpoint_enabled=adaptive_checkpoint_enabled,
            policy_mode="bayes",
        )
























# from __future__ import annotations

# import json
# import time
# from typing import Any, Dict

# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import END, START, StateGraph
# from langchain_openai import ChatOpenAI

# from startup_pitch_refinery.agents import (
#     AdaptiveControllerAgent,
#     BusinessModelAgent,
#     ClaimRepairAgent,
#     DirectStrategyAgent,
#     IdeaRefinementAgent,
#     MarketResearchAgent,
#     PitchDeckGeneratorAgent,
#     SourceValidatorAgent,
#     SupervisorAgent,
#     build_claim_repair_plan,
# )
# from startup_pitch_refinery.state import PitchState


# class StartupPitchRefinery:
#     def __init__(
#         self,
#         model: str = "gpt-4.1-nano",
#         secondary_judge_model: str | None = None,
#         temperature: float = 0.0,
#         seed: int = 42,
#         strict_tools: bool = True,
#         enable_trends: bool = True,
#         generate_pitch: bool = True,
#         controller_policy: str = "fixed",
#         output_dir: str = "output",
#     ):
#         llm = ChatOpenAI(model=model, temperature=temperature, seed=seed)
#         secondary_judge_llm = (
#             ChatOpenAI(
#                 model=secondary_judge_model,
#                 temperature=temperature,
#                 seed=seed + 101,
#             )
#             if secondary_judge_model
#             else None
#         )
#         self.generate_pitch = generate_pitch
#         self.controller_policy = controller_policy.strip().lower()
#         if self.controller_policy not in {"fixed", "adaptive"}:
#             raise ValueError(
#                 f"Unsupported controller_policy `{controller_policy}`. Allowed: fixed, adaptive."
#             )

#         self.supervisor = SupervisorAgent()
#         self.idea_agent = IdeaRefinementAgent(llm)
#         self.controller_agent = (
#             AdaptiveControllerAgent(llm) if self.controller_policy == "adaptive" else None
#         )
#         self.market_agent = MarketResearchAgent(
#             llm,
#             strict_tools=strict_tools,
#             enable_trends=enable_trends,
#         )
#         self.claim_repair_agent = ClaimRepairAgent(llm, strict_tools=strict_tools)
#         self.validator_agent = SourceValidatorAgent(
#             llm,
#             secondary_judge_llm=secondary_judge_llm,
#         )
#         self.direct_agent = (
#             DirectStrategyAgent(
#                 llm,
#                 strict_tools=strict_tools,
#                 enable_trends=enable_trends,
#                 secondary_judge_llm=secondary_judge_llm,
#             )
#             if self.controller_policy == "adaptive"
#             else None
#         )
#         self.business_agent = BusinessModelAgent(llm, strict_tools=strict_tools)
#         self.pitch_agent = (
#             PitchDeckGeneratorAgent(llm, output_dir=output_dir)
#             if self.generate_pitch
#             else None
#         )
#         self.checkpointer = MemorySaver()

#         self.graph = self._build_graph()

#     @staticmethod
#     def _sget(state: PitchState | Dict[str, Any], key: str, default: Any = None) -> Any:
#         if isinstance(state, dict):
#             return state.get(key, default)
#         return getattr(state, key, default)

#     @staticmethod
#     def _estimate_tokens_proxy(state: Dict[str, Any]) -> int:
#         chunks = [
#             str(state.get("idea", "")),
#             str(state.get("refined_idea", "")),
#             str(state.get("market_analysis", "")),
#             str(state.get("business_model", "")),
#             str(state.get("validated_market_analysis", "")),
#             json.dumps(state.get("pitch_content", {}), ensure_ascii=True),
#             json.dumps(state.get("trend_signals", {}), ensure_ascii=True),
#             json.dumps(state.get("validation_report", {}), ensure_ascii=True),
#             json.dumps(state.get("repair_plan", {}), ensure_ascii=True),
#             json.dumps(state.get("repair_patches", []), ensure_ascii=True),
#             json.dumps(state.get("micro_validation", {}), ensure_ascii=True),
#         ]
#         total_chars = sum(len(c) for c in chunks)
#         return max(1, total_chars // 4)

#     def _compute_budget_updates(self, merged: Dict[str, Any]) -> Dict[str, Any]:
#         max_tool_calls = self._sget(merged, "max_tool_calls")
#         max_token_proxy = self._sget(merged, "max_token_proxy")
#         max_total_tokens = self._sget(merged, "max_total_tokens")
#         max_runtime_seconds = self._sget(merged, "max_runtime_seconds")

#         tool_calls_current = len(self._sget(merged, "tool_audit", []))
#         token_proxy_current = self._estimate_tokens_proxy(merged)
#         token_usage = self._sget(merged, "token_usage", {}) or {}
#         prompt_tokens_current = int(token_usage.get("prompt_tokens", 0) or 0)
#         completion_tokens_current = int(token_usage.get("completion_tokens", 0) or 0)
#         total_tokens_current = int(token_usage.get("total_tokens", 0) or 0)
#         started_at = self._sget(merged, "runtime_started_at")
#         if isinstance(started_at, (int, float)) and started_at > 0:
#             runtime_elapsed_seconds = max(0.0, time.time() - float(started_at))
#         else:
#             runtime_elapsed_seconds = 0.0

#         reasons = list(self._sget(merged, "budget_hit_reasons", []))
#         budget_hit = bool(self._sget(merged, "budget_hit", False))

#         if isinstance(max_tool_calls, int) and max_tool_calls >= 0:
#             if tool_calls_current > max_tool_calls:
#                 budget_hit = True
#                 reasons.append(
#                     f"max_tool_calls_exceeded:{tool_calls_current}>{max_tool_calls}"
#                 )
#         if isinstance(max_token_proxy, int) and max_token_proxy >= 0:
#             if token_proxy_current > max_token_proxy:
#                 budget_hit = True
#                 reasons.append(
#                     f"max_token_proxy_exceeded:{token_proxy_current}>{max_token_proxy}"
#                 )
#         if isinstance(max_total_tokens, int) and max_total_tokens >= 0:
#             if total_tokens_current > max_total_tokens:
#                 budget_hit = True
#                 reasons.append(
#                     f"max_total_tokens_exceeded:{total_tokens_current}>{max_total_tokens}"
#                 )
#         if isinstance(max_runtime_seconds, (int, float)) and max_runtime_seconds >= 0:
#             if runtime_elapsed_seconds > float(max_runtime_seconds):
#                 budget_hit = True
#                 reasons.append(
#                     f"max_runtime_seconds_exceeded:{runtime_elapsed_seconds:.3f}>{float(max_runtime_seconds):.3f}"
#                 )

#         dedup_reasons = []
#         seen = set()
#         for r in reasons:
#             if r not in seen:
#                 dedup_reasons.append(r)
#                 seen.add(r)

#         remaining = {
#             "tool_calls": (
#                 None
#                 if not isinstance(max_tool_calls, int) or max_tool_calls < 0
#                 else max_tool_calls - tool_calls_current
#             ),
#             "token_proxy": (
#                 None
#                 if not isinstance(max_token_proxy, int) or max_token_proxy < 0
#                 else max_token_proxy - token_proxy_current
#             ),
#             "total_tokens": (
#                 None
#                 if not isinstance(max_total_tokens, int) or max_total_tokens < 0
#                 else max_total_tokens - total_tokens_current
#             ),
#             "runtime_seconds": (
#                 None
#                 if not isinstance(max_runtime_seconds, (int, float)) or max_runtime_seconds < 0
#                 else float(max_runtime_seconds) - runtime_elapsed_seconds
#             ),
#         }

#         return {
#             "runtime_elapsed_seconds": round(runtime_elapsed_seconds, 3),
#             "tool_calls_current": tool_calls_current,
#             "token_proxy_current": token_proxy_current,
#             "prompt_tokens_current": prompt_tokens_current,
#             "completion_tokens_current": completion_tokens_current,
#             "total_tokens_current": total_tokens_current,
#             "budget_hit": budget_hit,
#             "budget_hit_reasons": dedup_reasons,
#             "budget_remaining": remaining,
#         }

#     def _run_node_with_budget(
#         self,
#         state: PitchState | Dict[str, Any],
#         runner,
#         node_name: str,
#     ) -> Dict[str, Any]:
#         base_state = state if isinstance(state, dict) else state.model_dump()
#         precheck = self._compute_budget_updates(base_state)
#         if precheck["budget_hit"]:
#             return {
#                 **precheck,
#                 "controller_rationale": self._sget(base_state, "controller_rationale"),
#             }

#         enriched_state = {**base_state, **precheck}
#         updates = runner(enriched_state)
#         merged = {**enriched_state, **updates}
#         postcheck = self._compute_budget_updates(merged)
#         if postcheck["budget_hit"]:
#             audit = list(self._sget(merged, "tool_audit", []))
#             audit.append(
#                 {
#                     "agent": "budget_guard",
#                     "tool": "runtime_budget_check",
#                     "status": "halted",
#                     "node": node_name,
#                     "reasons": postcheck["budget_hit_reasons"],
#                 }
#             )
#             postcheck["tool_audit"] = audit
#         return {**updates, **postcheck}

#     @staticmethod
#     def _last_retry_decision(state: PitchState | Dict[str, Any]) -> Dict[str, Any]:
#         decisions = StartupPitchRefinery._sget(state, "retry_budget_decisions", []) or []
#         last = decisions[-1] if decisions else {}
#         return last if isinstance(last, dict) else {}

#     def _build_market_repair_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
#         validation = self._sget(state, "validation_report", {}) or {}
#         claims = validation.get("claim_units", []) or validation.get("claims", []) if isinstance(validation, dict) else []
#         weak_claims = []
#         supported_claims = []
#         for idx, claim in enumerate(claims or [], start=1):
#             if not isinstance(claim, dict):
#                 continue
#             text = str(claim.get("claim") or claim.get("claim_text") or "").strip()
#             if not text:
#                 continue
#             verdict = str(claim.get("verdict", "") or "").strip().lower()
#             item = {
#                 "claim_id": str(claim.get("claim_id") or f"c{idx}"),
#                 "claim": text,
#                 "claim_text": text,
#                 "verdict": verdict,
#                 "confidence": claim.get("confidence"),
#                 "category": claim.get("category"),
#                 "materiality": claim.get("materiality"),
#                 "failure_type": claim.get("failure_type") or claim.get("failure_type_hint"),
#                 "candidate_queries": claim.get("candidate_queries", []),
#                 "rationale": str(claim.get("rationale", "") or "")[:300],
#                 "supporting_sources": claim.get("supporting_sources", []),
#                 "source_diversity_count": claim.get("source_diversity_count"),
#                 "distinct_source_domains": claim.get("distinct_source_domains", []),
#                 "regulatory_flag": claim.get("regulatory_flag", False),
#             }
#             if verdict == "supported":
#                 supported_claims.append(item)
#             elif verdict in {"weakly_supported", "unsupported", "needs_review"}:
#                 weak_claims.append(item)
#         evidence_gaps = str(validation.get("evidence_gaps", "") or "") if isinstance(validation, dict) else ""
#         repair_plan = build_claim_repair_plan(weak_claims)
#         return {
#             "repair_strategy": "targeted_claim_repair: preserve supported claims, patch only weak claims, avoid broad rewrite",
#             "previous_validation_score": self._validation_score(validation),
#             "previous_supported_ratio": round(self._validation_supported_ratio(validation), 4),
#             "previous_market_analysis": self._sget(state, "market_analysis", "") or "",
#             "previous_market_sources": list(self._sget(state, "market_sources", []) or []),
#             "evidence_gaps": evidence_gaps,
#             "evidence_gap_signal": bool(evidence_gaps),
#             "weak_or_unsupported_claims": weak_claims[:8],
#             "supported_claims": supported_claims[:8],
#             "repair_plan": repair_plan[:8],
#             "repair_action_counts": {
#                 "search_and_replace": sum(1 for x in repair_plan if x.get("action") == "search_and_replace"),
#                 "qualify_or_remove": sum(1 for x in repair_plan if x.get("action") == "qualify_or_remove"),
#                 "remove": sum(1 for x in repair_plan if x.get("action") == "remove"),
#             },
#         }

#     def _market_retry(self, state: PitchState):
#         """Execute retry selected by the retry gate.

#         Adaptive retries use claim-level repair. Fixed recursive baselines keep
#         broad market retry behavior for a fair baseline comparison.
#         """
#         current_retry = int(self._sget(state, "retry_count", 0) or 0)
#         base_state = state if isinstance(state, dict) else state.model_dump()
#         last_decision = self._last_retry_decision(base_state)
#         retry_type = str(last_decision.get("retry_type", "") or "").strip().lower()
#         forced_mode = str(self._sget(base_state, "forced_controller_mode", "") or "").strip().lower()
#         repair_context = self._build_market_repair_context(base_state)
#         retry_state = {
#             **base_state,
#             "retry_count": current_retry + 1,
#             "market_repair_context": repair_context,
#         }
#         use_claim_repair = retry_type == "claim_micro_repair" and forced_mode != "recursive"
#         runner = self.claim_repair_agent.run if use_claim_repair else self.market_agent.run
#         node_name = "claim_micro_repair" if use_claim_repair else "market_retry"
#         updates = self._run_node_with_budget(retry_state, runner, node_name)
#         updates["retry_count"] = current_retry + 1
#         updates["market_repair_context"] = repair_context
#         repair_history = list(self._sget(base_state, "market_repair_history", []) or [])
#         repair_history.append({
#             "retry_count": current_retry + 1,
#             "retry_type": retry_type or "unknown",
#             "repair_node": node_name,
#             "forced_mode": forced_mode or None,
#             "weak_claim_count": len(repair_context.get("weak_or_unsupported_claims", [])),
#             "supported_claim_count": len(repair_context.get("supported_claims", [])),
#             "previous_validation_score": repair_context.get("previous_validation_score"),
#             "previous_supported_ratio": repair_context.get("previous_supported_ratio"),
#             "repair_action_counts": repair_context.get("repair_action_counts", {}),
#         })
#         updates["market_repair_history"] = repair_history
#         return updates

#     def _run_supervisor(self, state: PitchState):
#         return self._run_node_with_budget(state, self.supervisor.run, "supervisor")

#     def _run_idea(self, state: PitchState):
#         return self._run_node_with_budget(state, self.idea_agent.run, "idea")

#     def _run_controller(self, state: PitchState):
#         if self.controller_agent is None:
#             return {}
#         return self._run_node_with_budget(state, self.controller_agent.run, "controller")

#     def _run_market(self, state: PitchState):
#         return self._run_node_with_budget(state, self.market_agent.run, "market")

#     def _run_validator(self, state: PitchState):
#         base_state = state if isinstance(state, dict) else state.model_dump()
#         updates = self._run_node_with_budget(base_state, self.validator_agent.run, "validator")
#         merged = {**base_state, **updates}
#         snapshot = self._build_validation_snapshot(merged)
#         snapshot["validation_mode"] = "full_dual_judge"
#         snapshots = list(self._sget(base_state, "validation_snapshots", []) or [])
#         snapshots.append(snapshot)
#         updates["validation_snapshots"] = snapshots
#         updates["checkpoint_history"] = snapshots
#         return updates

#     def _run_lightweight_repair_validator(self, state: PitchState):
#         base_state = state if isinstance(state, dict) else state.model_dump()
#         updates = self._run_node_with_budget(base_state, self.validator_agent.run_repair_only, "repair_validator")
#         merged = {**base_state, **updates}
#         snapshot = self._build_validation_snapshot(merged)
#         snapshot["validation_mode"] = "lightweight_repair"
#         snapshots = list(self._sget(base_state, "validation_snapshots", []) or [])
#         snapshots.append(snapshot)
#         updates["validation_snapshots"] = snapshots
#         updates["checkpoint_history"] = snapshots
#         return updates

#     def _run_direct(self, state: PitchState):
#         if self.direct_agent is None:
#             return {}
#         return self._run_node_with_budget(state, self.direct_agent.run, "direct")

#     def _run_pitch(self, state: PitchState):
#         if self.pitch_agent is None:
#             return {}
#         return self._run_node_with_budget(state, self.pitch_agent.run, "pitch")

#     @staticmethod
#     def _validation_supported_ratio(validation: Dict[str, Any]) -> float:
#         claims = validation.get("claims", []) if isinstance(validation, dict) else []
#         if not claims:
#             return 0.0
#         supported = 0
#         for claim in claims:
#             if not isinstance(claim, dict):
#                 continue
#             verdict = str(claim.get("verdict", "")).strip().lower()
#             if verdict == "supported":
#                 supported += 1
#         return supported / max(1, len(claims))

#     @staticmethod
#     def _validation_score(validation: Dict[str, Any]) -> int:
#         if not isinstance(validation, dict):
#             return 0
#         try:
#             return int(validation.get("reliability_score", 0) or 0)
#         except (TypeError, ValueError):
#             return 0

#     def _build_validation_snapshot(self, state: Dict[str, Any]) -> Dict[str, Any]:
#         validation = self._sget(state, "validation_report", {}) or {}
#         agreement = (
#             validation.get("agreement_stats", {}).get("overall_agreement", 0.0)
#             if isinstance(validation, dict)
#             else 0.0
#         )
#         token_usage = self._sget(state, "token_usage", {}) or {}
#         return {
#             "checkpoint_index": len(self._sget(state, "validation_snapshots", []) or []),
#             "retry_count": int(self._sget(state, "retry_count", 0) or 0),
#             "controller_mode": self._sget(state, "controller_mode"),
#             "reliability_score": self._validation_score(validation),
#             "judge_agreement": float(agreement or 0.0),
#             "supported_ratio": round(self._validation_supported_ratio(validation), 4),
#             "total_tokens_at_checkpoint": int(token_usage.get("total_tokens", 0) or 0),
#             "market_analysis": self._sget(state, "market_analysis"),
#             "validated_market_analysis": self._sget(state, "validated_market_analysis"),
#             "validation_report": validation,
#             "market_sources": list(self._sget(state, "market_sources", []) or []),
#             "market_evidence": list(self._sget(state, "market_evidence", []) or []),
#             "trend_signals": dict(self._sget(state, "trend_signals", {}) or {}),
#         }

#     def _run_select_best_checkpoint(self, state: PitchState) -> Dict[str, Any]:
#         base_state = state if isinstance(state, dict) else state.model_dump()
#         forced_mode = str(self._sget(base_state, "forced_controller_mode", "") or "").strip().lower()
#         if forced_mode in {"direct", "shallow", "recursive"}:
#             return {}
#         snapshots = [
#             item
#             for item in (self._sget(base_state, "validation_snapshots", []) or [])
#             if isinstance(item, dict)
#         ]
#         if not snapshots:
#             return {}

#         def rank(snapshot: Dict[str, Any]) -> tuple[float, float, float, int]:
#             return (
#                 float(snapshot.get("reliability_score", 0) or 0),
#                 float(snapshot.get("judge_agreement", 0.0) or 0.0),
#                 float(snapshot.get("supported_ratio", 0.0) or 0.0),
#                 -int(snapshot.get("total_tokens_at_checkpoint", 0) or 0),
#             )

#         best = max(snapshots, key=rank)
#         current = snapshots[-1]
#         best_score = int(best.get("reliability_score", 0) or 0)
#         current_score = int(current.get("reliability_score", 0) or 0)
#         selected_previous = best.get("checkpoint_index") != current.get("checkpoint_index")
#         threshold = int(self._sget(base_state, "validation_threshold", 70) or 70)
#         selection = {
#             "selected_checkpoint_index": best.get("checkpoint_index"),
#             "selected_retry_count": best.get("retry_count"),
#             "selected_previous_checkpoint": bool(selected_previous),
#             "best_validation_score": best_score,
#             "current_validation_score_before_selection": current_score,
#             "score_delta_vs_current": best_score - current_score,
#             "selection_rule": (
#                 "highest reliability score, then judge agreement, supported ratio, "
#                 "then lower token cost"
#             ),
#         }
#         tool_audit = list(self._sget(base_state, "tool_audit", []) or [])
#         tool_audit.append(
#             {
#                 "agent": "adaptive_controller",
#                 "tool": "checkpoint_selector",
#                 "status": "ok",
#                 **selection,
#             }
#         )

#         updates: Dict[str, Any] = {
#             "selected_validation_checkpoint": best,
#             "adaptive_checkpoint_selection": selection,
#             "tool_audit": tool_audit,
#             # The retry gate has already finished, so the selected checkpoint is
#             # the final available validation state for this run.
#             "needs_revision": False,
#         }
#         if selected_previous:
#             updates.update(
#                 {
#                     "market_analysis": best.get("market_analysis"),
#                     "validated_market_analysis": best.get("validated_market_analysis"),
#                     "validation_report": best.get("validation_report"),
#                     "market_sources": best.get("market_sources", []),
#                     "market_evidence": best.get("market_evidence", []),
#                     "trend_signals": best.get("trend_signals", {}),
#                 }
#             )
#         elif best_score < threshold:
#             updates["needs_revision"] = False
#         return updates

#     def _route_after_validation(self, state: PitchState) -> str:
#         if bool(self._sget(state, "budget_hit", False)):
#             return "stop"
#         return "retry_gate"

#     def _run_retry_gate(self, state: PitchState) -> Dict[str, Any]:
#         base_state = state if isinstance(state, dict) else state.model_dump()
#         controller_mode = str(self._sget(state, "controller_mode", "") or "").strip().lower()
#         forced_mode = str(self._sget(state, "forced_controller_mode", "") or "").strip().lower()
#         fixed_mode_baseline = forced_mode in {"direct", "shallow", "recursive"}
#         fixed_recursive_baseline = forced_mode == "recursive"
#         retry_enabled = bool(self._sget(state, "adaptive_retry_enabled", True))
#         needs_revision = bool(self._sget(state, "needs_revision", False))
#         retry_count = int(self._sget(state, "retry_count", 0) or 0)
#         max_retries = int(self._sget(state, "max_validation_retries", 1) or 1)
#         validation = self._sget(state, "validation_report", {}) or {}
#         validation_score = self._validation_score(validation)
#         validation_threshold = int(self._sget(state, "validation_threshold", 70) or 70)
#         validation_deficit = max(0, validation_threshold - validation_score)
#         agreement_stats = validation.get("agreement_stats", {}) if isinstance(validation, dict) else {}
#         judge_agreement = float(agreement_stats.get("overall_agreement", 0.0) or 0.0)
#         supported_ratio = self._validation_supported_ratio(validation)
#         claims = validation.get("claim_units", []) or validation.get("claims", []) if isinstance(validation, dict) else []
#         weak_claim_items = []
#         weak_claim_count = unsupported_claim_count = needs_review_claim_count = 0
#         material_failing_claim_count = specific_repair_claim_count = 0
#         low_claim_count_flag = False
#         claim_count_penalty = 0
#         if isinstance(validation, dict):
#             agg = validation.get("judge_scores", {}).get("aggregated", {}) if isinstance(validation.get("judge_scores", {}), dict) else {}
#             low_claim_count_flag = bool(agg.get("low_claim_count_flag", False))
#             claim_count_penalty = int(agg.get("claim_count_penalty", 0) or 0)
#         for claim in claims or []:
#             if not isinstance(claim, dict):
#                 continue
#             verdict = str(claim.get("verdict", "") or "").strip().lower()
#             if verdict in {"weakly_supported", "unsupported", "needs_review"}:
#                 weak_claim_items.append(claim)
#                 mat = int(claim.get("materiality", 1) or 1)
#                 if mat >= 4:
#                     material_failing_claim_count += 1
#                 failure_type = str(claim.get("failure_type") or claim.get("failure_type_hint") or "").strip().lower()
#                 if failure_type not in {"", "none", "too_broad", "unclear_attribution"}:
#                     specific_repair_claim_count += 1
#             if verdict == "weakly_supported":
#                 weak_claim_count += 1
#             elif verdict == "unsupported":
#                 unsupported_claim_count += 1
#             elif verdict == "needs_review":
#                 needs_review_claim_count += 1

#         budget_remaining = self._sget(state, "budget_remaining", {}) or {}
#         remaining_total_tokens = budget_remaining.get("total_tokens") if isinstance(budget_remaining, dict) else None
#         remaining_tool_calls = budget_remaining.get("tool_calls") if isinstance(budget_remaining, dict) else None
#         remaining_runtime = budget_remaining.get("runtime_seconds") if isinstance(budget_remaining, dict) else None
#         evidence_gaps = str(validation.get("evidence_gaps", "") or "") if isinstance(validation, dict) else ""
#         evidence_gap_signal = validation_deficit > 0 and any(m in evidence_gaps.lower() for m in ["lack", "limited", "gap", "weak", "indirect", "not specific"])
#         repair_plan = build_claim_repair_plan(weak_claim_items)
#         repair_action_counts = {
#             "search_and_replace": sum(1 for x in repair_plan if x.get("action") == "search_and_replace"),
#             "qualify_or_remove": sum(1 for x in repair_plan if x.get("action") == "qualify_or_remove"),
#             "remove": sum(1 for x in repair_plan if x.get("action") == "remove"),
#         }
#         repairable_claim_count = sum(repair_action_counts.values())
#         search_repair_count = repair_action_counts["search_and_replace"]
#         conservative_repair_count = repair_action_counts["qualify_or_remove"] + repair_action_counts["remove"]
#         concrete_claim_repair_target = len(weak_claim_items) > 0 and repairable_claim_count > 0
#         material_failure_signal = material_failing_claim_count > 0 or unsupported_claim_count > 0
#         specific_repair_signal = specific_repair_claim_count > 0 or material_failure_signal

#         estimated_retry_tokens = 1700 if search_repair_count == 0 else 2500
#         estimated_retry_tokens = min(estimated_retry_tokens, 2500)
#         estimated_retry_tool_calls = 1 if search_repair_count == 0 else 3
#         business_reserve_tokens = 900 if not self.generate_pitch else 2600
#         safety_buffer_tokens = 500 if not self.generate_pitch else 900
#         estimated_total_needed = estimated_retry_tokens + business_reserve_tokens + safety_buffer_tokens
#         estimated_tool_calls_needed = estimated_retry_tool_calls + (1 if not self.generate_pitch else 2)
#         estimated_runtime_needed = 18.0 if not self.generate_pitch else 42.0

#         quality_low_enough = validation_score <= validation_threshold - 4 or material_failure_signal or low_claim_count_flag
#         current_quality_close_enough = (
#             validation_score >= max(70, validation_threshold - 4)
#             and validation_deficit <= 4
#             and judge_agreement >= 0.88
#             and supported_ratio >= 0.66
#             and unsupported_claim_count == 0
#             and needs_review_claim_count <= 1
#             and weak_claim_count <= 1
#             and not low_claim_count_flag
#         )
#         retry_expected_gain = min(
#             18.0,
#             (1.10 * validation_deficit)
#             + (8.0 * max(0.0, 0.86 - judge_agreement))
#             + (9.0 * max(0.0, 0.75 - supported_ratio))
#             + (1.0 * weak_claim_count)
#             + (3.0 * unsupported_claim_count)
#             + (2.0 * needs_review_claim_count)
#             + (2.0 * material_failing_claim_count)
#             + (1.5 * search_repair_count)
#             + (1.0 * conservative_repair_count)
#             + (float(claim_count_penalty) * 0.6)
#             + (2.0 if evidence_gap_signal and concrete_claim_repair_target else 0.0),
#         )
#         retry_roi = retry_expected_gain / max(1.0, estimated_retry_tokens / 1000.0)

#         claim_micro_repair_candidate = (
#             controller_mode in {"shallow", "recursive"}
#             and not fixed_mode_baseline
#             and retry_enabled
#             and needs_revision
#             and retry_count < max_retries
#             and concrete_claim_repair_target
#             and quality_low_enough
#         )
#         fixed_recursive_retry = fixed_recursive_baseline and needs_revision and retry_count < max_retries and validation_deficit >= 3
#         retry_allowed = claim_micro_repair_candidate or fixed_recursive_retry
#         min_retry_expected_gain = 5.0 if claim_micro_repair_candidate else 0.0
#         min_retry_roi = 1.5 if claim_micro_repair_candidate else 0.0

#         block_reasons = []
#         if fixed_mode_baseline and not fixed_recursive_baseline and needs_revision:
#             block_reasons.append("fixed_baseline_no_adaptive_retry")
#         if not needs_revision:
#             block_reasons.append("validation_passed")
#         if needs_revision and not quality_low_enough and not fixed_recursive_retry:
#             block_reasons.append("quality_not_low_enough_for_repair")
#         if current_quality_close_enough and not material_failure_signal and not fixed_recursive_retry:
#             retry_allowed = False; block_reasons.append("quality_close_enough_accept_checkpoint")
#         if retry_allowed and not specific_repair_signal and not fixed_recursive_retry:
#             retry_allowed = False; block_reasons.append("no_specific_repair_signal")
#         if retry_allowed and retry_expected_gain < min_retry_expected_gain:
#             retry_allowed = False; block_reasons.append(f"low_expected_retry_gain:{retry_expected_gain:.3f}<{min_retry_expected_gain:.3f}")
#         if retry_allowed and retry_roi < min_retry_roi:
#             retry_allowed = False; block_reasons.append(f"low_retry_roi:{retry_roi:.3f}<{min_retry_roi:.3f}")
#         if retry_count >= max_retries:
#             retry_allowed = False; block_reasons.append("max_validation_retries_reached")
#         if isinstance(remaining_total_tokens, int) and remaining_total_tokens < estimated_total_needed:
#             retry_allowed = False; block_reasons.append(f"insufficient_total_tokens:{remaining_total_tokens}<{estimated_total_needed}")
#         if isinstance(remaining_tool_calls, int) and remaining_tool_calls < estimated_tool_calls_needed:
#             retry_allowed = False; block_reasons.append(f"insufficient_tool_calls:{remaining_tool_calls}<{estimated_tool_calls_needed}")
#         if isinstance(remaining_runtime, (int, float)) and float(remaining_runtime) < estimated_runtime_needed:
#             retry_allowed = False; block_reasons.append(f"insufficient_runtime:{float(remaining_runtime):.3f}<{estimated_runtime_needed:.3f}")

#         retry_type = "claim_micro_repair" if claim_micro_repair_candidate and retry_allowed else "fixed_recursive_market_retry" if fixed_recursive_retry and retry_allowed else "none"
#         decision = {
#             "controller_mode": controller_mode,
#             "forced_mode": forced_mode or None,
#             "fixed_mode_baseline": fixed_mode_baseline,
#             "fixed_recursive_baseline": fixed_recursive_baseline,
#             "adaptive_retry_enabled": retry_enabled,
#             "needs_revision": needs_revision,
#             "retry_count": retry_count,
#             "max_validation_retries": max_retries,
#             "validation_score": validation_score,
#             "validation_threshold": validation_threshold,
#             "validation_deficit": validation_deficit,
#             "judge_agreement": round(judge_agreement, 4),
#             "supported_ratio": round(supported_ratio, 4),
#             "weak_or_unsupported_claim_count": len(weak_claim_items),
#             "weak_claim_count": weak_claim_count,
#             "unsupported_claim_count": unsupported_claim_count,
#             "needs_review_claim_count": needs_review_claim_count,
#             "material_failing_claim_count": material_failing_claim_count,
#             "specific_repair_claim_count": specific_repair_claim_count,
#             "concrete_claim_repair_target": concrete_claim_repair_target,
#             "repairable_claim_count": repairable_claim_count,
#             "repair_action_counts": repair_action_counts,
#             "search_repair_count": search_repair_count,
#             "conservative_repair_count": conservative_repair_count,
#             "current_quality_close_enough": current_quality_close_enough,
#             "retry_expected_gain": round(retry_expected_gain, 4),
#             "retry_incremental_tokens": estimated_retry_tokens,
#             "retry_roi": round(retry_roi, 4),
#             "min_retry_expected_gain": min_retry_expected_gain,
#             "min_retry_roi": min_retry_roi,
#             "retry_allowed": retry_allowed,
#             "retry_type": retry_type,
#             "block_reasons": block_reasons,
#             "remaining_total_tokens": remaining_total_tokens,
#             "estimated_retry_tokens": estimated_retry_tokens,
#             "business_reserve_tokens": business_reserve_tokens,
#             "safety_buffer_tokens": safety_buffer_tokens,
#             "estimated_total_needed": estimated_total_needed,
#             "remaining_tool_calls": remaining_tool_calls,
#             "estimated_tool_calls_needed": estimated_tool_calls_needed,
#             "remaining_runtime_seconds": remaining_runtime,
#             "estimated_runtime_needed": estimated_runtime_needed,
#         }
#         decisions = list(self._sget(state, "retry_budget_decisions", []))
#         decisions.append(decision)
#         updates: Dict[str, Any] = {
#             "retry_budget_decisions": decisions,
#             "retry_roi_estimate": round(retry_roi, 4),
#         }
#         # Do not flip shallow to recursive for claim micro-repair. Realized depth
#         # will be recorded as 2 after retry, but controller_mode remains shallow.
#         if not retry_allowed and needs_revision:
#             updates["needs_revision"] = False
#         return updates

#     def _route_after_retry_gate(self, state: PitchState) -> str:
#         if bool(self._sget(state, "budget_hit", False)):
#             return "stop"
#         decisions = self._sget(state, "retry_budget_decisions", []) or []
#         last = decisions[-1] if decisions else {}
#         if isinstance(last, dict) and bool(last.get("retry_allowed", False)):
#             return "retry_market"
#         return "continue"


#     def _route_after_market_retry(self, state: PitchState) -> str:
#         if bool(self._sget(state, "budget_hit", False)):
#             return "stop"
#         last = self._last_retry_decision(state)
#         if last.get("retry_type") == "claim_micro_repair":
#             return "repair_validator"
#         return "validator"

#     def _route_after_repair_validation(self, state: PitchState) -> str:
#         if bool(self._sget(state, "budget_hit", False)):
#             return "stop"
#         # After one lightweight patch validation, select the best checkpoint and
#         # continue. This prevents costly repeated repair loops.
#         return "continue"

#     @staticmethod
#     def _estimate_retry_tokens(state: Dict[str, Any]) -> int:
#         audits = state.get("tool_audit", []) or []
#         validator_totals = [
#             int(audit.get("total_tokens", 0) or 0)
#             for audit in audits
#             if isinstance(audit, dict) and audit.get("tool") == "llm_claim_verifier_dual_judge"
#         ]
#         last_validator_tokens = validator_totals[-1] if validator_totals else 4500
#         # Claim-level repair is intentionally capped. Broad market retry is only
#         # used for fixed-recursive baselines.
#         estimated_validation = max(1200, min(2200, last_validator_tokens // 2))
#         estimated_market = 800
#         return int(min(2500, estimated_validation + estimated_market))

#     def _route_after_controller(self, state: PitchState) -> str:
#         if bool(self._sget(state, "budget_hit", False)):
#             return "stop"
#         mode = str(self._sget(state, "controller_mode", "shallow")).strip().lower()
#         if mode not in {"direct", "shallow", "recursive"}:
#             return "shallow"
#         return mode

#     def _route_by_budget(self, state: PitchState) -> str:
#         if bool(self._sget(state, "budget_hit", False)):
#             return "stop"
#         return "continue"

#     def _route_after_business(self, state: PitchState) -> str:
#         if bool(self._sget(state, "budget_hit", False)):
#             return "stop"
#         if self.generate_pitch and self.pitch_agent is not None:
#             return "pitch"
#         return "end"

#     def _route_after_direct(self, state: PitchState) -> str:
#         if bool(self._sget(state, "budget_hit", False)):
#             return "stop"
#         if self.generate_pitch and self.pitch_agent is not None:
#             return "pitch"
#         return "end"

#     def _run_business_with_depth(self, state: PitchState):
#         updates = self._run_node_with_budget(state, self.business_agent.run, "business")
#         mode = str(self._sget(state, "controller_mode", "")).strip().lower()
#         retry_count = int(self._sget(state, "retry_count", 0))

#         if mode == "shallow":
#             depth = 2 if retry_count > 0 else 1
#         elif mode == "recursive":
#             depth = 2 if retry_count > 0 else 1
#         elif mode == "direct":
#             depth = 0
#         else:
#             depth = 2 if retry_count > 0 else 1
#         updates["decomposition_depth_realized"] = depth
#         return updates

#     def _build_graph(self):
#         workflow = StateGraph(PitchState)

#         workflow.add_node("supervisor", self._run_supervisor)
#         workflow.add_node("idea", self._run_idea)
#         if self.controller_policy == "adaptive" and self.controller_agent is not None:
#             workflow.add_node("controller", self._run_controller)
#         workflow.add_node("market", self._run_market)
#         workflow.add_node("validator", self._run_validator)
#         workflow.add_node("repair_validator", self._run_lightweight_repair_validator)
#         workflow.add_node("retry_gate", self._run_retry_gate)
#         workflow.add_node("market_retry", self._market_retry)
#         workflow.add_node("select_best_checkpoint", self._run_select_best_checkpoint)
#         workflow.add_node("business", self._run_business_with_depth)
#         if self.controller_policy == "adaptive" and self.direct_agent is not None:
#             workflow.add_node("direct", self._run_direct)
#         if self.generate_pitch and self.pitch_agent is not None:
#             workflow.add_node("pitch", self._run_pitch)

#         workflow.add_edge(START, "supervisor")
#         workflow.add_conditional_edges(
#             "supervisor",
#             self._route_by_budget,
#             {
#                 "stop": END,
#                 "continue": "idea",
#             },
#         )
#         if self.controller_policy == "adaptive" and self.controller_agent is not None:
#             workflow.add_conditional_edges(
#                 "idea",
#                 self._route_by_budget,
#                 {
#                     "stop": END,
#                     "continue": "controller",
#                 },
#             )
#             workflow.add_conditional_edges(
#                 "controller",
#                 self._route_after_controller,
#                 {
#                     "stop": END,
#                     "direct": "direct",
#                     "shallow": "market",
#                     "recursive": "market",
#                 },
#             )
#         else:
#             workflow.add_conditional_edges(
#                 "idea",
#                 self._route_by_budget,
#                 {
#                     "stop": END,
#                     "continue": "market",
#                 },
#             )
#         workflow.add_conditional_edges(
#             "market",
#             self._route_by_budget,
#             {
#                 "stop": END,
#                 "continue": "validator",
#             },
#         )
#         workflow.add_conditional_edges(
#             "validator",
#             self._route_after_validation,
#             {
#                 "stop": END,
#                 "retry_gate": "retry_gate",
#             },
#         )
#         workflow.add_conditional_edges(
#             "retry_gate",
#             self._route_after_retry_gate,
#             {
#                 "stop": END,
#                 "retry_market": "market_retry",
#                 "continue": "select_best_checkpoint",
#             },
#         )
#         workflow.add_conditional_edges(
#             "select_best_checkpoint",
#             self._route_by_budget,
#             {
#                 "stop": END,
#                 "continue": "business",
#             },
#         )
#         workflow.add_conditional_edges(
#             "market_retry",
#             self._route_after_market_retry,
#             {
#                 "stop": END,
#                 "repair_validator": "repair_validator",
#                 "validator": "validator",
#             },
#         )
#         workflow.add_conditional_edges(
#             "repair_validator",
#             self._route_after_repair_validation,
#             {
#                 "stop": END,
#                 "continue": "select_best_checkpoint",
#             },
#         )
#         direct_route_map = {"stop": END, "end": END}
#         if self.generate_pitch and self.pitch_agent is not None:
#             direct_route_map["pitch"] = "pitch"
#         if self.controller_policy == "adaptive" and self.direct_agent is not None:
#             workflow.add_conditional_edges(
#                 "direct",
#                 self._route_after_direct,
#                 direct_route_map,
#             )
#         business_route_map = {"stop": END, "end": END}
#         if self.generate_pitch and self.pitch_agent is not None:
#             business_route_map["pitch"] = "pitch"
#         workflow.add_conditional_edges(
#             "business",
#             self._route_after_business,
#             business_route_map,
#         )
#         if self.generate_pitch and self.pitch_agent is not None:
#             workflow.add_edge("pitch", END)

#         return workflow.compile(checkpointer=self.checkpointer)

#     def run(
#         self,
#         idea: str,
#         thread_id: str = "default-thread",
#         max_validation_retries: int = 1,
#         validation_threshold: int = 70,
#         max_tool_calls: int | None = None,
#         max_token_proxy: int | None = None,
#         max_total_tokens: int | None = None,
#         max_runtime_seconds: float | None = None,
#         forced_controller_mode: str | None = None,
#         adaptive_retry_enabled: bool = True,
#         adaptive_checkpoint_enabled: bool = True,
#         shared_refined_idea: str | None = None,
#         shared_refinement_token_usage: Dict[str, int] | None = None,
#         shared_refinement_tool_audit: list[Dict[str, Any]] | None = None,
#     ) -> PitchState:
#         normalized_max_tool_calls = (
#             None if max_tool_calls is None or max_tool_calls < 0 else int(max_tool_calls)
#         )
#         normalized_max_token_proxy = (
#             None if max_token_proxy is None or max_token_proxy < 0 else int(max_token_proxy)
#         )
#         normalized_max_total_tokens = (
#             None if max_total_tokens is None or max_total_tokens < 0 else int(max_total_tokens)
#         )
#         normalized_max_runtime_seconds = (
#             None
#             if max_runtime_seconds is None or max_runtime_seconds < 0
#             else float(max_runtime_seconds)
#         )

#         initial_state = PitchState(
#             idea=idea,
#             controller_policy=self.controller_policy,
#             forced_controller_mode=forced_controller_mode,
#             max_validation_retries=max_validation_retries,
#             validation_threshold=validation_threshold,
#             max_tool_calls=normalized_max_tool_calls,
#             max_token_proxy=normalized_max_token_proxy,
#             max_total_tokens=normalized_max_total_tokens,
#             max_runtime_seconds=normalized_max_runtime_seconds,
#             runtime_started_at=time.time(),
#             adaptive_retry_enabled=adaptive_retry_enabled,
#             adaptive_checkpoint_enabled=adaptive_checkpoint_enabled,
#         ).model_dump()
#         if shared_refined_idea:
#             initial_state["refined_idea"] = shared_refined_idea
#             initial_state["shared_refinement_locked"] = True
#             initial_state["token_usage"] = dict(shared_refinement_token_usage or {})
#             initial_state["tool_audit"] = list(shared_refinement_tool_audit or [])

#         return self.graph.invoke(
#             initial_state,
#             config={"configurable": {"thread_id": thread_id}},
#         )


#     def continue_adaptive_from_validated_state(
#         self,
#         state: PitchState | Dict[str, Any],
#         *,
#         adaptive_retry_enabled: bool = True,
#         adaptive_checkpoint_enabled: bool = True,
#     ) -> Dict[str, Any]:
#         """Continue an adaptive-no-retry state with retry/checkpoint logic only."""
#         current = dict(state if isinstance(state, dict) else state.model_dump())
#         validation = current.get("validation_report", {}) or {}
#         threshold = int(current.get("validation_threshold", 70) or 70)
#         current_score = self._validation_score(validation)
#         current.update({
#             "adaptive_retry_enabled": adaptive_retry_enabled,
#             "adaptive_checkpoint_enabled": adaptive_checkpoint_enabled,
#             "needs_revision": current_score < threshold,
#             "retry_budget_decisions": [],
#             "selected_validation_checkpoint": {},
#             "adaptive_checkpoint_selection": {},
#         })
#         retry_updates = self._run_retry_gate(current)
#         current.update(retry_updates)
#         if bool(current.get("budget_hit", False)):
#             return current
#         last = self._last_retry_decision(current)
#         if isinstance(last, dict) and last.get("retry_allowed", False):
#             retry_updates = self._market_retry(current)
#             current.update(retry_updates)
#             if bool(current.get("budget_hit", False)):
#                 return current
#             if last.get("retry_type") == "claim_micro_repair":
#                 val_updates = self._run_lightweight_repair_validator(current)
#             else:
#                 val_updates = self._run_validator(current)
#             current.update(val_updates)
#             if bool(current.get("budget_hit", False)):
#                 return current
#         select_updates = self._run_select_best_checkpoint(current)
#         current.update(select_updates)
#         if bool(current.get("budget_hit", False)):
#             return current
#         biz_updates = self._run_business_with_depth(current)
#         current.update(biz_updates)
#         if self.generate_pitch and self.pitch_agent is not None and not bool(current.get("budget_hit", False)):
#             pitch_updates = self._run_pitch(current)
#             current.update(pitch_updates)
#         return current



















# # from __future__ import annotations

# # import json
# # import time
# # from typing import Any, Dict

# # from langgraph.checkpoint.memory import MemorySaver
# # from langgraph.graph import END, START, StateGraph
# # from langchain_openai import ChatOpenAI

# # from startup_pitch_refinery.agents import (
# #     AdaptiveControllerAgent,
# #     BusinessModelAgent,
# #     ClaimRepairAgent,
# #     DirectStrategyAgent,
# #     IdeaRefinementAgent,
# #     MarketResearchAgent,
# #     PitchDeckGeneratorAgent,
# #     SourceValidatorAgent,
# #     SupervisorAgent,
# #     build_claim_repair_plan,
# # )
# # from startup_pitch_refinery.state import PitchState


# # class StartupPitchRefinery:
# #     def __init__(
# #         self,
# #         model: str = "gpt-4.1-nano",
# #         secondary_judge_model: str | None = None,
# #         temperature: float = 0.0,
# #         seed: int = 42,
# #         strict_tools: bool = True,
# #         enable_trends: bool = True,
# #         generate_pitch: bool = True,
# #         controller_policy: str = "fixed",
# #         output_dir: str = "output",
# #     ):
# #         llm = ChatOpenAI(model=model, temperature=temperature, seed=seed)
# #         secondary_judge_llm = (
# #             ChatOpenAI(model=secondary_judge_model, temperature=temperature, seed=seed + 101)
# #             if secondary_judge_model
# #             else None
# #         )
# #         self.generate_pitch = generate_pitch
# #         self.controller_policy = controller_policy.strip().lower()
# #         if self.controller_policy not in {"fixed", "adaptive"}:
# #             raise ValueError(
# #                 f"Unsupported controller_policy `{controller_policy}`. Allowed: fixed, adaptive."
# #             )

# #         self.supervisor = SupervisorAgent()
# #         self.idea_agent = IdeaRefinementAgent(llm)
# #         self.controller_agent = (
# #             AdaptiveControllerAgent(llm) if self.controller_policy == "adaptive" else None
# #         )
# #         self.market_agent = MarketResearchAgent(
# #             llm,
# #             strict_tools=strict_tools,
# #             enable_trends=enable_trends,
# #         )
# #         self.claim_repair_agent = ClaimRepairAgent(llm, strict_tools=strict_tools)
# #         self.validator_agent = SourceValidatorAgent(
# #             llm,
# #             secondary_judge_llm=secondary_judge_llm,
# #         )
# #         self.direct_agent = (
# #             DirectStrategyAgent(
# #                 llm,
# #                 strict_tools=strict_tools,
# #                 enable_trends=enable_trends,
# #                 secondary_judge_llm=secondary_judge_llm,
# #             )
# #             if self.controller_policy == "adaptive"
# #             else None
# #         )
# #         self.business_agent = BusinessModelAgent(llm, strict_tools=strict_tools)
# #         self.pitch_agent = (
# #             PitchDeckGeneratorAgent(llm, output_dir=output_dir)
# #             if self.generate_pitch
# #             else None
# #         )
# #         self.checkpointer = MemorySaver()

# #         self.graph = self._build_graph()

# #     @staticmethod
# #     def _sget(state: PitchState | Dict[str, Any], key: str, default: Any = None) -> Any:
# #         if isinstance(state, dict):
# #             return state.get(key, default)
# #         return getattr(state, key, default)

# #     @staticmethod
# #     def _estimate_tokens_proxy(state: Dict[str, Any]) -> int:
# #         chunks = [
# #             str(state.get("idea", "")),
# #             str(state.get("refined_idea", "")),
# #             str(state.get("market_analysis", "")),
# #             str(state.get("business_model", "")),
# #             str(state.get("validated_market_analysis", "")),
# #             json.dumps(state.get("pitch_content", {}), ensure_ascii=True),
# #             json.dumps(state.get("trend_signals", {}), ensure_ascii=True),
# #             json.dumps(state.get("validation_report", {}), ensure_ascii=True),
# #         ]
# #         total_chars = sum(len(c) for c in chunks)
# #         return max(1, total_chars // 4)

# #     def _compute_budget_updates(self, merged: Dict[str, Any]) -> Dict[str, Any]:
# #         max_tool_calls = self._sget(merged, "max_tool_calls")
# #         max_token_proxy = self._sget(merged, "max_token_proxy")
# #         max_total_tokens = self._sget(merged, "max_total_tokens")
# #         max_runtime_seconds = self._sget(merged, "max_runtime_seconds")

# #         tool_calls_current = len(self._sget(merged, "tool_audit", []))
# #         token_proxy_current = self._estimate_tokens_proxy(merged)
# #         token_usage = self._sget(merged, "token_usage", {}) or {}
# #         prompt_tokens_current = int(token_usage.get("prompt_tokens", 0) or 0)
# #         completion_tokens_current = int(token_usage.get("completion_tokens", 0) or 0)
# #         total_tokens_current = int(token_usage.get("total_tokens", 0) or 0)
# #         started_at = self._sget(merged, "runtime_started_at")
# #         if isinstance(started_at, (int, float)) and started_at > 0:
# #             runtime_elapsed_seconds = max(0.0, time.time() - float(started_at))
# #         else:
# #             runtime_elapsed_seconds = 0.0

# #         reasons = list(self._sget(merged, "budget_hit_reasons", []))
# #         budget_hit = bool(self._sget(merged, "budget_hit", False))

# #         if isinstance(max_tool_calls, int) and max_tool_calls >= 0:
# #             if tool_calls_current > max_tool_calls:
# #                 budget_hit = True
# #                 reasons.append(
# #                     f"max_tool_calls_exceeded:{tool_calls_current}>{max_tool_calls}"
# #                 )
# #         if isinstance(max_token_proxy, int) and max_token_proxy >= 0:
# #             if token_proxy_current > max_token_proxy:
# #                 budget_hit = True
# #                 reasons.append(
# #                     f"max_token_proxy_exceeded:{token_proxy_current}>{max_token_proxy}"
# #                 )
# #         if isinstance(max_total_tokens, int) and max_total_tokens >= 0:
# #             if total_tokens_current > max_total_tokens:
# #                 budget_hit = True
# #                 reasons.append(
# #                     f"max_total_tokens_exceeded:{total_tokens_current}>{max_total_tokens}"
# #                 )
# #         if isinstance(max_runtime_seconds, (int, float)) and max_runtime_seconds >= 0:
# #             if runtime_elapsed_seconds > float(max_runtime_seconds):
# #                 budget_hit = True
# #                 reasons.append(
# #                     f"max_runtime_seconds_exceeded:{runtime_elapsed_seconds:.3f}>{float(max_runtime_seconds):.3f}"
# #                 )

# #         dedup_reasons = []
# #         seen = set()
# #         for r in reasons:
# #             if r not in seen:
# #                 dedup_reasons.append(r)
# #                 seen.add(r)

# #         remaining = {
# #             "tool_calls": (
# #                 None
# #                 if not isinstance(max_tool_calls, int) or max_tool_calls < 0
# #                 else max_tool_calls - tool_calls_current
# #             ),
# #             "token_proxy": (
# #                 None
# #                 if not isinstance(max_token_proxy, int) or max_token_proxy < 0
# #                 else max_token_proxy - token_proxy_current
# #             ),
# #             "total_tokens": (
# #                 None
# #                 if not isinstance(max_total_tokens, int) or max_total_tokens < 0
# #                 else max_total_tokens - total_tokens_current
# #             ),
# #             "runtime_seconds": (
# #                 None
# #                 if not isinstance(max_runtime_seconds, (int, float)) or max_runtime_seconds < 0
# #                 else float(max_runtime_seconds) - runtime_elapsed_seconds
# #             ),
# #         }

# #         return {
# #             "runtime_elapsed_seconds": round(runtime_elapsed_seconds, 3),
# #             "tool_calls_current": tool_calls_current,
# #             "token_proxy_current": token_proxy_current,
# #             "prompt_tokens_current": prompt_tokens_current,
# #             "completion_tokens_current": completion_tokens_current,
# #             "total_tokens_current": total_tokens_current,
# #             "budget_hit": budget_hit,
# #             "budget_hit_reasons": dedup_reasons,
# #             "budget_remaining": remaining,
# #         }

# #     def _run_node_with_budget(
# #         self,
# #         state: PitchState | Dict[str, Any],
# #         runner,
# #         node_name: str,
# #     ) -> Dict[str, Any]:
# #         base_state = state if isinstance(state, dict) else state.model_dump()
# #         precheck = self._compute_budget_updates(base_state)
# #         if precheck["budget_hit"]:
# #             return {
# #                 **precheck,
# #                 "controller_rationale": self._sget(base_state, "controller_rationale"),
# #             }

# #         enriched_state = {**base_state, **precheck}
# #         updates = runner(enriched_state)
# #         merged = {**enriched_state, **updates}
# #         postcheck = self._compute_budget_updates(merged)
# #         if postcheck["budget_hit"]:
# #             audit = list(self._sget(merged, "tool_audit", []))
# #             audit.append(
# #                 {
# #                     "agent": "budget_guard",
# #                     "tool": "runtime_budget_check",
# #                     "status": "halted",
# #                     "node": node_name,
# #                     "reasons": postcheck["budget_hit_reasons"],
# #                 }
# #             )
# #             postcheck["tool_audit"] = audit
# #         return {**updates, **postcheck}

# #     def _market_retry(self, state: PitchState):
# #         current_retry = self._sget(state, "retry_count", 0)
# #         base_state = state if isinstance(state, dict) else state.model_dump()
# #         repair_context = self._build_market_repair_context(base_state)
# #         retry_state = {
# #             **base_state,
# #             "retry_count": current_retry + 1,
# #             "market_repair_context": repair_context,
# #         }
# #         weak_claim_count = len(repair_context.get("weak_or_unsupported_claims", []))
# #         has_repair_target = weak_claim_count > 0
# #         repair_runner = (
# #             self.claim_repair_agent.run if has_repair_target else self.market_agent.run
# #         )
# #         repair_node = "claim_micro_repair" if has_repair_target else "market_retry"
# #         updates = self._run_node_with_budget(retry_state, repair_runner, repair_node)
# #         updates["retry_count"] = current_retry + 1
# #         updates["market_repair_context"] = repair_context
# #         repair_history = list(self._sget(base_state, "market_repair_history", []) or [])
# #         repair_history.append(
# #             {
# #                 "retry_count": current_retry + 1,
# #                 "repair_node": repair_node,
# #                 "weak_claim_count": weak_claim_count,
# #                 "supported_claim_count": len(repair_context.get("supported_claims", [])),
# #                 "previous_validation_score": repair_context.get("previous_validation_score"),
# #                 "previous_supported_ratio": repair_context.get("previous_supported_ratio"),
# #                 "evidence_gap_signal": bool(repair_context.get("evidence_gap_signal", False)),
# #                 "repair_strategy": repair_context.get("repair_strategy", ""),
# #                 "repair_action_counts": repair_context.get("repair_action_counts", {}),
# #             }
# #         )
# #         updates["market_repair_history"] = repair_history
# #         return updates

# #     def _build_market_repair_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
# #         validation = self._sget(state, "validation_report", {}) or {}
# #         claims = validation.get("claims", []) if isinstance(validation, dict) else []
# #         weak_claims = []
# #         supported_claims = []
# #         claim_units = validation.get("claim_units", []) if isinstance(validation, dict) else []
# #         claim_source = claim_units if claim_units else claims
# #         for claim in claim_source or []:
# #             if not isinstance(claim, dict):
# #                 continue
# #             text = str(claim.get("claim") or claim.get("claim_text") or "").strip()
# #             if not text:
# #                 continue
# #             verdict = str(claim.get("verdict", "")).strip().lower()
# #             item = {
# #                 "claim_id": str(claim.get("claim_id") or f"c{len(weak_claims) + len(supported_claims) + 1}"),
# #                 "claim": text,
# #                 "claim_text": text,
# #                 "verdict": verdict,
# #                 "confidence": claim.get("confidence"),
# #                 "category": claim.get("category"),
# #                 "materiality": claim.get("materiality"),
# #                 "failure_type": claim.get("failure_type") or claim.get("failure_type_hint"),
# #                 "candidate_queries": claim.get("candidate_queries", []),
# #                 "rationale": str(claim.get("rationale", "")).strip()[:300],
# #                 "supporting_sources": claim.get("supporting_sources", []),
# #             }
# #             if verdict == "supported":
# #                 supported_claims.append(item)
# #             elif verdict in {"weakly_supported", "unsupported", "needs_review"}:
# #                 weak_claims.append(item)

# #         evidence_gaps = (
# #             str(validation.get("evidence_gaps", "") or "").strip()
# #             if isinstance(validation, dict)
# #             else ""
# #         )
# #         evidence_gap_signal = bool(
# #             evidence_gaps
# #             and any(
# #                 marker in evidence_gaps.lower()
# #                 for marker in ["lack", "limited", "gap", "weak", "indirect", "not specific"]
# #             )
# #         )
# #         repair_plan = build_claim_repair_plan(weak_claims)
# #         repair_action_counts = {
# #             "search_and_replace": sum(
# #                 1 for item in repair_plan if item.get("action") == "search_and_replace"
# #             ),
# #             "qualify_or_remove": sum(
# #                 1 for item in repair_plan if item.get("action") == "qualify_or_remove"
# #             ),
# #             "remove": sum(1 for item in repair_plan if item.get("action") == "remove"),
# #         }
# #         return {
# #             "repair_strategy": (
# #                 "targeted_claim_repair: preserve supported claims, repair or remove weak "
# #                 "claims, and avoid broad market-analysis rewrites"
# #             ),
# #             "previous_validation_score": self._validation_score(validation),
# #             "previous_supported_ratio": round(self._validation_supported_ratio(validation), 4),
# #             "previous_market_analysis": self._sget(state, "market_analysis", "") or "",
# #             "previous_market_sources": list(self._sget(state, "market_sources", []) or []),
# #             "evidence_gaps": evidence_gaps,
# #             "evidence_gap_signal": evidence_gap_signal,
# #             "weak_or_unsupported_claims": weak_claims[:8],
# #             "supported_claims": supported_claims[:8],
# #             "repair_plan": repair_plan[:8],
# #             "repair_action_counts": repair_action_counts,
# #         }

# #     def _run_supervisor(self, state: PitchState):
# #         return self._run_node_with_budget(state, self.supervisor.run, "supervisor")

# #     def _run_idea(self, state: PitchState):
# #         if bool(self._sget(state, "shared_refinement_locked", False)) and self._sget(
# #             state, "refined_idea"
# #         ):
# #             return {}
# #         return self._run_node_with_budget(state, self.idea_agent.run, "idea")

# #     def _run_controller(self, state: PitchState):
# #         if self.controller_agent is None:
# #             return {}
# #         return self._run_node_with_budget(state, self.controller_agent.run, "controller")

# #     def _run_market(self, state: PitchState):
# #         return self._run_node_with_budget(state, self.market_agent.run, "market")

# #     def _run_validator(self, state: PitchState):
# #         base_state = state if isinstance(state, dict) else state.model_dump()
# #         cascade_context = self._repair_validation_cascade_context(base_state)
# #         updates = self._run_node_with_budget(base_state, self.validator_agent.run, "validator")
# #         if cascade_context.get("eligible"):
# #             validation = updates.get("validation_report", {}) or {}
# #             tool_audit = list(updates.get("tool_audit", self._sget(base_state, "tool_audit", [])) or [])
# #             tool_audit.append(
# #                 {
# #                     "agent": "adaptive_controller",
# #                     "tool": "repair_validator_cascade",
# #                     "status": "escalated_to_full_dual_judge",
# #                     "reason": cascade_context.get("reason", ""),
# #                     "previous_reliability_score": cascade_context.get("previous_score", 0),
# #                     "lightweight_candidate_score": cascade_context.get("candidate_score", 0),
# #                     "full_validation_score": self._validation_score(validation),
# #                     "retry_expected_gain": cascade_context.get("retry_expected_gain", 0.0),
# #                     "retry_roi": cascade_context.get("retry_roi", 0.0),
# #                 }
# #             )
# #             updates["tool_audit"] = tool_audit
# #         merged = {**base_state, **updates}
# #         snapshot = self._build_validation_snapshot(merged)
# #         if cascade_context.get("eligible"):
# #             snapshot["validation_mode"] = "full_dual_judge_after_lightweight_repair"
# #         snapshots = list(self._sget(base_state, "validation_snapshots", []) or [])
# #         snapshots.append(snapshot)
# #         updates["validation_snapshots"] = snapshots
# #         updates["checkpoint_history"] = snapshots
# #         return updates

# #     def _run_lightweight_repair_validator(self, state: PitchState):
# #         base_state = state if isinstance(state, dict) else state.model_dump()
# #         updates = self._run_node_with_budget(
# #             base_state,
# #             self.validator_agent.run_repair_only,
# #             "repair_validator",
# #         )
# #         merged = {**base_state, **updates}
# #         snapshot = self._build_validation_snapshot(merged)
# #         snapshot["validation_mode"] = "lightweight_repair"
# #         snapshots = list(self._sget(base_state, "validation_snapshots", []) or [])
# #         snapshots.append(snapshot)
# #         updates["validation_snapshots"] = snapshots
# #         updates["checkpoint_history"] = snapshots
# #         return updates

# #     def _run_direct(self, state: PitchState):
# #         if self.direct_agent is None:
# #             return {}
# #         return self._run_node_with_budget(state, self.direct_agent.run, "direct")

# #     def _run_pitch(self, state: PitchState):
# #         if self.pitch_agent is None:
# #             return {}
# #         return self._run_node_with_budget(state, self.pitch_agent.run, "pitch")

# #     @staticmethod
# #     def _validation_supported_ratio(validation: Dict[str, Any]) -> float:
# #         claims = validation.get("claims", []) if isinstance(validation, dict) else []
# #         if not claims:
# #             return 0.0
# #         supported = 0
# #         for claim in claims:
# #             if not isinstance(claim, dict):
# #                 continue
# #             verdict = str(claim.get("verdict", "")).strip().lower()
# #             if verdict == "supported":
# #                 supported += 1
# #         return supported / max(1, len(claims))

# #     @staticmethod
# #     def _validation_score(validation: Dict[str, Any]) -> int:
# #         if not isinstance(validation, dict):
# #             return 0
# #         try:
# #             return int(validation.get("reliability_score", 0) or 0)
# #         except (TypeError, ValueError):
# #             return 0

# #     def _build_validation_snapshot(self, state: Dict[str, Any]) -> Dict[str, Any]:
# #         validation = self._sget(state, "validation_report", {}) or {}
# #         agreement = (
# #             validation.get("agreement_stats", {}).get("overall_agreement", 0.0)
# #             if isinstance(validation, dict)
# #             else 0.0
# #         )
# #         token_usage = self._sget(state, "token_usage", {}) or {}
# #         return {
# #             "checkpoint_index": len(self._sget(state, "validation_snapshots", []) or []),
# #             "retry_count": int(self._sget(state, "retry_count", 0) or 0),
# #             "controller_mode": self._sget(state, "controller_mode"),
# #             "reliability_score": self._validation_score(validation),
# #             "judge_agreement": float(agreement or 0.0),
# #             "supported_ratio": round(self._validation_supported_ratio(validation), 4),
# #             "total_tokens_at_checkpoint": int(token_usage.get("total_tokens", 0) or 0),
# #             "market_analysis": self._sget(state, "market_analysis"),
# #             "validated_market_analysis": self._sget(state, "validated_market_analysis"),
# #             "validation_report": validation,
# #             "market_sources": list(self._sget(state, "market_sources", []) or []),
# #             "market_evidence": list(self._sget(state, "market_evidence", []) or []),
# #             "trend_signals": dict(self._sget(state, "trend_signals", {}) or {}),
# #         }

# #     def _run_select_best_checkpoint(self, state: PitchState) -> Dict[str, Any]:
# #         base_state = state if isinstance(state, dict) else state.model_dump()
# #         forced_mode = str(self._sget(base_state, "forced_controller_mode", "") or "").strip().lower()
# #         if forced_mode in {"direct", "shallow", "recursive"}:
# #             return {}
# #         checkpoint_enabled = bool(self._sget(base_state, "adaptive_checkpoint_enabled", True))
# #         snapshots = [
# #             item
# #             for item in (self._sget(base_state, "validation_snapshots", []) or [])
# #             if isinstance(item, dict)
# #         ]
# #         if not snapshots:
# #             return {}

# #         def rank(snapshot: Dict[str, Any]) -> tuple[float, float, float, int]:
# #             return (
# #                 float(snapshot.get("reliability_score", 0) or 0),
# #                 float(snapshot.get("judge_agreement", 0.0) or 0.0),
# #                 float(snapshot.get("supported_ratio", 0.0) or 0.0),
# #                 -int(snapshot.get("total_tokens_at_checkpoint", 0) or 0),
# #             )

# #         current = snapshots[-1]
# #         best = max(snapshots, key=rank) if checkpoint_enabled else current
# #         best_score = int(best.get("reliability_score", 0) or 0)
# #         current_score = int(current.get("reliability_score", 0) or 0)
# #         selected_previous = best.get("checkpoint_index") != current.get("checkpoint_index")
# #         threshold = int(self._sget(base_state, "validation_threshold", 70) or 70)
# #         selection = {
# #             "selected_checkpoint_index": best.get("checkpoint_index"),
# #             "selected_retry_count": best.get("retry_count"),
# #             "selected_previous_checkpoint": bool(selected_previous),
# #             "best_validation_score": best_score,
# #             "current_validation_score_before_selection": current_score,
# #             "score_delta_vs_current": best_score - current_score,
# #             "checkpoint_enabled": checkpoint_enabled,
# #             "selection_rule": (
# #                 "highest reliability score, then judge agreement, supported ratio, "
# #                 "then lower token cost"
# #                 if checkpoint_enabled
# #                 else "checkpoint rollback disabled; latest validation state retained"
# #             ),
# #         }
# #         tool_audit = list(self._sget(base_state, "tool_audit", []) or [])
# #         tool_audit.append(
# #             {
# #                 "agent": "adaptive_controller",
# #                 "tool": "checkpoint_selector",
# #                 "status": "ok",
# #                 **selection,
# #             }
# #         )

# #         updates: Dict[str, Any] = {
# #             "selected_validation_checkpoint": best,
# #             "selected_checkpoint": best,
# #             "adaptive_checkpoint_selection": selection,
# #             "tool_audit": tool_audit,
# #             # The retry gate has already finished, so the selected checkpoint is
# #             # the final available validation state for this run.
# #             "needs_revision": False,
# #         }
# #         if selected_previous:
# #             updates.update(
# #                 {
# #                     "market_analysis": best.get("market_analysis"),
# #                     "validated_market_analysis": best.get("validated_market_analysis"),
# #                     "validation_report": best.get("validation_report"),
# #                     "market_sources": best.get("market_sources", []),
# #                     "market_evidence": best.get("market_evidence", []),
# #                     "trend_signals": best.get("trend_signals", {}),
# #                 }
# #             )
# #         elif best_score < threshold:
# #             updates["needs_revision"] = False
# #         return updates

# #     def _route_after_validation(self, state: PitchState) -> str:
# #         if bool(self._sget(state, "budget_hit", False)):
# #             return "stop"
# #         return "retry_gate"

# #     def _should_use_lightweight_repair_validation(
# #         self, state: PitchState | Dict[str, Any]
# #     ) -> bool:
# #         decisions = self._sget(state, "retry_budget_decisions", []) or []
# #         last_decision = decisions[-1] if decisions else {}
# #         if not (
# #             isinstance(last_decision, dict)
# #             and last_decision.get("retry_type") == "claim_micro_repair"
# #         ):
# #             return False

# #         repair_history = self._sget(state, "market_repair_history", []) or []
# #         last_repair = repair_history[-1] if repair_history else {}
# #         action_counts = {}
# #         if isinstance(last_repair, dict):
# #             action_counts = last_repair.get("repair_action_counts", {}) or {}
# #         if not isinstance(action_counts, dict):
# #             return False
# #         search_count = int(action_counts.get("search_and_replace", 0) or 0)
# #         conservative_count = int(action_counts.get("qualify_or_remove", 0) or 0) + int(
# #             action_counts.get("remove", 0) or 0
# #         )
# #         return search_count == 0 and conservative_count > 0

# #     def _route_after_market_retry(self, state: PitchState) -> str:
# #         if bool(self._sget(state, "budget_hit", False)):
# #             return "stop"
# #         if self._should_use_lightweight_repair_validation(state):
# #             return "repair_validator"
# #         return "validator"

# #     def _repair_validation_cascade_context(
# #         self, state: PitchState | Dict[str, Any]
# #     ) -> Dict[str, Any]:
# #         """Return escalation context after a lightweight repair validator run.

# #         The cascade keeps conservative repairs cheap when the lightweight judge
# #         accepts them, but falls back to full dual-judge validation when the
# #         repair is rejected despite a high expected retry value.
# #         """
# #         validation = self._sget(state, "validation_report", {}) or {}
# #         if not isinstance(validation, dict):
# #             return {"eligible": False}
# #         repair_validation = validation.get("repair_validation", {}) or {}
# #         if not isinstance(repair_validation, dict):
# #             return {"eligible": False}
# #         judge_scores = validation.get("judge_scores", {}) or {}
# #         aggregated = (
# #             judge_scores.get("aggregated", {}) if isinstance(judge_scores, dict) else {}
# #         )
# #         if not bool(aggregated.get("lightweight_repair_validation", False)):
# #             return {"eligible": False}
# #         if bool(repair_validation.get("accepted", False)) or bool(
# #             aggregated.get("repair_patch_accepted", False)
# #         ):
# #             return {"eligible": False}

# #         decisions = self._sget(state, "retry_budget_decisions", []) or []
# #         last_decision = decisions[-1] if decisions else {}
# #         if not (
# #             isinstance(last_decision, dict)
# #             and last_decision.get("retry_type") == "claim_micro_repair"
# #             and bool(last_decision.get("retry_allowed", False))
# #         ):
# #             return {"eligible": False}

# #         previous_score = int(repair_validation.get("previous_reliability_score", 0) or 0)
# #         candidate_score = int(repair_validation.get("candidate_reliability_score", 0) or 0)
# #         threshold = int(self._sget(state, "validation_threshold", 70) or 70)
# #         deficit = max(0, threshold - previous_score)
# #         try:
# #             retry_expected_gain = float(last_decision.get("retry_expected_gain", 0.0) or 0.0)
# #         except (TypeError, ValueError):
# #             retry_expected_gain = 0.0
# #         try:
# #             retry_roi = float(last_decision.get("retry_roi", 0.0) or 0.0)
# #         except (TypeError, ValueError):
# #             retry_roi = 0.0

# #         # Escalate only when the original validation is materially below the
# #         # target and the retry gate predicted enough value to justify a full
# #         # dual-judge pass. A small candidate-score miss is acceptable here: the
# #         # point of the cascade is to let the stronger evaluator arbitrate
# #         # borderline or conservative claim rewrites.
# #         quality_need = deficit >= 7 or previous_score < 75
# #         promising_retry = retry_expected_gain >= 6.0 and retry_roi >= 1.0
# #         not_clearly_harmful = candidate_score >= previous_score - 8
# #         if not (quality_need and promising_retry and not_clearly_harmful):
# #             return {"eligible": False}

# #         if not self._has_budget_for_full_repair_validation(state):
# #             return {"eligible": False}

# #         return {
# #             "eligible": True,
# #             "reason": "lightweight_repair_rejected_but_retry_value_remains_high",
# #             "previous_score": previous_score,
# #             "candidate_score": candidate_score,
# #             "retry_expected_gain": round(retry_expected_gain, 4),
# #             "retry_roi": round(retry_roi, 4),
# #         }

# #     def _has_budget_for_full_repair_validation(
# #         self, state: PitchState | Dict[str, Any]
# #     ) -> bool:
# #         remaining = self._sget(state, "budget_remaining", {}) or {}
# #         if not isinstance(remaining, dict):
# #             return True
# #         remaining_total_tokens = remaining.get("total_tokens")
# #         remaining_tool_calls = remaining.get("tool_calls")
# #         remaining_runtime = remaining.get("runtime_seconds")

# #         audits = self._sget(state, "tool_audit", []) or []
# #         prior_dual_validator_tokens = [
# #             int(audit.get("total_tokens", 0) or 0)
# #             for audit in audits
# #             if isinstance(audit, dict) and audit.get("tool") == "llm_claim_verifier_dual_judge"
# #         ]
# #         estimated_full_validation_tokens = (
# #             prior_dual_validator_tokens[-1] if prior_dual_validator_tokens else 4500
# #         )
# #         estimated_full_validation_tokens = max(
# #             3000, min(7000, estimated_full_validation_tokens)
# #         )
# #         downstream_reserve_tokens = 900 if not self.generate_pitch else 2600
# #         if (
# #             isinstance(remaining_total_tokens, int)
# #             and remaining_total_tokens
# #             < estimated_full_validation_tokens + downstream_reserve_tokens
# #         ):
# #             return False
# #         if isinstance(remaining_tool_calls, int) and remaining_tool_calls < 2:
# #             return False
# #         if isinstance(remaining_runtime, (int, float)) and float(remaining_runtime) < (
# #             25.0 if not self.generate_pitch else 45.0
# #         ):
# #             return False
# #         return True

# #     def _route_after_repair_validation(self, state: PitchState) -> str:
# #         if bool(self._sget(state, "budget_hit", False)):
# #             return "stop"
# #         if self._repair_validation_cascade_context(state).get("eligible"):
# #             return "validator"
# #         return "retry_gate"

# #     def _run_retry_gate(self, state: PitchState) -> Dict[str, Any]:
# #         base_state = state if isinstance(state, dict) else state.model_dump()
# #         controller_mode = str(self._sget(state, "controller_mode", "")).strip().lower()
# #         forced_mode = str(self._sget(state, "forced_controller_mode", "") or "").strip().lower()
# #         fixed_mode_baseline = forced_mode in {"direct", "shallow", "recursive"}
# #         retry_enabled = bool(self._sget(state, "adaptive_retry_enabled", True))
# #         needs_revision = bool(self._sget(state, "needs_revision", False))
# #         retry_count = int(self._sget(state, "retry_count", 0))
# #         max_retries = int(self._sget(state, "max_validation_retries", 1))
# #         validation = self._sget(state, "validation_report", {}) or {}
# #         validation_score = 0
# #         if isinstance(validation, dict):
# #             try:
# #                 validation_score = int(validation.get("reliability_score", 0) or 0)
# #             except (TypeError, ValueError):
# #                 validation_score = 0
# #         agreement_stats = (
# #             validation.get("agreement_stats", {}) if isinstance(validation, dict) else {}
# #         )
# #         judge_agreement = float(agreement_stats.get("overall_agreement", 0.0) or 0.0)
# #         supported_ratio = self._validation_supported_ratio(validation)
# #         weak_or_unsupported_claim_count = 0
# #         weak_claim_count = 0
# #         unsupported_claim_count = 0
# #         needs_review_claim_count = 0
# #         material_failing_claim_count = 0
# #         specific_repair_claim_count = 0
# #         claims_total = 0
# #         weak_claim_items = []
# #         low_claim_count_flag = False
# #         claim_count_penalty = 0
# #         if isinstance(validation, dict):
# #             claims = validation.get("claim_units", []) or validation.get("claims", []) or []
# #             claims_total = len(claims)
# #             aggregated_scores = (
# #                 validation.get("judge_scores", {}).get("aggregated", {})
# #                 if isinstance(validation.get("judge_scores", {}), dict)
# #                 else {}
# #             )
# #             low_claim_count_flag = bool(aggregated_scores.get("low_claim_count_flag", False))
# #             try:
# #                 claim_count_penalty = int(aggregated_scores.get("claim_count_penalty", 0) or 0)
# #             except (TypeError, ValueError):
# #                 claim_count_penalty = 0
# #             for claim in claims:
# #                 if not isinstance(claim, dict):
# #                     continue
# #                 verdict = str(claim.get("verdict", "")).strip().lower()
# #                 if verdict in {"weakly_supported", "unsupported", "needs_review"}:
# #                     weak_or_unsupported_claim_count += 1
# #                     weak_claim_items.append(claim)
# #                     try:
# #                         materiality = int(claim.get("materiality", 1) or 1)
# #                     except (TypeError, ValueError):
# #                         materiality = 1
# #                     if materiality >= 4:
# #                         material_failing_claim_count += 1
# #                     failure_type = str(
# #                         claim.get("failure_type") or claim.get("failure_type_hint") or ""
# #                     ).strip().lower()
# #                     if failure_type not in {"", "none", "too_broad", "unclear_attribution"}:
# #                         specific_repair_claim_count += 1
# #                 if verdict == "weakly_supported":
# #                     weak_claim_count += 1
# #                 elif verdict == "unsupported":
# #                     unsupported_claim_count += 1
# #                 elif verdict == "needs_review":
# #                     needs_review_claim_count += 1
# #         validation_threshold = int(self._sget(state, "validation_threshold", 70) or 70)
# #         validation_deficit = max(0, validation_threshold - validation_score)
# #         budget_remaining = self._sget(state, "budget_remaining", {}) or {}
# #         remaining_total_tokens = (
# #             budget_remaining.get("total_tokens")
# #             if isinstance(budget_remaining, dict)
# #             else None
# #         )
# #         remaining_tool_calls = (
# #             budget_remaining.get("tool_calls")
# #             if isinstance(budget_remaining, dict)
# #             else None
# #         )
# #         remaining_runtime = (
# #             budget_remaining.get("runtime_seconds")
# #             if isinstance(budget_remaining, dict)
# #             else None
# #         )
# #         controller_scorecard = self._sget(state, "controller_scorecard", {}) or {}
# #         controller_features = (
# #             controller_scorecard.get("features", {})
# #             if isinstance(controller_scorecard, dict)
# #             else {}
# #         )
# #         recursive_cost_efficient = bool(
# #             controller_scorecard.get("recursive_cost_efficient", False)
# #         )
# #         try:
# #             controller_structural_complexity = float(
# #                 controller_features.get("structural_complexity", 0.0) or 0.0
# #             )
# #         except (TypeError, ValueError):
# #             controller_structural_complexity = 0.0

# #         # A retry means another targeted market pass, another dual-judge
# #         # validation, and enough reserve to reach business-model generation.
# #         # Fixed baselines stay fixed-depth; only adaptive runs may use
# #         # feedback-driven repair/escalation.
# #         estimated_retry_tokens = self._estimate_retry_tokens(base_state)
# #         if validation_deficit >= 8:
# #             business_reserve_tokens = 2200
# #         elif validation_deficit >= 4:
# #             business_reserve_tokens = 2600
# #         else:
# #             business_reserve_tokens = 3000
# #         if not self.generate_pitch:
# #             # Compare-mode methodology runs usually skip deck generation. Keep
# #             # enough room for business-model synthesis without blocking the
# #             # recursive evidence-repair loop unnecessarily.
# #             business_reserve_tokens = min(business_reserve_tokens, 900)
# #         safety_buffer_tokens = 500 if not self.generate_pitch else 900
# #         estimated_total_needed = (
# #             estimated_retry_tokens + business_reserve_tokens + safety_buffer_tokens
# #         )
# #         estimated_retry_tool_calls = 4
# #         business_reserve_tool_calls = 1 if not self.generate_pitch else 2
# #         estimated_tool_calls_needed = estimated_retry_tool_calls + business_reserve_tool_calls
# #         estimated_runtime_needed = 35.0 if not self.generate_pitch else 55.0

# #         shallow_escalation = (
# #             controller_mode == "shallow"
# #             and not fixed_mode_baseline
# #             and needs_revision
# #             and validation_deficit >= 3
# #             and retry_count < max_retries
# #         )
# #         evidence_gaps = ""
# #         if isinstance(validation, dict):
# #             evidence_gaps = str(validation.get("evidence_gaps", "") or "").strip()
# #         evidence_gap_signal = (
# #             validation_deficit > 0
# #             and len(evidence_gaps) >= 20
# #             and any(
# #                 marker in evidence_gaps.lower()
# #                 for marker in ["lack", "limited", "gap", "weak", "indirect", "not specific"]
# #             )
# #         )
# #         severe_revision_need = (
# #             validation_deficit >= 9
# #             or judge_agreement < 0.82
# #             or supported_ratio < 0.68
# #             or unsupported_claim_count > 0
# #             or needs_review_claim_count >= 2
# #         )
# #         corroborated_revision_need = (
# #             validation_deficit >= 5
# #             and (
# #                 supported_ratio < 0.75
# #                 or judge_agreement < 0.86
# #                 or weak_or_unsupported_claim_count >= 2
# #                 or evidence_gap_signal
# #                 or low_claim_count_flag
# #             )
# #         )
# #         material_revision_need = severe_revision_need or corroborated_revision_need
# #         concrete_claim_repair_target = weak_or_unsupported_claim_count > 0
# #         material_failure_signal = material_failing_claim_count > 0
# #         specific_repair_signal = specific_repair_claim_count > 0
# #         evidence_gap_only_without_claim_target = (
# #             evidence_gap_signal
# #             and not concrete_claim_repair_target
# #             and not low_claim_count_flag
# #         )
# #         retry_repair_plan = build_claim_repair_plan(weak_claim_items)
# #         retry_repair_action_counts = {
# #             "search_and_replace": sum(
# #                 1 for item in retry_repair_plan if item.get("action") == "search_and_replace"
# #             ),
# #             "qualify_or_remove": sum(
# #                 1 for item in retry_repair_plan if item.get("action") == "qualify_or_remove"
# #             ),
# #             "remove": sum(1 for item in retry_repair_plan if item.get("action") == "remove"),
# #         }
# #         repairable_claim_count = sum(retry_repair_action_counts.values())
# #         search_repair_count = retry_repair_action_counts["search_and_replace"]
# #         conservative_repair_count = (
# #             retry_repair_action_counts["qualify_or_remove"]
# #             + retry_repair_action_counts["remove"]
# #         )
# #         claim_micro_repair_candidate = (
# #             controller_mode in {"recursive", "shallow"}
# #             and not fixed_mode_baseline
# #             and needs_revision
# #             and retry_count < max_retries
# #             and concrete_claim_repair_target
# #             and repairable_claim_count > 0
# #         )
# #         if claim_micro_repair_candidate:
# #             # The recursive retry is now a claim-level patch plus revalidation,
# #             # not a broad market-research rerun. Keep the gate calibrated to
# #             # that cheaper, more targeted action.
# #             if search_repair_count == 0 and conservative_repair_count > 0:
# #                 estimated_retry_tokens = min(estimated_retry_tokens, 2200)
# #                 estimated_retry_tool_calls = 1
# #             else:
# #                 estimated_retry_tokens = min(estimated_retry_tokens, 3000)
# #                 estimated_retry_tool_calls = 3
# #             estimated_tool_calls_needed = (
# #                 estimated_retry_tool_calls + business_reserve_tool_calls
# #             )
# #             estimated_runtime_needed = (
# #                 18.0
# #                 if search_repair_count == 0 and not self.generate_pitch
# #                 else 25.0
# #                 if not self.generate_pitch
# #                 else 42.0
# #             )
# #             estimated_total_needed = (
# #                 estimated_retry_tokens + business_reserve_tokens + safety_buffer_tokens
# #             )
# #         near_threshold_strong_checkpoint = (
# #             needs_revision
# #             and validation_deficit <= 3
# #             and supported_ratio >= 0.90
# #             and judge_agreement >= 0.90
# #             and weak_or_unsupported_claim_count == 0
# #             and not low_claim_count_flag
# #         )
# #         recursive_retry = (
# #             controller_mode == "recursive"
# #             and not fixed_mode_baseline
# #             and needs_revision
# #             and material_revision_need
# #             and not evidence_gap_only_without_claim_target
# #             and not near_threshold_strong_checkpoint
# #             and retry_count < max_retries
# #             and (
# #                 concrete_claim_repair_target
# #                 or validation_deficit >= 15
# #                 or supported_ratio < 0.45
# #                 or (low_claim_count_flag and claims_total < 2)
# #             )
# #         )
# #         retry_allowed = claim_micro_repair_candidate or recursive_retry
# #         current_total_tokens = int(
# #             (self._sget(base_state, "token_usage", {}) or {}).get("total_tokens", 0) or 0
# #         )
# #         retry_expected_gain = min(
# #             24.0,
# #             (1.10 * validation_deficit)
# #             + (8.0 * max(0.0, 0.86 - judge_agreement))
# #             + (10.0 * max(0.0, 0.75 - supported_ratio))
# #             + (1.0 * weak_claim_count)
# #             + (3.0 * unsupported_claim_count)
# #             + (2.0 * needs_review_claim_count)
# #             + (2.0 * material_failing_claim_count)
# #             + (2.0 * search_repair_count)
# #             + (1.5 * conservative_repair_count)
# #             + (float(claim_count_penalty) * 0.6)
# #             + (3.0 if evidence_gap_signal and concrete_claim_repair_target else 0.0),
# #         )
# #         retry_incremental_tokens = estimated_retry_tokens
# #         retry_roi = retry_expected_gain / max(1.0, retry_incremental_tokens / 1000.0)
# #         current_quality_close_enough = (
# #             validation_score >= max(70, validation_threshold - 6)
# #             and validation_deficit <= 6
# #             and judge_agreement >= 0.88
# #             and supported_ratio >= 0.66
# #             and unsupported_claim_count == 0
# #             and needs_review_claim_count <= 1
# #             and weak_claim_count <= 1
# #             and not low_claim_count_flag
# #         )
# #         if claim_micro_repair_candidate:
# #             min_retry_expected_gain = 4.0
# #             min_retry_roi = 0.8
# #         else:
# #             min_retry_expected_gain = 7.0 if shallow_escalation else 10.5
# #             min_retry_roi = 1.35 if shallow_escalation else 2.0
# #         block_reasons = []
# #         if fixed_mode_baseline and needs_revision:
# #             block_reasons.append("fixed_baseline_no_adaptive_retry")
# #         if controller_mode != "recursive" and not shallow_escalation:
# #             block_reasons.append(f"mode_{controller_mode or 'unknown'}_does_not_retry")
# #         if not needs_revision:
# #             block_reasons.append("validation_passed")
# #         if (
# #             needs_revision
# #             and controller_mode == "recursive"
# #             and not material_revision_need
# #             and not claim_micro_repair_candidate
# #         ):
# #             block_reasons.append(
# #                 "minor_deficit_high_confidence_accept_best_checkpoint"
# #             )
# #         if evidence_gap_only_without_claim_target and needs_revision:
# #             retry_allowed = False
# #             block_reasons.append("evidence_gap_only_no_claim_repair_target")
# #         if near_threshold_strong_checkpoint:
# #             retry_allowed = False
# #             block_reasons.append("near_threshold_strong_checkpoint_accept")
# #         if retry_allowed and validation_deficit < 3 and supported_ratio >= 0.70 and not material_failure_signal:
# #             retry_allowed = False
# #             block_reasons.append("no_material_retry_need")
# #         if retry_allowed and judge_agreement < 0.75:
# #             retry_allowed = False
# #             block_reasons.append("judge_agreement_too_low_for_repair")
# #         if retry_allowed and not specific_repair_signal and not material_failure_signal:
# #             retry_allowed = False
# #             block_reasons.append("no_specific_repair_signal")
# #         if current_quality_close_enough and not claim_micro_repair_candidate:
# #             retry_allowed = False
# #             block_reasons.append("quality_close_enough_without_retry")
# #         if (
# #             retry_allowed
# #             and shallow_escalation
# #             and not recursive_cost_efficient
# #             and controller_structural_complexity < 70.0
# #         ):
# #             retry_allowed = False
# #             block_reasons.append(
# #                 "shallow_escalation_blocked_recursive_not_cost_efficient"
# #             )
# #         if retry_allowed and retry_expected_gain < min_retry_expected_gain:
# #             retry_allowed = False
# #             block_reasons.append(
# #                 f"low_expected_retry_gain:{retry_expected_gain:.3f}<{min_retry_expected_gain:.3f}"
# #             )
# #         if retry_allowed and retry_roi < min_retry_roi:
# #             retry_allowed = False
# #             block_reasons.append(f"low_retry_roi:{retry_roi:.3f}<{min_retry_roi:.3f}")
# #         if retry_allowed and not retry_enabled:
# #             retry_allowed = False
# #             block_reasons.append("adaptive_retry_disabled_by_ablation")
# #         if (
# #             retry_allowed
# #             and controller_mode == "recursive"
# #             and not claim_micro_repair_candidate
# #             and current_total_tokens >= 12_000
# #             and validation_score >= max(68, validation_threshold - 10)
# #             and unsupported_claim_count == 0
# #         ):
# #             retry_allowed = False
# #             block_reasons.append("recursive_retry_token_cap_accept_checkpoint")
# #         if retry_count >= max_retries:
# #             block_reasons.append("max_validation_retries_reached")
# #         if isinstance(remaining_total_tokens, int) and remaining_total_tokens < estimated_total_needed:
# #             retry_allowed = False
# #             block_reasons.append(
# #                 f"insufficient_total_tokens:{remaining_total_tokens}<{estimated_total_needed}"
# #             )
# #         if isinstance(remaining_tool_calls, int) and remaining_tool_calls < estimated_tool_calls_needed:
# #             retry_allowed = False
# #             block_reasons.append(
# #                 f"insufficient_tool_calls:{remaining_tool_calls}<{estimated_tool_calls_needed}"
# #             )
# #         if isinstance(remaining_runtime, (int, float)) and float(remaining_runtime) < estimated_runtime_needed:
# #             retry_allowed = False
# #             block_reasons.append(
# #                 f"insufficient_runtime:{float(remaining_runtime):.3f}<{estimated_runtime_needed:.3f}"
# #             )

# #         decision = {
# #             "controller_mode": controller_mode,
# #             "fixed_mode_baseline": fixed_mode_baseline,
# #             "adaptive_retry_enabled": retry_enabled,
# #             "needs_revision": needs_revision,
# #             "retry_count": retry_count,
# #             "max_validation_retries": max_retries,
# #             "validation_score": validation_score,
# #             "validation_threshold": validation_threshold,
# #             "validation_deficit": validation_deficit,
# #             "judge_agreement": round(judge_agreement, 4),
# #             "supported_ratio": round(supported_ratio, 4),
# #             "weak_or_unsupported_claim_count": weak_or_unsupported_claim_count,
# #             "weak_claim_count": weak_claim_count,
# #             "unsupported_claim_count": unsupported_claim_count,
# #             "needs_review_claim_count": needs_review_claim_count,
# #             "material_failing_claim_count": material_failing_claim_count,
# #             "specific_repair_claim_count": specific_repair_claim_count,
# #             "claims_total": claims_total,
# #             "low_claim_count_flag": low_claim_count_flag,
# #             "claim_count_penalty": claim_count_penalty,
# #             "evidence_gap_signal": evidence_gap_signal,
# #             "severe_revision_need": severe_revision_need,
# #             "corroborated_revision_need": corroborated_revision_need,
# #             "material_revision_need": material_revision_need,
# #             "concrete_claim_repair_target": concrete_claim_repair_target,
# #             "material_failure_signal": material_failure_signal,
# #             "specific_repair_signal": specific_repair_signal,
# #             "evidence_gap_only_without_claim_target": evidence_gap_only_without_claim_target,
# #             "claim_micro_repair_candidate": claim_micro_repair_candidate,
# #             "repairable_claim_count": repairable_claim_count,
# #             "repair_action_counts": retry_repair_action_counts,
# #             "search_repair_count": search_repair_count,
# #             "conservative_repair_count": conservative_repair_count,
# #             "near_threshold_strong_checkpoint": near_threshold_strong_checkpoint,
# #             "current_quality_close_enough": current_quality_close_enough,
# #             "controller_structural_complexity": round(controller_structural_complexity, 4),
# #             "recursive_cost_efficient": recursive_cost_efficient,
# #             "retry_expected_gain": round(retry_expected_gain, 4),
# #             "retry_incremental_tokens": retry_incremental_tokens,
# #             "retry_roi": round(retry_roi, 4),
# #             "min_retry_expected_gain": min_retry_expected_gain,
# #             "min_retry_roi": min_retry_roi,
# #             "current_total_tokens": current_total_tokens,
# #             "retry_allowed": retry_allowed,
# #             "retry_type": (
# #                 "claim_micro_repair"
# #                 if claim_micro_repair_candidate and retry_allowed
# #                 else "shallow_escalation"
# #                 if shallow_escalation and retry_allowed
# #                 else "recursive_retry"
# #                 if recursive_retry and retry_allowed
# #                 else "none"
# #             ),
# #             "block_reasons": block_reasons,
# #             "remaining_total_tokens": remaining_total_tokens,
# #             "estimated_retry_tokens": estimated_retry_tokens,
# #             "business_reserve_tokens": business_reserve_tokens,
# #             "safety_buffer_tokens": safety_buffer_tokens,
# #             "estimated_total_needed": estimated_total_needed,
# #             "remaining_tool_calls": remaining_tool_calls,
# #             "estimated_tool_calls_needed": estimated_tool_calls_needed,
# #             "remaining_runtime_seconds": remaining_runtime,
# #             "estimated_runtime_needed": estimated_runtime_needed,
# #         }
# #         decisions = list(self._sget(state, "retry_budget_decisions", []))
# #         decisions.append(decision)
# #         updates: Dict[str, Any] = {
# #             "retry_budget_decisions": decisions,
# #             "retry_roi_estimate": round(retry_roi, 4),
# #             "over_decomposition_flag": bool(
# #                 controller_mode == "recursive"
# #                 and retry_count == 0
# #                 and not retry_allowed
# #                 and validation_score >= max(70, validation_threshold - 6)
# #             ),
# #         }
# #         if retry_allowed and shallow_escalation:
# #             updates["controller_mode"] = "recursive"
# #             updates["decomposition_depth_target"] = 2
# #         if not retry_allowed and needs_revision:
# #             updates["needs_revision"] = False
# #         return updates

# #     def _route_after_retry_gate(self, state: PitchState) -> str:
# #         if bool(self._sget(state, "budget_hit", False)):
# #             return "stop"
# #         decisions = self._sget(state, "retry_budget_decisions", []) or []
# #         last = decisions[-1] if decisions else {}
# #         if isinstance(last, dict) and bool(last.get("retry_allowed", False)):
# #             return "retry_market"
# #         return "continue"

# #     @staticmethod
# #     def _estimate_retry_tokens(state: Dict[str, Any]) -> int:
# #         audits = state.get("tool_audit", []) or []
# #         validator_totals = [
# #             int(audit.get("total_tokens", 0) or 0)
# #             for audit in audits
# #             if isinstance(audit, dict) and audit.get("tool") == "llm_claim_verifier_dual_judge"
# #         ]
# #         last_validator_tokens = validator_totals[-1] if validator_totals else 4500
# #         # Market retry prompt plus keyword extraction typically adds a few thousand
# #         # tokens. Clamp to avoid one unusually large judge call overestimating forever.
# #         estimated_validation = max(2600, min(3600, last_validator_tokens))
# #         estimated_market = 1400
# #         return int(estimated_validation + estimated_market)

# #     def _route_after_controller(self, state: PitchState) -> str:
# #         if bool(self._sget(state, "budget_hit", False)):
# #             return "stop"
# #         mode = str(self._sget(state, "controller_mode", "shallow")).strip().lower()
# #         if mode not in {"direct", "shallow", "recursive"}:
# #             return "shallow"
# #         return mode

# #     def _route_by_budget(self, state: PitchState) -> str:
# #         if bool(self._sget(state, "budget_hit", False)):
# #             return "stop"
# #         return "continue"

# #     def _route_after_business(self, state: PitchState) -> str:
# #         if bool(self._sget(state, "budget_hit", False)):
# #             return "stop"
# #         if self.generate_pitch and self.pitch_agent is not None:
# #             return "pitch"
# #         return "end"

# #     def _route_after_direct(self, state: PitchState) -> str:
# #         if bool(self._sget(state, "budget_hit", False)):
# #             return "stop"
# #         if self.generate_pitch and self.pitch_agent is not None:
# #             return "pitch"
# #         return "end"

# #     def _run_business_with_depth(self, state: PitchState):
# #         updates = self._run_node_with_budget(state, self.business_agent.run, "business")
# #         mode = str(self._sget(state, "controller_mode", "")).strip().lower()
# #         retry_count = int(self._sget(state, "retry_count", 0))

# #         if mode == "shallow":
# #             depth = 2 if retry_count > 0 else 1
# #         elif mode == "recursive":
# #             depth = 2 if retry_count > 0 else 1
# #         elif mode == "direct":
# #             depth = 0
# #         else:
# #             depth = 2 if retry_count > 0 else 1
# #         updates["decomposition_depth_realized"] = depth
# #         return updates

# #     def _build_graph(self):
# #         workflow = StateGraph(PitchState)

# #         workflow.add_node("supervisor", self._run_supervisor)
# #         workflow.add_node("idea", self._run_idea)
# #         if self.controller_policy == "adaptive" and self.controller_agent is not None:
# #             workflow.add_node("controller", self._run_controller)
# #         workflow.add_node("market", self._run_market)
# #         workflow.add_node("validator", self._run_validator)
# #         workflow.add_node("repair_validator", self._run_lightweight_repair_validator)
# #         workflow.add_node("retry_gate", self._run_retry_gate)
# #         workflow.add_node("market_retry", self._market_retry)
# #         workflow.add_node("select_best_checkpoint", self._run_select_best_checkpoint)
# #         workflow.add_node("business", self._run_business_with_depth)
# #         if self.controller_policy == "adaptive" and self.direct_agent is not None:
# #             workflow.add_node("direct", self._run_direct)
# #         if self.generate_pitch and self.pitch_agent is not None:
# #             workflow.add_node("pitch", self._run_pitch)

# #         workflow.add_edge(START, "supervisor")
# #         workflow.add_conditional_edges(
# #             "supervisor",
# #             self._route_by_budget,
# #             {
# #                 "stop": END,
# #                 "continue": "idea",
# #             },
# #         )
# #         if self.controller_policy == "adaptive" and self.controller_agent is not None:
# #             workflow.add_conditional_edges(
# #                 "idea",
# #                 self._route_by_budget,
# #                 {
# #                     "stop": END,
# #                     "continue": "controller",
# #                 },
# #             )
# #             workflow.add_conditional_edges(
# #                 "controller",
# #                 self._route_after_controller,
# #                 {
# #                     "stop": END,
# #                     "direct": "direct",
# #                     "shallow": "market",
# #                     "recursive": "market",
# #                 },
# #             )
# #         else:
# #             workflow.add_conditional_edges(
# #                 "idea",
# #                 self._route_by_budget,
# #                 {
# #                     "stop": END,
# #                     "continue": "market",
# #                 },
# #             )
# #         workflow.add_conditional_edges(
# #             "market",
# #             self._route_by_budget,
# #             {
# #                 "stop": END,
# #                 "continue": "validator",
# #             },
# #         )
# #         workflow.add_conditional_edges(
# #             "validator",
# #             self._route_after_validation,
# #             {
# #                 "stop": END,
# #                 "retry_gate": "retry_gate",
# #             },
# #         )
# #         workflow.add_conditional_edges(
# #             "retry_gate",
# #             self._route_after_retry_gate,
# #             {
# #                 "stop": END,
# #                 "retry_market": "market_retry",
# #                 "continue": "select_best_checkpoint",
# #             },
# #         )
# #         workflow.add_conditional_edges(
# #             "select_best_checkpoint",
# #             self._route_by_budget,
# #             {
# #                 "stop": END,
# #                 "continue": "business",
# #             },
# #         )
# #         workflow.add_conditional_edges(
# #             "market_retry",
# #             self._route_after_market_retry,
# #             {
# #                 "stop": END,
# #                 "repair_validator": "repair_validator",
# #                 "validator": "validator",
# #             },
# #         )
# #         workflow.add_conditional_edges(
# #             "repair_validator",
# #             self._route_after_repair_validation,
# #             {
# #                 "stop": END,
# #                 "validator": "validator",
# #                 "retry_gate": "retry_gate",
# #             },
# #         )
# #         direct_route_map = {"stop": END, "end": END}
# #         if self.generate_pitch and self.pitch_agent is not None:
# #             direct_route_map["pitch"] = "pitch"
# #         if self.controller_policy == "adaptive" and self.direct_agent is not None:
# #             workflow.add_conditional_edges(
# #                 "direct",
# #                 self._route_after_direct,
# #                 direct_route_map,
# #             )
# #         business_route_map = {"stop": END, "end": END}
# #         if self.generate_pitch and self.pitch_agent is not None:
# #             business_route_map["pitch"] = "pitch"
# #         workflow.add_conditional_edges(
# #             "business",
# #             self._route_after_business,
# #             business_route_map,
# #         )
# #         if self.generate_pitch and self.pitch_agent is not None:
# #             workflow.add_edge("pitch", END)

# #         return workflow.compile(checkpointer=self.checkpointer)

# #     def run(
# #         self,
# #         idea: str,
# #         thread_id: str = "default-thread",
# #         max_validation_retries: int = 1,
# #         validation_threshold: int = 70,
# #         max_tool_calls: int | None = None,
# #         max_token_proxy: int | None = None,
# #         max_total_tokens: int | None = None,
# #         max_runtime_seconds: float | None = None,
# #         forced_controller_mode: str | None = None,
# #         adaptive_retry_enabled: bool = True,
# #         adaptive_checkpoint_enabled: bool = True,
# #         shared_refined_idea: str | None = None,
# #         shared_refinement_token_usage: Dict[str, int] | None = None,
# #         shared_refinement_tool_audit: list[Dict[str, Any]] | None = None,
# #     ) -> PitchState:
# #         normalized_max_tool_calls = (
# #             None if max_tool_calls is None or max_tool_calls < 0 else int(max_tool_calls)
# #         )
# #         normalized_max_token_proxy = (
# #             None if max_token_proxy is None or max_token_proxy < 0 else int(max_token_proxy)
# #         )
# #         normalized_max_total_tokens = (
# #             None if max_total_tokens is None or max_total_tokens < 0 else int(max_total_tokens)
# #         )
# #         normalized_max_runtime_seconds = (
# #             None
# #             if max_runtime_seconds is None or max_runtime_seconds < 0
# #             else float(max_runtime_seconds)
# #         )

# #         initial_state = PitchState(
# #             idea=idea,
# #             controller_policy=self.controller_policy,
# #             forced_controller_mode=forced_controller_mode,
# #             max_validation_retries=max_validation_retries,
# #             validation_threshold=validation_threshold,
# #             max_tool_calls=normalized_max_tool_calls,
# #             max_token_proxy=normalized_max_token_proxy,
# #             max_total_tokens=normalized_max_total_tokens,
# #             max_runtime_seconds=normalized_max_runtime_seconds,
# #             runtime_started_at=time.time(),
# #             adaptive_retry_enabled=adaptive_retry_enabled,
# #             adaptive_checkpoint_enabled=adaptive_checkpoint_enabled,
# #         ).model_dump()
# #         if shared_refined_idea:
# #             initial_state["refined_idea"] = shared_refined_idea
# #             initial_state["shared_refinement_locked"] = True
# #             initial_state["token_usage"] = dict(shared_refinement_token_usage or {})
# #             initial_state["tool_audit"] = list(shared_refinement_tool_audit or [])

# #         return self.graph.invoke(
# #             initial_state,
# #             config={"configurable": {"thread_id": thread_id}},
# #         )

# #     def continue_adaptive_from_validated_state(
# #         self,
# #         state: PitchState | Dict[str, Any],
# #         *,
# #         adaptive_retry_enabled: bool = True,
# #         adaptive_checkpoint_enabled: bool = True,
# #     ) -> Dict[str, Any]:
# #         """Continue a completed adaptive-no-retry state with full adaptive logic.

# #         This is used for paired ablations: `adaptive_no_retry` and
# #         `adaptive_controller` share the exact same initial decomposition,
# #         research, and validation result. The full controller then only adds its
# #         feedback-driven retry/checkpoint continuation, which makes the ablation
# #         measure the added policy component rather than independent LLM variance.
# #         """
# #         base_state = state if isinstance(state, dict) else state.model_dump()
# #         current = dict(base_state)
# #         validation = current.get("validation_report", {}) or {}
# #         threshold = int(current.get("validation_threshold", 70) or 70)
# #         current_score = self._validation_score(validation)
# #         current.update(
# #             {
# #                 "adaptive_retry_enabled": adaptive_retry_enabled,
# #                 "adaptive_checkpoint_enabled": adaptive_checkpoint_enabled,
# #                 "needs_revision": current_score < threshold,
# #                 "retry_budget_decisions": [],
# #                 "selected_validation_checkpoint": {},
# #                 "adaptive_checkpoint_selection": {},
# #             }
# #         )

# #         retry_updates = self._run_retry_gate(current)
# #         current.update(retry_updates)
# #         if bool(current.get("budget_hit", False)):
# #             return current

# #         retry_decisions = current.get("retry_budget_decisions", []) or []
# #         last_retry_decision = retry_decisions[-1] if retry_decisions else {}
# #         retried = bool(
# #             isinstance(last_retry_decision, dict)
# #             and last_retry_decision.get("retry_allowed", False)
# #         )

# #         if retried:
# #             retry_market_updates = self._market_retry(current)
# #             current.update(retry_market_updates)
# #             if bool(current.get("budget_hit", False)):
# #                 return current

# #             if self._should_use_lightweight_repair_validation(current):
# #                 validation_updates = self._run_lightweight_repair_validator(current)
# #                 current.update(validation_updates)
# #                 if bool(current.get("budget_hit", False)):
# #                     return current
# #                 if self._repair_validation_cascade_context(current).get("eligible"):
# #                     validation_updates = self._run_validator(current)
# #             else:
# #                 validation_updates = self._run_validator(current)
# #             current.update(validation_updates)
# #             if bool(current.get("budget_hit", False)):
# #                 return current

# #             final_retry_updates = self._run_retry_gate(current)
# #             current.update(final_retry_updates)

# #         checkpoint_updates = self._run_select_best_checkpoint(current)
# #         current.update(checkpoint_updates)
# #         if bool(current.get("budget_hit", False)):
# #             return current

# #         if retried or not current.get("business_model"):
# #             business_updates = self._run_business_with_depth(current)
# #             current.update(business_updates)
# #         else:
# #             # Preserve the no-retry business output when no continuation work was
# #             # performed; this keeps the paired full-controller cost identical to
# #             # the paired no-retry baseline unless a retry actually occurs.
# #             mode = str(current.get("controller_mode", "")).strip().lower()
# #             retry_count = int(current.get("retry_count", 0) or 0)
# #             if mode == "direct":
# #                 depth = 0
# #             else:
# #                 depth = 2 if retry_count > 0 else 1
# #             current["decomposition_depth_realized"] = depth

# #         audit = list(current.get("tool_audit", []) or [])
# #         audit.append(
# #             {
# #                 "agent": "adaptive_controller",
# #                 "tool": "paired_adaptive_continuation",
# #                 "status": "ok",
# #                 "paired_from": "adaptive_no_retry",
# #                 "retried": retried,
# #                 "checkpoint_enabled": adaptive_checkpoint_enabled,
# #                 "retry_enabled": adaptive_retry_enabled,
# #             }
# #         )
# #         current["tool_audit"] = audit
# #         return current
