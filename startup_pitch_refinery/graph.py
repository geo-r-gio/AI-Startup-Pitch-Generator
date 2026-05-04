from __future__ import annotations

import json
import time
from typing import Any, Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI

from startup_pitch_refinery.agents import (
    AdaptiveControllerAgent,
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
        if isinstance(started_at, (int, float)) and started_at > 0:
            runtime_elapsed_seconds = max(0.0, time.time() - float(started_at))
        else:
            runtime_elapsed_seconds = 0.0

        reasons = list(self._sget(merged, "budget_hit_reasons", []))
        budget_hit = bool(self._sget(merged, "budget_hit", False))

        if isinstance(max_tool_calls, int) and max_tool_calls >= 0:
            if tool_calls_current > max_tool_calls:
                budget_hit = True
                reasons.append(
                    f"max_tool_calls_exceeded:{tool_calls_current}>{max_tool_calls}"
                )
        if isinstance(max_token_proxy, int) and max_token_proxy >= 0:
            if token_proxy_current > max_token_proxy:
                budget_hit = True
                reasons.append(
                    f"max_token_proxy_exceeded:{token_proxy_current}>{max_token_proxy}"
                )
        if isinstance(max_total_tokens, int) and max_total_tokens >= 0:
            if total_tokens_current > max_total_tokens:
                budget_hit = True
                reasons.append(
                    f"max_total_tokens_exceeded:{total_tokens_current}>{max_total_tokens}"
                )
        if isinstance(max_runtime_seconds, (int, float)) and max_runtime_seconds >= 0:
            if runtime_elapsed_seconds > float(max_runtime_seconds):
                budget_hit = True
                reasons.append(
                    f"max_runtime_seconds_exceeded:{runtime_elapsed_seconds:.3f}>{float(max_runtime_seconds):.3f}"
                )

        dedup_reasons = []
        seen = set()
        for r in reasons:
            if r not in seen:
                dedup_reasons.append(r)
                seen.add(r)

        remaining = {
            "tool_calls": (
                None
                if not isinstance(max_tool_calls, int) or max_tool_calls < 0
                else max_tool_calls - tool_calls_current
            ),
            "token_proxy": (
                None
                if not isinstance(max_token_proxy, int) or max_token_proxy < 0
                else max_token_proxy - token_proxy_current
            ),
            "total_tokens": (
                None
                if not isinstance(max_total_tokens, int) or max_total_tokens < 0
                else max_total_tokens - total_tokens_current
            ),
            "runtime_seconds": (
                None
                if not isinstance(max_runtime_seconds, (int, float)) or max_runtime_seconds < 0
                else float(max_runtime_seconds) - runtime_elapsed_seconds
            ),
        }

        return {
            "runtime_elapsed_seconds": round(runtime_elapsed_seconds, 3),
            "tool_calls_current": tool_calls_current,
            "token_proxy_current": token_proxy_current,
            "prompt_tokens_current": prompt_tokens_current,
            "completion_tokens_current": completion_tokens_current,
            "total_tokens_current": total_tokens_current,
            "budget_hit": budget_hit,
            "budget_hit_reasons": dedup_reasons,
            "budget_remaining": remaining,
        }

    def _run_node_with_budget(
        self,
        state: PitchState | Dict[str, Any],
        runner,
        node_name: str,
    ) -> Dict[str, Any]:
        base_state = state if isinstance(state, dict) else state.model_dump()
        precheck = self._compute_budget_updates(base_state)
        if precheck["budget_hit"]:
            return {
                **precheck,
                "controller_rationale": self._sget(base_state, "controller_rationale"),
            }

        enriched_state = {**base_state, **precheck}
        updates = runner(enriched_state)
        merged = {**enriched_state, **updates}
        postcheck = self._compute_budget_updates(merged)
        if postcheck["budget_hit"]:
            audit = list(self._sget(merged, "tool_audit", []))
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

    def _market_retry(self, state: PitchState):
        current_retry = self._sget(state, "retry_count", 0)
        base_state = state if isinstance(state, dict) else state.model_dump()
        repair_context = self._build_market_repair_context(base_state)
        retry_state = {
            **base_state,
            "retry_count": current_retry + 1,
            "market_repair_context": repair_context,
        }
        weak_claim_count = len(repair_context.get("weak_or_unsupported_claims", []))
        has_repair_target = weak_claim_count > 0
        repair_runner = (
            self.claim_repair_agent.run if has_repair_target else self.market_agent.run
        )
        repair_node = "claim_micro_repair" if has_repair_target else "market_retry"
        updates = self._run_node_with_budget(retry_state, repair_runner, repair_node)
        updates["retry_count"] = current_retry + 1
        updates["market_repair_context"] = repair_context
        repair_history = list(self._sget(base_state, "market_repair_history", []) or [])
        repair_history.append(
            {
                "retry_count": current_retry + 1,
                "repair_node": repair_node,
                "weak_claim_count": weak_claim_count,
                "supported_claim_count": len(repair_context.get("supported_claims", [])),
                "previous_validation_score": repair_context.get("previous_validation_score"),
                "previous_supported_ratio": repair_context.get("previous_supported_ratio"),
                "evidence_gap_signal": bool(repair_context.get("evidence_gap_signal", False)),
                "repair_strategy": repair_context.get("repair_strategy", ""),
                "repair_action_counts": repair_context.get("repair_action_counts", {}),
            }
        )
        updates["market_repair_history"] = repair_history
        return updates

    def _build_market_repair_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        validation = self._sget(state, "validation_report", {}) or {}
        claims = validation.get("claims", []) if isinstance(validation, dict) else []
        weak_claims = []
        supported_claims = []
        for claim in claims or []:
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("claim", "")).strip()
            if not text:
                continue
            verdict = str(claim.get("verdict", "")).strip().lower()
            item = {
                "claim": text,
                "verdict": verdict,
                "confidence": claim.get("confidence"),
                "rationale": str(claim.get("rationale", "")).strip()[:300],
                "supporting_sources": claim.get("supporting_sources", []),
            }
            if verdict == "supported":
                supported_claims.append(item)
            elif verdict in {"weakly_supported", "unsupported", "needs_review"}:
                weak_claims.append(item)

        evidence_gaps = (
            str(validation.get("evidence_gaps", "") or "").strip()
            if isinstance(validation, dict)
            else ""
        )
        evidence_gap_signal = bool(
            evidence_gaps
            and any(
                marker in evidence_gaps.lower()
                for marker in ["lack", "limited", "gap", "weak", "indirect", "not specific"]
            )
        )
        repair_plan = build_claim_repair_plan(weak_claims)
        repair_action_counts = {
            "search_and_replace": sum(
                1 for item in repair_plan if item.get("action") == "search_and_replace"
            ),
            "qualify_or_remove": sum(
                1 for item in repair_plan if item.get("action") == "qualify_or_remove"
            ),
            "remove": sum(1 for item in repair_plan if item.get("action") == "remove"),
        }
        return {
            "repair_strategy": (
                "targeted_claim_repair: preserve supported claims, repair or remove weak "
                "claims, and avoid broad market-analysis rewrites"
            ),
            "previous_validation_score": self._validation_score(validation),
            "previous_supported_ratio": round(self._validation_supported_ratio(validation), 4),
            "previous_market_analysis": self._sget(state, "market_analysis", "") or "",
            "previous_market_sources": list(self._sget(state, "market_sources", []) or []),
            "evidence_gaps": evidence_gaps,
            "evidence_gap_signal": evidence_gap_signal,
            "weak_or_unsupported_claims": weak_claims[:8],
            "supported_claims": supported_claims[:8],
            "repair_plan": repair_plan[:8],
            "repair_action_counts": repair_action_counts,
        }

    def _run_supervisor(self, state: PitchState):
        return self._run_node_with_budget(state, self.supervisor.run, "supervisor")

    def _run_idea(self, state: PitchState):
        if bool(self._sget(state, "shared_refinement_locked", False)) and self._sget(
            state, "refined_idea"
        ):
            return {}
        return self._run_node_with_budget(state, self.idea_agent.run, "idea")

    def _run_controller(self, state: PitchState):
        if self.controller_agent is None:
            return {}
        return self._run_node_with_budget(state, self.controller_agent.run, "controller")

    def _run_market(self, state: PitchState):
        return self._run_node_with_budget(state, self.market_agent.run, "market")

    def _run_validator(self, state: PitchState):
        base_state = state if isinstance(state, dict) else state.model_dump()
        cascade_context = self._repair_validation_cascade_context(base_state)
        updates = self._run_node_with_budget(base_state, self.validator_agent.run, "validator")
        if cascade_context.get("eligible"):
            validation = updates.get("validation_report", {}) or {}
            tool_audit = list(updates.get("tool_audit", self._sget(base_state, "tool_audit", [])) or [])
            tool_audit.append(
                {
                    "agent": "adaptive_controller",
                    "tool": "repair_validator_cascade",
                    "status": "escalated_to_full_dual_judge",
                    "reason": cascade_context.get("reason", ""),
                    "previous_reliability_score": cascade_context.get("previous_score", 0),
                    "lightweight_candidate_score": cascade_context.get("candidate_score", 0),
                    "full_validation_score": self._validation_score(validation),
                    "retry_expected_gain": cascade_context.get("retry_expected_gain", 0.0),
                    "retry_roi": cascade_context.get("retry_roi", 0.0),
                }
            )
            updates["tool_audit"] = tool_audit
        merged = {**base_state, **updates}
        snapshot = self._build_validation_snapshot(merged)
        if cascade_context.get("eligible"):
            snapshot["validation_mode"] = "full_dual_judge_after_lightweight_repair"
        snapshots = list(self._sget(base_state, "validation_snapshots", []) or [])
        snapshots.append(snapshot)
        updates["validation_snapshots"] = snapshots
        return updates

    def _run_lightweight_repair_validator(self, state: PitchState):
        base_state = state if isinstance(state, dict) else state.model_dump()
        updates = self._run_node_with_budget(
            base_state,
            self.validator_agent.run_repair_only,
            "repair_validator",
        )
        merged = {**base_state, **updates}
        snapshot = self._build_validation_snapshot(merged)
        snapshot["validation_mode"] = "lightweight_repair"
        snapshots = list(self._sget(base_state, "validation_snapshots", []) or [])
        snapshots.append(snapshot)
        updates["validation_snapshots"] = snapshots
        return updates

    def _run_direct(self, state: PitchState):
        if self.direct_agent is None:
            return {}
        return self._run_node_with_budget(state, self.direct_agent.run, "direct")

    def _run_pitch(self, state: PitchState):
        if self.pitch_agent is None:
            return {}
        return self._run_node_with_budget(state, self.pitch_agent.run, "pitch")

    @staticmethod
    def _validation_supported_ratio(validation: Dict[str, Any]) -> float:
        claims = validation.get("claims", []) if isinstance(validation, dict) else []
        if not claims:
            return 0.0
        supported = 0
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            verdict = str(claim.get("verdict", "")).strip().lower()
            if verdict == "supported":
                supported += 1
        return supported / max(1, len(claims))

    @staticmethod
    def _validation_score(validation: Dict[str, Any]) -> int:
        if not isinstance(validation, dict):
            return 0
        try:
            return int(validation.get("reliability_score", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _build_validation_snapshot(self, state: Dict[str, Any]) -> Dict[str, Any]:
        validation = self._sget(state, "validation_report", {}) or {}
        agreement = (
            validation.get("agreement_stats", {}).get("overall_agreement", 0.0)
            if isinstance(validation, dict)
            else 0.0
        )
        token_usage = self._sget(state, "token_usage", {}) or {}
        return {
            "checkpoint_index": len(self._sget(state, "validation_snapshots", []) or []),
            "retry_count": int(self._sget(state, "retry_count", 0) or 0),
            "controller_mode": self._sget(state, "controller_mode"),
            "reliability_score": self._validation_score(validation),
            "judge_agreement": float(agreement or 0.0),
            "supported_ratio": round(self._validation_supported_ratio(validation), 4),
            "total_tokens_at_checkpoint": int(token_usage.get("total_tokens", 0) or 0),
            "market_analysis": self._sget(state, "market_analysis"),
            "validated_market_analysis": self._sget(state, "validated_market_analysis"),
            "validation_report": validation,
            "market_sources": list(self._sget(state, "market_sources", []) or []),
            "market_evidence": list(self._sget(state, "market_evidence", []) or []),
            "trend_signals": dict(self._sget(state, "trend_signals", {}) or {}),
        }

    def _run_select_best_checkpoint(self, state: PitchState) -> Dict[str, Any]:
        base_state = state if isinstance(state, dict) else state.model_dump()
        forced_mode = str(self._sget(base_state, "forced_controller_mode", "") or "").strip().lower()
        if forced_mode in {"direct", "shallow", "recursive"}:
            return {}
        checkpoint_enabled = bool(self._sget(base_state, "adaptive_checkpoint_enabled", True))
        snapshots = [
            item
            for item in (self._sget(base_state, "validation_snapshots", []) or [])
            if isinstance(item, dict)
        ]
        if not snapshots:
            return {}

        def rank(snapshot: Dict[str, Any]) -> tuple[float, float, float, int]:
            return (
                float(snapshot.get("reliability_score", 0) or 0),
                float(snapshot.get("judge_agreement", 0.0) or 0.0),
                float(snapshot.get("supported_ratio", 0.0) or 0.0),
                -int(snapshot.get("total_tokens_at_checkpoint", 0) or 0),
            )

        current = snapshots[-1]
        best = max(snapshots, key=rank) if checkpoint_enabled else current
        best_score = int(best.get("reliability_score", 0) or 0)
        current_score = int(current.get("reliability_score", 0) or 0)
        selected_previous = best.get("checkpoint_index") != current.get("checkpoint_index")
        threshold = int(self._sget(base_state, "validation_threshold", 70) or 70)
        selection = {
            "selected_checkpoint_index": best.get("checkpoint_index"),
            "selected_retry_count": best.get("retry_count"),
            "selected_previous_checkpoint": bool(selected_previous),
            "best_validation_score": best_score,
            "current_validation_score_before_selection": current_score,
            "score_delta_vs_current": best_score - current_score,
            "checkpoint_enabled": checkpoint_enabled,
            "selection_rule": (
                "highest reliability score, then judge agreement, supported ratio, "
                "then lower token cost"
                if checkpoint_enabled
                else "checkpoint rollback disabled; latest validation state retained"
            ),
        }
        tool_audit = list(self._sget(base_state, "tool_audit", []) or [])
        tool_audit.append(
            {
                "agent": "adaptive_controller",
                "tool": "checkpoint_selector",
                "status": "ok",
                **selection,
            }
        )

        updates: Dict[str, Any] = {
            "selected_validation_checkpoint": best,
            "adaptive_checkpoint_selection": selection,
            "tool_audit": tool_audit,
            # The retry gate has already finished, so the selected checkpoint is
            # the final available validation state for this run.
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
        elif best_score < threshold:
            updates["needs_revision"] = False
        return updates

    def _route_after_validation(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        return "retry_gate"

    def _should_use_lightweight_repair_validation(
        self, state: PitchState | Dict[str, Any]
    ) -> bool:
        decisions = self._sget(state, "retry_budget_decisions", []) or []
        last_decision = decisions[-1] if decisions else {}
        if not (
            isinstance(last_decision, dict)
            and last_decision.get("retry_type") == "claim_micro_repair"
        ):
            return False

        repair_history = self._sget(state, "market_repair_history", []) or []
        last_repair = repair_history[-1] if repair_history else {}
        action_counts = {}
        if isinstance(last_repair, dict):
            action_counts = last_repair.get("repair_action_counts", {}) or {}
        if not isinstance(action_counts, dict):
            return False
        search_count = int(action_counts.get("search_and_replace", 0) or 0)
        conservative_count = int(action_counts.get("qualify_or_remove", 0) or 0) + int(
            action_counts.get("remove", 0) or 0
        )
        return search_count == 0 and conservative_count > 0

    def _route_after_market_retry(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        if self._should_use_lightweight_repair_validation(state):
            return "repair_validator"
        return "validator"

    def _repair_validation_cascade_context(
        self, state: PitchState | Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return escalation context after a lightweight repair validator run.

        The cascade keeps conservative repairs cheap when the lightweight judge
        accepts them, but falls back to full dual-judge validation when the
        repair is rejected despite a high expected retry value.
        """
        validation = self._sget(state, "validation_report", {}) or {}
        if not isinstance(validation, dict):
            return {"eligible": False}
        repair_validation = validation.get("repair_validation", {}) or {}
        if not isinstance(repair_validation, dict):
            return {"eligible": False}
        judge_scores = validation.get("judge_scores", {}) or {}
        aggregated = (
            judge_scores.get("aggregated", {}) if isinstance(judge_scores, dict) else {}
        )
        if not bool(aggregated.get("lightweight_repair_validation", False)):
            return {"eligible": False}
        if bool(repair_validation.get("accepted", False)) or bool(
            aggregated.get("repair_patch_accepted", False)
        ):
            return {"eligible": False}

        decisions = self._sget(state, "retry_budget_decisions", []) or []
        last_decision = decisions[-1] if decisions else {}
        if not (
            isinstance(last_decision, dict)
            and last_decision.get("retry_type") == "claim_micro_repair"
            and bool(last_decision.get("retry_allowed", False))
        ):
            return {"eligible": False}

        previous_score = int(repair_validation.get("previous_reliability_score", 0) or 0)
        candidate_score = int(repair_validation.get("candidate_reliability_score", 0) or 0)
        threshold = int(self._sget(state, "validation_threshold", 70) or 70)
        deficit = max(0, threshold - previous_score)
        try:
            retry_expected_gain = float(last_decision.get("retry_expected_gain", 0.0) or 0.0)
        except (TypeError, ValueError):
            retry_expected_gain = 0.0
        try:
            retry_roi = float(last_decision.get("retry_roi", 0.0) or 0.0)
        except (TypeError, ValueError):
            retry_roi = 0.0

        # Escalate only when the original validation is materially below the
        # target and the retry gate predicted enough value to justify a full
        # dual-judge pass. A small candidate-score miss is acceptable here: the
        # point of the cascade is to let the stronger evaluator arbitrate
        # borderline or conservative claim rewrites.
        quality_need = deficit >= 7 or previous_score < 75
        promising_retry = retry_expected_gain >= 6.0 and retry_roi >= 1.0
        not_clearly_harmful = candidate_score >= previous_score - 8
        if not (quality_need and promising_retry and not_clearly_harmful):
            return {"eligible": False}

        if not self._has_budget_for_full_repair_validation(state):
            return {"eligible": False}

        return {
            "eligible": True,
            "reason": "lightweight_repair_rejected_but_retry_value_remains_high",
            "previous_score": previous_score,
            "candidate_score": candidate_score,
            "retry_expected_gain": round(retry_expected_gain, 4),
            "retry_roi": round(retry_roi, 4),
        }

    def _has_budget_for_full_repair_validation(
        self, state: PitchState | Dict[str, Any]
    ) -> bool:
        remaining = self._sget(state, "budget_remaining", {}) or {}
        if not isinstance(remaining, dict):
            return True
        remaining_total_tokens = remaining.get("total_tokens")
        remaining_tool_calls = remaining.get("tool_calls")
        remaining_runtime = remaining.get("runtime_seconds")

        audits = self._sget(state, "tool_audit", []) or []
        prior_dual_validator_tokens = [
            int(audit.get("total_tokens", 0) or 0)
            for audit in audits
            if isinstance(audit, dict) and audit.get("tool") == "llm_claim_verifier_dual_judge"
        ]
        estimated_full_validation_tokens = (
            prior_dual_validator_tokens[-1] if prior_dual_validator_tokens else 4500
        )
        estimated_full_validation_tokens = max(
            3000, min(7000, estimated_full_validation_tokens)
        )
        downstream_reserve_tokens = 900 if not self.generate_pitch else 2600
        if (
            isinstance(remaining_total_tokens, int)
            and remaining_total_tokens
            < estimated_full_validation_tokens + downstream_reserve_tokens
        ):
            return False
        if isinstance(remaining_tool_calls, int) and remaining_tool_calls < 2:
            return False
        if isinstance(remaining_runtime, (int, float)) and float(remaining_runtime) < (
            25.0 if not self.generate_pitch else 45.0
        ):
            return False
        return True

    def _route_after_repair_validation(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        if self._repair_validation_cascade_context(state).get("eligible"):
            return "validator"
        return "retry_gate"

    def _run_retry_gate(self, state: PitchState) -> Dict[str, Any]:
        base_state = state if isinstance(state, dict) else state.model_dump()
        controller_mode = str(self._sget(state, "controller_mode", "")).strip().lower()
        forced_mode = str(self._sget(state, "forced_controller_mode", "") or "").strip().lower()
        fixed_mode_baseline = forced_mode in {"direct", "shallow", "recursive"}
        retry_enabled = bool(self._sget(state, "adaptive_retry_enabled", True))
        needs_revision = bool(self._sget(state, "needs_revision", False))
        retry_count = int(self._sget(state, "retry_count", 0))
        max_retries = int(self._sget(state, "max_validation_retries", 1))
        validation = self._sget(state, "validation_report", {}) or {}
        validation_score = 0
        if isinstance(validation, dict):
            try:
                validation_score = int(validation.get("reliability_score", 0) or 0)
            except (TypeError, ValueError):
                validation_score = 0
        agreement_stats = (
            validation.get("agreement_stats", {}) if isinstance(validation, dict) else {}
        )
        judge_agreement = float(agreement_stats.get("overall_agreement", 0.0) or 0.0)
        supported_ratio = self._validation_supported_ratio(validation)
        weak_or_unsupported_claim_count = 0
        weak_claim_count = 0
        unsupported_claim_count = 0
        needs_review_claim_count = 0
        claims_total = 0
        weak_claim_items = []
        low_claim_count_flag = False
        claim_count_penalty = 0
        if isinstance(validation, dict):
            claims = validation.get("claims", []) or []
            claims_total = len(claims)
            aggregated_scores = (
                validation.get("judge_scores", {}).get("aggregated", {})
                if isinstance(validation.get("judge_scores", {}), dict)
                else {}
            )
            low_claim_count_flag = bool(aggregated_scores.get("low_claim_count_flag", False))
            try:
                claim_count_penalty = int(aggregated_scores.get("claim_count_penalty", 0) or 0)
            except (TypeError, ValueError):
                claim_count_penalty = 0
            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                verdict = str(claim.get("verdict", "")).strip().lower()
                if verdict in {"weakly_supported", "unsupported", "needs_review"}:
                    weak_or_unsupported_claim_count += 1
                    weak_claim_items.append(claim)
                if verdict == "weakly_supported":
                    weak_claim_count += 1
                elif verdict == "unsupported":
                    unsupported_claim_count += 1
                elif verdict == "needs_review":
                    needs_review_claim_count += 1
        validation_threshold = int(self._sget(state, "validation_threshold", 70) or 70)
        validation_deficit = max(0, validation_threshold - validation_score)
        budget_remaining = self._sget(state, "budget_remaining", {}) or {}
        remaining_total_tokens = (
            budget_remaining.get("total_tokens")
            if isinstance(budget_remaining, dict)
            else None
        )
        remaining_tool_calls = (
            budget_remaining.get("tool_calls")
            if isinstance(budget_remaining, dict)
            else None
        )
        remaining_runtime = (
            budget_remaining.get("runtime_seconds")
            if isinstance(budget_remaining, dict)
            else None
        )
        controller_scorecard = self._sget(state, "controller_scorecard", {}) or {}
        controller_features = (
            controller_scorecard.get("features", {})
            if isinstance(controller_scorecard, dict)
            else {}
        )
        recursive_cost_efficient = bool(
            controller_scorecard.get("recursive_cost_efficient", False)
        )
        try:
            controller_structural_complexity = float(
                controller_features.get("structural_complexity", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            controller_structural_complexity = 0.0

        # A retry means another targeted market pass, another dual-judge
        # validation, and enough reserve to reach business-model generation.
        # Fixed baselines stay fixed-depth; only adaptive runs may use
        # feedback-driven repair/escalation.
        estimated_retry_tokens = self._estimate_retry_tokens(base_state)
        if validation_deficit >= 8:
            business_reserve_tokens = 2200
        elif validation_deficit >= 4:
            business_reserve_tokens = 2600
        else:
            business_reserve_tokens = 3000
        if not self.generate_pitch:
            # Compare-mode methodology runs usually skip deck generation. Keep
            # enough room for business-model synthesis without blocking the
            # recursive evidence-repair loop unnecessarily.
            business_reserve_tokens = min(business_reserve_tokens, 900)
        safety_buffer_tokens = 500 if not self.generate_pitch else 900
        estimated_total_needed = (
            estimated_retry_tokens + business_reserve_tokens + safety_buffer_tokens
        )
        estimated_retry_tool_calls = 4
        business_reserve_tool_calls = 1 if not self.generate_pitch else 2
        estimated_tool_calls_needed = estimated_retry_tool_calls + business_reserve_tool_calls
        estimated_runtime_needed = 35.0 if not self.generate_pitch else 55.0

        shallow_escalation = (
            controller_mode == "shallow"
            and not fixed_mode_baseline
            and needs_revision
            and validation_deficit >= 6
            and retry_count < max_retries
        )
        evidence_gaps = ""
        if isinstance(validation, dict):
            evidence_gaps = str(validation.get("evidence_gaps", "") or "").strip()
        evidence_gap_signal = (
            validation_deficit > 0
            and len(evidence_gaps) >= 20
            and any(
                marker in evidence_gaps.lower()
                for marker in ["lack", "limited", "gap", "weak", "indirect", "not specific"]
            )
        )
        severe_revision_need = (
            validation_deficit >= 9
            or judge_agreement < 0.82
            or supported_ratio < 0.68
            or unsupported_claim_count > 0
            or needs_review_claim_count >= 2
        )
        corroborated_revision_need = (
            validation_deficit >= 5
            and (
                supported_ratio < 0.75
                or judge_agreement < 0.86
                or weak_or_unsupported_claim_count >= 2
                or evidence_gap_signal
                or low_claim_count_flag
            )
        )
        material_revision_need = severe_revision_need or corroborated_revision_need
        concrete_claim_repair_target = weak_or_unsupported_claim_count > 0
        evidence_gap_only_without_claim_target = (
            evidence_gap_signal
            and not concrete_claim_repair_target
            and not low_claim_count_flag
        )
        retry_repair_plan = build_claim_repair_plan(weak_claim_items)
        retry_repair_action_counts = {
            "search_and_replace": sum(
                1 for item in retry_repair_plan if item.get("action") == "search_and_replace"
            ),
            "qualify_or_remove": sum(
                1 for item in retry_repair_plan if item.get("action") == "qualify_or_remove"
            ),
            "remove": sum(1 for item in retry_repair_plan if item.get("action") == "remove"),
        }
        repairable_claim_count = sum(retry_repair_action_counts.values())
        search_repair_count = retry_repair_action_counts["search_and_replace"]
        conservative_repair_count = (
            retry_repair_action_counts["qualify_or_remove"]
            + retry_repair_action_counts["remove"]
        )
        claim_micro_repair_candidate = (
            controller_mode == "recursive"
            and not fixed_mode_baseline
            and needs_revision
            and retry_count < max_retries
            and concrete_claim_repair_target
            and repairable_claim_count > 0
        )
        if claim_micro_repair_candidate:
            # The recursive retry is now a claim-level patch plus revalidation,
            # not a broad market-research rerun. Keep the gate calibrated to
            # that cheaper, more targeted action.
            if search_repair_count == 0 and conservative_repair_count > 0:
                estimated_retry_tokens = min(estimated_retry_tokens, 2200)
                estimated_retry_tool_calls = 1
            else:
                estimated_retry_tokens = min(estimated_retry_tokens, 3000)
                estimated_retry_tool_calls = 3
            estimated_tool_calls_needed = (
                estimated_retry_tool_calls + business_reserve_tool_calls
            )
            estimated_runtime_needed = (
                18.0
                if search_repair_count == 0 and not self.generate_pitch
                else 25.0
                if not self.generate_pitch
                else 42.0
            )
            estimated_total_needed = (
                estimated_retry_tokens + business_reserve_tokens + safety_buffer_tokens
            )
        near_threshold_strong_checkpoint = (
            needs_revision
            and validation_deficit <= 3
            and supported_ratio >= 0.90
            and judge_agreement >= 0.90
            and weak_or_unsupported_claim_count == 0
            and not low_claim_count_flag
        )
        recursive_retry = (
            controller_mode == "recursive"
            and not fixed_mode_baseline
            and needs_revision
            and material_revision_need
            and not evidence_gap_only_without_claim_target
            and not near_threshold_strong_checkpoint
            and retry_count < max_retries
        )
        retry_allowed = recursive_retry or shallow_escalation or claim_micro_repair_candidate
        current_total_tokens = int(
            (self._sget(base_state, "token_usage", {}) or {}).get("total_tokens", 0) or 0
        )
        retry_expected_gain = min(
            24.0,
            (1.10 * validation_deficit)
            + (8.0 * max(0.0, 0.86 - judge_agreement))
            + (10.0 * max(0.0, 0.75 - supported_ratio))
            + (1.0 * weak_claim_count)
            + (3.0 * unsupported_claim_count)
            + (2.0 * needs_review_claim_count)
            + (2.0 * search_repair_count)
            + (1.5 * conservative_repair_count)
            + (float(claim_count_penalty) * 0.6)
            + (3.0 if evidence_gap_signal and concrete_claim_repair_target else 0.0),
        )
        retry_incremental_tokens = estimated_retry_tokens
        retry_roi = retry_expected_gain / max(1.0, retry_incremental_tokens / 1000.0)
        current_quality_close_enough = (
            validation_score >= max(70, validation_threshold - 6)
            and validation_deficit <= 6
            and judge_agreement >= 0.88
            and supported_ratio >= 0.66
            and unsupported_claim_count == 0
            and needs_review_claim_count <= 1
            and weak_claim_count <= 1
            and not low_claim_count_flag
        )
        if claim_micro_repair_candidate:
            min_retry_expected_gain = 4.0
            min_retry_roi = 1.0
        else:
            min_retry_expected_gain = 7.0 if shallow_escalation else 10.5
            min_retry_roi = 1.35 if shallow_escalation else 2.0
        block_reasons = []
        if fixed_mode_baseline and needs_revision:
            block_reasons.append("fixed_baseline_no_adaptive_retry")
        if controller_mode != "recursive" and not shallow_escalation:
            block_reasons.append(f"mode_{controller_mode or 'unknown'}_does_not_retry")
        if not needs_revision:
            block_reasons.append("validation_passed")
        if (
            needs_revision
            and controller_mode == "recursive"
            and not material_revision_need
            and not claim_micro_repair_candidate
        ):
            block_reasons.append(
                "minor_deficit_high_confidence_accept_best_checkpoint"
            )
        if evidence_gap_only_without_claim_target and needs_revision:
            retry_allowed = False
            block_reasons.append("evidence_gap_only_no_claim_repair_target")
        if near_threshold_strong_checkpoint:
            retry_allowed = False
            block_reasons.append("near_threshold_strong_checkpoint_accept")
        if current_quality_close_enough and not claim_micro_repair_candidate:
            retry_allowed = False
            block_reasons.append("quality_close_enough_without_retry")
        if (
            retry_allowed
            and shallow_escalation
            and not recursive_cost_efficient
            and controller_structural_complexity < 70.0
        ):
            retry_allowed = False
            block_reasons.append(
                "shallow_escalation_blocked_recursive_not_cost_efficient"
            )
        if retry_allowed and retry_expected_gain < min_retry_expected_gain:
            retry_allowed = False
            block_reasons.append(
                f"low_expected_retry_gain:{retry_expected_gain:.3f}<{min_retry_expected_gain:.3f}"
            )
        if retry_allowed and retry_roi < min_retry_roi:
            retry_allowed = False
            block_reasons.append(f"low_retry_roi:{retry_roi:.3f}<{min_retry_roi:.3f}")
        if retry_allowed and not retry_enabled:
            retry_allowed = False
            block_reasons.append("adaptive_retry_disabled_by_ablation")
        if (
            retry_allowed
            and controller_mode == "recursive"
            and not claim_micro_repair_candidate
            and current_total_tokens >= 12_000
            and validation_score >= max(68, validation_threshold - 10)
            and unsupported_claim_count == 0
        ):
            retry_allowed = False
            block_reasons.append("recursive_retry_token_cap_accept_checkpoint")
        if retry_count >= max_retries:
            block_reasons.append("max_validation_retries_reached")
        if isinstance(remaining_total_tokens, int) and remaining_total_tokens < estimated_total_needed:
            retry_allowed = False
            block_reasons.append(
                f"insufficient_total_tokens:{remaining_total_tokens}<{estimated_total_needed}"
            )
        if isinstance(remaining_tool_calls, int) and remaining_tool_calls < estimated_tool_calls_needed:
            retry_allowed = False
            block_reasons.append(
                f"insufficient_tool_calls:{remaining_tool_calls}<{estimated_tool_calls_needed}"
            )
        if isinstance(remaining_runtime, (int, float)) and float(remaining_runtime) < estimated_runtime_needed:
            retry_allowed = False
            block_reasons.append(
                f"insufficient_runtime:{float(remaining_runtime):.3f}<{estimated_runtime_needed:.3f}"
            )

        decision = {
            "controller_mode": controller_mode,
            "fixed_mode_baseline": fixed_mode_baseline,
            "adaptive_retry_enabled": retry_enabled,
            "needs_revision": needs_revision,
            "retry_count": retry_count,
            "max_validation_retries": max_retries,
            "validation_score": validation_score,
            "validation_threshold": validation_threshold,
            "validation_deficit": validation_deficit,
            "judge_agreement": round(judge_agreement, 4),
            "supported_ratio": round(supported_ratio, 4),
            "weak_or_unsupported_claim_count": weak_or_unsupported_claim_count,
            "weak_claim_count": weak_claim_count,
            "unsupported_claim_count": unsupported_claim_count,
            "needs_review_claim_count": needs_review_claim_count,
            "claims_total": claims_total,
            "low_claim_count_flag": low_claim_count_flag,
            "claim_count_penalty": claim_count_penalty,
            "evidence_gap_signal": evidence_gap_signal,
            "severe_revision_need": severe_revision_need,
            "corroborated_revision_need": corroborated_revision_need,
            "material_revision_need": material_revision_need,
            "concrete_claim_repair_target": concrete_claim_repair_target,
            "evidence_gap_only_without_claim_target": evidence_gap_only_without_claim_target,
            "claim_micro_repair_candidate": claim_micro_repair_candidate,
            "repairable_claim_count": repairable_claim_count,
            "repair_action_counts": retry_repair_action_counts,
            "search_repair_count": search_repair_count,
            "conservative_repair_count": conservative_repair_count,
            "near_threshold_strong_checkpoint": near_threshold_strong_checkpoint,
            "current_quality_close_enough": current_quality_close_enough,
            "controller_structural_complexity": round(controller_structural_complexity, 4),
            "recursive_cost_efficient": recursive_cost_efficient,
            "retry_expected_gain": round(retry_expected_gain, 4),
            "retry_incremental_tokens": retry_incremental_tokens,
            "retry_roi": round(retry_roi, 4),
            "min_retry_expected_gain": min_retry_expected_gain,
            "min_retry_roi": min_retry_roi,
            "current_total_tokens": current_total_tokens,
            "retry_allowed": retry_allowed,
            "retry_type": (
                "shallow_escalation"
                if shallow_escalation and retry_allowed
                else "claim_micro_repair"
                if claim_micro_repair_candidate and retry_allowed
                else "recursive_retry"
                if recursive_retry and retry_allowed
                else "none"
            ),
            "block_reasons": block_reasons,
            "remaining_total_tokens": remaining_total_tokens,
            "estimated_retry_tokens": estimated_retry_tokens,
            "business_reserve_tokens": business_reserve_tokens,
            "safety_buffer_tokens": safety_buffer_tokens,
            "estimated_total_needed": estimated_total_needed,
            "remaining_tool_calls": remaining_tool_calls,
            "estimated_tool_calls_needed": estimated_tool_calls_needed,
            "remaining_runtime_seconds": remaining_runtime,
            "estimated_runtime_needed": estimated_runtime_needed,
        }
        decisions = list(self._sget(state, "retry_budget_decisions", []))
        decisions.append(decision)
        updates: Dict[str, Any] = {
            "retry_budget_decisions": decisions,
        }
        if retry_allowed and shallow_escalation:
            updates["controller_mode"] = "recursive"
            updates["decomposition_depth_target"] = 2
        if not retry_allowed and needs_revision:
            updates["needs_revision"] = False
        return updates

    def _route_after_retry_gate(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        decisions = self._sget(state, "retry_budget_decisions", []) or []
        last = decisions[-1] if decisions else {}
        if isinstance(last, dict) and bool(last.get("retry_allowed", False)):
            return "retry_market"
        return "continue"

    @staticmethod
    def _estimate_retry_tokens(state: Dict[str, Any]) -> int:
        audits = state.get("tool_audit", []) or []
        validator_totals = [
            int(audit.get("total_tokens", 0) or 0)
            for audit in audits
            if isinstance(audit, dict) and audit.get("tool") == "llm_claim_verifier_dual_judge"
        ]
        last_validator_tokens = validator_totals[-1] if validator_totals else 4500
        # Market retry prompt plus keyword extraction typically adds a few thousand
        # tokens. Clamp to avoid one unusually large judge call overestimating forever.
        estimated_validation = max(2600, min(3600, last_validator_tokens))
        estimated_market = 1400
        return int(estimated_validation + estimated_market)

    def _route_after_controller(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        mode = str(self._sget(state, "controller_mode", "shallow")).strip().lower()
        if mode not in {"direct", "shallow", "recursive"}:
            return "shallow"
        return mode

    def _route_by_budget(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        return "continue"

    def _route_after_business(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        if self.generate_pitch and self.pitch_agent is not None:
            return "pitch"
        return "end"

    def _route_after_direct(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        if self.generate_pitch and self.pitch_agent is not None:
            return "pitch"
        return "end"

    def _run_business_with_depth(self, state: PitchState):
        updates = self._run_node_with_budget(state, self.business_agent.run, "business")
        mode = str(self._sget(state, "controller_mode", "")).strip().lower()
        retry_count = int(self._sget(state, "retry_count", 0))

        if mode == "shallow":
            depth = 2 if retry_count > 0 else 1
        elif mode == "recursive":
            depth = 2 if retry_count > 0 else 1
        elif mode == "direct":
            depth = 0
        else:
            depth = 2 if retry_count > 0 else 1
        updates["decomposition_depth_realized"] = depth
        return updates

    def _build_graph(self):
        workflow = StateGraph(PitchState)

        workflow.add_node("supervisor", self._run_supervisor)
        workflow.add_node("idea", self._run_idea)
        if self.controller_policy == "adaptive" and self.controller_agent is not None:
            workflow.add_node("controller", self._run_controller)
        workflow.add_node("market", self._run_market)
        workflow.add_node("validator", self._run_validator)
        workflow.add_node("repair_validator", self._run_lightweight_repair_validator)
        workflow.add_node("retry_gate", self._run_retry_gate)
        workflow.add_node("market_retry", self._market_retry)
        workflow.add_node("select_best_checkpoint", self._run_select_best_checkpoint)
        workflow.add_node("business", self._run_business_with_depth)
        if self.controller_policy == "adaptive" and self.direct_agent is not None:
            workflow.add_node("direct", self._run_direct)
        if self.generate_pitch and self.pitch_agent is not None:
            workflow.add_node("pitch", self._run_pitch)

        workflow.add_edge(START, "supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            self._route_by_budget,
            {
                "stop": END,
                "continue": "idea",
            },
        )
        if self.controller_policy == "adaptive" and self.controller_agent is not None:
            workflow.add_conditional_edges(
                "idea",
                self._route_by_budget,
                {
                    "stop": END,
                    "continue": "controller",
                },
            )
            workflow.add_conditional_edges(
                "controller",
                self._route_after_controller,
                {
                    "stop": END,
                    "direct": "direct",
                    "shallow": "market",
                    "recursive": "market",
                },
            )
        else:
            workflow.add_conditional_edges(
                "idea",
                self._route_by_budget,
                {
                    "stop": END,
                    "continue": "market",
                },
            )
        workflow.add_conditional_edges(
            "market",
            self._route_by_budget,
            {
                "stop": END,
                "continue": "validator",
            },
        )
        workflow.add_conditional_edges(
            "validator",
            self._route_after_validation,
            {
                "stop": END,
                "retry_gate": "retry_gate",
            },
        )
        workflow.add_conditional_edges(
            "retry_gate",
            self._route_after_retry_gate,
            {
                "stop": END,
                "retry_market": "market_retry",
                "continue": "select_best_checkpoint",
            },
        )
        workflow.add_conditional_edges(
            "select_best_checkpoint",
            self._route_by_budget,
            {
                "stop": END,
                "continue": "business",
            },
        )
        workflow.add_conditional_edges(
            "market_retry",
            self._route_after_market_retry,
            {
                "stop": END,
                "repair_validator": "repair_validator",
                "validator": "validator",
            },
        )
        workflow.add_conditional_edges(
            "repair_validator",
            self._route_after_repair_validation,
            {
                "stop": END,
                "validator": "validator",
                "retry_gate": "retry_gate",
            },
        )
        direct_route_map = {"stop": END, "end": END}
        if self.generate_pitch and self.pitch_agent is not None:
            direct_route_map["pitch"] = "pitch"
        if self.controller_policy == "adaptive" and self.direct_agent is not None:
            workflow.add_conditional_edges(
                "direct",
                self._route_after_direct,
                direct_route_map,
            )
        business_route_map = {"stop": END, "end": END}
        if self.generate_pitch and self.pitch_agent is not None:
            business_route_map["pitch"] = "pitch"
        workflow.add_conditional_edges(
            "business",
            self._route_after_business,
            business_route_map,
        )
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
    ) -> PitchState:
        normalized_max_tool_calls = (
            None if max_tool_calls is None or max_tool_calls < 0 else int(max_tool_calls)
        )
        normalized_max_token_proxy = (
            None if max_token_proxy is None or max_token_proxy < 0 else int(max_token_proxy)
        )
        normalized_max_total_tokens = (
            None if max_total_tokens is None or max_total_tokens < 0 else int(max_total_tokens)
        )
        normalized_max_runtime_seconds = (
            None
            if max_runtime_seconds is None or max_runtime_seconds < 0
            else float(max_runtime_seconds)
        )

        initial_state = PitchState(
            idea=idea,
            controller_policy=self.controller_policy,
            forced_controller_mode=forced_controller_mode,
            max_validation_retries=max_validation_retries,
            validation_threshold=validation_threshold,
            max_tool_calls=normalized_max_tool_calls,
            max_token_proxy=normalized_max_token_proxy,
            max_total_tokens=normalized_max_total_tokens,
            max_runtime_seconds=normalized_max_runtime_seconds,
            runtime_started_at=time.time(),
            adaptive_retry_enabled=adaptive_retry_enabled,
            adaptive_checkpoint_enabled=adaptive_checkpoint_enabled,
        ).model_dump()
        if shared_refined_idea:
            initial_state["refined_idea"] = shared_refined_idea
            initial_state["shared_refinement_locked"] = True
            initial_state["token_usage"] = dict(shared_refinement_token_usage or {})
            initial_state["tool_audit"] = list(shared_refinement_tool_audit or [])

        return self.graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )

    def continue_adaptive_from_validated_state(
        self,
        state: PitchState | Dict[str, Any],
        *,
        adaptive_retry_enabled: bool = True,
        adaptive_checkpoint_enabled: bool = True,
    ) -> Dict[str, Any]:
        """Continue a completed adaptive-no-retry state with full adaptive logic.

        This is used for paired ablations: `adaptive_no_retry` and
        `adaptive_controller` share the exact same initial decomposition,
        research, and validation result. The full controller then only adds its
        feedback-driven retry/checkpoint continuation, which makes the ablation
        measure the added policy component rather than independent LLM variance.
        """
        base_state = state if isinstance(state, dict) else state.model_dump()
        current = dict(base_state)
        validation = current.get("validation_report", {}) or {}
        threshold = int(current.get("validation_threshold", 70) or 70)
        current_score = self._validation_score(validation)
        current.update(
            {
                "adaptive_retry_enabled": adaptive_retry_enabled,
                "adaptive_checkpoint_enabled": adaptive_checkpoint_enabled,
                "needs_revision": current_score < threshold,
                "retry_budget_decisions": [],
                "selected_validation_checkpoint": {},
                "adaptive_checkpoint_selection": {},
            }
        )

        retry_updates = self._run_retry_gate(current)
        current.update(retry_updates)
        if bool(current.get("budget_hit", False)):
            return current

        retry_decisions = current.get("retry_budget_decisions", []) or []
        last_retry_decision = retry_decisions[-1] if retry_decisions else {}
        retried = bool(
            isinstance(last_retry_decision, dict)
            and last_retry_decision.get("retry_allowed", False)
        )

        if retried:
            retry_market_updates = self._market_retry(current)
            current.update(retry_market_updates)
            if bool(current.get("budget_hit", False)):
                return current

            if self._should_use_lightweight_repair_validation(current):
                validation_updates = self._run_lightweight_repair_validator(current)
                current.update(validation_updates)
                if bool(current.get("budget_hit", False)):
                    return current
                if self._repair_validation_cascade_context(current).get("eligible"):
                    validation_updates = self._run_validator(current)
            else:
                validation_updates = self._run_validator(current)
            current.update(validation_updates)
            if bool(current.get("budget_hit", False)):
                return current

            final_retry_updates = self._run_retry_gate(current)
            current.update(final_retry_updates)

        checkpoint_updates = self._run_select_best_checkpoint(current)
        current.update(checkpoint_updates)
        if bool(current.get("budget_hit", False)):
            return current

        if retried or not current.get("business_model"):
            business_updates = self._run_business_with_depth(current)
            current.update(business_updates)
        else:
            # Preserve the no-retry business output when no continuation work was
            # performed; this keeps the paired full-controller cost identical to
            # the paired no-retry baseline unless a retry actually occurs.
            mode = str(current.get("controller_mode", "")).strip().lower()
            retry_count = int(current.get("retry_count", 0) or 0)
            if mode == "direct":
                depth = 0
            else:
                depth = 2 if retry_count > 0 else 1
            current["decomposition_depth_realized"] = depth

        audit = list(current.get("tool_audit", []) or [])
        audit.append(
            {
                "agent": "adaptive_controller",
                "tool": "paired_adaptive_continuation",
                "status": "ok",
                "paired_from": "adaptive_no_retry",
                "retried": retried,
                "checkpoint_enabled": adaptive_checkpoint_enabled,
                "retry_enabled": adaptive_retry_enabled,
            }
        )
        current["tool_audit"] = audit
        return current
