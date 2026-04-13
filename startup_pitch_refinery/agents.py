from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from startup_pitch_refinery.state import PitchState
from startup_pitch_refinery.tools import (
    BusinessCalcTool,
    GoogleTrendsTool,
    MarketSearchTool,
    ScenarioAnalysisTool,
    generate_pitch_deck,
)


def _sget(state: PitchState | Dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


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


def _merge_token_usage(
    state: PitchState | Dict[str, Any], usage_delta: Dict[str, int]
) -> Dict[str, int]:
    base = _sget(state, "token_usage", {}) or {}
    merged = {
        "prompt_tokens": int(base.get("prompt_tokens", 0)) + int(usage_delta.get("prompt_tokens", 0)),
        "completion_tokens": int(base.get("completion_tokens", 0))
        + int(usage_delta.get("completion_tokens", 0)),
        "total_tokens": int(base.get("total_tokens", 0)) + int(usage_delta.get("total_tokens", 0)),
    }
    return merged


def _deterministic_reliability_score(claims: List[Dict[str, Any]]) -> int:
    """
    Deterministic score from claim-level outputs.
    Formula:
      claim_score = verdict_weight * confidence
      reliability = average(claim_score) * 100
    """
    if not claims:
        return 0

    verdict_weights = {
        "supported": 1.0,
        "weakly_supported": 0.6,
        "needs_review": 0.3,
        "unsupported": 0.0,
    }

    total = 0.0
    for claim in claims:
        verdict = str(claim.get("verdict", "needs_review"))
        confidence = claim.get("confidence", 0.0)
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        total += verdict_weights.get(verdict, 0.3) * conf

    return int(round((total / len(claims)) * 100))


def _fallback_keywords(text: str) -> List[str]:
    """Domain-agnostic fallback if LLM keyword extraction fails."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", text.lower())
    stop = {
        "problem",
        "solution",
        "summary",
        "value",
        "proposition",
        "startup",
        "idea",
        "with",
        "that",
        "this",
        "from",
        "into",
        "your",
        "for",
        "and",
        "the",
    }
    keywords: List[str] = []
    for token in tokens:
        if token in stop:
            continue
        if token not in keywords:
            keywords.append(token)
    if not keywords:
        return ["startup market", "industry trends", "competitor landscape"]
    return keywords[:5]


class RefinedIdea(BaseModel):
    problem: str = Field(..., description="Core user/customer problem")
    solution: str = Field(..., description="Proposed startup solution")
    value_proposition: str = Field(..., description="Why this solution wins")
    refined_summary: str = Field(..., description="Short integrated startup concept")


class MarketOutput(BaseModel):
    target_market: str
    market_size: str
    trends: str
    competitors: str
    differentiation_gaps: str


class BusinessOutput(BaseModel):
    revenue_streams: str
    pricing_strategy: str
    cost_structure: str
    unit_economics: str
    financial_projection: str


class DirectStrategyOutput(BaseModel):
    target_market: str
    market_size: str
    trends: str
    competitors: str
    differentiation_gaps: str
    revenue_streams: str
    pricing_strategy: str
    cost_structure: str
    unit_economics: str
    financial_projection: str
    users_year1: int = Field(..., ge=1000, le=500000)
    arpu_monthly: float = Field(..., ge=2.0, le=300.0)
    gross_margin: float = Field(..., ge=0.2, le=0.95)
    assumptions_rationale: str


class FinancialAssumptions(BaseModel):
    users_year1: int = Field(..., ge=1000, le=500000)
    arpu_monthly: float = Field(..., ge=2.0, le=300.0)
    gross_margin: float = Field(..., ge=0.2, le=0.95)
    rationale: str


class PitchSlides(BaseModel):
    title: str
    subtitle: str
    problem: str
    solution: str
    market: str
    business_model: str
    competitive_advantage: str
    financials: str


class VerifiedClaim(BaseModel):
    claim: str
    verdict: Literal["supported", "weakly_supported", "unsupported", "needs_review"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    supporting_sources: List[str]


class ValidationOutput(BaseModel):
    validated_summary: str
    reliability_score: int = Field(..., ge=0, le=100)
    evidence_gaps: str
    claims: List[VerifiedClaim]


class TrendKeywords(BaseModel):
    keywords: List[str] = Field(
        ...,
        description="3 to 7 concise, domain-agnostic trend keywords for the startup idea.",
        min_length=3,
        max_length=7,
    )


class ControllerDecision(BaseModel):
    mode: Literal["direct", "shallow", "recursive"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    estimated_complexity: int = Field(..., ge=0, le=100)
    expected_tool_calls_delta: int = Field(..., ge=0, le=20)
    expected_token_proxy_delta: int = Field(..., ge=50, le=6000)
    expected_runtime_seconds_delta: float = Field(..., ge=0.1, le=600.0)
    triggers: List[str] = Field(default_factory=list)


class SupervisorAgent:
    def run(self, state: PitchState) -> PitchState:
        plan = [
            "1. Understand and refine the startup idea.",
            "2. Research market size, trends, and competitors.",
            "3. Validate claims against cited sources and score reliability.",
            "4. Build business model and basic financial projection.",
            "5. Structure investor-ready pitch content.",
            "6. Generate PowerPoint pitch deck (.pptx).",
        ]
        return {"task_plan": plan}


class IdeaRefinementAgent:
    def __init__(self, llm: ChatOpenAI):
        self.structured_llm = llm.with_structured_output(RefinedIdea, include_raw=True)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "# Role\n"
                    "You are a startup concept refiner.\n\n"
                    "# Instructions\n"
                    "- Transform the raw startup idea into an investor-ready concept.\n"
                    "- Keep outputs concise, concrete, and specific.\n"
                    "- Cover problem, solution, value proposition, and a unified summary.\n\n"
                    "# Output Format\n"
                    "- Return content that strictly matches the structured schema fields.",
                ),
                (
                    "human",
                    "Raw idea: {idea}\n\n"
                    "Define: problem, solution, value proposition, and refined summary.",
                ),
            ]
        )

    def run(self, state: PitchState) -> PitchState:
        idea = _sget(state, "idea", "")
        result, usage = _invoke_structured_with_usage(
            self.structured_llm, self.prompt.format_messages(idea=idea)
        )
        token_usage = _merge_token_usage(state, usage)
        refined = (
            f"Problem: {result.problem}\n"
            f"Solution: {result.solution}\n"
            f"Value Proposition: {result.value_proposition}\n"
            f"Summary: {result.refined_summary}"
        )
        return {"refined_idea": refined, "token_usage": token_usage}


class AdaptiveControllerAgent:
    MODE_ORDER = ["direct", "shallow", "recursive"]
    MODE_QUALITY_PRIORITY = ["recursive", "shallow", "direct"]
    MODE_PRIORS = {
        "direct": {
            "tool_calls": 4,
            "token_proxy": 1700,
            "runtime_seconds": 28.0,
        },
        "shallow": {
            "tool_calls": 6,
            "token_proxy": 3000,
            "runtime_seconds": 38.0,
        },
        "recursive": {
            "tool_calls": 8,
            "token_proxy": 3900,
            "runtime_seconds": 55.0,
        },
    }

    def __init__(self, llm: ChatOpenAI):
        self.structured_llm = llm.with_structured_output(ControllerDecision, include_raw=True)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "# Role\n"
                    "You are an adaptive decomposition controller for a controller-executor startup workflow.\n\n"
                    "# Instructions\n"
                    "- Choose one execution mode: direct, shallow, or recursive.\n"
                    "- `direct`: lowest decomposition and lowest cost for straightforward ideas.\n"
                    "- `shallow`: one-pass decomposition with validation and no recursive retry.\n"
                    "- `recursive`: decomposition with validation-driven retry for higher uncertainty.\n"
                    "- Balance expected quality gains against remaining budget.\n"
                    "- Estimate complexity (0-100), confidence, concise rationale, and expected incremental costs.\n"
                    "- Include short trigger phrases that justify the choice.\n\n"
                    "# Output Format\n"
                    "- Return content that strictly matches the structured schema fields.",
                ),
                (
                    "human",
                    "Raw idea:\n{idea}\n\n"
                    "Refined idea:\n{refined_idea}\n\n"
                    "Validation threshold: {validation_threshold}\n"
                    "Max retries available: {max_validation_retries}\n\n"
                    "Budget state:\n{budget_snapshot}\n\n"
                    "Select mode and provide rationale.",
                ),
            ]
        )

    @staticmethod
    def _fallback_decision(idea_text: str) -> ControllerDecision:
        text = idea_text.lower()
        tokens = re.findall(r"[a-zA-Z0-9]+", text)
        token_count = len(tokens)
        complexity_keywords = [
            "enterprise",
            "compliance",
            "risk",
            "workflow",
            "contract",
            "integration",
            "platform",
            "multi",
            "automation",
            "regulatory",
            "b2b",
            "agent",
            "marketplace",
            "prediction",
        ]
        match_count = sum(1 for key in complexity_keywords if key in text)
        score = min(100, int(token_count * 0.6 + match_count * 8))

        if score < 30:
            mode = "direct"
            confidence = 0.78
            rationale = "Idea appears relatively narrow and can be handled with low decomposition overhead."
            expected_tool_calls_delta = 3
            expected_token_proxy_delta = 1400
            expected_runtime_seconds_delta = 18.0
        elif score < 60:
            mode = "shallow"
            confidence = 0.72
            rationale = "Idea has moderate complexity, so one decomposition pass with validation is appropriate."
            expected_tool_calls_delta = 5
            expected_token_proxy_delta = 2600
            expected_runtime_seconds_delta = 32.0
        else:
            mode = "recursive"
            confidence = 0.69
            rationale = "Idea is high-complexity or high-uncertainty and benefits from validation-driven recursion."
            expected_tool_calls_delta = 7
            expected_token_proxy_delta = 3600
            expected_runtime_seconds_delta = 48.0

        triggers = []
        if token_count >= 35:
            triggers.append("long_problem_description")
        if match_count >= 3:
            triggers.append("multiple_complexity_markers")
        if "compliance" in text or "risk" in text:
            triggers.append("high_stakes_domain")
        if not triggers:
            triggers = ["low_complexity_signal"]

        return ControllerDecision(
            mode=mode,
            confidence=confidence,
            rationale=rationale,
            estimated_complexity=score,
            expected_tool_calls_delta=expected_tool_calls_delta,
            expected_token_proxy_delta=expected_token_proxy_delta,
            expected_runtime_seconds_delta=expected_runtime_seconds_delta,
            triggers=triggers,
        )

    @staticmethod
    def _compute_remaining_budget(state: PitchState | Dict[str, Any]) -> Dict[str, Any]:
        max_tool_calls = _sget(state, "max_tool_calls")
        max_token_proxy = _sget(state, "max_token_proxy")
        max_total_tokens = _sget(state, "max_total_tokens")
        max_runtime_seconds = _sget(state, "max_runtime_seconds")
        tool_calls_current = int(_sget(state, "tool_calls_current", 0) or 0)
        token_proxy_current = int(_sget(state, "token_proxy_current", 0) or 0)
        total_tokens_current = int(_sget(state, "total_tokens_current", 0) or 0)
        runtime_elapsed_seconds = float(_sget(state, "runtime_elapsed_seconds", 0.0) or 0.0)

        return {
            "max_tool_calls": max_tool_calls,
            "max_token_proxy": max_token_proxy,
            "max_total_tokens": max_total_tokens,
            "max_runtime_seconds": max_runtime_seconds,
            "tool_calls_current": tool_calls_current,
            "token_proxy_current": token_proxy_current,
            "total_tokens_current": total_tokens_current,
            "runtime_elapsed_seconds": round(runtime_elapsed_seconds, 3),
            "remaining_tool_calls": (
                None
                if max_tool_calls is None
                else int(max_tool_calls) - tool_calls_current
            ),
            "remaining_token_proxy": (
                None
                if max_token_proxy is None
                else int(max_token_proxy) - token_proxy_current
            ),
            "remaining_total_tokens": (
                None
                if max_total_tokens is None
                else int(max_total_tokens) - total_tokens_current
            ),
            "remaining_runtime_seconds": (
                None
                if max_runtime_seconds is None
                else round(float(max_runtime_seconds) - runtime_elapsed_seconds, 3)
            ),
        }

    @staticmethod
    def _budget_guardrail_mode(budget_snapshot: Dict[str, Any]) -> Optional[str]:
        rem_calls = budget_snapshot.get("remaining_tool_calls")
        rem_tokens = budget_snapshot.get("remaining_token_proxy")
        rem_total_tokens = budget_snapshot.get("remaining_total_tokens")
        rem_runtime = budget_snapshot.get("remaining_runtime_seconds")

        tight = (
            (isinstance(rem_calls, int) and rem_calls <= 2)
            or (isinstance(rem_tokens, int) and rem_tokens <= 900)
            or (isinstance(rem_total_tokens, int) and rem_total_tokens <= 900)
            or (isinstance(rem_runtime, (int, float)) and rem_runtime <= 10.0)
        )
        moderate = (
            (isinstance(rem_calls, int) and rem_calls <= 4)
            or (isinstance(rem_tokens, int) and rem_tokens <= 1900)
            or (isinstance(rem_total_tokens, int) and rem_total_tokens <= 1900)
            or (isinstance(rem_runtime, (int, float)) and rem_runtime <= 22.0)
        )

        if tight:
            return "direct"
        if moderate:
            return "shallow"
        return None

    @classmethod
    def _prior_cost_for_mode(cls, mode: str, complexity: int) -> Dict[str, Any]:
        base = cls.MODE_PRIORS.get(mode, cls.MODE_PRIORS["shallow"])
        # Scale priors mildly by complexity (0-100 -> 0.9x to 1.25x).
        scale = 0.9 + (max(0, min(100, complexity)) / 100.0) * 0.35
        return {
            "tool_calls": max(1, int(round(base["tool_calls"] * scale))),
            "token_proxy": max(100, int(round(base["token_proxy"] * scale))),
            "runtime_seconds": round(max(0.5, base["runtime_seconds"] * scale), 3),
        }

    @classmethod
    def _calibrate_selected_cost(
        cls,
        decision: ControllerDecision,
        mode_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        mode_for_prior = mode_override or decision.mode
        prior = cls._prior_cost_for_mode(mode_for_prior, decision.estimated_complexity)
        raw = {
            "tool_calls": int(decision.expected_tool_calls_delta),
            "token_proxy": int(decision.expected_token_proxy_delta),
            "runtime_seconds": float(decision.expected_runtime_seconds_delta),
        }
        # Blend with prior and enforce non-trivial floor (80% of prior).
        tool_calls = max(int(round(prior["tool_calls"] * 0.8)), int(round(0.75 * prior["tool_calls"] + 0.25 * raw["tool_calls"])))
        token_proxy = max(int(round(prior["token_proxy"] * 0.8)), int(round(0.75 * prior["token_proxy"] + 0.25 * raw["token_proxy"])))
        runtime_seconds = max(round(prior["runtime_seconds"] * 0.8, 3), round(0.75 * prior["runtime_seconds"] + 0.25 * raw["runtime_seconds"], 3))
        return {
            "tool_calls": tool_calls,
            "token_proxy": token_proxy,
            "runtime_seconds": runtime_seconds,
            "prior": prior,
            "raw": raw,
        }

    @staticmethod
    def _complexity_features(text: str) -> Dict[str, Any]:
        text_l = (text or "").lower()
        tokens = re.findall(r"[a-zA-Z0-9]+", text_l)
        marker_terms = [
            "recursive",
            "adversarial",
            "uncertainty",
            "uncertain",
            "cross-border",
            "jurisdiction",
            "compliance",
            "regulatory",
            "critical infrastructure",
            "rollback",
            "re-plan",
            "replanning",
            "conflicting",
            "safety",
            "failure",
            "multi-agent",
        ]
        marker_count = sum(1 for term in marker_terms if term in text_l)
        return {
            "token_count": len(tokens),
            "marker_count": marker_count,
            "has_high_stakes": marker_count >= 4,
        }

    @classmethod
    def _should_promote_recursive(
        cls,
        *,
        decision: ControllerDecision,
        combined_text: str,
        validation_threshold: int,
        max_validation_retries: int,
        budget_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Deterministic promotion only in high-assurance setups.
        if validation_threshold < 85 or max_validation_retries < 2:
            return {"promote": False, "reason": None}

        features = cls._complexity_features(combined_text)
        complexity_signal = (
            decision.estimated_complexity >= 40
            or features["token_count"] >= 45
            or features["marker_count"] >= 3
            or features["has_high_stakes"]
        )
        confidence_signal = decision.confidence <= 0.86
        rationale_l = (decision.rationale or "").lower()
        uncertainty_signal = any(
            token in rationale_l
            for token in ["complex", "uncertain", "risk", "adversarial", "conflicting"]
        )
        if not complexity_signal:
            return {"promote": False, "reason": None}
        if not (confidence_signal or uncertainty_signal):
            return {"promote": False, "reason": None}

        recursive_cost = cls._prior_cost_for_mode("recursive", decision.estimated_complexity)
        if not cls._is_mode_feasible(recursive_cost, budget_snapshot):
            return {"promote": False, "reason": "recursive_not_budget_feasible"}

        return {
            "promote": True,
            "reason": "deterministic_recursive_promotion_high_threshold",
        }

    @classmethod
    def _is_mode_feasible(
        cls,
        mode_cost: Dict[str, Any],
        budget_snapshot: Dict[str, Any],
    ) -> bool:
        rem_calls = budget_snapshot.get("remaining_tool_calls")
        rem_proxy_tokens = budget_snapshot.get("remaining_token_proxy")
        rem_total_tokens = budget_snapshot.get("remaining_total_tokens")
        # Prefer true-token budget when available; otherwise fallback to proxy budget.
        rem_tokens = rem_total_tokens if isinstance(rem_total_tokens, int) else rem_proxy_tokens
        rem_runtime = budget_snapshot.get("remaining_runtime_seconds")
        # Keep a risk buffer to reduce over-budget finishes caused by run-time variance.
        calls_buffer = 0.9
        tokens_buffer = 0.9
        runtime_buffer = 0.8
        if isinstance(rem_runtime, (int, float)):
            if float(rem_runtime) <= 45.0:
                runtime_buffer = 0.7
            elif float(rem_runtime) <= 60.0:
                runtime_buffer = 0.75

        if isinstance(rem_calls, int) and mode_cost["tool_calls"] > max(0, int(rem_calls * calls_buffer)):
            return False
        if isinstance(rem_tokens, int) and mode_cost["token_proxy"] > max(0, int(rem_tokens * tokens_buffer)):
            return False
        if isinstance(rem_runtime, (int, float)) and mode_cost["runtime_seconds"] > max(0.0, float(rem_runtime) * runtime_buffer):
            return False
        return True

    @classmethod
    def _best_feasible_mode(
        cls,
        initial_mode: str,
        budget_snapshot: Dict[str, Any],
        complexity: int,
    ) -> Dict[str, Any]:
        costs_by_mode = {
            mode: cls._prior_cost_for_mode(mode, complexity) for mode in cls.MODE_ORDER
        }
        initial_cost = costs_by_mode.get(initial_mode, costs_by_mode["shallow"])
        if cls._is_mode_feasible(initial_cost, budget_snapshot):
            return {
                "mode_final": initial_mode,
                "override": False,
                "override_reason": None,
                "costs_by_mode": costs_by_mode,
            }

        for mode in cls.MODE_QUALITY_PRIORITY:
            if cls._is_mode_feasible(costs_by_mode[mode], budget_snapshot):
                return {
                    "mode_final": mode,
                    "override": mode != initial_mode,
                    "override_reason": f"budget_feasible_mode_{mode}",
                    "costs_by_mode": costs_by_mode,
                }
        return {
            "mode_final": "direct",
            "override": initial_mode != "direct",
            "override_reason": "no_mode_fits_budget_forced_direct",
            "costs_by_mode": costs_by_mode,
        }

    def run(self, state: PitchState) -> PitchState:
        idea = _sget(state, "idea", "")
        refined = _sget(state, "refined_idea", "")
        combined = f"{idea}\n{refined}".strip()
        validation_threshold = int(_sget(state, "validation_threshold", 70) or 70)
        max_validation_retries = int(_sget(state, "max_validation_retries", 1) or 1)
        budget_snapshot = self._compute_remaining_budget(state)
        usage = _empty_token_usage()
        forced_mode = str(_sget(state, "forced_controller_mode", "") or "").strip().lower()
        forced_mode_applied = forced_mode in self.MODE_ORDER
        if forced_mode_applied:
            fallback = self._fallback_decision(combined)
            decision = ControllerDecision(
                mode=forced_mode,  # type: ignore[arg-type]
                confidence=1.0,
                rationale=f"Forced controller mode `{forced_mode}` for fixed-policy baseline.",
                estimated_complexity=fallback.estimated_complexity,
                expected_tool_calls_delta=fallback.expected_tool_calls_delta,
                expected_token_proxy_delta=fallback.expected_token_proxy_delta,
                expected_runtime_seconds_delta=fallback.expected_runtime_seconds_delta,
                triggers=["forced_mode_baseline"],
            )
        else:
            try:
                decision, usage = _invoke_structured_with_usage(
                    self.structured_llm,
                    self.prompt.format_messages(
                        idea=idea,
                        refined_idea=refined,
                        validation_threshold=validation_threshold,
                        max_validation_retries=max_validation_retries,
                        budget_snapshot=json.dumps(budget_snapshot, indent=2),
                    ),
                )
            except Exception:  # noqa: BLE001
                decision = self._fallback_decision(combined)
        token_usage = _merge_token_usage(state, usage)

        chosen_mode_initial = decision.mode
        chosen_mode_final = decision.mode
        budget_override = False
        deterministic_trigger = None
        guardrail_trigger = None

        if forced_mode_applied:
            chosen_mode_initial = forced_mode  # type: ignore[assignment]
            chosen_mode_final = forced_mode  # type: ignore[assignment]
            calibrated_selected = self._calibrate_selected_cost(
                decision,
                mode_override=chosen_mode_initial,
            )
            feasible_pick = {
                "mode_final": chosen_mode_final,
                "override": False,
                "override_reason": "forced_mode_no_override",
                "costs_by_mode": {
                    mode: self._prior_cost_for_mode(mode, decision.estimated_complexity)
                    for mode in self.MODE_ORDER
                },
            }
        else:
            promotion = self._should_promote_recursive(
                decision=decision,
                combined_text=combined,
                validation_threshold=validation_threshold,
                max_validation_retries=max_validation_retries,
                budget_snapshot=budget_snapshot,
            )
            if promotion["promote"] and chosen_mode_initial != "recursive":
                chosen_mode_initial = "recursive"
                deterministic_trigger = promotion["reason"]

            calibrated_selected = self._calibrate_selected_cost(
                decision,
                mode_override=chosen_mode_initial,
            )
            feasible_pick = self._best_feasible_mode(
                initial_mode=chosen_mode_initial,
                budget_snapshot=budget_snapshot,
                complexity=decision.estimated_complexity,
            )
            chosen_mode_final = feasible_pick["mode_final"]
            budget_override = feasible_pick["override"]
            budget_guard = self._budget_guardrail_mode(budget_snapshot)
            if budget_override and feasible_pick["override_reason"]:
                guardrail_trigger = feasible_pick["override_reason"]
            if budget_guard is not None and budget_guard != chosen_mode_final:
                chosen_mode_final = budget_guard
                budget_override = True
                guardrail_trigger = f"budget_guardrail_forced_{budget_guard}"

        depth_map = {"direct": 0, "shallow": 1, "recursive": 2}
        decisions = list(_sget(state, "controller_decisions", []))
        tool_audit = list(_sget(state, "tool_audit", []))
        trigger_chain = list(decision.triggers)
        if deterministic_trigger:
            trigger_chain.append(deterministic_trigger)
        if guardrail_trigger:
            trigger_chain.append(guardrail_trigger)
        decisions.append(
            {
                "mode_initial": chosen_mode_initial,
                "mode_final": chosen_mode_final,
                "confidence": decision.confidence,
                "estimated_complexity": decision.estimated_complexity,
                "triggers": trigger_chain,
                "rationale": decision.rationale,
                "expected_tool_calls_delta": calibrated_selected["tool_calls"],
                "expected_token_proxy_delta": calibrated_selected["token_proxy"],
                "expected_runtime_seconds_delta": calibrated_selected["runtime_seconds"],
                "costs_by_mode": feasible_pick["costs_by_mode"],
                "budget_override": budget_override,
                "deterministic_recursive_promotion": bool(deterministic_trigger),
                "forced_mode_applied": forced_mode_applied,
            }
        )
        tool_audit.append(
            {
                "agent": "adaptive_controller",
                "tool": "mode_selector",
                "status": "ok",
                "mode_initial": chosen_mode_initial,
                "mode_final": chosen_mode_final,
                "confidence": decision.confidence,
                "estimated_complexity": decision.estimated_complexity,
                "triggers": trigger_chain,
                "expected_tool_calls_delta": calibrated_selected["tool_calls"],
                "expected_token_proxy_delta": calibrated_selected["token_proxy"],
                "expected_runtime_seconds_delta": calibrated_selected["runtime_seconds"],
                "costs_by_mode": feasible_pick["costs_by_mode"],
                "budget_override": budget_override,
                "deterministic_recursive_promotion": bool(deterministic_trigger),
                "forced_mode_applied": forced_mode_applied,
            }
        )
        return {
            "controller_mode": chosen_mode_final,
            "controller_mode_initial": chosen_mode_initial,
            "controller_budget_override": budget_override,
            "controller_confidence": decision.confidence,
            "controller_rationale": decision.rationale,
            "controller_expected_cost": {
                "expected_tool_calls_delta": calibrated_selected["tool_calls"],
                "expected_token_proxy_delta": calibrated_selected["token_proxy"],
                "expected_runtime_seconds_delta": calibrated_selected["runtime_seconds"],
                "raw_model_estimate": calibrated_selected["raw"],
                "prior_for_mode": calibrated_selected["prior"],
                "costs_by_mode": feasible_pick["costs_by_mode"],
                "mode_for_estimate": chosen_mode_initial,
            },
            "controller_budget_snapshot": budget_snapshot,
            "controller_decisions": decisions,
            "decomposition_depth_target": depth_map.get(chosen_mode_final, 1),
            "tool_audit": tool_audit,
            "token_usage": token_usage,
        }


class MarketResearchAgent:
    def __init__(
        self,
        llm: ChatOpenAI,
        strict_tools: bool = True,
        enable_trends: bool = True,
    ):
        self.structured_llm = llm.with_structured_output(MarketOutput, include_raw=True)
        self.keyword_llm = llm.with_structured_output(TrendKeywords, include_raw=True)
        self.search_tool = MarketSearchTool()
        self.trends_tool = GoogleTrendsTool() if enable_trends else None
        self.strict_tools = strict_tools
        self.enable_trends = enable_trends
        self.keyword_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "# Role\n"
                    "You are a market keyword extractor.\n\n"
                    "# Instructions\n"
                    "- Extract domain-agnostic trend keywords for startup market research.\n"
                    "- Focus on product type, buyer segment, core technology, problem space, and industry context.\n"
                    "- Keep keywords short and directly searchable.\n\n"
                    "# Output Format\n"
                    "- Return content that strictly matches the structured schema fields.",
                ),
                (
                    "human",
                    "Startup concept:\n{refined_idea}\n\n"
                    "Return keyword phrases covering product type, buyer segment, core technology, "
                    "problem space, and industry context.",
                ),
            ]
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "# Role\n"
                    "You are a market research analyst.\n\n"
                    "# Instructions\n"
                    "- Use the provided web research snippets and trend signals.\n"
                    "- Synthesize target market, market size, trends, competitors, and differentiation gaps.\n"
                    "- Prioritize evidence-grounded statements and avoid speculation.\n\n"
                    "# Output Format\n"
                    "- Return content that strictly matches the structured schema fields.",
                ),
                (
                    "human",
                    "Startup concept:\n{refined_idea}\n\n"
                    "Web research snippets:\n{search_results}\n\n"
                    "Keyword trend signals:\n{trend_signals}\n\n"
                    "Produce market analysis for investors.",
                ),
            ]
        )

    def run(self, state: PitchState) -> PitchState:
        refined_idea = _sget(state, "refined_idea", "")
        total_usage = _empty_token_usage()
        retry_count = _sget(state, "retry_count", 0)
        query_suffix = (
            " prioritize authoritative and recent sources with concrete numbers"
            if retry_count > 0
            else ""
        )
        query = f"startup market size competitors trends for: {refined_idea}{query_suffix}"
        search_payload = self.search_tool.search(query)
        if self.strict_tools and search_payload["status"] != "ok":
            raise RuntimeError(f"Market search tool unavailable: {search_payload['error']}")
        search_results = search_payload["results_json"]
        try:
            kw_model, kw_usage = _invoke_structured_with_usage(
                self.keyword_llm,
                self.keyword_prompt.format_messages(refined_idea=refined_idea),
            )
            total_usage = _merge_token_usage({"token_usage": total_usage}, kw_usage)
            extracted_keywords = [k.strip() for k in kw_model.keywords if k.strip()]
        except Exception:  # noqa: BLE001
            extracted_keywords = _fallback_keywords(refined_idea)
        if self.enable_trends and self.trends_tool is not None:
            trend_payload = self.trends_tool.fetch(extracted_keywords[:5])
        else:
            trend_payload = {
                "status": "skipped",
                "keywords": extracted_keywords[:5],
                "data": {},
                "error": "Google Trends disabled by configuration.",
            }

        result, market_usage = _invoke_structured_with_usage(
            self.structured_llm,
            self.prompt.format_messages(
                refined_idea=refined_idea,
                search_results=search_results,
                trend_signals=json.dumps(trend_payload, indent=2),
            ),
        )
        total_usage = _merge_token_usage({"token_usage": total_usage}, market_usage)
        token_usage = _merge_token_usage(state, total_usage)
        market_analysis = (
            f"Target Market: {result.target_market}\n"
            f"Market Size: {result.market_size}\n"
            f"Trends: {result.trends}\n"
            f"Competitors: {result.competitors}\n"
            f"Differentiation Gaps: {result.differentiation_gaps}"
        )
        tool_audit = list(_sget(state, "tool_audit", []))
        tool_audit.append(
            {
                "agent": "market_research",
                "tool": "linkup_search",
                "query": query,
                "status": search_payload["status"],
                "source_count": len(search_payload["sources"]),
                "dropped_sources": search_payload.get("dropped_sources", 0),
                "error": search_payload["error"],
            }
        )
        tool_audit.append(
            {
                "agent": "market_research",
                "tool": "google_trends",
                "status": trend_payload["status"],
                "keyword_count": len(trend_payload.get("keywords", [])),
                "keywords": trend_payload.get("keywords", []),
                "error": trend_payload.get("error", ""),
            }
        )
        return {
            "market_analysis": market_analysis,
            "market_sources": search_payload["sources"],
            "market_evidence": search_payload["results"],
            "trend_signals": trend_payload,
            "tool_audit": tool_audit,
            "token_usage": token_usage,
        }


class SourceValidatorAgent:
    def __init__(self, llm: ChatOpenAI):
        self.structured_llm = llm.with_structured_output(ValidationOutput, include_raw=True)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "# Role\n"
                    "You are a source validation analyst.\n\n"
                    "# Instructions\n"
                    "- Verify material market claims only against provided evidence snippets and URLs.\n"
                    "- Do not invent sources or unsupported claims.\n"
                    "- Provide claim-level verdicts, confidence, rationale, and supporting sources.\n\n"
                    "# Output Format\n"
                    "- Return content that strictly matches the structured schema fields.",
                ),
                (
                    "human",
                    "Startup concept:\n{refined_idea}\n\n"
                    "Market analysis draft:\n{market_analysis}\n\n"
                    "Evidence snippets (title/url/content):\n{market_evidence}\n\n"
                    "Return a claim-by-claim validation report with confidence and source links.",
                ),
            ]
        )

    def run(self, state: PitchState) -> PitchState:
        evidence_json = json.dumps(_sget(state, "market_evidence", []), indent=2)
        result, usage = _invoke_structured_with_usage(
            self.structured_llm,
            self.prompt.format_messages(
                refined_idea=_sget(state, "refined_idea", ""),
                market_analysis=_sget(state, "market_analysis", ""),
                market_evidence=evidence_json,
            ),
        )
        token_usage = _merge_token_usage(state, usage)
        report: Dict[str, Any] = json.loads(result.model_dump_json())
        deterministic_score = _deterministic_reliability_score(report.get("claims", []))
        report["reliability_score"] = deterministic_score
        validated = (
            f"{_sget(state, 'market_analysis', '')}\n\n"
            f"Validation Score: {deterministic_score}/100\n"
            f"Validated Summary: {result.validated_summary}\n"
            f"Evidence Gaps: {result.evidence_gaps}"
        )
        tool_audit = list(_sget(state, "tool_audit", []))
        tool_audit.append(
            {
                "agent": "source_validator",
                "tool": "llm_claim_verifier",
                "status": "ok",
                "claims_checked": len(report.get("claims", [])),
                "reliability_score": deterministic_score,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        )
        return {
            "validation_report": report,
            "validated_market_analysis": validated,
            "needs_revision": deterministic_score < _sget(state, "validation_threshold", 70),
            "tool_audit": tool_audit,
            "token_usage": token_usage,
        }


class DirectStrategyAgent:
    def __init__(
        self,
        llm: ChatOpenAI,
        strict_tools: bool = True,
        enable_trends: bool = True,
    ):
        self.structured_llm = llm.with_structured_output(DirectStrategyOutput, include_raw=True)
        self.validator_agent = SourceValidatorAgent(llm)
        self.search_tool = MarketSearchTool()
        self.trends_tool = GoogleTrendsTool() if enable_trends else None
        self.calc_tool = BusinessCalcTool()
        self.scenario_tool = ScenarioAnalysisTool()
        self.strict_tools = strict_tools
        self.enable_trends = enable_trends
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "# Role\n"
                    "You are a direct (low-decomposition) startup strategist.\n\n"
                    "# Instructions\n"
                    "- Produce market analysis and business model in one synthesis pass.\n"
                    "- Also propose bounded Year-1 assumptions (users/arpu/gross margin).\n"
                    "- Keep outputs concise, evidence-aware, and investor-ready.\n\n"
                    "# Output Format\n"
                    "- Return content that strictly matches the structured schema fields.",
                ),
                (
                    "human",
                    "Startup concept:\n{refined_idea}\n\n"
                    "Web snippets:\n{search_results}\n\n"
                    "Trend signals:\n{trend_signals}\n\n"
                    "Provide integrated market + business + assumptions output.",
                ),
            ]
        )

    def run(self, state: PitchState) -> PitchState:
        refined_idea = _sget(state, "refined_idea", "")
        total_usage = _empty_token_usage()
        query = f"startup market size competitors trends for: {refined_idea}"
        search_payload = self.search_tool.search(query)
        if self.strict_tools and search_payload["status"] != "ok":
            raise RuntimeError(f"Market search tool unavailable: {search_payload['error']}")

        extracted_keywords = _fallback_keywords(refined_idea)
        if self.enable_trends and self.trends_tool is not None:
            trend_payload = self.trends_tool.fetch(extracted_keywords[:5])
        else:
            trend_payload = {
                "status": "skipped",
                "keywords": extracted_keywords[:5],
                "data": {},
                "error": "Google Trends disabled by configuration.",
            }

        result, direct_usage = _invoke_structured_with_usage(
            self.structured_llm,
            self.prompt.format_messages(
                refined_idea=refined_idea,
                search_results=search_payload["results_json"],
                trend_signals=json.dumps(trend_payload, indent=2),
            ),
        )
        total_usage = _merge_token_usage({"token_usage": total_usage}, direct_usage)
        token_usage = _merge_token_usage(state, total_usage)
        market_analysis = (
            f"Target Market: {result.target_market}\n"
            f"Market Size: {result.market_size}\n"
            f"Trends: {result.trends}\n"
            f"Competitors: {result.competitors}\n"
            f"Differentiation Gaps: {result.differentiation_gaps}"
        )
        assumptions = {
            "users_year1": max(1000, min(500000, int(result.users_year1))),
            "arpu_monthly": round(max(2.0, min(300.0, float(result.arpu_monthly))), 2),
            "gross_margin": round(max(0.2, min(0.95, float(result.gross_margin))), 3),
            "rationale": result.assumptions_rationale,
        }
        calc_script = f"""
users_year1 = {assumptions["users_year1"]}
arpu_monthly = {assumptions["arpu_monthly"]}
annual_revenue = users_year1 * arpu_monthly * 12
gross_margin = {assumptions["gross_margin"]}
gross_profit = annual_revenue * gross_margin
print(f'Year1 Revenue: ${{annual_revenue:,.0f}}')
print(f'Year1 Gross Profit: ${{gross_profit:,.0f}}')
"""
        calc_output = self.calc_tool.run(calc_script)
        if self.strict_tools and calc_output.startswith("Python calc failed:"):
            raise RuntimeError(calc_output)
        scenario_output = self.scenario_tool.run(
            users_year1=assumptions["users_year1"],
            arpu_monthly=assumptions["arpu_monthly"],
            gross_margin=assumptions["gross_margin"],
        )
        business_model = (
            f"Revenue Streams: {result.revenue_streams}\n"
            f"Pricing Strategy: {result.pricing_strategy}\n"
            f"Cost Structure: {result.cost_structure}\n"
            f"Unit Economics: {result.unit_economics}\n"
            f"Financial Projection: {result.financial_projection}\n"
            f"Financial Assumptions:\n{json.dumps(assumptions, indent=2)}\n"
            f"Calculator Baseline:\n{calc_output}\n"
            f"Scenario Analysis:\n{json.dumps(scenario_output, indent=2)}"
        )
        tool_audit = list(_sget(state, "tool_audit", []))
        tool_audit.append(
            {
                "agent": "adaptive_direct_strategy",
                "tool": "llm_direct_synthesis",
                "status": "ok",
                "prompt_tokens": direct_usage.get("prompt_tokens", 0),
                "completion_tokens": direct_usage.get("completion_tokens", 0),
                "total_tokens": direct_usage.get("total_tokens", 0),
            }
        )
        tool_audit.append(
            {
                "agent": "adaptive_direct_strategy",
                "tool": "linkup_search",
                "query": query,
                "status": search_payload["status"],
                "source_count": len(search_payload["sources"]),
                "dropped_sources": search_payload.get("dropped_sources", 0),
                "error": search_payload["error"],
            }
        )
        tool_audit.append(
            {
                "agent": "adaptive_direct_strategy",
                "tool": "google_trends",
                "status": trend_payload["status"],
                "keyword_count": len(trend_payload.get("keywords", [])),
                "keywords": trend_payload.get("keywords", []),
                "error": trend_payload.get("error", ""),
            }
        )
        tool_audit.append(
            {
                "agent": "adaptive_direct_strategy",
                "tool": "python_calc",
                "status": "ok" if not calc_output.startswith("Python calc failed:") else "error",
                "assumptions": assumptions,
                "error": calc_output if calc_output.startswith("Python calc failed:") else "",
            }
        )
        tool_audit.append(
            {
                "agent": "adaptive_direct_strategy",
                "tool": "scenario_analysis",
                "status": "ok",
                "scenario_count": len(scenario_output),
                "error": "",
            }
        )
        candidate_state = {
            **(state if isinstance(state, dict) else state.model_dump()),
            "market_analysis": market_analysis,
            "market_sources": search_payload["sources"],
            "market_evidence": search_payload["results"],
            "trend_signals": trend_payload,
            "business_model": business_model,
            "financial_assumptions": assumptions,
            "scenario_analysis": scenario_output,
            "tool_audit": tool_audit,
            "token_usage": token_usage,
            "decomposition_depth_realized": 0,
        }
        validated_update = self.validator_agent.run(candidate_state)
        candidate_state.update(validated_update)
        return {
            "market_analysis": candidate_state["market_analysis"],
            "market_sources": candidate_state["market_sources"],
            "market_evidence": candidate_state["market_evidence"],
            "trend_signals": candidate_state["trend_signals"],
            "business_model": candidate_state["business_model"],
            "financial_assumptions": candidate_state["financial_assumptions"],
            "scenario_analysis": candidate_state["scenario_analysis"],
            "validation_report": candidate_state.get("validation_report"),
            "validated_market_analysis": candidate_state.get("validated_market_analysis"),
            "needs_revision": candidate_state.get("needs_revision", False),
            "tool_audit": candidate_state["tool_audit"],
            "token_usage": candidate_state.get("token_usage", token_usage),
            "decomposition_depth_realized": 0,
        }


class BusinessModelAgent:
    def __init__(self, llm: ChatOpenAI, strict_tools: bool = True):
        self.structured_llm = llm.with_structured_output(BusinessOutput, include_raw=True)
        self.assumptions_llm = llm.with_structured_output(FinancialAssumptions, include_raw=True)
        self.calc_tool = BusinessCalcTool()
        self.scenario_tool = ScenarioAnalysisTool()
        self.strict_tools = strict_tools
        self.assumptions_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "# Role\n"
                    "You are a startup financial assumptions analyst.\n\n"
                    "# Instructions\n"
                    "- Generate realistic Year-1 assumptions from startup and market context.\n"
                    "- Use conservative, explainable values.\n"
                    "- Stay strictly within schema bounds.\n\n"
                    "# Output Format\n"
                    "- Return content that strictly matches the structured schema fields.",
                ),
                (
                    "human",
                    "Startup concept:\n{refined_idea}\n\n"
                    "Validated market analysis:\n{market_analysis}\n\n"
                    "Trend signals:\n{trend_signals}\n\n"
                    "Validation score:\n{validation_score}\n\n"
                    "Return assumptions for users_year1, arpu_monthly, gross_margin and rationale.",
                ),
            ]
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "# Role\n"
                    "You are a startup business model strategist.\n\n"
                    "# Instructions\n"
                    "- Use supplied market analysis, assumptions, calculator output, and scenario output.\n"
                    "- Produce practical revenue model, pricing, costs, unit economics, and projection narrative.\n"
                    "- Keep recommendations coherent with the provided quantitative inputs.\n\n"
                    "# Output Format\n"
                    "- Return content that strictly matches the structured schema fields.",
                ),
                (
                    "human",
                    "Startup concept:\n{refined_idea}\n\n"
                    "Market analysis:\n{market_analysis}\n\n"
                    "Financial assumptions:\n{financial_assumptions}\n\n"
                    "Financial calculator output:\n{calc_output}\n\n"
                    "Scenario analysis output:\n{scenario_output}\n\n"
                    "Provide revenue model, pricing, costs, and projection.",
                ),
            ]
        )

    def run(self, state: PitchState) -> PitchState:
        assumptions_model, assumptions_usage = _invoke_structured_with_usage(
            self.assumptions_llm,
            self.assumptions_prompt.format_messages(
                refined_idea=_sget(state, "refined_idea", ""),
                market_analysis=(
                    _sget(state, "validated_market_analysis")
                    or _sget(state, "market_analysis", "")
                ),
                trend_signals=json.dumps(_sget(state, "trend_signals", {}), indent=2),
                validation_score=_sget(state, "validation_report", {}).get(
                    "reliability_score", 0
                ),
            ),
        )
        total_usage = _merge_token_usage({"token_usage": _empty_token_usage()}, assumptions_usage)
        assumptions = {
            "users_year1": max(1000, min(500000, int(assumptions_model.users_year1))),
            "arpu_monthly": round(max(2.0, min(300.0, float(assumptions_model.arpu_monthly))), 2),
            "gross_margin": round(max(0.2, min(0.95, float(assumptions_model.gross_margin))), 3),
            "rationale": assumptions_model.rationale,
        }
        calc_script = f"""
users_year1 = {assumptions["users_year1"]}
arpu_monthly = {assumptions["arpu_monthly"]}
annual_revenue = users_year1 * arpu_monthly * 12
gross_margin = {assumptions["gross_margin"]}
gross_profit = annual_revenue * gross_margin
print(f'Year1 Revenue: ${{annual_revenue:,.0f}}')
print(f'Year1 Gross Profit: ${{gross_profit:,.0f}}')
"""
        calc_output = self.calc_tool.run(calc_script)
        if self.strict_tools and calc_output.startswith("Python calc failed:"):
            raise RuntimeError(calc_output)
        scenario_output = self.scenario_tool.run(
            users_year1=assumptions["users_year1"],
            arpu_monthly=assumptions["arpu_monthly"],
            gross_margin=assumptions["gross_margin"],
        )

        result, business_usage = _invoke_structured_with_usage(
            self.structured_llm,
            self.prompt.format_messages(
                refined_idea=_sget(state, "refined_idea", ""),
                market_analysis=(
                    _sget(state, "validated_market_analysis")
                    or _sget(state, "market_analysis", "")
                ),
                financial_assumptions=json.dumps(assumptions, indent=2),
                calc_output=calc_output,
                scenario_output=json.dumps(scenario_output, indent=2),
            ),
        )
        total_usage = _merge_token_usage({"token_usage": total_usage}, business_usage)
        token_usage = _merge_token_usage(state, total_usage)
        business_model = (
            f"Revenue Streams: {result.revenue_streams}\n"
            f"Pricing Strategy: {result.pricing_strategy}\n"
            f"Cost Structure: {result.cost_structure}\n"
            f"Unit Economics: {result.unit_economics}\n"
            f"Financial Projection: {result.financial_projection}\n"
            f"Financial Assumptions:\n{json.dumps(assumptions, indent=2)}\n"
            f"Calculator Baseline:\n{calc_output}\n"
            f"Scenario Analysis:\n{json.dumps(scenario_output, indent=2)}"
        )
        tool_audit = list(_sget(state, "tool_audit", []))
        tool_audit.append(
            {
                "agent": "business_model",
                "tool": "python_calc",
                "status": "ok" if not calc_output.startswith("Python calc failed:") else "error",
                "assumptions": assumptions,
                "error": calc_output if calc_output.startswith("Python calc failed:") else "",
                "prompt_tokens_assumptions": assumptions_usage.get("prompt_tokens", 0),
                "completion_tokens_assumptions": assumptions_usage.get("completion_tokens", 0),
                "total_tokens_assumptions": assumptions_usage.get("total_tokens", 0),
                "prompt_tokens_business": business_usage.get("prompt_tokens", 0),
                "completion_tokens_business": business_usage.get("completion_tokens", 0),
                "total_tokens_business": business_usage.get("total_tokens", 0),
            }
        )
        tool_audit.append(
            {
                "agent": "business_model",
                "tool": "scenario_analysis",
                "status": "ok",
                "scenario_count": len(scenario_output),
                "error": "",
            }
        )
        return {
            "business_model": business_model,
            "financial_assumptions": assumptions,
            "scenario_analysis": scenario_output,
            "tool_audit": tool_audit,
            "token_usage": token_usage,
        }


class PitchDeckGeneratorAgent:
    def __init__(self, llm: ChatOpenAI, output_dir: str = "output"):
        self.structured_llm = llm.with_structured_output(PitchSlides, include_raw=True)
        self.output_dir = Path(output_dir)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "# Role\n"
                    "You are an investor pitch deck strategist.\n\n"
                    "# Instructions\n"
                    "- Convert refined idea, market analysis, and business model into concise slide-ready content.\n"
                    "- Keep language clear, specific, and investor-oriented.\n"
                    "- Cover all required sections.\n\n"
                    "# Output Format\n"
                    "- Return content that strictly matches the structured schema fields.",
                ),
                (
                    "human",
                    "Refined idea:\n{refined_idea}\n\n"
                    "Market analysis:\n{market_analysis}\n\n"
                    "Business model:\n{business_model}\n\n"
                    "Return all required slide sections.",
                ),
            ]
        )

    def run(self, state: PitchState) -> PitchState:
        slide_model, usage = _invoke_structured_with_usage(
            self.structured_llm,
            self.prompt.format_messages(
                refined_idea=_sget(state, "refined_idea", ""),
                market_analysis=(
                    _sget(state, "validated_market_analysis")
                    or _sget(state, "market_analysis", "")
                ),
                business_model=_sget(state, "business_model", ""),
            ),
        )
        token_usage = _merge_token_usage(state, usage)
        slide_dict: Dict[str, str] = json.loads(slide_model.model_dump_json())

        summary_line = _sget(state, "refined_idea", "")
        words = re.findall(r"[a-zA-Z0-9]+", summary_line.lower())
        safe_name = "startup_pitch_deck"
        if words:
            safe_name = "pitch_" + "_".join(words[:6])
        ppt_path = self.output_dir / f"{safe_name}.pptx"
        saved = generate_pitch_deck(slide_dict, str(ppt_path))

        tool_audit = list(_sget(state, "tool_audit", []))
        tool_audit.append(
            {
                "agent": "pitch_deck_generator",
                "tool": "python_pptx",
                "status": "ok",
                "output_path": saved,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        )
        return {
            "pitch_content": slide_dict,
            "ppt_path": saved,
            "tool_audit": tool_audit,
            "token_usage": token_usage,
        }
