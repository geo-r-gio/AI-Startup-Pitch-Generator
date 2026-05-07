from __future__ import annotations

import json
import math
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


def _limit_evidence_items(
    evidence: List[Dict[str, Any]],
    *,
    max_items: int,
    content_chars: int,
) -> List[Dict[str, Any]]:
    """Keep evidence prompts bounded while preserving titles and source URLs."""
    limited: List[Dict[str, Any]] = []
    seen_urls = set()
    for item in evidence:
        if len(limited) >= max_items:
            break
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        limited.append(
            {
                "title": str(item.get("title", ""))[:160],
                "url": url,
                "domain": str(item.get("domain", ""))[:120],
                "content": str(item.get("content", ""))[:content_chars],
            }
        )
    return limited


def _limited_search_payload(
    payload: Dict[str, Any],
    *,
    max_items: int,
    content_chars: int,
) -> Dict[str, Any]:
    """Return a compact search payload for LLM prompts and downstream validation."""
    evidence = _limit_evidence_items(
        payload.get("results", []) or [],
        max_items=max_items,
        content_chars=content_chars,
    )
    compact = dict(payload)
    compact["results"] = evidence
    compact["results_json"] = json.dumps(evidence, indent=2)
    compact["sources"] = [item["url"] for item in evidence if item.get("url")]
    compact["source_count_raw"] = len(payload.get("sources", []) or [])
    compact["evidence_limit"] = {
        "max_items": max_items,
        "content_chars": content_chars,
        "raw_items": len(payload.get("results", []) or []),
        "used_items": len(evidence),
    }
    return compact


def _evidence_budget_for_mode(mode: str, remaining_total_tokens: Any = None) -> Dict[str, int]:
    """Mode- and budget-aware evidence limits for synthesis/validation prompts."""
    mode_l = str(mode or "").strip().lower()
    if mode_l == "direct":
        max_items, content_chars = 5, 220
    elif mode_l == "recursive":
        max_items, content_chars = 10, 280
    else:
        max_items, content_chars = 8, 260

    if isinstance(remaining_total_tokens, int):
        if remaining_total_tokens <= 3500:
            max_items = min(max_items, 3)
            content_chars = min(content_chars, 180)
        elif remaining_total_tokens <= 7000:
            max_items = min(max_items, 5)
            content_chars = min(content_chars, 220)

    return {
        "max_items": max_items,
        "content_chars": content_chars,
    }



def _merge_evidence_items(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge evidence lists while preserving order and dropping duplicate URLs."""
    merged: List[Dict[str, Any]] = []
    seen = set()
    for item in list(primary or []) + list(secondary or []):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "") or "").strip()
        key = url or f"{item.get('title', '')}:{str(item.get('content', ''))[:80]}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _merge_unique(a: List[str], b: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in list(a or []) + list(b or []):
        sx = str(x or "").strip()
        if sx and sx not in seen:
            seen.add(sx)
            out.append(sx)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _token_overlap_score(a: str, b: str) -> float:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "are", "is",
        "can", "may", "could", "would", "their", "using", "based", "market",
    }
    ta = {t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", (a or "").lower()) if t not in stop}
    tb = {t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", (b or "").lower()) if t not in stop}
    if not ta or not tb:
        return 0.0
    return round(len(ta & tb) / max(1, len(ta)), 4)


def _claim_repair_category(claim_text: str) -> str:
    text = (claim_text or "").lower()
    numeric_markers = ["$", "%", "cagr", "billion", "million", "trillion", "market size", "forecast", "projected", "growth"]
    gap_markers = ["gap", "opportunity", "underserved", "differentiation", "white space", "lack", "limited", "specific"]
    competitor_markers = ["competitor", "competitors", "platforms", "vendors", "solutions", "players", "offerings"]
    trend_markers = ["trend", "adoption", "increasing", "demand", "shift", "predictive", "automation", "real-time"]
    regulatory_markers = ["hipaa", "compliance", "regulatory", "regulation", "audit", "privacy", "security", "authorization"]
    if any(m in text for m in numeric_markers):
        return "market_size_or_numeric"
    if any(m in text for m in regulatory_markers):
        return "regulatory_or_compliance"
    if any(m in text for m in competitor_markers):
        return "competitor_landscape"
    if any(m in text for m in trend_markers):
        return "trend_or_adoption"
    if any(m in text for m in gap_markers):
        return "gap_or_opportunity"
    return "general_market_claim"


def _claim_materiality(claim_text: str, category: str | None = None) -> int:
    text = (claim_text or "").lower()
    category = category or _claim_repair_category(text)
    materiality = 2
    if category == "market_size_or_numeric":
        materiality = 5
    elif category == "regulatory_or_compliance":
        materiality = 5
    elif category in {"competitor_landscape", "trend_or_adoption"}:
        materiality = 4
    elif category == "gap_or_opportunity":
        materiality = 3
    if any(m in text for m in ["regulatory", "compliance", "clinical", "healthcare", "risk", "patient"]):
        materiality = min(5, materiality + 1)
    if any(m in text for m in ["may", "could", "potential", "hypothesis"]):
        materiality = max(1, materiality - 1)
    return max(1, min(5, materiality))


def _claim_failure_type(claim: Dict[str, Any], evidence_gaps: str = "") -> str:
    text = str(claim.get("claim", "") or claim.get("claim_text", "") or "").lower()
    verdict = str(claim.get("verdict", "") or "").strip().lower()
    rationale = str(claim.get("rationale", "") or "").lower()
    combined = " ".join([text, rationale, evidence_gaps.lower()])
    if verdict == "supported":
        return "none"
    if any(m in combined for m in ["contradict", "conflict", "inconsistent"]):
        return "contradictory_evidence"
    if any(m in combined for m in ["cagr", "percent", "%", "market size", "billion", "million", "forecast"]):
        return "speculative_number"
    if any(m in combined for m in ["outdated", "older", "old source"]):
        return "outdated"
    if any(m in combined for m in ["broad", "generic", "not specific", "indirect", "adjacent"]):
        return "too_broad"
    if any(m in combined for m in ["unclear", "ambiguous", "attribution"]):
        return "unclear_attribution"
    return "missing_evidence"


def _repair_action_for_claim(claim: Dict[str, Any]) -> str:
    """Choose the cheapest safe repair action for a weak claim."""
    if bool(claim.get("coverage_enhancement")) or str(claim.get("failure_type", "")).strip() == "coverage_gap":
        return "coverage_addition"
    claim_text = str(claim.get("claim", "") or claim.get("claim_text", "") or "")
    text = claim_text.lower()
    verdict = str(claim.get("verdict", "") or "").strip().lower()
    category = _claim_repair_category(claim_text)
    supporting_sources = claim.get("supporting_sources", []) or []
    speculative_markers = ["may", "might", "could", "potentially", "likely", "appears", "suggests", "opportunity", "white space", "gap", "underserved", "differentiation"]
    negative_capability_markers = ["lack", "lacks", "lacking", "limited", "without", "do not", "does not", "not offer", "not provide", "missing", "fails to", "few solutions"]
    numeric_markers = ["$", "%", "cagr", "billion", "million", "market size", "forecast", "projected"]
    competitor_markers = ["competitor", "competitors", "vendors", "platforms", "solutions", "offerings"]
    is_negative_competitor_gap = any(m in text for m in negative_capability_markers) and any(m in text for m in competitor_markers + ["solution", "solutions", "platform", "platforms"])
    is_speculative_gap = category == "gap_or_opportunity" or any(m in text for m in speculative_markers)
    if category in {"market_size_or_numeric", "regulatory_or_compliance"}:
        return "search_and_replace"
    if verdict == "unsupported" and not supporting_sources:
        return "qualify_or_remove" if (is_negative_competitor_gap or is_speculative_gap) else "remove"
    if any(m in text for m in numeric_markers):
        return "search_and_replace"
    if category == "trend_or_adoption" and not is_speculative_gap:
        return "search_and_replace"
    if category == "competitor_landscape":
        return "qualify_or_remove" if is_negative_competitor_gap else "search_and_replace"
    if verdict == "needs_review" or is_speculative_gap:
        return "qualify_or_remove"
    return "search_and_replace" if supporting_sources else "qualify_or_remove"


def build_claim_repair_plan(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert weak validator claims into explicit, bounded repair actions."""
    plan: List[Dict[str, Any]] = []
    for claim in claims or []:
        if not isinstance(claim, dict):
            continue
        claim_text = str(claim.get("claim") or claim.get("claim_text") or "").strip()
        if not claim_text:
            continue
        category = str(claim.get("category") or _claim_repair_category(claim_text))
        action = _repair_action_for_claim(claim)
        materiality = int(claim.get("materiality") or _claim_materiality(claim_text, category))
        failure_type = str(claim.get("failure_type") or claim.get("failure_type_hint") or _claim_failure_type(claim))
        if action == "search_and_replace":
            instruction = "Search for direct evidence, then replace only if the evidence supports the revised wording."
        elif action == "coverage_addition":
            target_add_count = min(max(1, int(claim.get("target_add_count") or claim.get("missing_claim_count") or 1)), 2)
            instruction = (
                f"Add {target_add_count} concise, source-backed coverage claim(s). "
                "Do not remove or rewrite existing supported claims."
            )
        elif action == "remove":
            instruction = "Remove this unsupported claim; do not replace it with a broader speculative claim."
        else:
            instruction = "Narrow to a conservative hypothesis/limitation or remove it if direct support is absent."
        plan.append({
            "claim_id": str(claim.get("claim_id") or f"c{len(plan)+1}"),
            "claim": claim_text,
            "claim_text": claim_text,
            "verdict": str(claim.get("verdict", "") or "").strip().lower(),
            "confidence": claim.get("confidence"),
            "category": category,
            "materiality": materiality,
            "failure_type": failure_type,
            "action": action,
            "instruction": instruction,
            "supporting_sources": claim.get("supporting_sources", []),
            "rationale": str(claim.get("rationale", "") or "")[:300],
            "coverage_enhancement": bool(claim.get("coverage_enhancement", False)),
            "missing_claim_count": int(claim.get("missing_claim_count", 0) or 0),
            "claims_total_before_repair": int(claim.get("claims_total_before_repair", 0) or 0),
            "target_add_count": min(max(1, int(claim.get("target_add_count") or claim.get("missing_claim_count") or 1)), 2),
            "candidate_queries": claim.get("candidate_queries", []) or [
                f"{claim_text} evidence source",
                f"{category.replace('_', ' ')} market evidence",
            ],
        })
    return plan


def _retrieval_bucket(score: float) -> str:
    if score >= 0.70:
        return "strong"
    if score >= 0.55:
        return "medium"
    if score > 0.0:
        return "weak"
    return "none"


def _source_bound_replacement_ready(
    *,
    category: str,
    materiality: int,
    retrieval_score: float,
    unique_sources: int,
    high_quality_sources: int,
    overlap_score: float,
) -> bool:
    category_key = str(category or "").strip()
    if category_key not in {"market_size_or_numeric", "regulatory_or_compliance"}:
        return False
    if int(unique_sources or 0) < 2:
        return False
    if int(high_quality_sources or 0) < 1:
        return False
    materiality_int = int(materiality or 0)
    overlap = float(overlap_score or 0.0)
    retrieval = float(retrieval_score or 0.0)
    high_quality = int(high_quality_sources or 0)
    if overlap < 0.12:
        return False
    if materiality_int >= 5 and high_quality >= 2 and overlap >= 0.18 and retrieval >= 0.65:
        return True
    if retrieval < 0.75:
        return False
    return materiality_int >= 5


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


class ValidationRubricScores(BaseModel):
    evidence_grounding: int = Field(..., ge=0, le=100)
    source_credibility: int = Field(..., ge=0, le=100)
    claim_specificity: int = Field(..., ge=0, le=100)
    internal_consistency: int = Field(..., ge=0, le=100)


class JudgeValidationOutput(BaseModel):
    validated_summary: str
    evidence_gaps: str
    overall_reliability: int = Field(..., ge=0, le=100)
    rubric_scores: ValidationRubricScores
    claims: List[VerifiedClaim]


class SecondJudgeOutput(BaseModel):
    evidence_gaps: str
    overall_reliability: int = Field(..., ge=0, le=100)
    rubric_scores: ValidationRubricScores
    claim_assessments: List[VerifiedClaim]



class ClaimPatchOutput(BaseModel):
    original_claim: str
    action: Literal["search_and_replace", "coverage_addition", "qualify_or_remove", "remove", "preserve"]
    replacement: str
    expected_reliability_effect: Literal[
        "improve_to_supported",
        "add_source_backed_claim",
        "reduce_to_conservative_claim",
        "remove_unsupported_claim",
        "no_change",
    ]
    evidence_urls: List[str] = Field(default_factory=list)
    rationale: str


class MarketRepairOutput(MarketOutput):
    patch_summary: str
    patches: List[ClaimPatchOutput] = Field(default_factory=list)


class RepairPatchValidationOutput(BaseModel):
    evidence_gaps: str
    patch_reliability: int = Field(..., ge=0, le=100)
    rubric_scores: ValidationRubricScores
    repaired_claim_assessments: List[VerifiedClaim]
    can_accept_patch: bool
    rationale: str


class RetrievalDiagnostic(BaseModel):
    claim_id: str
    claim: str
    category: str
    action: Literal["search_and_replace", "coverage_addition", "qualify_or_remove", "remove", "preserve"]
    failure_type: str
    materiality: int = Field(..., ge=1, le=5)
    coverage_enhancement: bool = False
    target_add_count: int = 1
    missing_claim_count: int = 0
    claims_total_before_repair: int = 0
    retrieval_score: float = Field(..., ge=0.0, le=1.0)
    unique_sources: int = 0
    high_quality_sources: int = 0
    mean_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overlap_score: float = Field(default=0.0, ge=0.0, le=1.0)
    domain_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_strength: Literal["none", "weak", "medium", "strong"] = "none"
    query_count: int = 0
    status: str = "unknown"
    reasons: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class RetryPolicyDecision(BaseModel):
    action: Literal["stop", "repair_bundle", "broad_retry"]
    selected_claim_ids: List[str] = Field(default_factory=list)
    action_type: Optional[Literal["search_and_replace", "coverage_addition", "qualify_or_remove", "remove"]] = None
    posterior_acceptance: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_gain: float = 0.0
    expected_tokens: int = 0
    expected_seconds: float = 0.0
    expected_utility: float = 0.0
    roi_per_1k: float = 0.0
    current_score: int = 0
    threshold: int = 75
    retrieval_score: float = 0.0
    candidate_count: int = 0
    evaluated_candidate_count: int = 0
    viable_candidate_count: int = 0
    rejected_candidate_count: int = 0
    reason: str = ""
    block_reasons: List[str] = Field(default_factory=list)
    rejected_candidate_block_reasons: List[str] = Field(default_factory=list)
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    rejected_candidates: List[Dict[str, Any]] = Field(default_factory=list)


class BayesianRetryPolicy:
    DEFAULT_THRESHOLDS: Dict[str, float] = {
        "posterior_accept_min": 0.35,
        "retrieval_quality_min": 0.55,
        "roi_min": 1.25,
        "eu_margin": 0.50,
        "max_bundle_claims": 1,
        "max_repair_rounds": 1,
        "tail_token_reserve": 3500,
        "min_remaining_tokens_for_repair": 4000,
        "threshold_crossing_margin": 1.0,
        "large_deficit_repair_min_deficit": 10.0,
        "large_deficit_repair_min_materiality": 5.0,
        "large_deficit_repair_min_retrieval": 0.75,
        "large_deficit_repair_min_expected_gain": 5.0,
        "large_deficit_repair_min_expected_utility": 1.0,
        "large_deficit_repair_min_roi": 1.25,
        "large_deficit_repair_min_unique_sources": 2.0,
        "large_deficit_repair_min_high_quality_sources": 2.0,
        "large_deficit_repair_reserve_slack_tokens": 1000.0,
        "threshold_crossing_repair_min_materiality": 5.0,
        "threshold_crossing_repair_min_retrieval": 0.75,
        "threshold_crossing_repair_min_expected_utility": 1.0,
        "threshold_crossing_repair_min_roi": 1.25,
        "threshold_crossing_repair_min_unique_sources": 2.0,
        "threshold_crossing_repair_min_high_quality_sources": 2.0,
        "threshold_crossing_repair_reserve_slack_tokens": 1000.0,
        "source_backed_material_repair_min_materiality": 5.0,
        "source_backed_material_repair_min_retrieval": 0.65,
        "source_backed_material_repair_min_overlap": 0.18,
        "source_backed_material_repair_min_expected_utility": -0.25,
        "source_backed_material_repair_min_roi": 0.60,
        "source_backed_material_repair_min_unique_sources": 2.0,
        "source_backed_material_repair_min_high_quality_sources": 2.0,
        "source_backed_material_repair_reserve_slack_tokens": 1800.0,
        "coverage_addition_min_retrieval": 0.60,
        "coverage_addition_min_unique_sources": 2.0,
        "coverage_addition_min_high_quality_sources": 1.0,
        "coverage_addition_min_high_quality_sources_regulated": 2.0,
        "coverage_addition_reserve_slack_tokens": 1800.0,
    }
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "lambda_tok": 0.50,
        "lambda_time": 0.03,
        "lambda_crit": 4.00,
        "ucb_exploration": 0.15,
        "gain_kappa": 2.0,
        "token_kappa": 3.0,
        "seconds_kappa": 5.0,
    }
    DEFAULT_PRIORS: Dict[str, Dict[str, float]] = {
        "qualify_or_remove": {"alpha": 3.0, "beta": 4.0, "gain": 1.5, "tokens": 2800.0, "seconds": 15.0},
        "remove": {"alpha": 3.0, "beta": 3.0, "gain": 1.0, "tokens": 2200.0, "seconds": 12.0},
        "search_and_replace_strong": {"alpha": 4.0, "beta": 3.0, "gain": 4.0, "tokens": 3000.0, "seconds": 19.0},
        "search_and_replace_medium": {"alpha": 3.0, "beta": 4.0, "gain": 3.0, "tokens": 4000.0, "seconds": 21.0},
        "search_and_replace_weak": {"alpha": 1.0, "beta": 5.0, "gain": 1.0, "tokens": 4500.0, "seconds": 23.0},
        "coverage_addition": {"alpha": 4.0, "beta": 3.0, "gain": 5.0, "tokens": 4200.0, "seconds": 18.0},
        "broad_retry": {"alpha": 2.0, "beta": 5.0, "gain": 5.0, "tokens": 9500.0, "seconds": 36.0},
    }

    def __init__(self, thresholds: Optional[Dict[str, float]] = None, utility_weights: Optional[Dict[str, float]] = None) -> None:
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.weights = {**self.DEFAULT_WEIGHTS, **(utility_weights or {})}

    @staticmethod
    def _score_and_threshold(state: PitchState | Dict[str, Any]) -> tuple[int, int]:
        validation = _sget(state, "validation_report", {}) or {}
        score = int(validation.get("reliability_score", 0) or 0) if isinstance(validation, dict) else 0
        threshold = int(_sget(state, "validation_threshold", 75) or 75)
        return score, threshold

    @staticmethod
    def _remaining_total_tokens(state: PitchState | Dict[str, Any]) -> Optional[int]:
        max_total = _sget(state, "max_total_tokens")
        usage = _sget(state, "token_usage", {}) or {}
        current = int(usage.get("total_tokens", 0) or _sget(state, "total_tokens_current", 0) or 0)
        if isinstance(max_total, int):
            return int(max_total) - current
        return None

    @classmethod
    def base_bucket_key(cls, bucket_key: str) -> str:
        key = str(bucket_key or "").strip()
        for base in sorted(cls.DEFAULT_PRIORS.keys(), key=len, reverse=True):
            if key == base or key.startswith(f"{base}:"):
                return base
        return "qualify_or_remove"

    @staticmethod
    def _bucket_component(value: Any, default: str = "general") -> str:
        text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
        return text or default

    @staticmethod
    def _materiality_tier(materiality: int) -> str:
        if materiality >= 5:
            return "m5"
        if materiality >= 4:
            return "m4"
        return "m_low"

    def _bucket_key(self, diag: RetrievalDiagnostic, action: Optional[str] = None) -> str:
        scoring_action = str(action or diag.action or "").strip()
        if scoring_action == "search_and_replace":
            base = f"search_and_replace_{_retrieval_bucket(diag.retrieval_score)}"
        elif scoring_action == "coverage_addition":
            base = "coverage_addition"
        elif scoring_action == "remove":
            base = "remove"
        else:
            base = "qualify_or_remove"
        category = self._bucket_component(diag.category)
        materiality = self._materiality_tier(int(diag.materiality or 0))
        return f"{base}:{category}:{materiality}"

    @staticmethod
    def _guard_adjusted_action(diag: RetrievalDiagnostic) -> tuple[str, List[str]]:
        """Mirror source-bound repair guards before pricing a repair candidate."""
        requested_action = str(diag.action or "").strip()
        if requested_action != "search_and_replace":
            return requested_action, []

        fragile_categories = {
            "competitor_landscape",
            "trend_or_adoption",
            "gap_or_opportunity",
            "general_market_claim",
        }
        fragile_failures = {
            "speculative_number",
            "missing_evidence",
            "too_broad",
            "unclear_attribution",
            "contradictory_evidence",
        }
        guard_reasons: List[str] = []
        source_ready = _source_bound_replacement_ready(
            category=diag.category,
            materiality=int(diag.materiality or 0),
            retrieval_score=float(diag.retrieval_score or 0.0),
            unique_sources=int(diag.unique_sources or 0),
            high_quality_sources=int(diag.high_quality_sources or 0),
            overlap_score=float(diag.overlap_score or 0.0),
        )
        if diag.category in fragile_categories and diag.failure_type in fragile_failures and not source_ready:
            guard_reasons.append("fragile_claim_type_requires_qualification")
        if diag.category != "market_size_or_numeric" and int(diag.materiality or 0) < 5 and not source_ready:
            guard_reasons.append("non_numeric_claim_not_material_enough_for_strong_replacement")
        if int(diag.unique_sources or 0) < 2:
            guard_reasons.append("insufficient_unique_sources_for_replacement")
        if int(diag.high_quality_sources or 0) < 1:
            guard_reasons.append("no_high_quality_source_for_replacement")
        if float(diag.overlap_score or 0.0) < 0.12:
            guard_reasons.append("low_claim_evidence_overlap")
        if float(diag.retrieval_score or 0.0) < 0.75 and not source_ready:
            guard_reasons.append("retrieval_score_below_source_bound_threshold")
        if guard_reasons:
            return "qualify_or_remove", guard_reasons
        return "search_and_replace", []

    def _posterior_values(self, *, bucket_key: str, policy_stats: Dict[str, Any]) -> Dict[str, float]:
        base_bucket = self.base_bucket_key(bucket_key)
        prior = self.DEFAULT_PRIORS.get(base_bucket, self.DEFAULT_PRIORS["qualify_or_remove"])
        buckets = policy_stats.get("repair_buckets", {}) if isinstance(policy_stats, dict) else {}
        observed = buckets.get(bucket_key, {}) if isinstance(buckets, dict) else {}
        successes = _safe_float(observed.get("accepted", 0.0))
        failures = _safe_float(observed.get("rejected", 0.0))
        n_empirical = max(0.0, successes + failures)
        total_attempts = _safe_float(policy_stats.get("repair_observations", 0.0), 0.0) if isinstance(policy_stats, dict) else 0.0
        alpha = prior["alpha"] + successes
        beta = prior["beta"] + failures
        p_mean = alpha / max(1e-6, alpha + beta)
        exploration_weight = 1.0 / max(1.0, math.sqrt(1.0 + n_empirical))
        raw_ucb_bonus = exploration_weight * math.sqrt(
            2.0 * math.log(max(2.0, total_attempts + 1.0)) / max(1.0, n_empirical + 1.0)
        )
        ucb_bonus = min(0.20, float(self.weights["ucb_exploration"]) * raw_ucb_bonus)
        p_accept = min(1.0, p_mean + ucb_bonus)
        n = n_empirical
        gain_kappa = max(0.0, _safe_float(self.weights.get("gain_kappa", 2.0), 2.0))
        token_kappa = max(0.0, _safe_float(self.weights.get("token_kappa", 3.0), 3.0))
        seconds_kappa = max(0.0, _safe_float(self.weights.get("seconds_kappa", 5.0), 5.0))
        obs_gain = _safe_float(observed.get("mean_gain", prior["gain"]), prior["gain"])
        obs_tokens = _safe_float(observed.get("mean_tokens", prior["tokens"]), prior["tokens"])
        obs_seconds = _safe_float(observed.get("mean_seconds", prior["seconds"]), prior["seconds"])
        expected_gain = (gain_kappa * prior["gain"] + n * obs_gain) / max(1e-6, gain_kappa + n)
        expected_tokens = (token_kappa * prior["tokens"] + n * obs_tokens) / max(1e-6, token_kappa + n)
        expected_seconds = (seconds_kappa * prior["seconds"] + n * obs_seconds) / max(1e-6, seconds_kappa + n)
        return {
            "posterior_acceptance": round(p_accept, 4),
            "posterior_mean": round(p_mean, 4),
            "ucb_bonus": round(ucb_bonus, 4),
            "n_empirical": round(n_empirical, 4),
            "expected_gain": round(expected_gain, 4),
            "expected_tokens": round(expected_tokens, 4),
            "expected_seconds": round(expected_seconds, 4),
            "gain_kappa": round(gain_kappa, 4),
            "token_kappa": round(token_kappa, 4),
            "seconds_kappa": round(seconds_kappa, 4),
            "p_critical_degradation": round(_safe_float(observed.get("critical_degradation_rate", 0.05), 0.05), 4),
        }

    def _candidate_from_diag(self, *, diag: RetrievalDiagnostic, state: PitchState | Dict[str, Any], current_score: int, threshold: int) -> Optional[Dict[str, Any]]:
        retrieval_min = float(self.thresholds["retrieval_quality_min"])
        prefilter_reasons: List[str] = []
        if diag.action == "preserve":
            return None
        is_coverage_addition = bool(
            diag.action == "coverage_addition"
            or diag.failure_type == "coverage_gap"
            or diag.coverage_enhancement
            or str(diag.claim_id).startswith("coverage_gap")
        )
        if is_coverage_addition:
            effective_action = "coverage_addition"
            guard_reasons: List[str] = []
        else:
            effective_action, guard_reasons = self._guard_adjusted_action(diag)
        if effective_action in {"qualify_or_remove", "remove"}:
            prefilter_reasons.append("blocked_by_non_improvement_action")
        if effective_action == "search_and_replace" and diag.retrieval_score < retrieval_min:
            prefilter_reasons.append("blocked_by_retrieval_quality")
        if effective_action == "coverage_addition":
            coverage_min_retrieval = float(self.thresholds.get("coverage_addition_min_retrieval", 0.60) or 0.60)
            coverage_min_sources = int(self.thresholds.get("coverage_addition_min_unique_sources", 2.0) or 2.0)
            regulated_category = diag.category in {"regulatory_or_compliance"}
            coverage_min_hq = int(
                self.thresholds.get(
                    "coverage_addition_min_high_quality_sources_regulated"
                    if regulated_category
                    else "coverage_addition_min_high_quality_sources",
                    1.0,
                )
                or 1.0
            )
            if diag.retrieval_score < coverage_min_retrieval:
                prefilter_reasons.append("blocked_by_coverage_retrieval_quality")
            if int(diag.unique_sources or 0) < coverage_min_sources:
                prefilter_reasons.append("blocked_by_coverage_unique_sources")
            if int(diag.high_quality_sources or 0) < coverage_min_hq:
                prefilter_reasons.append("blocked_by_coverage_high_quality_sources")
        if effective_action in {"qualify_or_remove", "remove"} and diag.retrieval_score < 0.20 and diag.materiality < 4:
            prefilter_reasons.append("blocked_by_low_retrieval_low_materiality")
        policy_stats = _sget(state, "policy_stats", {}) or {}
        bucket_key = self._bucket_key(diag, effective_action)
        vals = self._posterior_values(bucket_key=bucket_key, policy_stats=policy_stats)
        score_deficit = max(0.0, float(threshold - current_score))
        if effective_action in {"qualify_or_remove", "remove"} and diag.unique_sources == 0 and score_deficit < 8.0:
            prefilter_reasons.append("blocked_by_low_deficit_no_sources")
        if effective_action == "coverage_addition":
            target_add_count = min(max(1, int(diag.target_add_count or diag.missing_claim_count or 1)), 2)
            missing_claim_count = max(0, int(diag.missing_claim_count or 0))
            coverage_bonus = (
                1.50 * target_add_count
                + 0.70 * min(2, missing_claim_count)
                + 1.00 * max(0.0, diag.retrieval_score - 0.60)
                + 0.05 * score_deficit
            )
            expected_gain = min(9.0, vals["expected_gain"] + coverage_bonus)
        else:
            target_add_count = 0
            missing_claim_count = 0
            expected_gain = min(12.0, vals["expected_gain"] + 0.35 * max(0, diag.materiality - 2) + 1.0 * max(0.0, diag.retrieval_score - 0.55) + 0.12 * score_deficit)
        if diag.action == "search_and_replace" and effective_action != "search_and_replace":
            expected_gain = min(expected_gain, 2.5)
        expected_tokens = vals["expected_tokens"]
        expected_seconds = vals["expected_seconds"]
        p_accept = vals["posterior_acceptance"]
        eu = p_accept * max(0.0, expected_gain) - float(self.weights["lambda_tok"]) * (expected_tokens / 1000.0) - float(self.weights["lambda_time"]) * expected_seconds - float(self.weights["lambda_crit"]) * vals["p_critical_degradation"]
        roi = (p_accept * max(0.0, expected_gain)) / max(1e-6, expected_tokens / 1000.0)
        projected_score_after = float(current_score) + float(expected_gain)
        threshold_crossing_margin = float(self.thresholds.get("threshold_crossing_margin", 1.0) or 0.0)
        threshold_crossing_required = bool(effective_action == "search_and_replace" and score_deficit > 0.0)
        large_deficit_incremental_exception = bool(
            threshold_crossing_required
            and score_deficit >= float(self.thresholds.get("large_deficit_repair_min_deficit", 10.0) or 10.0)
            and int(diag.materiality or 0) >= int(self.thresholds.get("large_deficit_repair_min_materiality", 5.0) or 5.0)
            and float(diag.retrieval_score or 0.0) >= float(self.thresholds.get("large_deficit_repair_min_retrieval", 0.75) or 0.75)
            and int(diag.unique_sources or 0) >= int(self.thresholds.get("large_deficit_repair_min_unique_sources", 2.0) or 2.0)
            and int(diag.high_quality_sources or 0) >= int(self.thresholds.get("large_deficit_repair_min_high_quality_sources", 2.0) or 2.0)
            and expected_gain >= float(self.thresholds.get("large_deficit_repair_min_expected_gain", 5.0) or 5.0)
            and eu >= float(self.thresholds.get("large_deficit_repair_min_expected_utility", 1.0) or 1.0)
            and roi >= float(self.thresholds.get("large_deficit_repair_min_roi", self.thresholds.get("roi_min", 1.25)) or 1.25)
            and not guard_reasons
        )
        threshold_crossing_met = bool(projected_score_after >= float(threshold) - threshold_crossing_margin)
        threshold_crossing_repair_exception = bool(
            threshold_crossing_required
            and threshold_crossing_met
            and int(diag.materiality or 0) >= int(self.thresholds.get("threshold_crossing_repair_min_materiality", 5.0) or 5.0)
            and float(diag.retrieval_score or 0.0) >= float(self.thresholds.get("threshold_crossing_repair_min_retrieval", 0.75) or 0.75)
            and int(diag.unique_sources or 0) >= int(self.thresholds.get("threshold_crossing_repair_min_unique_sources", 2.0) or 2.0)
            and int(diag.high_quality_sources or 0) >= int(self.thresholds.get("threshold_crossing_repair_min_high_quality_sources", 2.0) or 2.0)
            and eu >= float(self.thresholds.get("threshold_crossing_repair_min_expected_utility", 1.0) or 1.0)
            and roi >= float(self.thresholds.get("threshold_crossing_repair_min_roi", self.thresholds.get("roi_min", 1.25)) or 1.25)
            and not guard_reasons
        )
        source_backed_material_repair_exception = bool(
            threshold_crossing_required
            and threshold_crossing_met
            and diag.category in {"market_size_or_numeric", "regulatory_or_compliance"}
            and int(diag.materiality or 0) >= int(self.thresholds.get("source_backed_material_repair_min_materiality", 5.0) or 5.0)
            and float(diag.retrieval_score or 0.0) >= float(self.thresholds.get("source_backed_material_repair_min_retrieval", 0.65) or 0.65)
            and float(diag.overlap_score or 0.0) >= float(self.thresholds.get("source_backed_material_repair_min_overlap", 0.18) or 0.18)
            and int(diag.unique_sources or 0) >= int(self.thresholds.get("source_backed_material_repair_min_unique_sources", 2.0) or 2.0)
            and int(diag.high_quality_sources or 0) >= int(self.thresholds.get("source_backed_material_repair_min_high_quality_sources", 2.0) or 2.0)
            and eu >= float(self.thresholds.get("source_backed_material_repair_min_expected_utility", -0.10) or -0.10)
            and roi >= float(self.thresholds.get("source_backed_material_repair_min_roi", 0.70) or 0.70)
            and not guard_reasons
        )
        coverage_addition_exception = bool(
            effective_action == "coverage_addition"
            and not prefilter_reasons
            and missing_claim_count > 0
            and target_add_count > 0
        )
        return {
            "claim_id": diag.claim_id,
            "action_type": effective_action,
            "requested_action_type": diag.action,
            "guard_adjusted_action": effective_action,
            "source_bound_guard_reasons": guard_reasons,
            "bucket_key": bucket_key,
            "base_bucket_key": self.base_bucket_key(bucket_key),
            "posterior_acceptance": round(p_accept, 4),
            "posterior_mean": vals["posterior_mean"],
            "ucb_bonus": vals["ucb_bonus"],
            "n_empirical": vals["n_empirical"],
            "expected_gain": round(expected_gain, 4),
            "expected_tokens": int(round(expected_tokens)),
            "expected_seconds": round(expected_seconds, 4),
            "expected_utility": round(eu, 4),
            "roi_per_1k": round(roi, 4),
            "score_deficit": round(score_deficit, 4),
            "projected_score_after": round(projected_score_after, 4),
            "threshold_crossing_required": 1 if threshold_crossing_required else 0,
            "threshold_crossing_met": 1 if threshold_crossing_met else 0,
            "threshold_crossing_margin": round(threshold_crossing_margin, 4),
            "threshold_crossing_exception": 0,
            "threshold_crossing_repair_exception": 1 if threshold_crossing_repair_exception else 0,
            "source_backed_material_repair_exception": 1 if source_backed_material_repair_exception else 0,
            "large_deficit_incremental_exception": 1 if large_deficit_incremental_exception else 0,
            "coverage_addition_exception": 1 if coverage_addition_exception else 0,
            "target_add_count": int(target_add_count),
            "missing_claim_count": int(missing_claim_count),
            "claims_total_before_repair": int(diag.claims_total_before_repair or 0),
            "retrieval_score": round(diag.retrieval_score, 4),
            "unique_sources": int(diag.unique_sources or 0),
            "high_quality_sources": int(diag.high_quality_sources or 0),
            "overlap_score": round(float(diag.overlap_score or 0.0), 4),
            "materiality": diag.materiality,
            "category": diag.category,
            "failure_type": diag.failure_type,
            "blocking_gates": prefilter_reasons,
            "gate_result": "blocked" if prefilter_reasons else "candidate",
        }

    def decide(self, *, state: PitchState | Dict[str, Any], claim_diagnostics: List[RetrievalDiagnostic | Dict[str, Any]]) -> RetryPolicyDecision:
        current_score, threshold = self._score_and_threshold(state)
        repair_rounds = int(_sget(state, "repair_rounds", 0) or 0)
        if repair_rounds >= int(self.thresholds["max_repair_rounds"]):
            return RetryPolicyDecision(
                action="stop",
                current_score=current_score,
                threshold=threshold,
                reason="max_repair_rounds_reached",
                block_reasons=["max_repair_rounds_reached"],
            )
        remaining_tokens = self._remaining_total_tokens(state)
        evaluated_candidates: List[Dict[str, Any]] = []
        rejected_candidates: List[Dict[str, Any]] = []
        candidates: List[Dict[str, Any]] = []
        for raw_diag in claim_diagnostics or []:
            diag = raw_diag if isinstance(raw_diag, RetrievalDiagnostic) else RetrievalDiagnostic(**raw_diag)
            cand = self._candidate_from_diag(diag=diag, state=state, current_score=current_score, threshold=threshold)
            if cand is None:
                continue
            blocking_gates = list(cand.get("blocking_gates", []) or [])
            source_backed_material_exception = bool(cand.get("source_backed_material_repair_exception"))
            coverage_addition_exception = bool(cand.get("coverage_addition_exception"))
            if cand["posterior_acceptance"] < float(self.thresholds["posterior_accept_min"]):
                blocking_gates.append("blocked_by_posterior_acceptance")
            if isinstance(remaining_tokens, int):
                min_remaining = int(float(self.thresholds.get("min_remaining_tokens_for_repair", 4000) or 4000))
                if remaining_tokens < min_remaining:
                    blocking_gates.append("blocked_by_insufficient_repair_budget")
            cand["blocking_gates"] = blocking_gates
            cand["gate_result"] = "blocked" if blocking_gates else "passed"
            evaluated_candidates.append(cand)
            if blocking_gates:
                rejected_candidates.append(cand)
            else:
                candidates.append(cand)
        candidates = sorted(
            candidates,
            key=lambda c: (
                int(bool(c.get("coverage_addition_exception"))),
                float(c.get("retrieval_score", 0.0) or 0.0),
                int(c.get("high_quality_sources", 0) or 0),
                int(c.get("unique_sources", 0) or 0),
                int(c.get("materiality", 0) or 0),
                float(c.get("posterior_acceptance", 0.0) or 0.0),
                float(c.get("expected_gain", 0.0) or 0.0),
            ),
            reverse=True,
        )
        selected = candidates[: max(1, int(self.thresholds["max_bundle_claims"]))]
        block_reason_counts: Dict[str, int] = {}
        for cand in rejected_candidates:
            for reason in cand.get("blocking_gates", []) or []:
                block_reason_counts[str(reason)] = block_reason_counts.get(str(reason), 0) + 1
        rejected_candidate_block_reasons = [
            reason
            for reason, _ in sorted(
                block_reason_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        if not selected:
            reason = "no_evidence_gated_repair_candidate"
            return RetryPolicyDecision(
                action="stop",
                current_score=current_score,
                threshold=threshold,
                reason=reason,
                candidate_count=len(candidates),
                evaluated_candidate_count=len(evaluated_candidates),
                viable_candidate_count=len(candidates),
                rejected_candidate_count=len(rejected_candidates),
                block_reasons=rejected_candidate_block_reasons or [reason],
                rejected_candidate_block_reasons=rejected_candidate_block_reasons,
                candidates=evaluated_candidates,
                rejected_candidates=rejected_candidates,
            )
        best = selected[0]
        if any(bool(c.get("coverage_addition_exception")) for c in selected):
            selected_reason = "coverage_addition_selected"
        elif any(bool(c.get("source_backed_material_repair_exception")) for c in selected):
            selected_reason = "source_backed_material_repair_selected"
        elif any(bool(c.get("threshold_crossing_repair_exception")) for c in selected):
            selected_reason = "threshold_crossing_repair_selected"
        elif any(bool(c.get("large_deficit_incremental_exception")) for c in selected):
            selected_reason = "large_deficit_incremental_repair_selected"
        else:
            selected_reason = "evidence_gated_repair_selected"
        return RetryPolicyDecision(
            action="repair_bundle",
            selected_claim_ids=[str(c["claim_id"]) for c in selected],
            action_type=best["action_type"],  # type: ignore[arg-type]
            posterior_acceptance=round(sum(float(c["posterior_acceptance"]) for c in selected) / len(selected), 4),
            expected_gain=round(sum(float(c["expected_gain"]) for c in selected), 4),
            expected_tokens=int(sum(int(c["expected_tokens"]) for c in selected)),
            expected_seconds=round(sum(float(c["expected_seconds"]) for c in selected), 4),
            expected_utility=round(sum(float(c["expected_utility"]) for c in selected), 4),
            roi_per_1k=round(sum(float(c["posterior_acceptance"]) * float(c["expected_gain"]) for c in selected) / max(1e-6, sum(int(c["expected_tokens"]) for c in selected) / 1000.0), 4),
            current_score=current_score,
            threshold=threshold,
            retrieval_score=round(sum(float(c["retrieval_score"]) for c in selected) / len(selected), 4),
            candidate_count=len(selected),
            evaluated_candidate_count=len(evaluated_candidates),
            viable_candidate_count=len(candidates),
            rejected_candidate_count=len(rejected_candidates),
            reason=selected_reason,
            block_reasons=[],
            rejected_candidate_block_reasons=rejected_candidate_block_reasons,
            candidates=evaluated_candidates,
            rejected_candidates=rejected_candidates,
        )


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
    @staticmethod
    def _claim_ledger(claims: List[Dict[str, Any]], evidence: List[Dict[str, Any]], evidence_gaps: str = "") -> List[Dict[str, Any]]:
        ledger: List[Dict[str, Any]] = []
        evidence_domains = [str(item.get("domain")) for item in evidence or [] if isinstance(item, dict) and item.get("domain")]
        for idx, claim in enumerate(claims or [], start=1):
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("claim", "") or "").strip()
            if not text:
                continue
            category = _claim_repair_category(text)
            item = dict(claim)
            item["claim_id"] = str(claim.get("claim_id") or f"c{idx}")
            item["claim_text"] = text
            item["category"] = category
            item["materiality"] = int(claim.get("materiality") or _claim_materiality(text, category))
            item["failure_type"] = str(claim.get("failure_type") or _claim_failure_type(claim, evidence_gaps))
            item["source_diversity_count"] = len(set(evidence_domains))
            item["distinct_source_domains"] = sorted(set(evidence_domains))[:6]
            if str(item.get("verdict", "")).strip().lower() != "supported" and not item.get("candidate_queries"):
                item["candidate_queries"] = [f"{text} evidence source", f"{category.replace('_', ' ')} market evidence"]
            ledger.append(item)
        return ledger

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
        complexity_terms = [
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
            "enterprise",
            "integration",
            "workflow",
            "marketplace",
            "prediction",
            "optimization",
            "supply chain",
            "healthcare",
            "finance",
            "legal",
            "cybersecurity",
            "security",
            "public sector",
            "government",
            "city",
            "manufacturing",
            "manufacturers",
            "logistics",
            "shipment",
            "customs",
            "vendor",
            "geopolitical",
            "audit",
            "traceability",
            "real-time",
            "multi-sided",
            "stakeholders",
            "hipaa",
            "insurance",
            "pre-authorization",
            "authorization",
            "clinical",
            "documentation",
            "specialty",
            "governance",
            "fraud",
            "banks",
            "transaction",
            "incident",
            "response",
            "alerts",
            "logs",
            "containment",
            "renewable",
            "energy",
            "grid",
            "equipment",
            "maintenance",
            "scientific",
            "literature",
            "token budgets",
        ]
        evidence_terms = [
            "market",
            "competitor",
            "competitors",
            "regulation",
            "regulatory",
            "compliance",
            "clinical",
            "hipaa",
            "insurance",
            "pre-authorization",
            "authorization",
            "patient",
            "governance",
            "legal",
            "financial",
            "bank",
            "banks",
            "fraud",
            "audit",
            "risk",
            "safety",
            "security",
            "certification",
            "benchmark",
            "accuracy",
            "claims",
            "evidence",
            "scientific",
            "literature",
            "customs",
            "geopolitical",
        ]
        uncertainty_terms = [
            "uncertain",
            "uncertainty",
            "predict",
            "prediction",
            "risk",
            "failure",
            "adversarial",
            "conflicting",
            "dynamic",
            "adaptive",
            "real-time",
            "optimize",
            "optimization",
            "trade-off",
            "tradeoff",
            "multi-step",
            "long-horizon",
            "fallback",
            "replan",
            "re-plan",
            "shipment",
            "delays",
            "fraud",
            "incident",
            "alerts",
            "failures",
            "emergency",
            "constraints",
            "coverage",
            "budget",
            "budgets",
        ]
        workflow_terms = [
            "platform",
            "workflow",
            "coordinates",
            "coordination",
            "management",
            "monitoring",
            "dashboard",
            "assistant",
            "copilot",
            "agent",
            "multi-agent",
            "integration",
            "api",
            "tool",
            "automation",
            "pipeline",
            "orchestration",
            "enterprise",
            "team",
            "teams",
            "case-management",
            "case management",
            "scheduling",
            "documentation",
            "reports",
            "reviews",
            "coverage",
        ]

        def count_terms(terms: List[str]) -> int:
            return sum(1 for term in terms if term in text_l)

        marker_count = count_terms(complexity_terms)
        evidence_count = count_terms(evidence_terms)
        uncertainty_count = count_terms(uncertainty_terms)
        workflow_count = count_terms(workflow_terms)

        token_count = len(tokens)
        length_score = min(1.0, token_count / 45.0)
        complexity_marker_score = min(1.0, marker_count / 6.0)
        evidence_need_score = min(1.0, evidence_count / 4.0)
        uncertainty_score = min(1.0, uncertainty_count / 4.0)
        workflow_score = min(1.0, workflow_count / 5.0)
        structural_complexity = round(
            100.0
            * (
                0.30 * length_score
                + 0.25 * complexity_marker_score
                + 0.20 * evidence_need_score
                + 0.15 * uncertainty_score
                + 0.10 * workflow_score
            ),
            2,
        )
        uncertainty_need = round(
            100.0
            * (
                0.45 * uncertainty_score
                + 0.30 * evidence_need_score
                + 0.25 * workflow_score
            ),
            2,
        )
        return {
            "token_count": token_count,
            "marker_count": marker_count,
            "evidence_count": evidence_count,
            "uncertainty_count": uncertainty_count,
            "workflow_count": workflow_count,
            "length_score": round(length_score, 4),
            "complexity_marker_score": round(complexity_marker_score, 4),
            "evidence_need_score": round(evidence_need_score, 4),
            "uncertainty_score": round(uncertainty_score, 4),
            "workflow_score": round(workflow_score, 4),
            "structural_complexity": structural_complexity,
            "uncertainty_need": uncertainty_need,
            "has_high_stakes": marker_count >= 4 or evidence_count >= 3,
        }

    @classmethod
    def _score_modes(
        cls,
        *,
        idea_text: str,
        llm_decision: ControllerDecision,
        budget_snapshot: Dict[str, Any],
        validation_threshold: int,
        max_validation_retries: int,
    ) -> Dict[str, Any]:
        """
        Math-backed adaptive controller.

        Utility(mode) = expected_quality(mode) - lambda_cost * normalized_cost(mode) * 100
        where expected quality is estimated from structural complexity, uncertainty,
        evidence need, and workflow coupling. Costs are deterministic mode priors
        scaled by complexity and checked against the current budget.
        """
        raw_idea_text = ""
        for line in str(idea_text or "").splitlines():
            stripped = line.strip()
            if stripped:
                raw_idea_text = stripped
                break
        raw_features = cls._complexity_features(raw_idea_text or idea_text)
        features = cls._complexity_features(idea_text)
        complexity = max(
            int(round(float(features["structural_complexity"]))),
            int(llm_decision.estimated_complexity * 0.25),
        )
        complexity = max(0, min(100, complexity))

        c = complexity / 100.0
        u = float(features["uncertainty_need"]) / 100.0
        e = float(features["evidence_need_score"])
        w = float(features["workflow_score"])
        complexity_marker = float(features["complexity_marker_score"])

        mode_costs = {mode: cls._prior_cost_for_mode(mode, complexity) for mode in cls.MODE_ORDER}

        max_cost_tokens = max(cost["token_proxy"] for cost in mode_costs.values()) or 1
        max_cost_calls = max(cost["tool_calls"] for cost in mode_costs.values()) or 1
        max_cost_runtime = max(cost["runtime_seconds"] for cost in mode_costs.values()) or 1.0

        remaining_total = budget_snapshot.get("remaining_total_tokens")
        remaining_proxy = budget_snapshot.get("remaining_token_proxy")
        remaining_tokens = (
            remaining_total if isinstance(remaining_total, int) else remaining_proxy
        )
        token_budget_pressure = 0.0
        if isinstance(remaining_tokens, int) and remaining_tokens > 0:
            token_budget_pressure = max(
                0.0,
                min(1.0, 1.0 - (remaining_tokens / max(remaining_tokens, max_cost_tokens * 2.5))),
            )
        elif isinstance(remaining_tokens, int) and remaining_tokens <= 0:
            token_budget_pressure = 1.0

        remaining_calls = budget_snapshot.get("remaining_tool_calls")
        call_budget_pressure = 0.0
        if isinstance(remaining_calls, int) and remaining_calls > 0:
            call_budget_pressure = max(
                0.0,
                min(1.0, 1.0 - (remaining_calls / max(remaining_calls, max_cost_calls * 2.0))),
            )
        elif isinstance(remaining_calls, int) and remaining_calls <= 0:
            call_budget_pressure = 1.0

        budget_pressure = round(max(token_budget_pressure, call_budget_pressure), 4)
        lambda_cost = round(0.12 + 0.28 * budget_pressure, 4)

        # Expected reliability is deliberately quality-first when the experiment
        # asks for high validation confidence. Direct mode is cheap, but prior
        # runs showed it is fragile under evidence-grounded judge scoring.
        expected_quality = {
            "direct": (
                53.0
                + 7.0 * (1.0 - c)
                + 2.0 * (1.0 - u)
                - 7.0 * e
                - 4.0 * w
                - 3.0 * complexity_marker
            ),
            "shallow": 66.0 + 10.0 * c + 8.0 * e + 5.0 * w + 3.0 * u,
            "recursive": (
                65.0
                + 14.0 * c
                + 11.0 * u
                + 6.0 * e
                + 3.0 * complexity_marker
            ),
        }

        # Recursion only gives value if the graph is allowed to revise.
        if max_validation_retries <= 0:
            expected_quality["recursive"] -= 5.0
        elif validation_threshold >= 75:
            expected_quality["shallow"] += 2.0
            expected_quality["recursive"] += 3.0
            expected_quality["direct"] -= 6.0

        raw_complexity = float(raw_features["structural_complexity"])
        raw_uncertainty = float(raw_features["uncertainty_need"])
        raw_evidence_count = int(raw_features["evidence_count"])
        raw_workflow_count = int(raw_features["workflow_count"])
        raw_marker_count = int(raw_features["marker_count"])
        raw_token_count = int(raw_features["token_count"])

        # Direct is eligible from the raw idea, not the expanded refined summary.
        # Otherwise the refinement step itself can make simple ideas look too
        # verbose/complex and direct is never tested.
        direct_eligible = (
            raw_complexity <= 22
            and raw_uncertainty <= 12
            and raw_evidence_count == 0
            and raw_workflow_count <= 1
            and raw_marker_count == 0
            and raw_token_count <= 24
            and complexity <= 45
        ) or budget_pressure >= 0.72
        if direct_eligible:
            expected_quality["direct"] += 12.0
            if validation_threshold >= 75:
                # Keep direct viable for simple tasks, but not unrealistically
                # dominant under a high evidence-grounding threshold.
                expected_quality["direct"] -= 2.0
            if budget_pressure >= 0.50:
                expected_quality["direct"] += 2.0
        elif not direct_eligible:
            expected_quality["direct"] -= 12.0

        recursive_upfront_allowed = (
            complexity >= 82
            or float(features["uncertainty_need"]) >= 78
            or (
                bool(features["has_high_stakes"])
                and complexity >= 64
                and (
                    float(features["uncertainty_need"]) >= 55
                    or e >= 0.75
                    or w >= 0.70
                )
            )
            or (
                complexity >= 64
                and float(features["uncertainty_need"]) >= 70
                and e >= 0.50
            )
        )

        # Hard, uncertain, evidence-heavy tasks should be allowed to recurse when feasible.
        if recursive_upfront_allowed:
            expected_quality["recursive"] += 4.0
        else:
            # For non-hard cases, recursion should usually be a validation-driven
            # escalation rather than the initial plan.
            expected_quality["recursive"] -= 4.0
        if complexity >= 45 and e >= 0.50:
            expected_quality["shallow"] += 1.5
            expected_quality["recursive"] += 2.0

        mode_scores: Dict[str, Dict[str, Any]] = {}
        selected_mode = "direct"
        selected_utility = -10_000.0
        for mode in cls.MODE_ORDER:
            cost = mode_costs[mode]
            token_norm = cost["token_proxy"] / max_cost_tokens
            call_norm = cost["tool_calls"] / max_cost_calls
            runtime_norm = cost["runtime_seconds"] / max_cost_runtime
            normalized_cost = round(
                0.60 * token_norm + 0.25 * call_norm + 0.15 * runtime_norm,
                4,
            )
            feasible = cls._is_mode_feasible(cost, budget_snapshot)
            advisory_bonus = (
                round(1.5 * float(llm_decision.confidence), 4)
                if mode == llm_decision.mode
                else 0.0
            )
            utility = (
                expected_quality[mode]
                - lambda_cost * normalized_cost * 100.0
                + advisory_bonus
            )
            if not feasible:
                utility -= 100.0
            mode_scores[mode] = {
                "expected_quality": round(expected_quality[mode], 4),
                "expected_cost": cost,
                "normalized_cost": normalized_cost,
                "advisory_bonus": advisory_bonus,
                "feasible": feasible,
                "utility": round(utility, 4),
            }
            if utility > selected_utility:
                selected_utility = utility
                selected_mode = mode

        sorted_scores = sorted(
            mode_scores.items(),
            key=lambda kv: float(kv[1]["utility"]),
            reverse=True,
        )
        utility_margin = 0.0
        if len(sorted_scores) > 1:
            utility_margin = round(
                float(sorted_scores[0][1]["utility"]) - float(sorted_scores[1][1]["utility"]),
                4,
            )

        policy_adjustments: List[str] = []
        if selected_mode == "direct" and not direct_eligible:
            non_direct = [
                item
                for item in sorted_scores
                if item[0] != "direct" and bool(item[1].get("feasible", False))
            ]
            if non_direct:
                selected_mode = non_direct[0][0]
                selected_utility = float(non_direct[0][1]["utility"])
                policy_adjustments.append("direct_not_eligible_promoted_to_non_direct")

        direct_score = mode_scores["direct"]
        recursive_score = mode_scores["recursive"]
        shallow_score = mode_scores["shallow"]
        direct_expected_quality = float(direct_score.get("expected_quality", 0.0) or 0.0)
        shallow_expected_quality = float(shallow_score.get("expected_quality", 0.0) or 0.0)
        direct_quality_floor = max(62.0, float(validation_threshold) - 5.0)
        direct_quality_ready = direct_expected_quality >= direct_quality_floor
        shallow_quality_advantage = shallow_expected_quality - direct_expected_quality
        if (
            selected_mode == "shallow"
            and direct_eligible
            and bool(direct_score.get("feasible", False))
            and direct_quality_ready
            and shallow_quality_advantage <= 4.0
            and float(direct_score["utility"]) >= float(shallow_score["utility"]) - 3.5
        ):
            selected_mode = "direct"
            selected_utility = float(direct_score["utility"])
            policy_adjustments.append("validation_ready_close_call_prefers_direct")
        elif (
            selected_mode == "direct"
            and (
                not direct_quality_ready
                or (
                    bool(shallow_score.get("feasible", False))
                    and shallow_quality_advantage > 4.0
                    and budget_pressure < 0.65
                )
            )
        ):
            non_direct = [
                item
                for item in sorted_scores
                if item[0] != "direct" and bool(item[1].get("feasible", False))
            ]
            if non_direct:
                selected_mode = non_direct[0][0]
                selected_utility = float(non_direct[0][1]["utility"])
                policy_adjustments.append("direct_deferred_until_validation_ready")

        if (
            selected_mode == "shallow"
            and bool(recursive_score.get("feasible", False))
            and max_validation_retries > 0
            and validation_threshold >= 75
            and budget_pressure <= 0.45
            and recursive_upfront_allowed
            and float(recursive_score["utility"]) >= float(shallow_score["utility"]) - 4.0
        ):
            selected_mode = "recursive"
            selected_utility = float(recursive_score["utility"])
            policy_adjustments.append("high_assurance_close_call_promoted_to_recursive")
        elif selected_mode == "recursive" and not recursive_upfront_allowed:
            if bool(shallow_score.get("feasible", False)):
                selected_mode = "shallow"
                selected_utility = float(shallow_score["utility"])
                policy_adjustments.append("recursive_deferred_until_validation_failure")

        if policy_adjustments:
            best_alternative = max(
                (
                    float(score["utility"])
                    for mode, score in mode_scores.items()
                    if mode != selected_mode
                ),
                default=float(selected_utility),
            )
            utility_margin = round(float(selected_utility) - best_alternative, 4)

        formula = (
            "utility(mode)=expected_quality(mode)-lambda_cost*normalized_cost(mode)*100"
            "+llm_advisory_bonus; normalized_cost=0.60*token_norm+0.25*tool_call_norm"
            "+0.15*runtime_norm"
        )
        return {
            "selected_mode": selected_mode,
            "selected_utility": round(selected_utility, 4),
            "utility_margin": utility_margin,
            "features": features,
            "raw_features": raw_features,
            "estimated_complexity": complexity,
            "budget_pressure": budget_pressure,
            "lambda_cost": lambda_cost,
            "direct_eligible": direct_eligible,
            "direct_quality_ready": direct_quality_ready,
            "direct_quality_floor": round(direct_quality_floor, 4),
            "shallow_quality_advantage": round(shallow_quality_advantage, 4),
            "recursive_upfront_allowed": recursive_upfront_allowed,
            "policy_adjustments": policy_adjustments,
            "mode_scores": mode_scores,
            "formula": formula,
            "llm_advisory_mode": llm_decision.mode,
            "llm_advisory_confidence": llm_decision.confidence,
            "llm_advisory_complexity": llm_decision.estimated_complexity,
        }

    @staticmethod
    def _build_decomposition_graph(mode: str) -> Dict[str, Any]:
        """
        Persist the decomposition object D=(V,E,tau,rho) used by the controller.
        This is intentionally compact and inspectable for methodology reporting.
        """
        if mode == "direct":
            nodes = [
                {
                    "id": "direct_synthesis",
                    "task": "Generate startup analysis in one integrated pass",
                    "interface": "structured_llm_output",
                    "executor": "direct_strategy_agent",
                    "atomicity": "coarse",
                }
            ]
            edges: List[Dict[str, str]] = []
        else:
            nodes = [
                {
                    "id": "idea_refinement",
                    "task": "Refine raw idea into problem, solution, value proposition, and summary",
                    "interface": "RefinedIdea schema",
                    "executor": "idea_refinement_agent",
                    "atomicity": "atomic",
                },
                {
                    "id": "market_research",
                    "task": "Retrieve market sources and synthesize market, competitors, and trends",
                    "interface": "MarketOutput schema plus search/trends tools",
                    "executor": "market_research_agent",
                    "atomicity": "composite",
                },
                {
                    "id": "source_validation",
                    "task": "Evaluate evidence support and assign reliability score",
                    "interface": "JudgeValidationOutput schema",
                    "executor": "source_validator_agent",
                    "atomicity": "atomic",
                },
                {
                    "id": "business_model",
                    "task": "Generate bounded financial assumptions and deterministic scenarios",
                    "interface": "FinancialAssumptions schema plus calculator",
                    "executor": "business_model_agent",
                    "atomicity": "atomic",
                },
                {
                    "id": "pitch_content",
                    "task": "Create investor-ready pitch narrative and optional deck",
                    "interface": "PitchSlides schema",
                    "executor": "pitch_deck_generator_agent",
                    "atomicity": "atomic",
                },
            ]
            edges = [
                {"from": "idea_refinement", "to": "market_research"},
                {"from": "market_research", "to": "source_validation"},
                {"from": "source_validation", "to": "business_model"},
                {"from": "business_model", "to": "pitch_content"},
            ]
            if mode == "recursive":
                nodes.append(
                    {
                        "id": "market_revision",
                        "task": "Revise market research if validation score falls below threshold",
                        "interface": "retry prompt with prior evidence gaps",
                        "executor": "market_research_agent",
                        "atomicity": "conditional",
                    }
                )
                edges.extend(
                    [
                        {"from": "source_validation", "to": "market_revision"},
                        {"from": "market_revision", "to": "source_validation"},
                    ]
                )

        depth = {"direct": 0, "shallow": 1, "recursive": 2}.get(mode, 1)
        node_count = len(nodes)
        atomic_nodes = sum(1 for node in nodes if node.get("atomicity") == "atomic")
        atomicity_ratio = round(atomic_nodes / max(1, node_count), 4)
        return {
            "formalization": "D=(V,E,tau,rho), where V=subtasks, E=dependencies, tau=interfaces, rho=executors",
            "mode": mode,
            "depth_target": depth,
            "nodes": nodes,
            "edges": edges,
            "metrics": {
                "node_count": node_count,
                "edge_count": len(edges),
                "atomicity_ratio": atomicity_ratio,
                "branching_factor_proxy": round(len(edges) / max(1, node_count), 4),
            },
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
        scorecard: Dict[str, Any] = {}

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
            scorecard = self._score_modes(
                idea_text=combined,
                llm_decision=decision,
                budget_snapshot=budget_snapshot,
                validation_threshold=validation_threshold,
                max_validation_retries=max_validation_retries,
            )
            scorecard["forced_mode_applied"] = True
            scorecard["selected_mode_before_forcing"] = scorecard.get("selected_mode")
            scorecard["selected_mode"] = chosen_mode_final
        else:
            scorecard = self._score_modes(
                idea_text=combined,
                llm_decision=decision,
                budget_snapshot=budget_snapshot,
                validation_threshold=validation_threshold,
                max_validation_retries=max_validation_retries,
            )
            chosen_mode_initial = scorecard["selected_mode"]
            if chosen_mode_initial != decision.mode:
                deterministic_trigger = "utility_scorecard_selected_mode"

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
        feature_summary = scorecard.get("features", {}) if isinstance(scorecard, dict) else {}
        decomposition_graph = self._build_decomposition_graph(chosen_mode_final)
        decisions.append(
            {
                "mode_initial": chosen_mode_initial,
                "mode_final": chosen_mode_final,
                "confidence": decision.confidence,
                "estimated_complexity": scorecard.get(
                    "estimated_complexity",
                    decision.estimated_complexity,
                ),
                "llm_advisory_mode": decision.mode,
                "llm_advisory_complexity": decision.estimated_complexity,
                "structural_complexity": feature_summary.get("structural_complexity"),
                "uncertainty_need": feature_summary.get("uncertainty_need"),
                "utility_margin": scorecard.get("utility_margin"),
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
                "estimated_complexity": scorecard.get(
                    "estimated_complexity",
                    decision.estimated_complexity,
                ),
                "llm_advisory_mode": decision.mode,
                "llm_advisory_complexity": decision.estimated_complexity,
                "structural_complexity": feature_summary.get("structural_complexity"),
                "uncertainty_need": feature_summary.get("uncertainty_need"),
                "utility_margin": scorecard.get("utility_margin"),
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
            "controller_mode_realized": chosen_mode_final,
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
            "controller_scorecard": scorecard,
            "controller_decisions": decisions,
            "decomposition_graph": decomposition_graph,
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
                    "Controller mode: {controller_mode}\n"
                    "Retry count: {retry_count}\n"
                    "Previous validation gaps, if any:\n{validation_gaps}\n\n"
                    "Web research snippets:\n{search_results}\n\n"
                    "Keyword trend signals:\n{trend_signals}\n\n"
                    "Produce market analysis for investors. If this is a retry, repair the previous "
                    "evidence gaps directly and avoid claims that are not supported by the snippets.",
                ),
            ]
        )

    def run(self, state: PitchState) -> PitchState:
        refined_idea = _sget(state, "refined_idea", "")
        total_usage = _empty_token_usage()
        retry_count = _sget(state, "retry_count", 0)
        controller_mode = str(_sget(state, "controller_mode", "shallow") or "shallow")
        prior_validation = _sget(state, "validation_report", {}) or {}
        validation_gaps = ""
        if isinstance(prior_validation, dict):
            validation_gaps = str(prior_validation.get("evidence_gaps", "") or "")
            weak_claims = [
                str(claim.get("claim", "")).strip()
                for claim in prior_validation.get("claims", []) or []
                if str(claim.get("verdict", "")).strip().lower()
                in {"weakly_supported", "unsupported"}
            ]
            if weak_claims:
                validation_gaps = (
                    f"{validation_gaps}\nWeak or unsupported claims to repair:\n"
                    + "\n".join(f"- {claim}" for claim in weak_claims[:5])
                ).strip()
        query_suffix = (
            " prioritize authoritative and recent sources with concrete numbers"
            if retry_count > 0
            else ""
        )
        if retry_count > 0 and validation_gaps:
            query_suffix += f" evidence gaps to resolve: {validation_gaps[:240]}"
        query = f"startup market size competitors trends for: {refined_idea}{query_suffix}"
        search_payload = self.search_tool.search(query)
        if self.strict_tools and search_payload["status"] != "ok":
            raise RuntimeError(f"Market search tool unavailable: {search_payload['error']}")
        evidence_budget = _evidence_budget_for_mode(
            _sget(state, "controller_mode", "shallow"),
            _sget(state, "budget_remaining", {}).get("total_tokens")
            if isinstance(_sget(state, "budget_remaining", {}), dict)
            else None,
        )
        search_payload = _limited_search_payload(search_payload, **evidence_budget)
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
                controller_mode=controller_mode,
                retry_count=retry_count,
                validation_gaps=validation_gaps or "None.",
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
                "source_count_raw": search_payload.get("source_count_raw"),
                "evidence_limit": search_payload.get("evidence_limit"),
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


class ClaimRepairAgent:
    """Targeted claim-level market analysis repair agent."""

    def __init__(self, llm: ChatOpenAI, strict_tools: bool = True):
        self.structured_llm = llm.with_structured_output(MarketRepairOutput, include_raw=True)
        self.search_tool = MarketSearchTool()
        self.strict_tools = strict_tools
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "# Role\n"
                    "You are a conservative claim-level market repair agent.\n\n"
                    "# Instructions\n"
                    "- Do not rewrite the whole market analysis.\n"
                    "- Preserve supported claims unless directly contradicted.\n"
                    "- Repair only the listed weak/unsupported claims.\n"
                    "- The adaptive controller only benefits from source-backed improvements: prefer coverage_addition or search_and_replace patches that can raise validator score.\n"
                    "- For action=coverage_addition, append target_add_count concise source-backed market-analysis claims that improve missing coverage or specificity; preserve the existing supported claims.\n"
                    "- Treat action=coverage_addition as additive: do not remove, soften, or replace existing supported claims while adding coverage.\n"
                    "- For action=qualify_or_remove or action=remove, make the smallest safe edit only if the plan still contains enough source-backed claims; otherwise preserve the checkpoint by making no substantive change.\n"
                    "- For action=search_and_replace, every replacement must be a direct paraphrase of the focused snippets and must include supporting evidence URLs in the patch.\n"
                    "- For action=coverage_addition, every added claim must cite at least one focused evidence URL in the patch.\n"
                    "- If evidence is weak, qualify or remove the claim instead of making it sound stronger.\n"
                    "- Do not add new market-size, CAGR, adoption-rate, or competitor claims unless directly supported by focused snippets.\n"
                    "- Do not introduce numbers, named competitors, adoption rates, compliance claims, or market forecasts that are absent from the provided snippets.\n"
                    "- For speculative negative competitor-gap claims, prefer conservative qualification over stronger replacement.\n"
                    "- A successful repair is usually shorter and more cautious than the original weak claim.\n"
                    "- Keep the output concise to control token cost.\n\n"
                    "# Output Format\n"
                    "- Return content that strictly matches the structured schema fields.",
                ),
                (
                    "human",
                    "Startup concept:\n{refined_idea}\n\n"
                    "Previous market analysis:\n{previous_market_analysis}\n\n"
                    "Supported claims to preserve:\n{supported_claims}\n\n"
                    "Repair plan:\n{repair_plan}\n\n"
                    "Retrieval diagnostics:\n{retrieval_diagnostics}\n\n"
                    "Focused evidence snippets:\n{focused_evidence}\n\n"
                    "Return a patched market analysis plus explicit patches.",
                ),
            ]
        )

    @staticmethod
    def _queries_for_plan(plan: List[Dict[str, Any]]) -> List[str]:
        queries: List[str] = []
        for item in plan:
            if item.get("action") not in {"search_and_replace", "coverage_addition"}:
                continue
            candidates = item.get("candidate_queries") or []
            q = str(candidates[0]).strip() if candidates else f"{item.get('claim', '')} evidence source"
            if q and q not in queries:
                queries.append(q)
            if len(queries) >= 2:
                break
        return queries

    @staticmethod
    def _apply_source_bound_guards(plan: List[Dict[str, Any]], diagnostic_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        guarded: List[Dict[str, Any]] = []
        fragile_categories = {"competitor_landscape", "trend_or_adoption", "gap_or_opportunity", "general_market_claim"}
        fragile_failures = {"speculative_number", "missing_evidence", "too_broad", "unclear_attribution", "contradictory_evidence"}
        for raw_item in plan:
            item = dict(raw_item)
            requested_action = str(item.get("action") or "qualify_or_remove").strip()
            item["requested_action"] = requested_action
            if requested_action == "coverage_addition":
                item["guarded_action"] = "coverage_addition"
                item.setdefault("guard_reason", "coverage_addition_source_search_allowed")
                guarded.append(item)
                continue
            if requested_action != "search_and_replace":
                item["guarded_action"] = requested_action
                item.setdefault("guard_reason", "already_conservative_action")
                guarded.append(item)
                continue

            diag = diagnostic_map.get(str(item.get("claim_id")), {}) if isinstance(diagnostic_map, dict) else {}
            category = str(item.get("category") or diag.get("category") or "").strip()
            failure_type = str(item.get("failure_type") or diag.get("failure_type") or "").strip()
            materiality = int(item.get("materiality") or diag.get("materiality") or 1)
            retrieval_score = _safe_float(diag.get("retrieval_score", 0.0), 0.0)
            unique_sources = int(diag.get("unique_sources", 0) or 0)
            high_quality_sources = int(diag.get("high_quality_sources", 0) or 0)
            overlap_score = _safe_float(diag.get("overlap_score", 0.0), 0.0)

            source_ready = _source_bound_replacement_ready(
                category=category,
                materiality=materiality,
                retrieval_score=retrieval_score,
                unique_sources=unique_sources,
                high_quality_sources=high_quality_sources,
                overlap_score=overlap_score,
            )
            guard_reasons: List[str] = []
            if category in fragile_categories and failure_type in fragile_failures and not source_ready:
                guard_reasons.append("fragile_claim_type_requires_qualification")
            if category != "market_size_or_numeric" and materiality < 5 and not source_ready:
                guard_reasons.append("non_numeric_claim_not_material_enough_for_strong_replacement")
            if unique_sources < 2:
                guard_reasons.append("insufficient_unique_sources_for_replacement")
            if high_quality_sources < 1:
                guard_reasons.append("no_high_quality_source_for_replacement")
            if overlap_score < 0.12:
                guard_reasons.append("low_claim_evidence_overlap")
            if retrieval_score < 0.75 and not source_ready:
                guard_reasons.append("retrieval_score_below_source_bound_threshold")

            if guard_reasons:
                item["action"] = "qualify_or_remove"
                item["guarded_action"] = "qualify_or_remove"
                item["guard_reason"] = "|".join(guard_reasons)
                item["instruction"] = (
                    "Source-bound guard downgraded this from search_and_replace. "
                    "Do not strengthen it. Narrow it into a cautious limitation or remove it "
                    "from factual market claims unless direct snippet support is present."
                )
            else:
                item["guarded_action"] = "search_and_replace"
                item["guard_reason"] = "source_bound_search_allowed"
                item["instruction"] = (
                    "Replace only with directly source-supported wording. Use only facts, "
                    "numbers, competitors, and URLs visible in the focused snippets."
                )
            guarded.append(item)
        return guarded

    @staticmethod
    def _diagnostic_from_payload(*, plan_item: Dict[str, Any], payload: Dict[str, Any]) -> RetrievalDiagnostic:
        results = payload.get("results", []) or []
        sources = payload.get("sources", []) or []
        claim_text = str(plan_item.get("claim") or plan_item.get("claim_text") or "")
        quality_scores: List[float] = []
        domains: List[str] = []
        overlaps: List[float] = []
        source_urls: List[str] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "") or "").strip()
            if url:
                source_urls.append(url)
            q = _safe_float(item.get("quality_score", 0.0), 0.0)
            if q > 0:
                quality_scores.append(max(0.0, min(1.0, q)))
            domain = str(item.get("domain", "") or "").strip().lower()
            if domain:
                domains.append(domain)
            overlaps.append(_token_overlap_score(claim_text, f"{item.get('title', '')} {item.get('content', '')}"))
        unique_sources = len(set(source_urls or sources))
        high_quality_sources = sum(1 for q in quality_scores if q >= 0.65)
        mean_quality = round(sum(quality_scores) / max(1, len(quality_scores)), 4) if quality_scores else 0.0
        overlap = _safe_float(payload.get("mean_claim_overlap"), 0.0)
        if overlap <= 0.0 and overlaps:
            overlap = round(sum(overlaps) / max(1, len(overlaps)), 4)
        domain_diversity = round(len(set(domains)) / max(1, unique_sources), 4) if unique_sources else 0.0
        action = str(plan_item.get("action", "qualify_or_remove"))
        if action in {"qualify_or_remove", "remove"} and unique_sources == 0:
            retrieval_score = 0.45 if int(plan_item.get("materiality", 1) or 1) >= 4 else 0.30
        else:
            retrieval_score = round(0.35 * min(1.0, unique_sources / 3.0) + 0.25 * min(1.0, high_quality_sources / 2.0) + 0.20 * max(0.0, min(1.0, overlap)) + 0.10 * max(0.0, min(1.0, domain_diversity)) + 0.10 * max(0.0, min(1.0, mean_quality)), 4)
        strength = "strong" if retrieval_score >= 0.70 else ("medium" if retrieval_score >= 0.55 else ("weak" if retrieval_score > 0 else "none"))
        reasons: List[str] = []
        if unique_sources < 2 and action in {"search_and_replace", "coverage_addition"}:
            reasons.append("fewer_than_two_unique_sources")
        if high_quality_sources == 0 and action in {"search_and_replace", "coverage_addition"}:
            reasons.append("no_high_quality_source")
        if overlap < 0.10 and action in {"search_and_replace", "coverage_addition"}:
            reasons.append("low_claim_snippet_overlap")
        probe_status = str(plan_item.get("evidence_probe_status", "") or "").strip()
        if probe_status:
            reasons.append(probe_status)
        return RetrievalDiagnostic(
            claim_id=str(plan_item.get("claim_id") or ""),
            claim=claim_text,
            category=str(plan_item.get("category") or _claim_repair_category(claim_text)),
            action=action if action in {"search_and_replace", "coverage_addition", "qualify_or_remove", "remove", "preserve"} else "qualify_or_remove",  # type: ignore[arg-type]
            failure_type=str(plan_item.get("failure_type") or "missing_evidence"),
            materiality=int(plan_item.get("materiality", 1) or 1),
            coverage_enhancement=bool(plan_item.get("coverage_enhancement", False)),
            target_add_count=min(max(1, int(plan_item.get("target_add_count", 1) or 1)), 2),
            missing_claim_count=max(0, int(plan_item.get("missing_claim_count", 0) or 0)),
            claims_total_before_repair=max(0, int(plan_item.get("claims_total_before_repair", 0) or 0)),
            retrieval_score=max(0.0, min(1.0, retrieval_score)),
            unique_sources=unique_sources,
            high_quality_sources=high_quality_sources,
            mean_quality_score=mean_quality,
            overlap_score=max(0.0, min(1.0, overlap)),
            domain_diversity=max(0.0, min(1.0, domain_diversity)),
            evidence_strength=strength,  # type: ignore[arg-type]
            query_count=len(payload.get("queries", []) or []),
            status=str(payload.get("status", "unknown") or "unknown"),
            reasons=reasons,
            source_urls=source_urls[:5],
        )

    def diagnose_repairability(self, *, refined_idea: str, claims: List[Dict[str, Any]], max_results_per_query: int = 2) -> List[RetrievalDiagnostic]:
        plan = build_claim_repair_plan(claims)
        diagnostics: List[RetrievalDiagnostic] = []
        for item in plan:
            action = str(item.get("action", "qualify_or_remove"))
            payload: Dict[str, Any] = {"status": "skipped", "results": [], "sources": [], "queries": [], "error": "Conservative repair does not require focused retrieval."}
            if action in {"search_and_replace", "coverage_addition"}:
                queries = item.get("candidate_queries") or [f"{item.get('claim', '')} evidence source"]
                queries = [str(q).strip() for q in queries if str(q).strip()][:2]
                try:
                    payload = self.search_tool.search_many(queries, claim_text=str(item.get("claim") or ""), max_results_per_query=max_results_per_query)
                except AttributeError:
                    merged_results: List[Dict[str, Any]] = []
                    merged_sources: List[str] = []
                    for q in queries:
                        one = self.search_tool.search(q)
                        merged_results.extend(one.get("results", []) or [])
                        merged_sources.extend(one.get("sources", []) or [])
                    payload = {"status": "ok", "queries": queries, "results": merged_results, "sources": merged_sources, "error": ""}
                except TypeError:
                    payload = self.search_tool.search_many(queries, max_results_per_query=max_results_per_query)
            elif int(item.get("materiality", 1) or 1) >= 4:
                queries = item.get("candidate_queries") or [f"{item.get('claim', '')} evidence source"]
                queries = [str(q).strip() for q in queries if str(q).strip()][:1]
                try:
                    payload = self.search_tool.search_many(
                        queries,
                        claim_text=str(item.get("claim") or ""),
                        max_results_per_query=2,
                    )
                except AttributeError:
                    merged_results: List[Dict[str, Any]] = []
                    merged_sources: List[str] = []
                    for q in queries:
                        one = self.search_tool.search(q, claim_text=str(item.get("claim") or ""))
                        merged_results.extend(one.get("results", []) or [])
                        merged_sources.extend(one.get("sources", []) or [])
                    payload = {"status": "ok", "queries": queries, "results": merged_results, "sources": merged_sources, "error": ""}
                except TypeError:
                    payload = self.search_tool.search_many(queries, max_results_per_query=2)
                probe_item = {**item, "action": "search_and_replace"}
                probe_diag = self._diagnostic_from_payload(plan_item=probe_item, payload=payload)
                source_ready = _source_bound_replacement_ready(
                    category=probe_diag.category,
                    materiality=probe_diag.materiality,
                    retrieval_score=probe_diag.retrieval_score,
                    unique_sources=probe_diag.unique_sources,
                    high_quality_sources=probe_diag.high_quality_sources,
                    overlap_score=probe_diag.overlap_score,
                )
                if source_ready:
                    item["action"] = "search_and_replace"
                    item["evidence_probe_status"] = "evidence_probe_upgraded_to_search_and_replace"
                else:
                    item["evidence_probe_status"] = "evidence_probe_insufficient_for_replacement"
            diagnostics.append(self._diagnostic_from_payload(plan_item=item, payload=payload))
        return diagnostics

    def run(self, state: PitchState) -> PitchState:
        repair_context = _sget(state, "market_repair_context", {}) or {}
        plan = repair_context.get("repair_plan") or _sget(state, "repair_plan", []) or build_claim_repair_plan(repair_context.get("weak_or_unsupported_claims", []) or _sget(state, "failing_claims", []) or [])
        plan = list(plan or [])
        selected_action = _sget(state, "selected_action", {}) or repair_context.get("selected_action", {}) or {}
        selected_claim_ids = set(str(x) for x in selected_action.get("selected_claim_ids", []) or [])
        selected_action_type = str(selected_action.get("action_type") or "").strip()
        if selected_claim_ids:
            plan = [p for p in plan if str(p.get("claim_id")) in selected_claim_ids]
        if selected_action_type in {"search_and_replace", "coverage_addition", "qualify_or_remove", "remove"}:
            for item in plan:
                item["policy_selected_action"] = selected_action_type
                if selected_action_type in {"search_and_replace", "coverage_addition"}:
                    item["action"] = selected_action_type
                elif selected_action_type in {"qualify_or_remove", "remove"}:
                    item["policy_conservative_action_not_forced"] = True
        plan = plan[:3]
        diagnostics_raw = _sget(state, "retrieval_diagnostics", []) or repair_context.get("retrieval_diagnostics", []) or []
        diagnostics = [d if isinstance(d, dict) else d.model_dump() for d in diagnostics_raw]
        diagnostic_map = {str(d.get("claim_id")): d for d in diagnostics if isinstance(d, dict)}
        plan = self._apply_source_bound_guards(plan, diagnostic_map)
        queries = self._queries_for_plan(plan)
        focused_payload = {"status": "skipped", "results": [], "sources": [], "results_json": "[]", "error": "No search-based repair actions requested."}
        if queries:
            claim_text = " ".join(str(item.get("claim", "")) for item in plan if item.get("action") in {"search_and_replace", "coverage_addition"})
            try:
                focused_payload = self.search_tool.search_many(queries[:2], claim_text=claim_text, max_results_per_query=2)
            except AttributeError:
                merged_results: List[Dict[str, Any]] = []
                merged_sources: List[str] = []
                for q in queries[:2]:
                    one = self.search_tool.search(q)
                    merged_results.extend(one.get("results", []) or [])
                    merged_sources.extend(one.get("sources", []) or [])
                focused_payload = {"status": "ok", "queries": queries[:2], "results": merged_results, "sources": merged_sources, "error": ""}
            except TypeError:
                focused_payload = self.search_tool.search_many(queries[:2], max_results_per_query=2)
            if self.strict_tools and focused_payload.get("status") != "ok":
                raise RuntimeError(f"Focused claim repair search failed: {focused_payload.get('error', '')}")
        focused_payload = _limited_search_payload(focused_payload, max_items=4, content_chars=220)
        if plan and not diagnostic_map:
            for item in plan:
                payload = focused_payload if item.get("action") in {"search_and_replace", "coverage_addition"} else {"status": "skipped", "results": [], "sources": [], "queries": [], "error": "Conservative repair does not require focused retrieval."}
                diagnostic_map[str(item.get("claim_id"))] = self._diagnostic_from_payload(plan_item=item, payload=payload).model_dump()
            plan = self._apply_source_bound_guards(plan, diagnostic_map)
        diagnostics_for_prompt = [diagnostic_map.get(str(item.get("claim_id")), {}) for item in plan]
        previous_market_evidence = _sget(state, "market_evidence", []) or []
        merged_evidence = _merge_evidence_items(previous_market_evidence, focused_payload.get("results", []))
        result, usage = _invoke_structured_with_usage(
            self.structured_llm,
            self.prompt.format_messages(
                refined_idea=_sget(state, "refined_idea", ""),
                previous_market_analysis=repair_context.get("previous_market_analysis") or _sget(state, "market_analysis", ""),
                supported_claims=json.dumps(repair_context.get("supported_claims", [])[:6], indent=2),
                repair_plan=json.dumps(plan, indent=2),
                retrieval_diagnostics=json.dumps(diagnostics_for_prompt, indent=2),
                focused_evidence=focused_payload.get("results_json", "[]"),
            ),
        )
        token_usage = _merge_token_usage(state, usage)
        market_analysis = f"Target Market: {result.target_market}\nMarket Size: {result.market_size}\nTrends: {result.trends}\nCompetitors: {result.competitors}\nDifferentiation Gaps: {result.differentiation_gaps}\nPatch Summary: {result.patch_summary}"
        repair_patches = json.loads(json.dumps([p.model_dump() for p in result.patches]))
        tool_audit = list(_sget(state, "tool_audit", []))
        if queries:
            tool_audit.append({"agent": "claim_repair", "tool": "focused_repair_search", "status": focused_payload.get("status", "unknown"), "query": " | ".join(queries), "source_count": len(focused_payload.get("sources", [])), "source_count_raw": focused_payload.get("source_count_raw"), "high_quality_source_count": focused_payload.get("high_quality_source_count"), "mean_claim_overlap": focused_payload.get("mean_claim_overlap"), "error": focused_payload.get("error", "")})
        action_counts = {"search_and_replace": sum(1 for item in plan if item.get("action") == "search_and_replace"), "coverage_addition": sum(1 for item in plan if item.get("action") == "coverage_addition"), "qualify_or_remove": sum(1 for item in plan if item.get("action") == "qualify_or_remove"), "remove": sum(1 for item in plan if item.get("action") == "remove")}
        tool_audit.append({"agent": "claim_repair", "tool": "claim_micro_repair", "status": "ok", "claim_count": len(plan), "source_count": len(focused_payload.get("sources", [])), "repair_action_counts": action_counts, "retrieval_diagnostics": diagnostics_for_prompt, "selected_action": selected_action, "patches": repair_patches, "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0), "total_tokens": usage.get("total_tokens", 0)})
        return {"market_analysis": market_analysis, "market_sources": _merge_unique(_sget(state, "market_sources", []) or [], focused_payload.get("sources", []) or []), "market_evidence": merged_evidence, "repair_plan": plan, "repair_patches": repair_patches, "retrieval_diagnostics": list(diagnostic_map.values()), "tool_audit": tool_audit, "token_usage": token_usage}



class SourceValidatorAgent:
    MINIMUM_CLAIMS_REQUIRED = 5

    def __init__(self, llm: ChatOpenAI, secondary_judge_llm: ChatOpenAI | None = None):
        judge_b_llm = secondary_judge_llm or llm
        self.judge_a_model = self._model_name(llm)
        self.judge_b_model = self._model_name(judge_b_llm)
        self.cross_model_judging = self.judge_a_model != self.judge_b_model
        self.primary_judge_llm = llm.with_structured_output(
            JudgeValidationOutput, include_raw=True
        )
        self.secondary_judge_llm = judge_b_llm.with_structured_output(
            SecondJudgeOutput, include_raw=True
        )
        self.repair_judge_llm = llm.with_structured_output(
            RepairPatchValidationOutput, include_raw=True
        )
        self.primary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "# Role\n"
                    "You are Judge A, a strict source validation analyst.\n\n"
                    "# Instructions\n"
                    "- Verify material market claims only against provided evidence snippets and URLs.\n"
                    "- Do not invent sources or unsupported claims.\n"
                    "- Extract at least 5 distinct, independently verifiable market claims when the draft contains enough material.\n"
                    "- If fewer than 5 verifiable claims are available, state this explicitly in evidence_gaps.\n"
                    "- Provide claim-level verdicts, confidence, rationale, and supporting sources.\n"
                    "- Score rubric dimensions (0-100): evidence grounding, source credibility, claim specificity, internal consistency.\n"
                    "- Provide overall reliability (0-100).\n\n"
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
        self.secondary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "# Role\n"
                    "You are Judge B, an independent methodology-focused reviewer.\n\n"
                    "# Instructions\n"
                    "- Independently evaluate each provided claim against the same evidence.\n"
                    "- Do not copy Judge A labels blindly; reassess verdict and confidence per claim.\n"
                    "- Score rubric dimensions (0-100): evidence grounding, source credibility, claim specificity, internal consistency.\n"
                    "- Provide overall reliability (0-100) and evidence gaps.\n\n"
                    "# Output Format\n"
                    "- Return content that strictly matches the structured schema fields.",
                ),
                (
                    "human",
                    "Startup concept:\n{refined_idea}\n\n"
                    "Market analysis draft:\n{market_analysis}\n\n"
                    "Evidence snippets (title/url/content):\n{market_evidence}\n\n"
                    "Claims to evaluate:\n{claims_to_review}\n\n"
                    "Return independent claim assessments and rubric scores.",
                ),
            ]
        )

    @staticmethod
    def _model_name(llm: ChatOpenAI) -> str:
        params = getattr(llm, "_default_params", {}) or {}
        return str(getattr(llm, "model_name", None) or getattr(llm, "model", None) or params.get("model") or "unknown")

    @staticmethod
    def _claim_ledger(
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
        evidence_gaps: str = "",
    ) -> List[Dict[str, Any]]:
        """Normalize judge claims into inspectable claim units for repair/evaluation."""
        ledger: List[Dict[str, Any]] = []
        evidence_domains = []
        for item in evidence or []:
            if isinstance(item, dict) and item.get("domain"):
                evidence_domains.append(str(item.get("domain")))

        distinct_domains = sorted(set(evidence_domains))
        for idx, claim in enumerate(claims or [], start=1):
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("claim") or claim.get("claim_text") or "").strip()
            if not text:
                continue

            category = _claim_repair_category(text)
            verdict = str(
                claim.get("verdict", "needs_review") or "needs_review"
            ).strip().lower()
            item = dict(claim)
            item["claim_id"] = str(claim.get("claim_id") or f"c{idx}")
            item["claim_text"] = text
            item["category"] = category
            item["materiality"] = int(
                claim.get("materiality") or _claim_materiality(text, category)
            )
            item["failure_type"] = str(
                claim.get("failure_type") or _claim_failure_type(claim, evidence_gaps)
            )
            item["source_diversity_count"] = len(distinct_domains)
            item["distinct_source_domains"] = distinct_domains[:6]
            item["regulatory_flag"] = bool(claim.get("regulatory_flag")) or any(
                marker in text.lower()
                for marker in [
                    "clinical",
                    "healthcare",
                    "patient",
                    "regulatory",
                    "compliance",
                    "hipaa",
                ]
            )
            if verdict != "supported" and not item.get("candidate_queries"):
                item["candidate_queries"] = [
                    f"{text} evidence source",
                    f"{category.replace('_', ' ')} market evidence",
                ]
            ledger.append(item)
        return ledger

    @staticmethod
    def _rubric_mean(rubric: Dict[str, Any]) -> float:
        keys = [
            "evidence_grounding",
            "source_credibility",
            "claim_specificity",
            "internal_consistency",
        ]
        vals: List[float] = []
        for key in keys:
            try:
                vals.append(float(rubric.get(key, 0)))
            except (TypeError, ValueError):
                vals.append(0.0)
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    @staticmethod
    def _clamp_score(value: float) -> int:
        return int(max(0, min(100, round(value))))

    @staticmethod
    def _judge_summary(
        report: Dict[str, Any],
        det_score: int,
        usage: Dict[str, int],
    ) -> Dict[str, Any]:
        return {
            "overall_reliability": int(report.get("overall_reliability", 0) or 0),
            "deterministic_claim_score": int(det_score),
            "rubric_scores": dict(report.get("rubric_scores", {}) or {}),
            "rubric_mean": round(
                SourceValidatorAgent._rubric_mean(report.get("rubric_scores", {}) or {}),
                3,
            ),
            "claim_count": len(report.get("claims", []) or report.get("claim_assessments", [])),
            "token_usage": usage,
        }

    @staticmethod
    def _agreement_stats(
        primary_claims: List[Dict[str, Any]],
        secondary_claims: List[Dict[str, Any]],
        rubric_a: Dict[str, Any],
        rubric_b: Dict[str, Any],
        score_a: int,
        score_b: int,
    ) -> Dict[str, Any]:
        rubric_dims = [
            "evidence_grounding",
            "source_credibility",
            "claim_specificity",
            "internal_consistency",
        ]
        rubric_diffs: Dict[str, float] = {}
        for dim in rubric_dims:
            va = float(rubric_a.get(dim, 0) or 0)
            vb = float(rubric_b.get(dim, 0) or 0)
            rubric_diffs[dim] = round(abs(va - vb), 3)

        primary_map = {
            str(c.get("claim", "")).strip().lower(): str(c.get("verdict", "")).strip().lower()
            for c in primary_claims
            if str(c.get("claim", "")).strip()
        }
        secondary_map = {
            str(c.get("claim", "")).strip().lower(): str(c.get("verdict", "")).strip().lower()
            for c in secondary_claims
            if str(c.get("claim", "")).strip()
        }
        shared = sorted(set(primary_map.keys()) & set(secondary_map.keys()))
        verdict_matches = 0
        for key in shared:
            if primary_map.get(key) == secondary_map.get(key):
                verdict_matches += 1
        verdict_agreement = (
            round(verdict_matches / len(shared), 4) if shared else 0.0
        )

        score_delta = abs(int(score_a) - int(score_b))
        score_agreement = round(max(0.0, 1.0 - (score_delta / 100.0)), 4)
        rubric_mae = (
            round(sum(rubric_diffs.values()) / max(1, len(rubric_diffs)), 4)
            if rubric_diffs
            else 0.0
        )
        rubric_agreement = round(max(0.0, 1.0 - (rubric_mae / 100.0)), 4)
        overall_agreement = round(
            (score_agreement + rubric_agreement + verdict_agreement) / 3.0, 4
        )
        return {
            "score_delta_abs": score_delta,
            "score_agreement": score_agreement,
            "rubric_mae": rubric_mae,
            "rubric_diffs": rubric_diffs,
            "rubric_agreement": rubric_agreement,
            "shared_claims": len(shared),
            "verdict_agreement_rate": verdict_agreement,
            "overall_agreement": overall_agreement,
        }

    def run(self, state: PitchState) -> PitchState:
        token_usage_current = _sget(state, "token_usage", {}) or {}
        max_total_tokens = _sget(state, "max_total_tokens")
        total_tokens_current = int(token_usage_current.get("total_tokens", 0) or 0)
        remaining_total_tokens = (
            int(max_total_tokens) - total_tokens_current
            if isinstance(max_total_tokens, int)
            else None
        )
        evidence_budget = _evidence_budget_for_mode(
            _sget(state, "controller_mode", "shallow"),
            remaining_total_tokens,
        )
        raw_evidence = _sget(state, "market_evidence", []) or []
        validation_evidence = _limit_evidence_items(raw_evidence, **evidence_budget)
        evidence_json = json.dumps(validation_evidence, indent=2)
        primary_result, usage_primary = _invoke_structured_with_usage(
            self.primary_judge_llm,
            self.primary_prompt.format_messages(
                refined_idea=_sget(state, "refined_idea", ""),
                market_analysis=_sget(state, "market_analysis", ""),
                market_evidence=evidence_json,
            ),
        )
        primary_report: Dict[str, Any] = json.loads(primary_result.model_dump_json())
        primary_claims = primary_report.get("claims", [])
        primary_det_score = _deterministic_reliability_score(primary_claims)
        claims_for_secondary = json.dumps(
            [{"claim": c.get("claim", "")} for c in primary_claims], indent=2
        )

        secondary_failed = False
        usage_secondary = _empty_token_usage()
        secondary_report: Dict[str, Any] = {}
        try:
            secondary_result, usage_secondary = _invoke_structured_with_usage(
                self.secondary_judge_llm,
                self.secondary_prompt.format_messages(
                    refined_idea=_sget(state, "refined_idea", ""),
                    market_analysis=_sget(state, "market_analysis", ""),
                    market_evidence=evidence_json,
                    claims_to_review=claims_for_secondary,
                ),
            )
            secondary_report = json.loads(secondary_result.model_dump_json())
        except Exception:  # noqa: BLE001
            secondary_failed = True
            # Fallback to primary output so pipeline remains robust.
            secondary_report = {
                "evidence_gaps": primary_report.get("evidence_gaps", ""),
                "overall_reliability": int(primary_report.get("overall_reliability", 0) or 0),
                "rubric_scores": dict(primary_report.get("rubric_scores", {}) or {}),
                "claim_assessments": list(primary_claims),
            }

        secondary_claims = secondary_report.get("claim_assessments", [])
        secondary_det_score = _deterministic_reliability_score(secondary_claims)
        score_a = int(primary_report.get("overall_reliability", 0) or 0)
        score_b = int(secondary_report.get("overall_reliability", 0) or 0)
        rubric_a = dict(primary_report.get("rubric_scores", {}) or {})
        rubric_b = dict(secondary_report.get("rubric_scores", {}) or {})

        agreement = self._agreement_stats(
            primary_claims=primary_claims,
            secondary_claims=secondary_claims,
            rubric_a=rubric_a,
            rubric_b=rubric_b,
            score_a=score_a,
            score_b=score_b,
        )
        rubric_mean_a = self._rubric_mean(rubric_a)
        rubric_mean_b = self._rubric_mean(rubric_b)
        blended_score = (
            0.35 * ((score_a + score_b) / 2.0)
            + 0.35 * ((primary_det_score + secondary_det_score) / 2.0)
            + 0.30 * ((rubric_mean_a + rubric_mean_b) / 2.0)
        )
        final_score = self._clamp_score(blended_score)

        claim_units = self._claim_ledger(primary_claims, validation_evidence, primary_report.get("evidence_gaps", ""))
        failing_claims = [c for c in claim_units if str(c.get("verdict", "")).strip().lower() in {"weakly_supported", "unsupported", "needs_review"}]
        unsupported_material_claim_count = sum(1 for c in failing_claims if str(c.get("verdict", "")).strip().lower() == "unsupported" and int(c.get("materiality", 1) or 1) >= 4)
        minimum_claims_required = self.MINIMUM_CLAIMS_REQUIRED
        missing_claims = max(0, minimum_claims_required - len(primary_claims))
        low_claim_count_flag = missing_claims > 0
        if low_claim_count_flag:
            gap_note = (
                f"Minimum claim coverage not met: {len(primary_claims)} verifiable "
                f"claims found, {minimum_claims_required} required."
            )
            existing_gaps = str(primary_report.get("evidence_gaps", "") or "").strip()
            primary_report["evidence_gaps"] = (
                f"{existing_gaps}\n{gap_note}".strip() if existing_gaps else gap_note
            )

        # Use primary claim list as canonical for downstream compatibility.
        report: Dict[str, Any] = {
            "validated_summary": primary_report.get("validated_summary", ""),
            "evidence_gaps": primary_report.get("evidence_gaps", ""),
            "claims": primary_claims,
            "claim_units": claim_units,
            "failing_claims": failing_claims,
            "unsupported_material_claim_count": unsupported_material_claim_count,
            "reliability_score": final_score,
            "judge_scores": {
                "judge_a": self._judge_summary(primary_report, primary_det_score, usage_primary),
                "judge_b": self._judge_summary(
                    {
                        "overall_reliability": score_b,
                        "rubric_scores": rubric_b,
                        "claim_assessments": secondary_claims,
                    },
                    secondary_det_score,
                    usage_secondary,
                ),
                "aggregated": {
                    "final_reliability_score": final_score,
                    "final_formula": (
                        "0.35*avg(judge_overall) + 0.35*avg(deterministic_claim_score) + "
                        "0.30*avg(rubric_mean)"
                    ),
                    "minimum_claims_required": minimum_claims_required,
                    "low_claim_count_flag": low_claim_count_flag,
                    "missing_claim_count": missing_claims,
                },
            },
            "agreement_stats": agreement,
            "judge_metadata": {
                "judge_a_model": self.judge_a_model,
                "judge_b_model": self.judge_b_model,
                "cross_model_judging": self.cross_model_judging,
                "secondary_judge_fallback": secondary_failed,
            },
            "evaluation_primary": {
                "used_for_decision": True,
                "primary_reliability_score": final_score,
                "primary_judge_agreement": agreement.get("overall_agreement"),
                "decision_rule": (
                    "needs_revision = primary_reliability_score < validation_threshold or low_claim_count_flag"
                ),
            },
        }
        total_usage = _merge_token_usage({"token_usage": usage_primary}, usage_secondary)
        token_usage = _merge_token_usage(state, total_usage)
        validated = (
            f"{_sget(state, 'market_analysis', '')}\n\n"
            f"Validation Score: {final_score}/100\n"
            f"Validated Summary: {report.get('validated_summary', '')}\n"
            f"Evidence Gaps: {report.get('evidence_gaps', '')}\n"
            f"Judge Agreement: {agreement.get('overall_agreement')}"
        )
        tool_audit = list(_sget(state, "tool_audit", []))
        tool_audit.append(
            {
                "agent": "source_validator",
                "tool": "llm_claim_verifier_dual_judge",
                "status": "ok",
                "judge_a_model": self.judge_a_model,
                "judge_b_model": self.judge_b_model,
                "cross_model_judging": self.cross_model_judging,
                "lightweight_repair_validation": False,
                "claims_checked": len(report.get("claims", [])),
                "minimum_claims_required": minimum_claims_required,
                "low_claim_count_flag": low_claim_count_flag,
                "missing_claim_count": missing_claims,
                "reliability_score": final_score,
                "judge_a_score": score_a,
                "judge_b_score": score_b,
                "judge_agreement": agreement.get("overall_agreement"),
                "secondary_judge_fallback": secondary_failed,
                "evidence_items_raw": len(raw_evidence),
                "evidence_items_used": len(validation_evidence),
                "evidence_limit": evidence_budget,
                "prompt_tokens": total_usage.get("prompt_tokens", 0),
                "completion_tokens": total_usage.get("completion_tokens", 0),
                "total_tokens": total_usage.get("total_tokens", 0),
            }
        )
        return {
            "validation_report": report,
            "validated_market_analysis": validated,
            "needs_revision": final_score < _sget(state, "validation_threshold", 70) or low_claim_count_flag,
            "tool_audit": tool_audit,
            "token_usage": token_usage,
        }


    def run_repair_only(self, state: PitchState) -> PitchState:
        """Lightweight validation for a candidate claim-level patch."""
        validation = _sget(state, "validation_report", {}) or {}
        repair_context = _sget(state, "market_repair_context", {}) or {}
        repair_patches = _sget(state, "repair_patches", []) or []
        previous_score = int(repair_context.get("previous_validation_score") or validation.get("reliability_score", 0) or 0)
        previous_supported_ratio = float(repair_context.get("previous_supported_ratio") or 0.0)
        focused_evidence = _limit_evidence_items(_sget(state, "market_evidence", []) or [], max_items=4, content_chars=220)
        repaired_claims = [{"original_claim": p.get("original_claim"), "action": p.get("action"), "replacement": p.get("replacement"), "evidence_urls": p.get("evidence_urls", [])} for p in repair_patches if isinstance(p, dict)]
        result, usage = _invoke_structured_with_usage(
            self.repair_judge_llm,
            ChatPromptTemplate.from_messages([
                ("system", "You are a lightweight patch validator. Validate only the repaired claims against the provided evidence. Accept only if the patch preserves or improves reliability and does not hide material unsupported claims."),
                ("human", "Startup concept:\n{refined_idea}\n\nPrevious market analysis:\n{previous_market_analysis}\n\nCandidate repaired market analysis:\n{candidate_market_analysis}\n\nClaims repaired:\n{repaired_claims}\n\nFocused evidence snippets:\n{focused_evidence}\n\nPrevious validation score: {previous_score}\nPrevious supported ratio: {previous_supported_ratio}")
            ]).format_messages(
                refined_idea=_sget(state, "refined_idea", ""),
                previous_market_analysis=repair_context.get("previous_market_analysis", ""),
                candidate_market_analysis=_sget(state, "market_analysis", ""),
                repaired_claims=json.dumps(repaired_claims, indent=2),
                focused_evidence=json.dumps(focused_evidence, indent=2),
                previous_score=previous_score,
                previous_supported_ratio=previous_supported_ratio,
            ),
        )
        patch_report: Dict[str, Any] = json.loads(result.model_dump_json())
        patch_report["judge_metadata"] = {"judge_a_model": self.judge_a_model, "judge_b_model": self.judge_b_model, "cross_model_judging": self.cross_model_judging, "lightweight_repair_validation": True}
        repaired_assessments = patch_report.get("repaired_claim_assessments", []) or []
        candidate_score = int(patch_report.get("patch_reliability", 0) or 0)
        candidate_ratio = 0.0
        if repaired_assessments:
            candidate_ratio = sum(1 for c in repaired_assessments if str(c.get("verdict", "")).strip().lower() == "supported") / max(1, len(repaired_assessments))
        accepted = bool(patch_report.get("can_accept_patch", False))
        accepted = accepted and candidate_score >= previous_score and candidate_ratio >= max(0.0, previous_supported_ratio - 0.05)
        previous_validation = validation if isinstance(validation, dict) else {}
        if accepted:
            final_validation = dict(previous_validation)
            final_validation["reliability_score"] = max(previous_score, candidate_score)
            final_validation["evidence_gaps"] = patch_report.get("evidence_gaps", "")
            if repaired_assessments:
                final_validation["claims"] = repaired_assessments
                final_validation["claim_units"] = self._claim_ledger(repaired_assessments, focused_evidence, patch_report.get("evidence_gaps", ""))
                final_validation["failing_claims"] = [c for c in final_validation["claim_units"] if str(c.get("verdict", "")).strip().lower() in {"weakly_supported", "unsupported", "needs_review"}]
            final_validation["repair_validation"] = {"accepted": True, "previous_reliability_score": previous_score, "candidate_reliability_score": candidate_score, "rationale": patch_report.get("rationale", "")}
            validated_market_analysis = f"{_sget(state, 'market_analysis', '')}\n\nValidation Score: {final_validation.get('reliability_score', previous_score)}/100\nPatch Validation: accepted"
            accepted_delta, rejected_delta = 1, 0
        else:
            final_validation = dict(previous_validation)
            final_validation["repair_validation"] = {"accepted": False, "previous_reliability_score": previous_score, "candidate_reliability_score": candidate_score, "rationale": patch_report.get("rationale", "")}
            validated_market_analysis = _sget(state, "validated_market_analysis", "")
            accepted_delta, rejected_delta = 0, 1
        token_usage = _merge_token_usage(state, usage)
        tool_audit = list(_sget(state, "tool_audit", []))
        tool_audit.append({"agent": "source_validator", "tool": "llm_claim_repair_validator", "status": "ok", "judge_a_model": self.judge_a_model, "judge_b_model": self.judge_b_model, "cross_model_judging": self.cross_model_judging, "secondary_judge_fallback": False, "lightweight_repair_validation": True, "repair_patch_accepted": accepted, "previous_reliability_score": previous_score, "candidate_reliability_score": candidate_score, "total_tokens": usage.get("total_tokens", 0), "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)})
        return {"validation_report": final_validation, "validated_market_analysis": validated_market_analysis, "micro_validation": patch_report, "accepted_patch_count": int(_sget(state, "accepted_patch_count", 0) or 0) + accepted_delta, "rejected_patch_count": int(_sget(state, "rejected_patch_count", 0) or 0) + rejected_delta, "needs_revision": False if not accepted else final_validation.get("reliability_score", previous_score) < _sget(state, "validation_threshold", 70), "tool_audit": tool_audit, "token_usage": token_usage}



class DirectStrategyAgent:
    def __init__(
        self,
        llm: ChatOpenAI,
        strict_tools: bool = True,
        enable_trends: bool = True,
        secondary_judge_llm: ChatOpenAI | None = None,
    ):
        self.structured_llm = llm.with_structured_output(DirectStrategyOutput, include_raw=True)
        self.validator_agent = SourceValidatorAgent(
            llm,
            secondary_judge_llm=secondary_judge_llm,
        )
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
        evidence_budget = _evidence_budget_for_mode(
            "direct",
            _sget(state, "budget_remaining", {}).get("total_tokens")
            if isinstance(_sget(state, "budget_remaining", {}), dict)
            else None,
        )
        search_payload = _limited_search_payload(search_payload, **evidence_budget)

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
                "source_count_raw": search_payload.get("source_count_raw"),
                "evidence_limit": search_payload.get("evidence_limit"),
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























# from __future__ import annotations

# import json
# import re
# from pathlib import Path
# from typing import Any, Dict, List, Literal, Optional

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from pydantic import BaseModel, Field

# from startup_pitch_refinery.state import PitchState
# from startup_pitch_refinery.tools import (
#     BusinessCalcTool,
#     GoogleTrendsTool,
#     MarketSearchTool,
#     ScenarioAnalysisTool,
#     generate_pitch_deck,
# )


# def _sget(state: PitchState | Dict[str, Any], key: str, default: Any = None) -> Any:
#     if isinstance(state, dict):
#         return state.get(key, default)
#     return getattr(state, key, default)


# def _empty_token_usage() -> Dict[str, int]:
#     return {
#         "prompt_tokens": 0,
#         "completion_tokens": 0,
#         "total_tokens": 0,
#     }


# def _extract_token_usage(raw_message: Any) -> Dict[str, int]:
#     usage = _empty_token_usage()
#     if raw_message is None:
#         return usage

#     usage_meta = getattr(raw_message, "usage_metadata", None) or {}
#     response_meta = getattr(raw_message, "response_metadata", None) or {}
#     token_usage = response_meta.get("token_usage", {}) if isinstance(response_meta, dict) else {}

#     prompt = (
#         usage_meta.get("input_tokens")
#         or usage_meta.get("prompt_tokens")
#         or token_usage.get("prompt_tokens")
#         or 0
#     )
#     completion = (
#         usage_meta.get("output_tokens")
#         or usage_meta.get("completion_tokens")
#         or token_usage.get("completion_tokens")
#         or 0
#     )
#     total = (
#         usage_meta.get("total_tokens")
#         or token_usage.get("total_tokens")
#         or (int(prompt) + int(completion))
#     )

#     usage["prompt_tokens"] = max(0, int(prompt))
#     usage["completion_tokens"] = max(0, int(completion))
#     usage["total_tokens"] = max(0, int(total))
#     return usage


# def _invoke_structured_with_usage(runnable: Any, messages: Any) -> tuple[Any, Dict[str, int]]:
#     payload = runnable.invoke(messages)
#     if isinstance(payload, dict) and "parsed" in payload:
#         parsed = payload.get("parsed")
#         if parsed is None:
#             raise ValueError(f"Structured output parsing failed: {payload.get('parsing_error')}")
#         usage = _extract_token_usage(payload.get("raw"))
#         return parsed, usage
#     return payload, _empty_token_usage()


# def _merge_token_usage(
#     state: PitchState | Dict[str, Any], usage_delta: Dict[str, int]
# ) -> Dict[str, int]:
#     base = _sget(state, "token_usage", {}) or {}
#     merged = {
#         "prompt_tokens": int(base.get("prompt_tokens", 0)) + int(usage_delta.get("prompt_tokens", 0)),
#         "completion_tokens": int(base.get("completion_tokens", 0))
#         + int(usage_delta.get("completion_tokens", 0)),
#         "total_tokens": int(base.get("total_tokens", 0)) + int(usage_delta.get("total_tokens", 0)),
#     }
#     return merged


# def _deterministic_reliability_score(claims: List[Dict[str, Any]]) -> int:
#     """
#     Deterministic score from claim-level outputs.
#     Formula:
#       claim_score = verdict_weight * confidence
#       reliability = average(claim_score) * 100
#     """
#     if not claims:
#         return 0

#     verdict_weights = {
#         "supported": 1.0,
#         "weakly_supported": 0.6,
#         "needs_review": 0.3,
#         "unsupported": 0.0,
#     }

#     total = 0.0
#     for claim in claims:
#         verdict = str(claim.get("verdict", "needs_review"))
#         confidence = claim.get("confidence", 0.0)
#         try:
#             conf = float(confidence)
#         except (TypeError, ValueError):
#             conf = 0.0
#         conf = max(0.0, min(1.0, conf))
#         total += verdict_weights.get(verdict, 0.3) * conf

#     return int(round((total / len(claims)) * 100))


# def _fallback_keywords(text: str) -> List[str]:
#     """Domain-agnostic fallback if LLM keyword extraction fails."""
#     tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", text.lower())
#     stop = {
#         "problem",
#         "solution",
#         "summary",
#         "value",
#         "proposition",
#         "startup",
#         "idea",
#         "with",
#         "that",
#         "this",
#         "from",
#         "into",
#         "your",
#         "for",
#         "and",
#         "the",
#     }
#     keywords: List[str] = []
#     for token in tokens:
#         if token in stop:
#             continue
#         if token not in keywords:
#             keywords.append(token)
#     if not keywords:
#         return ["startup market", "industry trends", "competitor landscape"]
#     return keywords[:5]


# def _limit_evidence_items(
#     evidence: List[Dict[str, Any]],
#     *,
#     max_items: int,
#     content_chars: int,
# ) -> List[Dict[str, Any]]:
#     """Keep evidence prompts bounded while preserving titles and source URLs."""
#     limited: List[Dict[str, Any]] = []
#     seen_urls = set()
#     for item in evidence:
#         if len(limited) >= max_items:
#             break
#         if not isinstance(item, dict):
#             continue
#         url = str(item.get("url", "")).strip()
#         if url and url in seen_urls:
#             continue
#         if url:
#             seen_urls.add(url)
#         limited.append(
#             {
#                 "title": str(item.get("title", ""))[:160],
#                 "url": url,
#                 "domain": str(item.get("domain", ""))[:120],
#                 "content": str(item.get("content", ""))[:content_chars],
#                 "quality_score": item.get("quality_score"),
#                 "claim_id": item.get("claim_id"),
#             }
#         )
#     return limited


# def _limited_search_payload(
#     payload: Dict[str, Any],
#     *,
#     max_items: int,
#     content_chars: int,
# ) -> Dict[str, Any]:
#     """Return a compact search payload for LLM prompts and downstream validation."""
#     evidence = _limit_evidence_items(
#         payload.get("results", []) or [],
#         max_items=max_items,
#         content_chars=content_chars,
#     )
#     compact = dict(payload)
#     compact["results"] = evidence
#     compact["results_json"] = json.dumps(evidence, indent=2)
#     compact["sources"] = [item["url"] for item in evidence if item.get("url")]
#     compact["source_count_raw"] = len(payload.get("sources", []) or [])
#     compact["evidence_limit"] = {
#         "max_items": max_items,
#         "content_chars": content_chars,
#         "raw_items": len(payload.get("results", []) or []),
#         "used_items": len(evidence),
#     }
#     return compact


# def _merge_evidence_items(
#     primary: List[Dict[str, Any]],
#     secondary: List[Dict[str, Any]],
# ) -> List[Dict[str, Any]]:
#     """Merge evidence lists while preserving order and dropping duplicate URLs."""
#     merged: List[Dict[str, Any]] = []
#     seen = set()
#     for item in list(primary or []) + list(secondary or []):
#         if not isinstance(item, dict):
#             continue
#         url = str(item.get("url", "") or "").strip()
#         key = url or f"{item.get('title', '')}:{str(item.get('content', ''))[:80]}"
#         if key in seen:
#             continue
#         seen.add(key)
#         merged.append(item)
#     return merged


# def _claim_repair_category(claim_text: str) -> str:
#     """Classify a validator claim into a repair-relevant category."""
#     text = (claim_text or "").lower()
#     numeric_markers = ["$", "%", "cagr", "billion", "million", "trillion", "market size", "forecast", "projected", "growth"]
#     gap_markers = ["gap", "opportunity", "underserved", "differentiation", "white space", "lack", "limited", "specific"]
#     competitor_markers = ["competitor", "competitors", "platforms", "vendors", "solutions", "players", "offerings"]
#     trend_markers = ["trend", "adoption", "increasing", "demand", "shift", "predictive", "automation", "real-time"]
#     if any(m in text for m in numeric_markers):
#         return "market_size_or_numeric"
#     if any(m in text for m in competitor_markers):
#         return "competitor_landscape"
#     if any(m in text for m in trend_markers):
#         return "trend_or_adoption"
#     if any(m in text for m in gap_markers):
#         return "gap_or_opportunity"
#     return "general_market_claim"


# def _claim_materiality(claim_text: str, category: str | None = None) -> int:
#     """Estimate business-criticality for claim-targeted repair routing."""
#     text = (claim_text or "").lower()
#     category = category or _claim_repair_category(text)
#     materiality = 2
#     if category == "market_size_or_numeric":
#         materiality = 5
#     elif category in {"competitor_landscape", "trend_or_adoption"}:
#         materiality = 4
#     elif category == "gap_or_opportunity":
#         materiality = 3
#     if any(m in text for m in ["regulatory", "compliance", "clinical", "healthcare", "risk", "patient"]):
#         materiality = min(5, materiality + 1)
#     if any(m in text for m in ["may", "could", "potential", "hypothesis"]):
#         materiality = max(1, materiality - 1)
#     return max(1, min(5, materiality))


# def _claim_failure_type(claim: Dict[str, Any], evidence_gaps: str = "") -> str:
#     text = str(claim.get("claim", "") or claim.get("claim_text", "") or "").lower()
#     verdict = str(claim.get("verdict", "") or "").strip().lower()
#     rationale = str(claim.get("rationale", "") or "").lower()
#     combined = " ".join([text, rationale, evidence_gaps.lower()])
#     if verdict == "supported":
#         return "none"
#     if any(m in combined for m in ["contradict", "conflict", "inconsistent"]):
#         return "contradictory_evidence"
#     if any(m in combined for m in ["cagr", "percent", "%", "market size", "billion", "million", "forecast"]):
#         return "speculative_number"
#     if any(m in combined for m in ["outdated", "older", "old source"]):
#         return "outdated"
#     if any(m in combined for m in ["broad", "generic", "not specific", "indirect", "adjacent"]):
#         return "too_broad"
#     if any(m in combined for m in ["unclear", "ambiguous", "attribution"]):
#         return "unclear_attribution"
#     return "missing_evidence"


# def _repair_action_for_claim(claim: Dict[str, Any]) -> str:
#     claim_text = str(claim.get("claim", "") or claim.get("claim_text", "") or "")
#     verdict = str(claim.get("verdict", "") or "").strip().lower()
#     category = _claim_repair_category(claim_text)
#     supporting_sources = claim.get("supporting_sources", []) or []
#     if category == "gap_or_opportunity":
#         return "qualify_or_remove"
#     if category in {"market_size_or_numeric", "competitor_landscape", "trend_or_adoption"}:
#         return "search_and_replace"
#     if verdict == "unsupported" and not supporting_sources:
#         return "remove"
#     if verdict == "needs_review":
#         return "qualify_or_remove"
#     return "search_and_replace" if supporting_sources else "qualify_or_remove"


# def build_claim_repair_plan(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """Convert weak validator claims into explicit, bounded repair actions."""
#     plan: List[Dict[str, Any]] = []
#     for claim in claims or []:
#         if not isinstance(claim, dict):
#             continue
#         claim_text = str(claim.get("claim") or claim.get("claim_text") or "").strip()
#         if not claim_text:
#             continue
#         category = _claim_repair_category(claim_text)
#         action = _repair_action_for_claim(claim)
#         materiality = int(claim.get("materiality") or _claim_materiality(claim_text, category))
#         failure_type = str(claim.get("failure_type") or claim.get("failure_type_hint") or _claim_failure_type(claim))
#         if action == "search_and_replace":
#             instruction = "Search for direct evidence, then replace only if the evidence supports the revised wording."
#         elif action == "remove":
#             instruction = "Remove this unsupported claim; do not replace it with a broader speculative claim."
#         else:
#             instruction = "Narrow to a conservative hypothesis/limitation or remove it if direct support is absent."
#         plan.append({
#             "claim_id": str(claim.get("claim_id") or f"c{len(plan)+1}"),
#             "claim": claim_text,
#             "claim_text": claim_text,
#             "verdict": str(claim.get("verdict", "") or "").strip().lower(),
#             "confidence": claim.get("confidence"),
#             "category": category,
#             "materiality": materiality,
#             "failure_type": failure_type,
#             "action": action,
#             "instruction": instruction,
#             "supporting_sources": claim.get("supporting_sources", []),
#             "rationale": str(claim.get("rationale", "") or "")[:300],
#             "candidate_queries": claim.get("candidate_queries", []),
#         })
#     return plan


# def _evidence_budget_for_mode(mode: str, remaining_total_tokens: Any = None) -> Dict[str, int]:
#     """Mode- and budget-aware evidence limits for synthesis/validation prompts."""
#     mode_l = str(mode or "").strip().lower()
#     if mode_l == "direct":
#         max_items, content_chars = 5, 220
#     elif mode_l == "recursive":
#         max_items, content_chars = 10, 280
#     else:
#         max_items, content_chars = 8, 260

#     if isinstance(remaining_total_tokens, int):
#         if remaining_total_tokens <= 3500:
#             max_items = min(max_items, 3)
#             content_chars = min(content_chars, 180)
#         elif remaining_total_tokens <= 7000:
#             max_items = min(max_items, 5)
#             content_chars = min(content_chars, 220)

#     return {
#         "max_items": max_items,
#         "content_chars": content_chars,
#     }


# class RefinedIdea(BaseModel):
#     problem: str = Field(..., description="Core user/customer problem")
#     solution: str = Field(..., description="Proposed startup solution")
#     value_proposition: str = Field(..., description="Why this solution wins")
#     refined_summary: str = Field(..., description="Short integrated startup concept")


# class MarketOutput(BaseModel):
#     target_market: str
#     market_size: str
#     trends: str
#     competitors: str
#     differentiation_gaps: str


# class ClaimPatchOutput(BaseModel):
#     original_claim: str
#     action: Literal["search_and_replace", "qualify_or_remove", "remove", "preserve"]
#     replacement: str
#     expected_reliability_effect: Literal[
#         "improve_to_supported",
#         "reduce_to_conservative_claim",
#         "remove_unsupported_claim",
#         "no_change",
#     ]
#     evidence_urls: List[str] = Field(default_factory=list)
#     rationale: str


# class MarketRepairOutput(MarketOutput):
#     patch_summary: str
#     patches: List[ClaimPatchOutput] = Field(default_factory=list)


# class BusinessOutput(BaseModel):
#     revenue_streams: str
#     pricing_strategy: str
#     cost_structure: str
#     unit_economics: str
#     financial_projection: str


# class DirectStrategyOutput(BaseModel):
#     target_market: str
#     market_size: str
#     trends: str
#     competitors: str
#     differentiation_gaps: str
#     revenue_streams: str
#     pricing_strategy: str
#     cost_structure: str
#     unit_economics: str
#     financial_projection: str
#     users_year1: int = Field(..., ge=1000, le=500000)
#     arpu_monthly: float = Field(..., ge=2.0, le=300.0)
#     gross_margin: float = Field(..., ge=0.2, le=0.95)
#     assumptions_rationale: str


# class FinancialAssumptions(BaseModel):
#     users_year1: int = Field(..., ge=1000, le=500000)
#     arpu_monthly: float = Field(..., ge=2.0, le=300.0)
#     gross_margin: float = Field(..., ge=0.2, le=0.95)
#     rationale: str


# class PitchSlides(BaseModel):
#     title: str
#     subtitle: str
#     problem: str
#     solution: str
#     market: str
#     business_model: str
#     competitive_advantage: str
#     financials: str


# class VerifiedClaim(BaseModel):
#     claim: str
#     verdict: Literal["supported", "weakly_supported", "unsupported", "needs_review"]
#     confidence: float = Field(..., ge=0.0, le=1.0)
#     rationale: str
#     supporting_sources: List[str]


# class ValidationOutput(BaseModel):
#     validated_summary: str
#     reliability_score: int = Field(..., ge=0, le=100)
#     evidence_gaps: str
#     claims: List[VerifiedClaim]


# class ValidationRubricScores(BaseModel):
#     evidence_grounding: int = Field(..., ge=0, le=100)
#     source_credibility: int = Field(..., ge=0, le=100)
#     claim_specificity: int = Field(..., ge=0, le=100)
#     internal_consistency: int = Field(..., ge=0, le=100)


# class JudgeValidationOutput(BaseModel):
#     validated_summary: str
#     evidence_gaps: str
#     overall_reliability: int = Field(..., ge=0, le=100)
#     rubric_scores: ValidationRubricScores
#     claims: List[VerifiedClaim]


# class SecondJudgeOutput(BaseModel):
#     evidence_gaps: str
#     overall_reliability: int = Field(..., ge=0, le=100)
#     rubric_scores: ValidationRubricScores
#     claim_assessments: List[VerifiedClaim]



# class RepairPatchValidationOutput(BaseModel):
#     evidence_gaps: str
#     patch_reliability: int = Field(..., ge=0, le=100)
#     rubric_scores: ValidationRubricScores
#     repaired_claim_assessments: List[VerifiedClaim]
#     can_accept_patch: bool
#     rationale: str


# class TrendKeywords(BaseModel):
#     keywords: List[str] = Field(
#         ...,
#         description="3 to 7 concise, domain-agnostic trend keywords for the startup idea.",
#         min_length=3,
#         max_length=7,
#     )


# class ControllerDecision(BaseModel):
#     mode: Literal["direct", "shallow", "recursive"]
#     confidence: float = Field(..., ge=0.0, le=1.0)
#     rationale: str
#     estimated_complexity: int = Field(..., ge=0, le=100)
#     expected_tool_calls_delta: int = Field(..., ge=0, le=20)
#     expected_token_proxy_delta: int = Field(..., ge=50, le=6000)
#     expected_runtime_seconds_delta: float = Field(..., ge=0.1, le=600.0)
#     triggers: List[str] = Field(default_factory=list)
#     predicted_shallow_reliability: int = Field(default=75, ge=0, le=100)
#     predicted_repair_gain: float = Field(default=0.0, ge=0.0, le=20.0)
#     predicted_repair_cost_tokens: int = Field(default=500, ge=50, le=4000)


# class SupervisorAgent:

#     @staticmethod
#     def _claim_ledger(
#         claims: List[Dict[str, Any]],
#         evidence: List[Dict[str, Any]],
#         evidence_gaps: str = "",
#     ) -> List[Dict[str, Any]]:
#         ledger: List[Dict[str, Any]] = []
#         evidence_domains = []
#         for item in evidence or []:
#             if isinstance(item, dict) and item.get("domain"):
#                 evidence_domains.append(str(item.get("domain")))
#         for idx, claim in enumerate(claims or [], start=1):
#             if not isinstance(claim, dict):
#                 continue
#             text = str(claim.get("claim", "") or "").strip()
#             if not text:
#                 continue
#             category = _claim_repair_category(text)
#             verdict = str(claim.get("verdict", "needs_review") or "needs_review").strip().lower()
#             item = dict(claim)
#             item["claim_id"] = str(claim.get("claim_id") or f"c{idx}")
#             item["claim_text"] = text
#             item["category"] = category
#             item["materiality"] = int(claim.get("materiality") or _claim_materiality(text, category))
#             item["failure_type"] = str(claim.get("failure_type") or _claim_failure_type(claim, evidence_gaps))
#             item["source_diversity_count"] = len(set(evidence_domains))
#             item["distinct_source_domains"] = sorted(set(evidence_domains))[:6]
#             item["regulatory_flag"] = any(m in text.lower() for m in ["clinical", "healthcare", "patient", "regulatory", "compliance", "hipaa"])
#             if verdict != "supported" and not item.get("candidate_queries"):
#                 item["candidate_queries"] = [
#                     f"{text} evidence source",
#                     f"{category.replace('_', ' ')} market evidence",
#                 ]
#             ledger.append(item)
#         return ledger

#     def run(self, state: PitchState) -> PitchState:
#         plan = [
#             "1. Understand and refine the startup idea.",
#             "2. Research market size, trends, and competitors.",
#             "3. Validate claims against cited sources and score reliability.",
#             "4. Build business model and basic financial projection.",
#             "5. Structure investor-ready pitch content.",
#             "6. Generate PowerPoint pitch deck (.pptx).",
#         ]
#         return {"task_plan": plan}


# class IdeaRefinementAgent:
#     def __init__(self, llm: ChatOpenAI):
#         self.structured_llm = llm.with_structured_output(RefinedIdea, include_raw=True)
#         self.prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "# Role\n"
#                     "You are a startup concept refiner.\n\n"
#                     "# Instructions\n"
#                     "- Transform the raw startup idea into an investor-ready concept.\n"
#                     "- Keep outputs concise, concrete, and specific.\n"
#                     "- Cover problem, solution, value proposition, and a unified summary.\n\n"
#                     "# Output Format\n"
#                     "- Return content that strictly matches the structured schema fields.",
#                 ),
#                 (
#                     "human",
#                     "Raw idea: {idea}\n\n"
#                     "Define: problem, solution, value proposition, and refined summary.",
#                 ),
#             ]
#         )

#     def run(self, state: PitchState) -> PitchState:
#         idea = _sget(state, "idea", "")
#         result, usage = _invoke_structured_with_usage(
#             self.structured_llm, self.prompt.format_messages(idea=idea)
#         )
#         token_usage = _merge_token_usage(state, usage)
#         refined = (
#             f"Problem: {result.problem}\n"
#             f"Solution: {result.solution}\n"
#             f"Value Proposition: {result.value_proposition}\n"
#             f"Summary: {result.refined_summary}"
#         )
#         return {"refined_idea": refined, "token_usage": token_usage}


# class AdaptiveControllerAgent:
#     MODE_ORDER = ["direct", "shallow", "recursive"]
#     MODE_QUALITY_PRIORITY = ["recursive", "shallow", "direct"]
#     MODE_PRIORS = {
#         "direct": {
#             "tool_calls": 4,
#             "token_proxy": 1700,
#             "runtime_seconds": 28.0,
#         },
#         "shallow": {
#             "tool_calls": 6,
#             "token_proxy": 3000,
#             "runtime_seconds": 38.0,
#         },
#         "recursive": {
#             "tool_calls": 8,
#             "token_proxy": 3900,
#             "runtime_seconds": 55.0,
#         },
#     }

#     def __init__(self, llm: ChatOpenAI):
#         self.structured_llm = llm.with_structured_output(ControllerDecision, include_raw=True)
#         self.prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "# Role\n"
#                     "You are an adaptive decomposition controller for a controller-executor startup workflow.\n\n"
#                     "# Instructions\n"
#                     "- Choose one execution mode: direct, shallow, or recursive.\n"
#                     "- `direct`: lowest decomposition and lowest cost for straightforward ideas.\n"
#                     "- `shallow`: one-pass decomposition with validation and no recursive retry.\n"
#                     "- `recursive`: decomposition with validation-driven retry for higher uncertainty.\n"
#                     "- Balance expected quality gains against remaining budget.\n"
#                     "- Estimate complexity (0-100), confidence, concise rationale, and expected incremental costs.\n"
#                     "- Include short trigger phrases that justify the choice.\n\n"
#                     "# Output Format\n"
#                     "- Return content that strictly matches the structured schema fields.",
#                 ),
#                 (
#                     "human",
#                     "Raw idea:\n{idea}\n\n"
#                     "Refined idea:\n{refined_idea}\n\n"
#                     "Validation threshold: {validation_threshold}\n"
#                     "Max retries available: {max_validation_retries}\n\n"
#                     "Budget state:\n{budget_snapshot}\n\n"
#                     "Select mode and provide rationale.",
#                 ),
#             ]
#         )

#     @staticmethod
#     def _fallback_decision(idea_text: str) -> ControllerDecision:
#         text = idea_text.lower()
#         tokens = re.findall(r"[a-zA-Z0-9]+", text)
#         token_count = len(tokens)
#         complexity_keywords = [
#             "enterprise",
#             "compliance",
#             "risk",
#             "workflow",
#             "contract",
#             "integration",
#             "platform",
#             "multi",
#             "automation",
#             "regulatory",
#             "b2b",
#             "agent",
#             "marketplace",
#             "prediction",
#         ]
#         match_count = sum(1 for key in complexity_keywords if key in text)
#         score = min(100, int(token_count * 0.6 + match_count * 8))

#         if score < 30:
#             mode = "direct"
#             confidence = 0.78
#             rationale = "Idea appears relatively narrow and can be handled with low decomposition overhead."
#             expected_tool_calls_delta = 3
#             expected_token_proxy_delta = 1400
#             expected_runtime_seconds_delta = 18.0
#         elif score < 60:
#             mode = "shallow"
#             confidence = 0.72
#             rationale = "Idea has moderate complexity, so one decomposition pass with validation is appropriate."
#             expected_tool_calls_delta = 5
#             expected_token_proxy_delta = 2600
#             expected_runtime_seconds_delta = 32.0
#         else:
#             mode = "recursive"
#             confidence = 0.69
#             rationale = "Idea is high-complexity or high-uncertainty and benefits from validation-driven recursion."
#             expected_tool_calls_delta = 7
#             expected_token_proxy_delta = 3600
#             expected_runtime_seconds_delta = 48.0

#         triggers = []
#         if token_count >= 35:
#             triggers.append("long_problem_description")
#         if match_count >= 3:
#             triggers.append("multiple_complexity_markers")
#         if "compliance" in text or "risk" in text:
#             triggers.append("high_stakes_domain")
#         if not triggers:
#             triggers = ["low_complexity_signal"]

#         return ControllerDecision(
#             mode=mode,
#             confidence=confidence,
#             rationale=rationale,
#             estimated_complexity=score,
#             expected_tool_calls_delta=expected_tool_calls_delta,
#             expected_token_proxy_delta=expected_token_proxy_delta,
#             expected_runtime_seconds_delta=expected_runtime_seconds_delta,
#             triggers=triggers,
#             predicted_shallow_reliability=80 if score < 30 else (75 if score < 78 else 70),
#             predicted_repair_gain=0.0 if score < 30 else (4.0 if score < 78 else 8.0),
#             predicted_repair_cost_tokens=50 if score < 30 else (400 if score < 78 else 800),
#         )

#     @staticmethod
#     def _compute_remaining_budget(state: PitchState | Dict[str, Any]) -> Dict[str, Any]:
#         max_tool_calls = _sget(state, "max_tool_calls")
#         max_token_proxy = _sget(state, "max_token_proxy")
#         max_total_tokens = _sget(state, "max_total_tokens")
#         max_runtime_seconds = _sget(state, "max_runtime_seconds")
#         tool_calls_current = int(_sget(state, "tool_calls_current", 0) or 0)
#         token_proxy_current = int(_sget(state, "token_proxy_current", 0) or 0)
#         total_tokens_current = int(_sget(state, "total_tokens_current", 0) or 0)
#         runtime_elapsed_seconds = float(_sget(state, "runtime_elapsed_seconds", 0.0) or 0.0)

#         return {
#             "max_tool_calls": max_tool_calls,
#             "max_token_proxy": max_token_proxy,
#             "max_total_tokens": max_total_tokens,
#             "max_runtime_seconds": max_runtime_seconds,
#             "tool_calls_current": tool_calls_current,
#             "token_proxy_current": token_proxy_current,
#             "total_tokens_current": total_tokens_current,
#             "runtime_elapsed_seconds": round(runtime_elapsed_seconds, 3),
#             "remaining_tool_calls": (
#                 None
#                 if max_tool_calls is None
#                 else int(max_tool_calls) - tool_calls_current
#             ),
#             "remaining_token_proxy": (
#                 None
#                 if max_token_proxy is None
#                 else int(max_token_proxy) - token_proxy_current
#             ),
#             "remaining_total_tokens": (
#                 None
#                 if max_total_tokens is None
#                 else int(max_total_tokens) - total_tokens_current
#             ),
#             "remaining_runtime_seconds": (
#                 None
#                 if max_runtime_seconds is None
#                 else round(float(max_runtime_seconds) - runtime_elapsed_seconds, 3)
#             ),
#         }

#     @staticmethod
#     def _budget_guardrail_mode(budget_snapshot: Dict[str, Any]) -> Optional[str]:
#         rem_calls = budget_snapshot.get("remaining_tool_calls")
#         rem_tokens = budget_snapshot.get("remaining_token_proxy")
#         rem_total_tokens = budget_snapshot.get("remaining_total_tokens")
#         rem_runtime = budget_snapshot.get("remaining_runtime_seconds")

#         tight = (
#             (isinstance(rem_calls, int) and rem_calls <= 2)
#             or (isinstance(rem_tokens, int) and rem_tokens <= 900)
#             or (isinstance(rem_total_tokens, int) and rem_total_tokens <= 900)
#             or (isinstance(rem_runtime, (int, float)) and rem_runtime <= 10.0)
#         )
#         moderate = (
#             (isinstance(rem_calls, int) and rem_calls <= 4)
#             or (isinstance(rem_tokens, int) and rem_tokens <= 1900)
#             or (isinstance(rem_total_tokens, int) and rem_total_tokens <= 1900)
#             or (isinstance(rem_runtime, (int, float)) and rem_runtime <= 22.0)
#         )

#         if tight:
#             return "direct"
#         if moderate:
#             return "shallow"
#         return None

#     @classmethod
#     def _prior_cost_for_mode(cls, mode: str, complexity: int) -> Dict[str, Any]:
#         base = cls.MODE_PRIORS.get(mode, cls.MODE_PRIORS["shallow"])
#         # Scale priors mildly by complexity (0-100 -> 0.9x to 1.25x).
#         scale = 0.9 + (max(0, min(100, complexity)) / 100.0) * 0.35
#         return {
#             "tool_calls": max(1, int(round(base["tool_calls"] * scale))),
#             "token_proxy": max(100, int(round(base["token_proxy"] * scale))),
#             "runtime_seconds": round(max(0.5, base["runtime_seconds"] * scale), 3),
#         }

#     @classmethod
#     def _calibrate_selected_cost(
#         cls,
#         decision: ControllerDecision,
#         mode_override: Optional[str] = None,
#     ) -> Dict[str, Any]:
#         mode_for_prior = mode_override or decision.mode
#         prior = cls._prior_cost_for_mode(mode_for_prior, decision.estimated_complexity)
#         raw = {
#             "tool_calls": int(decision.expected_tool_calls_delta),
#             "token_proxy": int(decision.expected_token_proxy_delta),
#             "runtime_seconds": float(decision.expected_runtime_seconds_delta),
#         }
#         # Blend with prior and enforce non-trivial floor (80% of prior).
#         tool_calls = max(int(round(prior["tool_calls"] * 0.8)), int(round(0.75 * prior["tool_calls"] + 0.25 * raw["tool_calls"])))
#         token_proxy = max(int(round(prior["token_proxy"] * 0.8)), int(round(0.75 * prior["token_proxy"] + 0.25 * raw["token_proxy"])))
#         runtime_seconds = max(round(prior["runtime_seconds"] * 0.8, 3), round(0.75 * prior["runtime_seconds"] + 0.25 * raw["runtime_seconds"], 3))
#         return {
#             "tool_calls": tool_calls,
#             "token_proxy": token_proxy,
#             "runtime_seconds": runtime_seconds,
#             "prior": prior,
#             "raw": raw,
#         }

#     @staticmethod
#     def _complexity_features(text: str) -> Dict[str, Any]:
#         text_l = (text or "").lower()
#         tokens = re.findall(r"[a-zA-Z0-9]+", text_l)
#         complexity_terms = [
#             "recursive",
#             "adversarial",
#             "uncertainty",
#             "uncertain",
#             "cross-border",
#             "jurisdiction",
#             "compliance",
#             "regulatory",
#             "critical infrastructure",
#             "rollback",
#             "re-plan",
#             "replanning",
#             "conflicting",
#             "safety",
#             "failure",
#             "multi-agent",
#             "enterprise",
#             "integration",
#             "workflow",
#             "marketplace",
#             "prediction",
#             "optimization",
#             "supply chain",
#             "healthcare",
#             "finance",
#             "legal",
#             "cybersecurity",
#             "security",
#             "public sector",
#             "government",
#             "city",
#             "manufacturing",
#             "manufacturers",
#             "logistics",
#             "shipment",
#             "customs",
#             "vendor",
#             "geopolitical",
#             "audit",
#             "traceability",
#             "real-time",
#             "multi-sided",
#             "stakeholders",
#             "hipaa",
#             "insurance",
#             "pre-authorization",
#             "authorization",
#             "clinical",
#             "documentation",
#             "specialty",
#             "governance",
#             "fraud",
#             "banks",
#             "transaction",
#             "incident",
#             "response",
#             "alerts",
#             "logs",
#             "containment",
#             "renewable",
#             "energy",
#             "grid",
#             "equipment",
#             "maintenance",
#             "scientific",
#             "literature",
#             "token budgets",
#         ]
#         evidence_terms = [
#             "market",
#             "competitor",
#             "competitors",
#             "regulation",
#             "regulatory",
#             "compliance",
#             "clinical",
#             "hipaa",
#             "insurance",
#             "pre-authorization",
#             "authorization",
#             "patient",
#             "governance",
#             "legal",
#             "financial",
#             "bank",
#             "banks",
#             "fraud",
#             "audit",
#             "risk",
#             "safety",
#             "security",
#             "certification",
#             "benchmark",
#             "accuracy",
#             "claims",
#             "evidence",
#             "scientific",
#             "literature",
#             "customs",
#             "geopolitical",
#         ]
#         uncertainty_terms = [
#             "uncertain",
#             "uncertainty",
#             "predict",
#             "prediction",
#             "risk",
#             "failure",
#             "adversarial",
#             "conflicting",
#             "dynamic",
#             "adaptive",
#             "real-time",
#             "optimize",
#             "optimization",
#             "trade-off",
#             "tradeoff",
#             "multi-step",
#             "long-horizon",
#             "fallback",
#             "replan",
#             "re-plan",
#             "shipment",
#             "delays",
#             "fraud",
#             "incident",
#             "alerts",
#             "failures",
#             "emergency",
#             "constraints",
#             "coverage",
#             "budget",
#             "budgets",
#         ]
#         workflow_terms = [
#             "platform",
#             "workflow",
#             "coordinates",
#             "coordination",
#             "management",
#             "monitoring",
#             "dashboard",
#             "assistant",
#             "copilot",
#             "agent",
#             "multi-agent",
#             "integration",
#             "api",
#             "tool",
#             "automation",
#             "pipeline",
#             "orchestration",
#             "enterprise",
#             "team",
#             "teams",
#             "case-management",
#             "case management",
#             "scheduling",
#             "documentation",
#             "reports",
#             "reviews",
#             "coverage",
#         ]

#         def count_terms(terms: List[str]) -> int:
#             return sum(1 for term in terms if term in text_l)

#         marker_count = count_terms(complexity_terms)
#         evidence_count = count_terms(evidence_terms)
#         uncertainty_count = count_terms(uncertainty_terms)
#         workflow_count = count_terms(workflow_terms)

#         token_count = len(tokens)
#         length_score = min(1.0, token_count / 45.0)
#         complexity_marker_score = min(1.0, marker_count / 6.0)
#         evidence_need_score = min(1.0, evidence_count / 4.0)
#         uncertainty_score = min(1.0, uncertainty_count / 4.0)
#         workflow_score = min(1.0, workflow_count / 5.0)
#         structural_complexity = round(
#             100.0
#             * (
#                 0.30 * length_score
#                 + 0.25 * complexity_marker_score
#                 + 0.20 * evidence_need_score
#                 + 0.15 * uncertainty_score
#                 + 0.10 * workflow_score
#             ),
#             2,
#         )
#         uncertainty_need = round(
#             100.0
#             * (
#                 0.45 * uncertainty_score
#                 + 0.30 * evidence_need_score
#                 + 0.25 * workflow_score
#             ),
#             2,
#         )
#         return {
#             "token_count": token_count,
#             "marker_count": marker_count,
#             "evidence_count": evidence_count,
#             "uncertainty_count": uncertainty_count,
#             "workflow_count": workflow_count,
#             "length_score": round(length_score, 4),
#             "complexity_marker_score": round(complexity_marker_score, 4),
#             "evidence_need_score": round(evidence_need_score, 4),
#             "uncertainty_score": round(uncertainty_score, 4),
#             "workflow_score": round(workflow_score, 4),
#             "structural_complexity": structural_complexity,
#             "uncertainty_need": uncertainty_need,
#             "has_high_stakes": marker_count >= 4 or evidence_count >= 3,
#         }

#     @classmethod
#     def _score_modes(
#         cls,
#         *,
#         idea_text: str,
#         llm_decision: ControllerDecision,
#         budget_snapshot: Dict[str, Any],
#         validation_threshold: int,
#         max_validation_retries: int,
#     ) -> Dict[str, Any]:
#         """
#         Math-backed adaptive controller.

#         Utility(mode) = expected_quality(mode) - lambda_cost * normalized_cost(mode) * 100
#         where expected quality is estimated from structural complexity, uncertainty,
#         evidence need, and workflow coupling. Costs are deterministic mode priors
#         scaled by complexity and checked against the current budget.
#         """
#         raw_idea_text = ""
#         for line in str(idea_text or "").splitlines():
#             stripped = line.strip()
#             if stripped:
#                 raw_idea_text = stripped
#                 break
#         raw_features = cls._complexity_features(raw_idea_text or idea_text)
#         features = cls._complexity_features(idea_text)
#         complexity = max(
#             int(round(float(features["structural_complexity"]))),
#             int(llm_decision.estimated_complexity * 0.25),
#         )
#         complexity = max(0, min(100, complexity))

#         c = complexity / 100.0
#         u = float(features["uncertainty_need"]) / 100.0
#         e = float(features["evidence_need_score"])
#         w = float(features["workflow_score"])
#         complexity_marker = float(features["complexity_marker_score"])

#         mode_costs = {mode: cls._prior_cost_for_mode(mode, complexity) for mode in cls.MODE_ORDER}

#         max_cost_tokens = max(cost["token_proxy"] for cost in mode_costs.values()) or 1
#         max_cost_calls = max(cost["tool_calls"] for cost in mode_costs.values()) or 1
#         max_cost_runtime = max(cost["runtime_seconds"] for cost in mode_costs.values()) or 1.0

#         remaining_total = budget_snapshot.get("remaining_total_tokens")
#         remaining_proxy = budget_snapshot.get("remaining_token_proxy")
#         remaining_tokens = (
#             remaining_total if isinstance(remaining_total, int) else remaining_proxy
#         )
#         token_budget_pressure = 0.0
#         if isinstance(remaining_tokens, int) and remaining_tokens > 0:
#             token_budget_pressure = max(
#                 0.0,
#                 min(1.0, 1.0 - (remaining_tokens / max(remaining_tokens, max_cost_tokens * 2.5))),
#             )
#         elif isinstance(remaining_tokens, int) and remaining_tokens <= 0:
#             token_budget_pressure = 1.0

#         remaining_calls = budget_snapshot.get("remaining_tool_calls")
#         call_budget_pressure = 0.0
#         if isinstance(remaining_calls, int) and remaining_calls > 0:
#             call_budget_pressure = max(
#                 0.0,
#                 min(1.0, 1.0 - (remaining_calls / max(remaining_calls, max_cost_calls * 2.0))),
#             )
#         elif isinstance(remaining_calls, int) and remaining_calls <= 0:
#             call_budget_pressure = 1.0

#         budget_pressure = round(max(token_budget_pressure, call_budget_pressure), 4)
#         lambda_cost = round(0.12 + 0.28 * budget_pressure, 4)

#         # Expected reliability is deliberately quality-first when the experiment
#         # asks for high validation confidence. Direct mode is cheap, but prior
#         # runs showed it is fragile under evidence-grounded judge scoring.
#         expected_quality = {
#             "direct": (
#                 53.0
#                 + 7.0 * (1.0 - c)
#                 + 2.0 * (1.0 - u)
#                 - 7.0 * e
#                 - 4.0 * w
#                 - 3.0 * complexity_marker
#             ),
#             "shallow": 66.0 + 10.0 * c + 8.0 * e + 5.0 * w + 3.0 * u,
#             "recursive": (
#                 57.0
#                 + 11.0 * c
#                 + 8.0 * u
#                 + 6.0 * e
#                 + 3.0 * complexity_marker
#             ),
#         }

#         # Recursion only gives value if the graph is allowed to revise.
#         if max_validation_retries <= 0:
#             expected_quality["recursive"] -= 5.0
#         elif validation_threshold >= 75:
#             expected_quality["shallow"] += 2.0
#             expected_quality["recursive"] += 3.0
#             expected_quality["direct"] -= 6.0

#         raw_complexity = float(raw_features["structural_complexity"])
#         raw_uncertainty = float(raw_features["uncertainty_need"])
#         raw_evidence_count = int(raw_features["evidence_count"])
#         raw_workflow_count = int(raw_features["workflow_count"])
#         raw_marker_count = int(raw_features["marker_count"])
#         raw_token_count = int(raw_features["token_count"])

#         # Direct is eligible from the raw idea, not the expanded refined summary.
#         # Otherwise the refinement step itself can make simple ideas look too
#         # verbose/complex and direct is never tested.
#         direct_eligible = (
#             raw_complexity <= 22
#             and raw_uncertainty <= 12
#             and raw_evidence_count == 0
#             and raw_workflow_count <= 1
#             and raw_marker_count == 0
#             and raw_token_count <= 24
#             and complexity <= 45
#         ) or budget_pressure >= 0.72
#         if direct_eligible:
#             expected_quality["direct"] += 12.0
#             if validation_threshold >= 75:
#                 # Keep direct viable for simple tasks, but not unrealistically
#                 # dominant under a high evidence-grounding threshold.
#                 expected_quality["direct"] -= 2.0
#             if budget_pressure >= 0.50:
#                 expected_quality["direct"] += 2.0
#         elif not direct_eligible:
#             expected_quality["direct"] -= 12.0

#         recursive_upfront_allowed = (
#             complexity >= 90
#             or float(features["uncertainty_need"]) >= 85
#             or (
#                 bool(features["has_high_stakes"])
#                 and complexity >= 85
#                 and float(features["uncertainty_need"]) >= 78
#                 and e >= 0.85
#                 and w >= 0.75
#             )
#             or (
#                 complexity >= 88
#                 and float(features["uncertainty_need"]) >= 80
#                 and e >= 0.75
#             )
#         )

#         # Hard, uncertain, evidence-heavy tasks should be allowed to recurse when feasible.
#         if recursive_upfront_allowed:
#             expected_quality["recursive"] += 4.0
#         else:
#             # For non-hard cases, recursion should usually be a validation-driven
#             # escalation rather than the initial plan.
#             expected_quality["recursive"] -= 10.0
#         if complexity >= 45 and e >= 0.50:
#             expected_quality["shallow"] += 1.5
#             expected_quality["recursive"] += 1.0

#         expected_quality = {mode: max(0.0, min(100.0, value)) for mode, value in expected_quality.items()}

#         predicted_repair_gain = float(getattr(llm_decision, "predicted_repair_gain", 0.0) or 0.0)
#         predicted_repair_cost_tokens = int(getattr(llm_decision, "predicted_repair_cost_tokens", 500) or 500)
#         rem_total_tokens = budget_snapshot.get("remaining_total_tokens")
#         repair_affordable = rem_total_tokens is None or int(rem_total_tokens) >= predicted_repair_cost_tokens + 500
#         repair_roi_per_1k = predicted_repair_gain / max(1.0, predicted_repair_cost_tokens / 1000.0)
#         repair_roi_bonus_shallow = (
#             round(min(2.0, predicted_repair_gain * 0.15), 4)
#             if predicted_repair_gain > 0.0 and repair_affordable and repair_roi_per_1k >= 1.0
#             else 0.0
#         )

#         mode_scores: Dict[str, Dict[str, Any]] = {}
#         selected_mode = "direct"
#         selected_utility = -10_000.0
#         for mode in cls.MODE_ORDER:
#             cost = mode_costs[mode]
#             token_norm = cost["token_proxy"] / max_cost_tokens
#             call_norm = cost["tool_calls"] / max_cost_calls
#             runtime_norm = cost["runtime_seconds"] / max_cost_runtime
#             normalized_cost = round(
#                 0.60 * token_norm + 0.25 * call_norm + 0.15 * runtime_norm,
#                 4,
#             )
#             feasible = cls._is_mode_feasible(cost, budget_snapshot)
#             advisory_bonus = (
#                 round(1.5 * float(llm_decision.confidence), 4)
#                 if mode == llm_decision.mode
#                 else 0.0
#             )
#             roi_bonus = repair_roi_bonus_shallow if mode == "shallow" else 0.0
#             utility = (
#                 expected_quality[mode]
#                 - lambda_cost * normalized_cost * 100.0
#                 + advisory_bonus
#                 + roi_bonus
#             )
#             if not feasible:
#                 utility -= 100.0
#             mode_scores[mode] = {
#                 "expected_quality": round(expected_quality[mode], 4),
#                 "expected_cost": cost,
#                 "normalized_cost": normalized_cost,
#                 "advisory_bonus": advisory_bonus,
#                 "repair_roi_bonus": roi_bonus,
#                 "feasible": feasible,
#                 "utility": round(utility, 4),
#             }
#             if utility > selected_utility:
#                 selected_utility = utility
#                 selected_mode = mode

#         sorted_scores = sorted(
#             mode_scores.items(),
#             key=lambda kv: float(kv[1]["utility"]),
#             reverse=True,
#         )
#         utility_margin = 0.0
#         if len(sorted_scores) > 1:
#             utility_margin = round(
#                 float(sorted_scores[0][1]["utility"]) - float(sorted_scores[1][1]["utility"]),
#                 4,
#             )

#         policy_adjustments: List[str] = []
#         if selected_mode == "direct" and not direct_eligible:
#             non_direct = [
#                 item
#                 for item in sorted_scores
#                 if item[0] != "direct" and bool(item[1].get("feasible", False))
#             ]
#             if non_direct:
#                 selected_mode = non_direct[0][0]
#                 selected_utility = float(non_direct[0][1]["utility"])
#                 policy_adjustments.append("direct_not_eligible_promoted_to_non_direct")

#         direct_score = mode_scores["direct"]
#         recursive_score = mode_scores["recursive"]
#         shallow_score = mode_scores["shallow"]
#         direct_expected_quality = float(direct_score.get("expected_quality", 0.0) or 0.0)
#         shallow_expected_quality = float(shallow_score.get("expected_quality", 0.0) or 0.0)
#         direct_quality_floor = max(62.0, float(validation_threshold) - 5.0)
#         direct_quality_ready = direct_expected_quality >= direct_quality_floor
#         shallow_quality_advantage = shallow_expected_quality - direct_expected_quality
#         if (
#             selected_mode == "shallow"
#             and direct_eligible
#             and bool(direct_score.get("feasible", False))
#             and direct_quality_ready
#             and shallow_quality_advantage <= 4.0
#             and float(direct_score["utility"]) >= float(shallow_score["utility"]) - 3.5
#         ):
#             selected_mode = "direct"
#             selected_utility = float(direct_score["utility"])
#             policy_adjustments.append("validation_ready_close_call_prefers_direct")
#         elif (
#             selected_mode == "direct"
#             and (
#                 not direct_quality_ready
#                 or (
#                     bool(shallow_score.get("feasible", False))
#                     and shallow_quality_advantage > 4.0
#                     and budget_pressure < 0.65
#                 )
#             )
#         ):
#             non_direct = [
#                 item
#                 for item in sorted_scores
#                 if item[0] != "direct" and bool(item[1].get("feasible", False))
#             ]
#             if non_direct:
#                 selected_mode = non_direct[0][0]
#                 selected_utility = float(non_direct[0][1]["utility"])
#                 policy_adjustments.append("direct_deferred_until_validation_ready")

#         if (
#             selected_mode == "shallow"
#             and bool(recursive_score.get("feasible", False))
#             and max_validation_retries > 0
#             and validation_threshold >= 75
#             and budget_pressure <= 0.45
#             and recursive_upfront_allowed
#             and float(recursive_score["utility"]) > float(shallow_score["utility"]) + 5.0
#         ):
#             selected_mode = "recursive"
#             selected_utility = float(recursive_score["utility"])
#             policy_adjustments.append("recursive_margin_cleared_after_shallow_first_gate")
#         elif selected_mode == "recursive" and not recursive_upfront_allowed:
#             if bool(shallow_score.get("feasible", False)):
#                 selected_mode = "shallow"
#                 selected_utility = float(shallow_score["utility"])
#                 policy_adjustments.append("recursive_deferred_until_validation_failure")

#         if policy_adjustments:
#             best_alternative = max(
#                 (
#                     float(score["utility"])
#                     for mode, score in mode_scores.items()
#                     if mode != selected_mode
#                 ),
#                 default=float(selected_utility),
#             )
#             utility_margin = round(float(selected_utility) - best_alternative, 4)

#         formula = (
#             "utility(mode)=expected_quality(mode)-lambda_cost*normalized_cost(mode)*100"
#             "+llm_advisory_bonus; normalized_cost=0.60*token_norm+0.25*tool_call_norm"
#             "+0.15*runtime_norm"
#         )
#         return {
#             "selected_mode": selected_mode,
#             "selected_utility": round(selected_utility, 4),
#             "utility_margin": utility_margin,
#             "features": features,
#             "raw_features": raw_features,
#             "estimated_complexity": complexity,
#             "budget_pressure": budget_pressure,
#             "lambda_cost": lambda_cost,
#             "direct_eligible": direct_eligible,
#             "direct_quality_ready": direct_quality_ready,
#             "direct_quality_floor": round(direct_quality_floor, 4),
#             "shallow_quality_advantage": round(shallow_quality_advantage, 4),
#             "recursive_upfront_allowed": recursive_upfront_allowed,
#             "recursive_quality_advantage": round(float(mode_scores["recursive"].get("expected_quality", 0.0)) - float(mode_scores["shallow"].get("expected_quality", 0.0)), 4),
#             "recursive_marginal_quality_per_1k_token": round((float(mode_scores["recursive"].get("expected_quality", 0.0)) - float(mode_scores["shallow"].get("expected_quality", 0.0))) / max(1.0, (mode_costs["recursive"]["token_proxy"] - mode_costs["shallow"]["token_proxy"]) / 1000.0), 4),
#             "recursive_cost_efficient": bool((float(mode_scores["recursive"].get("expected_quality", 0.0)) - float(mode_scores["shallow"].get("expected_quality", 0.0))) >= 5.0),
#             "predicted_shallow_reliability": int(getattr(llm_decision, "predicted_shallow_reliability", 75) or 75),
#             "predicted_repair_gain": round(predicted_repair_gain, 4),
#             "predicted_repair_cost_tokens": predicted_repair_cost_tokens,
#             "repair_roi_per_1k_tokens": round(repair_roi_per_1k, 4),
#             "repair_roi_bonus_shallow": repair_roi_bonus_shallow,
#             "repair_affordable": repair_affordable,
#             "policy_adjustments": policy_adjustments,
#             "mode_scores": mode_scores,
#             "formula": formula,
#             "llm_advisory_mode": llm_decision.mode,
#             "llm_advisory_confidence": llm_decision.confidence,
#             "llm_advisory_complexity": llm_decision.estimated_complexity,
#         }

#     @staticmethod
#     def _build_decomposition_graph(mode: str) -> Dict[str, Any]:
#         """
#         Persist the decomposition object D=(V,E,tau,rho) used by the controller.
#         This is intentionally compact and inspectable for methodology reporting.
#         """
#         if mode == "direct":
#             nodes = [
#                 {
#                     "id": "direct_synthesis",
#                     "task": "Generate startup analysis in one integrated pass",
#                     "interface": "structured_llm_output",
#                     "executor": "direct_strategy_agent",
#                     "atomicity": "coarse",
#                 }
#             ]
#             edges: List[Dict[str, str]] = []
#         else:
#             nodes = [
#                 {
#                     "id": "idea_refinement",
#                     "task": "Refine raw idea into problem, solution, value proposition, and summary",
#                     "interface": "RefinedIdea schema",
#                     "executor": "idea_refinement_agent",
#                     "atomicity": "atomic",
#                 },
#                 {
#                     "id": "market_research",
#                     "task": "Retrieve market sources and synthesize market, competitors, and trends",
#                     "interface": "MarketOutput schema plus search/trends tools",
#                     "executor": "market_research_agent",
#                     "atomicity": "composite",
#                 },
#                 {
#                     "id": "source_validation",
#                     "task": "Evaluate evidence support and assign reliability score",
#                     "interface": "JudgeValidationOutput schema",
#                     "executor": "source_validator_agent",
#                     "atomicity": "atomic",
#                 },
#                 {
#                     "id": "business_model",
#                     "task": "Generate bounded financial assumptions and deterministic scenarios",
#                     "interface": "FinancialAssumptions schema plus calculator",
#                     "executor": "business_model_agent",
#                     "atomicity": "atomic",
#                 },
#                 {
#                     "id": "pitch_content",
#                     "task": "Create investor-ready pitch narrative and optional deck",
#                     "interface": "PitchSlides schema",
#                     "executor": "pitch_deck_generator_agent",
#                     "atomicity": "atomic",
#                 },
#             ]
#             edges = [
#                 {"from": "idea_refinement", "to": "market_research"},
#                 {"from": "market_research", "to": "source_validation"},
#                 {"from": "source_validation", "to": "business_model"},
#                 {"from": "business_model", "to": "pitch_content"},
#             ]
#             if mode == "recursive":
#                 nodes.append(
#                     {
#                         "id": "market_revision",
#                         "task": "Revise market research if validation score falls below threshold",
#                         "interface": "retry prompt with prior evidence gaps",
#                         "executor": "market_research_agent",
#                         "atomicity": "conditional",
#                     }
#                 )
#                 edges.extend(
#                     [
#                         {"from": "source_validation", "to": "market_revision"},
#                         {"from": "market_revision", "to": "source_validation"},
#                     ]
#                 )

#         depth = {"direct": 0, "shallow": 1, "recursive": 2}.get(mode, 1)
#         node_count = len(nodes)
#         atomic_nodes = sum(1 for node in nodes if node.get("atomicity") == "atomic")
#         atomicity_ratio = round(atomic_nodes / max(1, node_count), 4)
#         return {
#             "formalization": "D=(V,E,tau,rho), where V=subtasks, E=dependencies, tau=interfaces, rho=executors",
#             "mode": mode,
#             "depth_target": depth,
#             "nodes": nodes,
#             "edges": edges,
#             "metrics": {
#                 "node_count": node_count,
#                 "edge_count": len(edges),
#                 "atomicity_ratio": atomicity_ratio,
#                 "branching_factor_proxy": round(len(edges) / max(1, node_count), 4),
#             },
#         }

#     @classmethod
#     def _should_promote_recursive(
#         cls,
#         *,
#         decision: ControllerDecision,
#         combined_text: str,
#         validation_threshold: int,
#         max_validation_retries: int,
#         budget_snapshot: Dict[str, Any],
#     ) -> Dict[str, Any]:
#         # Deterministic promotion only in high-assurance setups.
#         if validation_threshold < 85 or max_validation_retries < 2:
#             return {"promote": False, "reason": None}

#         features = cls._complexity_features(combined_text)
#         complexity_signal = (
#             decision.estimated_complexity >= 40
#             or features["token_count"] >= 45
#             or features["marker_count"] >= 3
#             or features["has_high_stakes"]
#         )
#         confidence_signal = decision.confidence <= 0.86
#         rationale_l = (decision.rationale or "").lower()
#         uncertainty_signal = any(
#             token in rationale_l
#             for token in ["complex", "uncertain", "risk", "adversarial", "conflicting"]
#         )
#         if not complexity_signal:
#             return {"promote": False, "reason": None}
#         if not (confidence_signal or uncertainty_signal):
#             return {"promote": False, "reason": None}

#         recursive_cost = cls._prior_cost_for_mode("recursive", decision.estimated_complexity)
#         if not cls._is_mode_feasible(recursive_cost, budget_snapshot):
#             return {"promote": False, "reason": "recursive_not_budget_feasible"}

#         return {
#             "promote": True,
#             "reason": "deterministic_recursive_promotion_high_threshold",
#         }

#     @classmethod
#     def _is_mode_feasible(
#         cls,
#         mode_cost: Dict[str, Any],
#         budget_snapshot: Dict[str, Any],
#     ) -> bool:
#         rem_calls = budget_snapshot.get("remaining_tool_calls")
#         rem_proxy_tokens = budget_snapshot.get("remaining_token_proxy")
#         rem_total_tokens = budget_snapshot.get("remaining_total_tokens")
#         # Prefer true-token budget when available; otherwise fallback to proxy budget.
#         rem_tokens = rem_total_tokens if isinstance(rem_total_tokens, int) else rem_proxy_tokens
#         rem_runtime = budget_snapshot.get("remaining_runtime_seconds")
#         # Keep a risk buffer to reduce over-budget finishes caused by run-time variance.
#         calls_buffer = 0.9
#         tokens_buffer = 0.9
#         runtime_buffer = 0.8
#         if isinstance(rem_runtime, (int, float)):
#             if float(rem_runtime) <= 45.0:
#                 runtime_buffer = 0.7
#             elif float(rem_runtime) <= 60.0:
#                 runtime_buffer = 0.75

#         if isinstance(rem_calls, int) and mode_cost["tool_calls"] > max(0, int(rem_calls * calls_buffer)):
#             return False
#         if isinstance(rem_tokens, int) and mode_cost["token_proxy"] > max(0, int(rem_tokens * tokens_buffer)):
#             return False
#         if isinstance(rem_runtime, (int, float)) and mode_cost["runtime_seconds"] > max(0.0, float(rem_runtime) * runtime_buffer):
#             return False
#         return True

#     @classmethod
#     def _best_feasible_mode(
#         cls,
#         initial_mode: str,
#         budget_snapshot: Dict[str, Any],
#         complexity: int,
#     ) -> Dict[str, Any]:
#         costs_by_mode = {
#             mode: cls._prior_cost_for_mode(mode, complexity) for mode in cls.MODE_ORDER
#         }
#         initial_cost = costs_by_mode.get(initial_mode, costs_by_mode["shallow"])
#         if cls._is_mode_feasible(initial_cost, budget_snapshot):
#             return {
#                 "mode_final": initial_mode,
#                 "override": False,
#                 "override_reason": None,
#                 "costs_by_mode": costs_by_mode,
#             }

#         for mode in cls.MODE_QUALITY_PRIORITY:
#             if cls._is_mode_feasible(costs_by_mode[mode], budget_snapshot):
#                 return {
#                     "mode_final": mode,
#                     "override": mode != initial_mode,
#                     "override_reason": f"budget_feasible_mode_{mode}",
#                     "costs_by_mode": costs_by_mode,
#                 }
#         return {
#             "mode_final": "direct",
#             "override": initial_mode != "direct",
#             "override_reason": "no_mode_fits_budget_forced_direct",
#             "costs_by_mode": costs_by_mode,
#         }

#     def run(self, state: PitchState) -> PitchState:
#         idea = _sget(state, "idea", "")
#         refined = _sget(state, "refined_idea", "")
#         combined = f"{idea}\n{refined}".strip()
#         validation_threshold = int(_sget(state, "validation_threshold", 70) or 70)
#         max_validation_retries = int(_sget(state, "max_validation_retries", 1) or 1)
#         budget_snapshot = self._compute_remaining_budget(state)
#         usage = _empty_token_usage()
#         forced_mode = str(_sget(state, "forced_controller_mode", "") or "").strip().lower()
#         forced_mode_applied = forced_mode in self.MODE_ORDER
#         if forced_mode_applied:
#             fallback = self._fallback_decision(combined)
#             decision = ControllerDecision(
#                 mode=forced_mode,  # type: ignore[arg-type]
#                 confidence=1.0,
#                 rationale=f"Forced controller mode `{forced_mode}` for fixed-policy baseline.",
#                 estimated_complexity=fallback.estimated_complexity,
#                 expected_tool_calls_delta=fallback.expected_tool_calls_delta,
#                 expected_token_proxy_delta=fallback.expected_token_proxy_delta,
#                 expected_runtime_seconds_delta=fallback.expected_runtime_seconds_delta,
#                 triggers=["forced_mode_baseline"],
#                 predicted_shallow_reliability=fallback.predicted_shallow_reliability,
#                 predicted_repair_gain=fallback.predicted_repair_gain,
#                 predicted_repair_cost_tokens=fallback.predicted_repair_cost_tokens,
#             )
#         else:
#             try:
#                 decision, usage = _invoke_structured_with_usage(
#                     self.structured_llm,
#                     self.prompt.format_messages(
#                         idea=idea,
#                         refined_idea=refined,
#                         validation_threshold=validation_threshold,
#                         max_validation_retries=max_validation_retries,
#                         budget_snapshot=json.dumps(budget_snapshot, indent=2),
#                     ),
#                 )
#             except Exception:  # noqa: BLE001
#                 decision = self._fallback_decision(combined)
#         token_usage = _merge_token_usage(state, usage)

#         chosen_mode_initial = decision.mode
#         chosen_mode_final = decision.mode
#         budget_override = False
#         deterministic_trigger = None
#         guardrail_trigger = None
#         scorecard: Dict[str, Any] = {}

#         if forced_mode_applied:
#             chosen_mode_initial = forced_mode  # type: ignore[assignment]
#             chosen_mode_final = forced_mode  # type: ignore[assignment]
#             calibrated_selected = self._calibrate_selected_cost(
#                 decision,
#                 mode_override=chosen_mode_initial,
#             )
#             feasible_pick = {
#                 "mode_final": chosen_mode_final,
#                 "override": False,
#                 "override_reason": "forced_mode_no_override",
#                 "costs_by_mode": {
#                     mode: self._prior_cost_for_mode(mode, decision.estimated_complexity)
#                     for mode in self.MODE_ORDER
#                 },
#             }
#             scorecard = self._score_modes(
#                 idea_text=combined,
#                 llm_decision=decision,
#                 budget_snapshot=budget_snapshot,
#                 validation_threshold=validation_threshold,
#                 max_validation_retries=max_validation_retries,
#             )
#             scorecard["forced_mode_applied"] = True
#             scorecard["selected_mode_before_forcing"] = scorecard.get("selected_mode")
#             scorecard["selected_mode"] = chosen_mode_final
#         else:
#             scorecard = self._score_modes(
#                 idea_text=combined,
#                 llm_decision=decision,
#                 budget_snapshot=budget_snapshot,
#                 validation_threshold=validation_threshold,
#                 max_validation_retries=max_validation_retries,
#             )
#             chosen_mode_initial = scorecard["selected_mode"]
#             if chosen_mode_initial != decision.mode:
#                 deterministic_trigger = "utility_scorecard_selected_mode"

#             calibrated_selected = self._calibrate_selected_cost(
#                 decision,
#                 mode_override=chosen_mode_initial,
#             )
#             feasible_pick = self._best_feasible_mode(
#                 initial_mode=chosen_mode_initial,
#                 budget_snapshot=budget_snapshot,
#                 complexity=decision.estimated_complexity,
#             )
#             chosen_mode_final = feasible_pick["mode_final"]
#             budget_override = feasible_pick["override"]
#             budget_guard = self._budget_guardrail_mode(budget_snapshot)
#             if budget_override and feasible_pick["override_reason"]:
#                 guardrail_trigger = feasible_pick["override_reason"]
#             if budget_guard is not None and budget_guard != chosen_mode_final:
#                 chosen_mode_final = budget_guard
#                 budget_override = True
#                 guardrail_trigger = f"budget_guardrail_forced_{budget_guard}"

#         depth_map = {"direct": 0, "shallow": 1, "recursive": 2}
#         decisions = list(_sget(state, "controller_decisions", []))
#         tool_audit = list(_sget(state, "tool_audit", []))
#         trigger_chain = list(decision.triggers)
#         if deterministic_trigger:
#             trigger_chain.append(deterministic_trigger)
#         if guardrail_trigger:
#             trigger_chain.append(guardrail_trigger)
#         feature_summary = scorecard.get("features", {}) if isinstance(scorecard, dict) else {}
#         decomposition_graph = self._build_decomposition_graph(chosen_mode_final)
#         decisions.append(
#             {
#                 "mode_initial": chosen_mode_initial,
#                 "mode_final": chosen_mode_final,
#                 "confidence": decision.confidence,
#                 "estimated_complexity": scorecard.get(
#                     "estimated_complexity",
#                     decision.estimated_complexity,
#                 ),
#                 "llm_advisory_mode": decision.mode,
#                 "llm_advisory_complexity": decision.estimated_complexity,
#                 "structural_complexity": feature_summary.get("structural_complexity"),
#                 "uncertainty_need": feature_summary.get("uncertainty_need"),
#                 "utility_margin": scorecard.get("utility_margin"),
#                 "triggers": trigger_chain,
#                 "rationale": decision.rationale,
#                 "expected_tool_calls_delta": calibrated_selected["tool_calls"],
#                 "expected_token_proxy_delta": calibrated_selected["token_proxy"],
#                 "expected_runtime_seconds_delta": calibrated_selected["runtime_seconds"],
#                 "costs_by_mode": feasible_pick["costs_by_mode"],
#                 "budget_override": budget_override,
#                 "deterministic_recursive_promotion": bool(deterministic_trigger),
#                 "forced_mode_applied": forced_mode_applied,
#             }
#         )
#         tool_audit.append(
#             {
#                 "agent": "adaptive_controller",
#                 "tool": "mode_selector",
#                 "status": "ok",
#                 "mode_initial": chosen_mode_initial,
#                 "mode_final": chosen_mode_final,
#                 "confidence": decision.confidence,
#                 "estimated_complexity": scorecard.get(
#                     "estimated_complexity",
#                     decision.estimated_complexity,
#                 ),
#                 "llm_advisory_mode": decision.mode,
#                 "llm_advisory_complexity": decision.estimated_complexity,
#                 "structural_complexity": feature_summary.get("structural_complexity"),
#                 "uncertainty_need": feature_summary.get("uncertainty_need"),
#                 "utility_margin": scorecard.get("utility_margin"),
#                 "triggers": trigger_chain,
#                 "expected_tool_calls_delta": calibrated_selected["tool_calls"],
#                 "expected_token_proxy_delta": calibrated_selected["token_proxy"],
#                 "expected_runtime_seconds_delta": calibrated_selected["runtime_seconds"],
#                 "costs_by_mode": feasible_pick["costs_by_mode"],
#                 "budget_override": budget_override,
#                 "deterministic_recursive_promotion": bool(deterministic_trigger),
#                 "forced_mode_applied": forced_mode_applied,
#             }
#         )
#         return {
#             "controller_mode": chosen_mode_final,
#             "controller_mode_initial": chosen_mode_initial,
#             "controller_budget_override": budget_override,
#             "controller_confidence": decision.confidence,
#             "controller_rationale": decision.rationale,
#             "controller_expected_cost": {
#                 "expected_tool_calls_delta": calibrated_selected["tool_calls"],
#                 "expected_token_proxy_delta": calibrated_selected["token_proxy"],
#                 "expected_runtime_seconds_delta": calibrated_selected["runtime_seconds"],
#                 "raw_model_estimate": calibrated_selected["raw"],
#                 "prior_for_mode": calibrated_selected["prior"],
#                 "costs_by_mode": feasible_pick["costs_by_mode"],
#                 "mode_for_estimate": chosen_mode_initial,
#             },
#             "controller_budget_snapshot": budget_snapshot,
#             "controller_scorecard": scorecard,
#             "controller_decisions": decisions,
#             "decomposition_graph": decomposition_graph,
#             "decomposition_depth_target": depth_map.get(chosen_mode_final, 1),
#             "tool_audit": tool_audit,
#             "token_usage": token_usage,
#         }


# class MarketResearchAgent:
#     def __init__(
#         self,
#         llm: ChatOpenAI,
#         strict_tools: bool = True,
#         enable_trends: bool = True,
#     ):
#         self.structured_llm = llm.with_structured_output(MarketOutput, include_raw=True)
#         self.keyword_llm = llm.with_structured_output(TrendKeywords, include_raw=True)
#         self.search_tool = MarketSearchTool()
#         self.trends_tool = GoogleTrendsTool() if enable_trends else None
#         self.strict_tools = strict_tools
#         self.enable_trends = enable_trends
#         self.keyword_prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "# Role\n"
#                     "You are a market keyword extractor.\n\n"
#                     "# Instructions\n"
#                     "- Extract domain-agnostic trend keywords for startup market research.\n"
#                     "- Focus on product type, buyer segment, core technology, problem space, and industry context.\n"
#                     "- Keep keywords short and directly searchable.\n\n"
#                     "# Output Format\n"
#                     "- Return content that strictly matches the structured schema fields.",
#                 ),
#                 (
#                     "human",
#                     "Startup concept:\n{refined_idea}\n\n"
#                     "Return keyword phrases covering product type, buyer segment, core technology, "
#                     "problem space, and industry context.",
#                 ),
#             ]
#         )
#         self.prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "# Role\n"
#                     "You are a market research analyst.\n\n"
#                     "# Instructions\n"
#                     "- Use the provided web research snippets and trend signals.\n"
#                     "- Synthesize target market, market size, trends, competitors, and differentiation gaps.\n"
#                     "- Prioritize evidence-grounded statements and avoid speculation.\n\n"
#                     "# Output Format\n"
#                     "- Return content that strictly matches the structured schema fields.",
#                 ),
#                 (
#                     "human",
#                     "Startup concept:\n{refined_idea}\n\n"
#                     "Controller mode: {controller_mode}\n"
#                     "Retry count: {retry_count}\n"
#                     "Previous validation gaps, if any:\n{validation_gaps}\n\n"
#                     "Web research snippets:\n{search_results}\n\n"
#                     "Keyword trend signals:\n{trend_signals}\n\n"
#                     "Produce market analysis for investors. If this is a retry, repair the previous "
#                     "evidence gaps directly and avoid claims that are not supported by the snippets.",
#                 ),
#             ]
#         )

#     def run(self, state: PitchState) -> PitchState:
#         refined_idea = _sget(state, "refined_idea", "")
#         total_usage = _empty_token_usage()
#         retry_count = _sget(state, "retry_count", 0)
#         controller_mode = str(_sget(state, "controller_mode", "shallow") or "shallow")
#         prior_validation = _sget(state, "validation_report", {}) or {}
#         validation_gaps = ""
#         if isinstance(prior_validation, dict):
#             validation_gaps = str(prior_validation.get("evidence_gaps", "") or "")
#             weak_claims = [
#                 str(claim.get("claim", "")).strip()
#                 for claim in prior_validation.get("claims", []) or []
#                 if str(claim.get("verdict", "")).strip().lower()
#                 in {"weakly_supported", "unsupported"}
#             ]
#             if weak_claims:
#                 validation_gaps = (
#                     f"{validation_gaps}\nWeak or unsupported claims to repair:\n"
#                     + "\n".join(f"- {claim}" for claim in weak_claims[:5])
#                 ).strip()
#         query_suffix = (
#             " prioritize authoritative and recent sources with concrete numbers"
#             if retry_count > 0
#             else ""
#         )
#         if retry_count > 0 and validation_gaps:
#             query_suffix += f" evidence gaps to resolve: {validation_gaps[:240]}"
#         query = f"startup market size competitors trends for: {refined_idea}{query_suffix}"
#         search_payload = self.search_tool.search(query)
#         if self.strict_tools and search_payload["status"] != "ok":
#             raise RuntimeError(f"Market search tool unavailable: {search_payload['error']}")
#         evidence_budget = _evidence_budget_for_mode(
#             _sget(state, "controller_mode", "shallow"),
#             _sget(state, "budget_remaining", {}).get("total_tokens")
#             if isinstance(_sget(state, "budget_remaining", {}), dict)
#             else None,
#         )
#         search_payload = _limited_search_payload(search_payload, **evidence_budget)
#         search_results = search_payload["results_json"]
#         try:
#             kw_model, kw_usage = _invoke_structured_with_usage(
#                 self.keyword_llm,
#                 self.keyword_prompt.format_messages(refined_idea=refined_idea),
#             )
#             total_usage = _merge_token_usage({"token_usage": total_usage}, kw_usage)
#             extracted_keywords = [k.strip() for k in kw_model.keywords if k.strip()]
#         except Exception:  # noqa: BLE001
#             extracted_keywords = _fallback_keywords(refined_idea)
#         if self.enable_trends and self.trends_tool is not None:
#             trend_payload = self.trends_tool.fetch(extracted_keywords[:5])
#         else:
#             trend_payload = {
#                 "status": "skipped",
#                 "keywords": extracted_keywords[:5],
#                 "data": {},
#                 "error": "Google Trends disabled by configuration.",
#             }

#         result, market_usage = _invoke_structured_with_usage(
#             self.structured_llm,
#             self.prompt.format_messages(
#                 refined_idea=refined_idea,
#                 controller_mode=controller_mode,
#                 retry_count=retry_count,
#                 validation_gaps=validation_gaps or "None.",
#                 search_results=search_results,
#                 trend_signals=json.dumps(trend_payload, indent=2),
#             ),
#         )
#         total_usage = _merge_token_usage({"token_usage": total_usage}, market_usage)
#         token_usage = _merge_token_usage(state, total_usage)
#         market_analysis = (
#             f"Target Market: {result.target_market}\n"
#             f"Market Size: {result.market_size}\n"
#             f"Trends: {result.trends}\n"
#             f"Competitors: {result.competitors}\n"
#             f"Differentiation Gaps: {result.differentiation_gaps}"
#         )
#         tool_audit = list(_sget(state, "tool_audit", []))
#         tool_audit.append(
#             {
#                 "agent": "market_research",
#                 "tool": "linkup_search",
#                 "query": query,
#                 "status": search_payload["status"],
#                 "source_count": len(search_payload["sources"]),
#                 "source_count_raw": search_payload.get("source_count_raw"),
#                 "evidence_limit": search_payload.get("evidence_limit"),
#                 "dropped_sources": search_payload.get("dropped_sources", 0),
#                 "error": search_payload["error"],
#             }
#         )
#         tool_audit.append(
#             {
#                 "agent": "market_research",
#                 "tool": "google_trends",
#                 "status": trend_payload["status"],
#                 "keyword_count": len(trend_payload.get("keywords", [])),
#                 "keywords": trend_payload.get("keywords", []),
#                 "error": trend_payload.get("error", ""),
#             }
#         )
#         return {
#             "market_analysis": market_analysis,
#             "market_sources": search_payload["sources"],
#             "market_evidence": search_payload["results"],
#             "trend_signals": trend_payload,
#             "tool_audit": tool_audit,
#             "token_usage": token_usage,
#         }


# class ClaimRepairAgent:
#     """Targeted claim-level market analysis repair agent."""

#     def __init__(self, llm: ChatOpenAI, strict_tools: bool = True):
#         self.structured_llm = llm.with_structured_output(MarketRepairOutput, include_raw=True)
#         self.search_tool = MarketSearchTool()
#         self.strict_tools = strict_tools
#         self.prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "# Role\n"
#                     "You are a conservative claim-level market repair agent.\n\n"
#                     "# Instructions\n"
#                     "- Do not rewrite the whole market analysis.\n"
#                     "- Preserve supported claims unless directly contradicted.\n"
#                     "- Repair only the listed weak/unsupported claims.\n"
#                     "- If evidence is weak, qualify or remove the claim instead of making it sound stronger.\n"
#                     "- Do not add new market-size, CAGR, adoption-rate, or competitor claims unless directly supported by focused snippets.\n"
#                     "- Keep the output concise to control token cost.\n\n"
#                     "# Output Format\n"
#                     "- Return content that strictly matches the structured schema fields.",
#                 ),
#                 (
#                     "human",
#                     "Startup concept:\n{refined_idea}\n\n"
#                     "Previous market analysis:\n{previous_market_analysis}\n\n"
#                     "Supported claims to preserve:\n{supported_claims}\n\n"
#                     "Repair plan:\n{repair_plan}\n\n"
#                     "Focused evidence snippets:\n{focused_evidence}\n\n"
#                     "Return a patched market analysis plus explicit patches.",
#                 ),
#             ]
#         )

#     @staticmethod
#     def _queries_for_plan(plan: List[Dict[str, Any]]) -> List[str]:
#         queries: List[str] = []
#         for item in plan:
#             if item.get("action") != "search_and_replace":
#                 continue
#             candidates = item.get("candidate_queries") or []
#             if candidates:
#                 q = str(candidates[0]).strip()
#             else:
#                 q = f"{item.get('claim', '')} evidence source"
#             if q and q not in queries:
#                 queries.append(q)
#             if len(queries) >= 2:
#                 break
#         return queries

#     def run(self, state: PitchState) -> PitchState:
#         repair_context = _sget(state, "market_repair_context", {}) or {}
#         plan = repair_context.get("repair_plan") or build_claim_repair_plan(
#             repair_context.get("weak_or_unsupported_claims", []) or []
#         )
#         plan = list(plan or [])[:3]
#         queries = self._queries_for_plan(plan)
#         focused_payload = {
#             "status": "skipped",
#             "results": [],
#             "sources": [],
#             "results_json": "[]",
#             "error": "No search-based repair actions requested.",
#         }
#         if queries:
#             focused_payload = self.search_tool.search_many(queries[:2], max_results_per_query=2)
#             if self.strict_tools and focused_payload.get("status") != "ok":
#                 raise RuntimeError(f"Focused claim repair search failed: {focused_payload.get('error', '')}")
#         focused_payload = _limited_search_payload(focused_payload, max_items=4, content_chars=220)
#         previous_market_evidence = _sget(state, "market_evidence", []) or []
#         merged_evidence = _merge_evidence_items(previous_market_evidence, focused_payload.get("results", []))
#         result, usage = _invoke_structured_with_usage(
#             self.structured_llm,
#             self.prompt.format_messages(
#                 refined_idea=_sget(state, "refined_idea", ""),
#                 previous_market_analysis=repair_context.get("previous_market_analysis") or _sget(state, "market_analysis", ""),
#                 supported_claims=json.dumps(repair_context.get("supported_claims", [])[:6], indent=2),
#                 repair_plan=json.dumps(plan, indent=2),
#                 focused_evidence=focused_payload.get("results_json", "[]"),
#             ),
#         )
#         token_usage = _merge_token_usage(state, usage)
#         market_analysis = (
#             f"Target Market: {result.target_market}\n"
#             f"Market Size: {result.market_size}\n"
#             f"Trends: {result.trends}\n"
#             f"Competitors: {result.competitors}\n"
#             f"Differentiation Gaps: {result.differentiation_gaps}\n"
#             f"Patch Summary: {result.patch_summary}"
#         )
#         repair_patches = json.loads(json.dumps([p.model_dump() for p in result.patches]))
#         tool_audit = list(_sget(state, "tool_audit", []))
#         if queries:
#             tool_audit.append({
#                 "agent": "claim_repair",
#                 "tool": "focused_repair_search",
#                 "status": focused_payload.get("status", "unknown"),
#                 "query": " | ".join(queries),
#                 "source_count": len(focused_payload.get("sources", [])),
#                 "error": focused_payload.get("error", ""),
#             })
#         action_counts = {
#             "search_and_replace": sum(1 for item in plan if item.get("action") == "search_and_replace"),
#             "qualify_or_remove": sum(1 for item in plan if item.get("action") == "qualify_or_remove"),
#             "remove": sum(1 for item in plan if item.get("action") == "remove"),
#         }
#         tool_audit.append({
#             "agent": "claim_repair",
#             "tool": "claim_micro_repair",
#             "status": "ok",
#             "claim_count": len(plan),
#             "source_count": len(focused_payload.get("sources", [])),
#             "repair_action_counts": action_counts,
#             "prompt_tokens": usage.get("prompt_tokens", 0),
#             "completion_tokens": usage.get("completion_tokens", 0),
#             "total_tokens": usage.get("total_tokens", 0),
#         })
#         return {
#             "market_analysis": market_analysis,
#             "market_sources": _merge_unique(_sget(state, "market_sources", []) or [], focused_payload.get("sources", []) or []),
#             "market_evidence": merged_evidence,
#             "repair_plan": plan,
#             "repair_patches": repair_patches,
#             "tool_audit": tool_audit,
#             "token_usage": token_usage,
#         }


# def _merge_unique(a: List[str], b: List[str]) -> List[str]:
#     out: List[str] = []
#     seen = set()
#     for x in list(a or []) + list(b or []):
#         sx = str(x or "").strip()
#         if sx and sx not in seen:
#             seen.add(sx)
#             out.append(sx)
#     return out


# class SourceValidatorAgent:
#     def __init__(self, llm: ChatOpenAI, secondary_judge_llm: ChatOpenAI | None = None):
#         self.primary_judge_llm = llm.with_structured_output(
#             JudgeValidationOutput, include_raw=True
#         )
#         self.secondary_judge_llm = (secondary_judge_llm or llm).with_structured_output(
#             SecondJudgeOutput, include_raw=True
#         )
#         self.repair_judge_llm = llm.with_structured_output(
#             RepairPatchValidationOutput, include_raw=True
#         )
#         self.primary_prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "# Role\n"
#                     "You are Judge A, a strict source validation analyst.\n\n"
#                     "# Instructions\n"
#                     "- Verify material market claims only against provided evidence snippets and URLs.\n"
#                     "- Do not invent sources or unsupported claims.\n"
#                     "- Provide claim-level verdicts, confidence, rationale, and supporting sources.\n"
#                     "- Score rubric dimensions (0-100): evidence grounding, source credibility, claim specificity, internal consistency.\n"
#                     "- Provide overall reliability (0-100).\n\n"
#                     "# Output Format\n"
#                     "- Return content that strictly matches the structured schema fields.",
#                 ),
#                 (
#                     "human",
#                     "Startup concept:\n{refined_idea}\n\n"
#                     "Market analysis draft:\n{market_analysis}\n\n"
#                     "Evidence snippets (title/url/content):\n{market_evidence}\n\n"
#                     "Return a claim-by-claim validation report with confidence and source links.",
#                 ),
#             ]
#         )
#         self.secondary_prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "# Role\n"
#                     "You are Judge B, an independent methodology-focused reviewer.\n\n"
#                     "# Instructions\n"
#                     "- Independently evaluate each provided claim against the same evidence.\n"
#                     "- Do not copy Judge A labels blindly; reassess verdict and confidence per claim.\n"
#                     "- Score rubric dimensions (0-100): evidence grounding, source credibility, claim specificity, internal consistency.\n"
#                     "- Provide overall reliability (0-100) and evidence gaps.\n\n"
#                     "# Output Format\n"
#                     "- Return content that strictly matches the structured schema fields.",
#                 ),
#                 (
#                     "human",
#                     "Startup concept:\n{refined_idea}\n\n"
#                     "Market analysis draft:\n{market_analysis}\n\n"
#                     "Evidence snippets (title/url/content):\n{market_evidence}\n\n"
#                     "Claims to evaluate:\n{claims_to_review}\n\n"
#                     "Return independent claim assessments and rubric scores.",
#                 ),
#             ]
#         )


#         self.repair_prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "# Role\n"
#                     "You are a lightweight patch validator.\n\n"
#                     "# Instructions\n"
#                     "- Validate only the repaired claims against the provided focused evidence.\n"
#                     "- Accept a patch only if it does not reduce supported-ratio, does not delete material facts to hide weakness, and improves or conservatively preserves reliability.\n"
#                     "- Be stricter with numeric, market-size, competitor, and healthcare/regulatory claims.\n"
#                     "- Return whether the patch can be accepted and explain why.\n\n"
#                     "# Output Format\n"
#                     "- Return content that strictly matches the structured schema fields.",
#                 ),
#                 (
#                     "human",
#                     "Startup concept:\n{refined_idea}\n\n"
#                     "Previous market analysis:\n{previous_market_analysis}\n\n"
#                     "Candidate repaired market analysis:\n{candidate_market_analysis}\n\n"
#                     "Claims repaired:\n{repaired_claims}\n\n"
#                     "Focused evidence snippets:\n{focused_evidence}\n\n"
#                     "Previous validation score: {previous_score}\n"
#                     "Previous supported ratio: {previous_supported_ratio}\n\n"
#                     "Validate the patch only.",
#                 ),
#             ]
#         )

#     @staticmethod
#     def _rubric_mean(rubric: Dict[str, Any]) -> float:
#         keys = [
#             "evidence_grounding",
#             "source_credibility",
#             "claim_specificity",
#             "internal_consistency",
#         ]
#         vals: List[float] = []
#         for key in keys:
#             try:
#                 vals.append(float(rubric.get(key, 0)))
#             except (TypeError, ValueError):
#                 vals.append(0.0)
#         if not vals:
#             return 0.0
#         return sum(vals) / len(vals)

#     @staticmethod
#     def _clamp_score(value: float) -> int:
#         return int(max(0, min(100, round(value))))

#     @staticmethod
#     def _judge_summary(
#         report: Dict[str, Any],
#         det_score: int,
#         usage: Dict[str, int],
#     ) -> Dict[str, Any]:
#         return {
#             "overall_reliability": int(report.get("overall_reliability", 0) or 0),
#             "deterministic_claim_score": int(det_score),
#             "rubric_scores": dict(report.get("rubric_scores", {}) or {}),
#             "rubric_mean": round(
#                 SourceValidatorAgent._rubric_mean(report.get("rubric_scores", {}) or {}),
#                 3,
#             ),
#             "claim_count": len(report.get("claims", []) or report.get("claim_assessments", [])),
#             "token_usage": usage,
#         }

#     @staticmethod
#     def _agreement_stats(
#         primary_claims: List[Dict[str, Any]],
#         secondary_claims: List[Dict[str, Any]],
#         rubric_a: Dict[str, Any],
#         rubric_b: Dict[str, Any],
#         score_a: int,
#         score_b: int,
#     ) -> Dict[str, Any]:
#         rubric_dims = [
#             "evidence_grounding",
#             "source_credibility",
#             "claim_specificity",
#             "internal_consistency",
#         ]
#         rubric_diffs: Dict[str, float] = {}
#         for dim in rubric_dims:
#             va = float(rubric_a.get(dim, 0) or 0)
#             vb = float(rubric_b.get(dim, 0) or 0)
#             rubric_diffs[dim] = round(abs(va - vb), 3)

#         primary_map = {
#             str(c.get("claim", "")).strip().lower(): str(c.get("verdict", "")).strip().lower()
#             for c in primary_claims
#             if str(c.get("claim", "")).strip()
#         }
#         secondary_map = {
#             str(c.get("claim", "")).strip().lower(): str(c.get("verdict", "")).strip().lower()
#             for c in secondary_claims
#             if str(c.get("claim", "")).strip()
#         }
#         shared = sorted(set(primary_map.keys()) & set(secondary_map.keys()))
#         verdict_matches = 0
#         for key in shared:
#             if primary_map.get(key) == secondary_map.get(key):
#                 verdict_matches += 1
#         verdict_agreement = (
#             round(verdict_matches / len(shared), 4) if shared else 0.0
#         )

#         score_delta = abs(int(score_a) - int(score_b))
#         score_agreement = round(max(0.0, 1.0 - (score_delta / 100.0)), 4)
#         rubric_mae = (
#             round(sum(rubric_diffs.values()) / max(1, len(rubric_diffs)), 4)
#             if rubric_diffs
#             else 0.0
#         )
#         rubric_agreement = round(max(0.0, 1.0 - (rubric_mae / 100.0)), 4)
#         overall_agreement = round(
#             (score_agreement + rubric_agreement + verdict_agreement) / 3.0, 4
#         )
#         return {
#             "score_delta_abs": score_delta,
#             "score_agreement": score_agreement,
#             "rubric_mae": rubric_mae,
#             "rubric_diffs": rubric_diffs,
#             "rubric_agreement": rubric_agreement,
#             "shared_claims": len(shared),
#             "verdict_agreement_rate": verdict_agreement,
#             "overall_agreement": overall_agreement,
#         }


#     @staticmethod
#     def _claim_ledger(
#         claims: List[Dict[str, Any]],
#         evidence: List[Dict[str, Any]],
#         evidence_gaps: str = "",
#     ) -> List[Dict[str, Any]]:
#         ledger: List[Dict[str, Any]] = []
#         evidence_domains = []
#         for item in evidence or []:
#             if isinstance(item, dict) and item.get("domain"):
#                 evidence_domains.append(str(item.get("domain")))
#         for idx, claim in enumerate(claims or [], start=1):
#             if not isinstance(claim, dict):
#                 continue
#             text = str(claim.get("claim", "") or "").strip()
#             if not text:
#                 continue
#             category = _claim_repair_category(text)
#             verdict = str(claim.get("verdict", "needs_review") or "needs_review").strip().lower()
#             item = dict(claim)
#             item["claim_id"] = str(claim.get("claim_id") or f"c{idx}")
#             item["claim_text"] = text
#             item["category"] = category
#             item["materiality"] = int(claim.get("materiality") or _claim_materiality(text, category))
#             item["failure_type"] = str(claim.get("failure_type") or _claim_failure_type(claim, evidence_gaps))
#             item["source_diversity_count"] = len(set(evidence_domains))
#             item["distinct_source_domains"] = sorted(set(evidence_domains))[:6]
#             item["regulatory_flag"] = any(
#                 m in text.lower()
#                 for m in ["clinical", "healthcare", "patient", "regulatory", "compliance", "hipaa"]
#             )
#             if verdict != "supported" and not item.get("candidate_queries"):
#                 item["candidate_queries"] = [
#                     f"{text} evidence source",
#                     f"{category.replace('_', ' ')} market evidence",
#                 ]
#             ledger.append(item)
#         return ledger

#     def run(self, state: PitchState) -> PitchState:
#         token_usage_current = _sget(state, "token_usage", {}) or {}
#         max_total_tokens = _sget(state, "max_total_tokens")
#         total_tokens_current = int(token_usage_current.get("total_tokens", 0) or 0)
#         remaining_total_tokens = (
#             int(max_total_tokens) - total_tokens_current
#             if isinstance(max_total_tokens, int)
#             else None
#         )
#         evidence_budget = _evidence_budget_for_mode(
#             _sget(state, "controller_mode", "shallow"),
#             remaining_total_tokens,
#         )
#         raw_evidence = _sget(state, "market_evidence", []) or []
#         validation_evidence = _limit_evidence_items(raw_evidence, **evidence_budget)
#         evidence_json = json.dumps(validation_evidence, indent=2)
#         primary_result, usage_primary = _invoke_structured_with_usage(
#             self.primary_judge_llm,
#             self.primary_prompt.format_messages(
#                 refined_idea=_sget(state, "refined_idea", ""),
#                 market_analysis=_sget(state, "market_analysis", ""),
#                 market_evidence=evidence_json,
#             ),
#         )
#         primary_report: Dict[str, Any] = json.loads(primary_result.model_dump_json())
#         primary_claims = primary_report.get("claims", [])
#         primary_det_score = _deterministic_reliability_score(primary_claims)
#         claims_for_secondary = json.dumps(
#             [{"claim": c.get("claim", "")} for c in primary_claims], indent=2
#         )

#         secondary_failed = False
#         usage_secondary = _empty_token_usage()
#         secondary_report: Dict[str, Any] = {}
#         try:
#             secondary_result, usage_secondary = _invoke_structured_with_usage(
#                 self.secondary_judge_llm,
#                 self.secondary_prompt.format_messages(
#                     refined_idea=_sget(state, "refined_idea", ""),
#                     market_analysis=_sget(state, "market_analysis", ""),
#                     market_evidence=evidence_json,
#                     claims_to_review=claims_for_secondary,
#                 ),
#             )
#             secondary_report = json.loads(secondary_result.model_dump_json())
#         except Exception:  # noqa: BLE001
#             secondary_failed = True
#             # Fallback to primary output so pipeline remains robust.
#             secondary_report = {
#                 "evidence_gaps": primary_report.get("evidence_gaps", ""),
#                 "overall_reliability": int(primary_report.get("overall_reliability", 0) or 0),
#                 "rubric_scores": dict(primary_report.get("rubric_scores", {}) or {}),
#                 "claim_assessments": list(primary_claims),
#             }

#         secondary_claims = secondary_report.get("claim_assessments", [])
#         secondary_det_score = _deterministic_reliability_score(secondary_claims)
#         score_a = int(primary_report.get("overall_reliability", 0) or 0)
#         score_b = int(secondary_report.get("overall_reliability", 0) or 0)
#         rubric_a = dict(primary_report.get("rubric_scores", {}) or {})
#         rubric_b = dict(secondary_report.get("rubric_scores", {}) or {})

#         agreement = self._agreement_stats(
#             primary_claims=primary_claims,
#             secondary_claims=secondary_claims,
#             rubric_a=rubric_a,
#             rubric_b=rubric_b,
#             score_a=score_a,
#             score_b=score_b,
#         )
#         rubric_mean_a = self._rubric_mean(rubric_a)
#         rubric_mean_b = self._rubric_mean(rubric_b)
#         blended_score = (
#             0.35 * ((score_a + score_b) / 2.0)
#             + 0.35 * ((primary_det_score + secondary_det_score) / 2.0)
#             + 0.30 * ((rubric_mean_a + rubric_mean_b) / 2.0)
#         )
#         final_score = self._clamp_score(blended_score)

#         claim_units = self._claim_ledger(
#             primary_claims,
#             validation_evidence,
#             primary_report.get("evidence_gaps", ""),
#         )
#         failing_claims = [
#             c for c in claim_units
#             if str(c.get("verdict", "")).strip().lower() in {"weakly_supported", "unsupported", "needs_review"}
#         ]
#         unsupported_material_claim_count = sum(
#             1 for c in failing_claims
#             if str(c.get("verdict", "")).strip().lower() == "unsupported"
#             and int(c.get("materiality", 1) or 1) >= 4
#         )

#         # Use primary claim list as canonical for downstream compatibility.
#         report: Dict[str, Any] = {
#             "validated_summary": primary_report.get("validated_summary", ""),
#             "evidence_gaps": primary_report.get("evidence_gaps", ""),
#             "claims": primary_claims,
#             "claim_units": claim_units,
#             "failing_claims": failing_claims,
#             "unsupported_material_claim_count": unsupported_material_claim_count,
#             "reliability_score": final_score,
#             "judge_scores": {
#                 "judge_a": self._judge_summary(primary_report, primary_det_score, usage_primary),
#                 "judge_b": self._judge_summary(
#                     {
#                         "overall_reliability": score_b,
#                         "rubric_scores": rubric_b,
#                         "claim_assessments": secondary_claims,
#                     },
#                     secondary_det_score,
#                     usage_secondary,
#                 ),
#                 "aggregated": {
#                     "final_reliability_score": final_score,
#                     "final_formula": (
#                         "0.35*avg(judge_overall) + 0.35*avg(deterministic_claim_score) + "
#                         "0.30*avg(rubric_mean)"
#                     ),
#                 },
#             },
#             "agreement_stats": agreement,
#             "evaluation_primary": {
#                 "used_for_decision": True,
#                 "primary_reliability_score": final_score,
#                 "primary_judge_agreement": agreement.get("overall_agreement"),
#                 "decision_rule": (
#                     "needs_revision = primary_reliability_score < validation_threshold"
#                 ),
#             },
#         }
#         total_usage = _merge_token_usage({"token_usage": usage_primary}, usage_secondary)
#         token_usage = _merge_token_usage(state, total_usage)
#         validated = (
#             f"{_sget(state, 'market_analysis', '')}\n\n"
#             f"Validation Score: {final_score}/100\n"
#             f"Validated Summary: {report.get('validated_summary', '')}\n"
#             f"Evidence Gaps: {report.get('evidence_gaps', '')}\n"
#             f"Judge Agreement: {agreement.get('overall_agreement')}"
#         )
#         tool_audit = list(_sget(state, "tool_audit", []))
#         tool_audit.append(
#             {
#                 "agent": "source_validator",
#                 "tool": "llm_claim_verifier_dual_judge",
#                 "status": "ok",
#                 "claims_checked": len(report.get("claims", [])),
#                 "reliability_score": final_score,
#                 "judge_a_score": score_a,
#                 "judge_b_score": score_b,
#                 "judge_agreement": agreement.get("overall_agreement"),
#                 "secondary_judge_fallback": secondary_failed,
#                 "evidence_items_raw": len(raw_evidence),
#                 "evidence_items_used": len(validation_evidence),
#                 "evidence_limit": evidence_budget,
#                 "prompt_tokens": total_usage.get("prompt_tokens", 0),
#                 "completion_tokens": total_usage.get("completion_tokens", 0),
#                 "total_tokens": total_usage.get("total_tokens", 0),
#             }
#         )
#         return {
#             "validation_report": report,
#             "validated_market_analysis": validated,
#             "needs_revision": final_score < _sget(state, "validation_threshold", 70),
#             "tool_audit": tool_audit,
#             "token_usage": token_usage,
#         }


#     def run_repair_only(self, state: PitchState) -> PitchState:
#         """Lightweight validation for a candidate claim-level patch."""
#         token_usage_current = _sget(state, "token_usage", {}) or {}
#         validation = _sget(state, "validation_report", {}) or {}
#         repair_context = _sget(state, "market_repair_context", {}) or {}
#         repair_patches = _sget(state, "repair_patches", []) or []
#         previous_score = int(repair_context.get("previous_validation_score") or validation.get("reliability_score", 0) or 0)
#         previous_supported_ratio = float(repair_context.get("previous_supported_ratio") or 0.0)
#         focused_evidence = _limit_evidence_items(
#             _sget(state, "market_evidence", []) or [],
#             max_items=4,
#             content_chars=220,
#         )
#         repaired_claims = [
#             {
#                 "original_claim": p.get("original_claim"),
#                 "action": p.get("action"),
#                 "replacement": p.get("replacement"),
#                 "evidence_urls": p.get("evidence_urls", []),
#             }
#             for p in repair_patches
#             if isinstance(p, dict)
#         ]
#         result, usage = _invoke_structured_with_usage(
#             self.repair_judge_llm,
#             self.repair_prompt.format_messages(
#                 refined_idea=_sget(state, "refined_idea", ""),
#                 previous_market_analysis=repair_context.get("previous_market_analysis", ""),
#                 candidate_market_analysis=_sget(state, "market_analysis", ""),
#                 repaired_claims=json.dumps(repaired_claims, indent=2),
#                 focused_evidence=json.dumps(focused_evidence, indent=2),
#                 previous_score=previous_score,
#                 previous_supported_ratio=previous_supported_ratio,
#             ),
#         )
#         patch_report: Dict[str, Any] = json.loads(result.model_dump_json())
#         repaired_assessments = patch_report.get("repaired_claim_assessments", []) or []
#         candidate_score = int(patch_report.get("patch_reliability", 0) or 0)
#         candidate_ratio = 0.0
#         if repaired_assessments:
#             candidate_ratio = sum(
#                 1 for c in repaired_assessments
#                 if str(c.get("verdict", "")).strip().lower() == "supported"
#             ) / max(1, len(repaired_assessments))
#         accepted = bool(patch_report.get("can_accept_patch", False))
#         accepted = accepted and candidate_score >= previous_score and candidate_ratio >= max(0.0, previous_supported_ratio - 0.05)
#         previous_validation = validation if isinstance(validation, dict) else {}
#         if accepted:
#             final_validation = dict(previous_validation)
#             final_validation["reliability_score"] = max(previous_score, candidate_score)
#             final_validation["evidence_gaps"] = patch_report.get("evidence_gaps", "")
#             if repaired_assessments:
#                 final_validation["claims"] = repaired_assessments
#                 final_validation["claim_units"] = self._claim_ledger(repaired_assessments, focused_evidence, patch_report.get("evidence_gaps", ""))
#                 final_validation["failing_claims"] = [
#                     c for c in final_validation["claim_units"]
#                     if str(c.get("verdict", "")).strip().lower() in {"weakly_supported", "unsupported", "needs_review"}
#                 ]
#             final_validation["repair_validation"] = {
#                 "accepted": True,
#                 "previous_reliability_score": previous_score,
#                 "candidate_reliability_score": candidate_score,
#                 "rationale": patch_report.get("rationale", ""),
#             }
#             validated_market_analysis = (
#                 f"{_sget(state, 'market_analysis', '')}\n\n"
#                 f"Validation Score: {final_validation.get('reliability_score', previous_score)}/100\n"
#                 f"Patch Validation: accepted"
#             )
#             accepted_delta, rejected_delta = 1, 0
#         else:
#             final_validation = dict(previous_validation)
#             final_validation["repair_validation"] = {
#                 "accepted": False,
#                 "previous_reliability_score": previous_score,
#                 "candidate_reliability_score": candidate_score,
#                 "rationale": patch_report.get("rationale", ""),
#             }
#             validated_market_analysis = _sget(state, "validated_market_analysis", "")
#             accepted_delta, rejected_delta = 0, 1

#         token_usage = _merge_token_usage(state, usage)
#         tool_audit = list(_sget(state, "tool_audit", []))
#         tool_audit.append({
#             "agent": "source_validator",
#             "tool": "llm_claim_repair_validator",
#             "status": "ok",
#             "repair_patch_accepted": accepted,
#             "previous_reliability_score": previous_score,
#             "candidate_reliability_score": candidate_score,
#             "total_tokens": usage.get("total_tokens", 0),
#             "prompt_tokens": usage.get("prompt_tokens", 0),
#             "completion_tokens": usage.get("completion_tokens", 0),
#         })
#         return {
#             "validation_report": final_validation,
#             "validated_market_analysis": validated_market_analysis,
#             "micro_validation": patch_report,
#             "accepted_patch_count": int(_sget(state, "accepted_patch_count", 0) or 0) + accepted_delta,
#             "rejected_patch_count": int(_sget(state, "rejected_patch_count", 0) or 0) + rejected_delta,
#             "needs_revision": False if not accepted else final_validation.get("reliability_score", previous_score) < _sget(state, "validation_threshold", 70),
#             "tool_audit": tool_audit,
#             "token_usage": token_usage,
#         }


# class DirectStrategyAgent:
#     def __init__(
#         self,
#         llm: ChatOpenAI,
#         strict_tools: bool = True,
#         enable_trends: bool = True,
#         secondary_judge_llm: ChatOpenAI | None = None,
#     ):
#         self.structured_llm = llm.with_structured_output(DirectStrategyOutput, include_raw=True)
#         self.validator_agent = SourceValidatorAgent(llm, secondary_judge_llm=secondary_judge_llm)
#         self.search_tool = MarketSearchTool()
#         self.trends_tool = GoogleTrendsTool() if enable_trends else None
#         self.calc_tool = BusinessCalcTool()
#         self.scenario_tool = ScenarioAnalysisTool()
#         self.strict_tools = strict_tools
#         self.enable_trends = enable_trends
#         self.prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "# Role\n"
#                     "You are a direct (low-decomposition) startup strategist.\n\n"
#                     "# Instructions\n"
#                     "- Produce market analysis and business model in one synthesis pass.\n"
#                     "- Also propose bounded Year-1 assumptions (users/arpu/gross margin).\n"
#                     "- Keep outputs concise, evidence-aware, and investor-ready.\n\n"
#                     "# Output Format\n"
#                     "- Return content that strictly matches the structured schema fields.",
#                 ),
#                 (
#                     "human",
#                     "Startup concept:\n{refined_idea}\n\n"
#                     "Web snippets:\n{search_results}\n\n"
#                     "Trend signals:\n{trend_signals}\n\n"
#                     "Provide integrated market + business + assumptions output.",
#                 ),
#             ]
#         )

#     def run(self, state: PitchState) -> PitchState:
#         refined_idea = _sget(state, "refined_idea", "")
#         total_usage = _empty_token_usage()
#         query = f"startup market size competitors trends for: {refined_idea}"
#         search_payload = self.search_tool.search(query)
#         if self.strict_tools and search_payload["status"] != "ok":
#             raise RuntimeError(f"Market search tool unavailable: {search_payload['error']}")
#         evidence_budget = _evidence_budget_for_mode(
#             "direct",
#             _sget(state, "budget_remaining", {}).get("total_tokens")
#             if isinstance(_sget(state, "budget_remaining", {}), dict)
#             else None,
#         )
#         search_payload = _limited_search_payload(search_payload, **evidence_budget)

#         extracted_keywords = _fallback_keywords(refined_idea)
#         if self.enable_trends and self.trends_tool is not None:
#             trend_payload = self.trends_tool.fetch(extracted_keywords[:5])
#         else:
#             trend_payload = {
#                 "status": "skipped",
#                 "keywords": extracted_keywords[:5],
#                 "data": {},
#                 "error": "Google Trends disabled by configuration.",
#             }

#         result, direct_usage = _invoke_structured_with_usage(
#             self.structured_llm,
#             self.prompt.format_messages(
#                 refined_idea=refined_idea,
#                 search_results=search_payload["results_json"],
#                 trend_signals=json.dumps(trend_payload, indent=2),
#             ),
#         )
#         total_usage = _merge_token_usage({"token_usage": total_usage}, direct_usage)
#         token_usage = _merge_token_usage(state, total_usage)
#         market_analysis = (
#             f"Target Market: {result.target_market}\n"
#             f"Market Size: {result.market_size}\n"
#             f"Trends: {result.trends}\n"
#             f"Competitors: {result.competitors}\n"
#             f"Differentiation Gaps: {result.differentiation_gaps}"
#         )
#         assumptions = {
#             "users_year1": max(1000, min(500000, int(result.users_year1))),
#             "arpu_monthly": round(max(2.0, min(300.0, float(result.arpu_monthly))), 2),
#             "gross_margin": round(max(0.2, min(0.95, float(result.gross_margin))), 3),
#             "rationale": result.assumptions_rationale,
#         }
#         calc_script = f"""
# users_year1 = {assumptions["users_year1"]}
# arpu_monthly = {assumptions["arpu_monthly"]}
# annual_revenue = users_year1 * arpu_monthly * 12
# gross_margin = {assumptions["gross_margin"]}
# gross_profit = annual_revenue * gross_margin
# print(f'Year1 Revenue: ${{annual_revenue:,.0f}}')
# print(f'Year1 Gross Profit: ${{gross_profit:,.0f}}')
# """
#         calc_output = self.calc_tool.run(calc_script)
#         if self.strict_tools and calc_output.startswith("Python calc failed:"):
#             raise RuntimeError(calc_output)
#         scenario_output = self.scenario_tool.run(
#             users_year1=assumptions["users_year1"],
#             arpu_monthly=assumptions["arpu_monthly"],
#             gross_margin=assumptions["gross_margin"],
#         )
#         business_model = (
#             f"Revenue Streams: {result.revenue_streams}\n"
#             f"Pricing Strategy: {result.pricing_strategy}\n"
#             f"Cost Structure: {result.cost_structure}\n"
#             f"Unit Economics: {result.unit_economics}\n"
#             f"Financial Projection: {result.financial_projection}\n"
#             f"Financial Assumptions:\n{json.dumps(assumptions, indent=2)}\n"
#             f"Calculator Baseline:\n{calc_output}\n"
#             f"Scenario Analysis:\n{json.dumps(scenario_output, indent=2)}"
#         )
#         tool_audit = list(_sget(state, "tool_audit", []))
#         tool_audit.append(
#             {
#                 "agent": "adaptive_direct_strategy",
#                 "tool": "llm_direct_synthesis",
#                 "status": "ok",
#                 "prompt_tokens": direct_usage.get("prompt_tokens", 0),
#                 "completion_tokens": direct_usage.get("completion_tokens", 0),
#                 "total_tokens": direct_usage.get("total_tokens", 0),
#             }
#         )
#         tool_audit.append(
#             {
#                 "agent": "adaptive_direct_strategy",
#                 "tool": "linkup_search",
#                 "query": query,
#                 "status": search_payload["status"],
#                 "source_count": len(search_payload["sources"]),
#                 "source_count_raw": search_payload.get("source_count_raw"),
#                 "evidence_limit": search_payload.get("evidence_limit"),
#                 "dropped_sources": search_payload.get("dropped_sources", 0),
#                 "error": search_payload["error"],
#             }
#         )
#         tool_audit.append(
#             {
#                 "agent": "adaptive_direct_strategy",
#                 "tool": "google_trends",
#                 "status": trend_payload["status"],
#                 "keyword_count": len(trend_payload.get("keywords", [])),
#                 "keywords": trend_payload.get("keywords", []),
#                 "error": trend_payload.get("error", ""),
#             }
#         )
#         tool_audit.append(
#             {
#                 "agent": "adaptive_direct_strategy",
#                 "tool": "python_calc",
#                 "status": "ok" if not calc_output.startswith("Python calc failed:") else "error",
#                 "assumptions": assumptions,
#                 "error": calc_output if calc_output.startswith("Python calc failed:") else "",
#             }
#         )
#         tool_audit.append(
#             {
#                 "agent": "adaptive_direct_strategy",
#                 "tool": "scenario_analysis",
#                 "status": "ok",
#                 "scenario_count": len(scenario_output),
#                 "error": "",
#             }
#         )
#         candidate_state = {
#             **(state if isinstance(state, dict) else state.model_dump()),
#             "market_analysis": market_analysis,
#             "market_sources": search_payload["sources"],
#             "market_evidence": search_payload["results"],
#             "trend_signals": trend_payload,
#             "business_model": business_model,
#             "financial_assumptions": assumptions,
#             "scenario_analysis": scenario_output,
#             "tool_audit": tool_audit,
#             "token_usage": token_usage,
#             "decomposition_depth_realized": 0,
#         }
#         validated_update = self.validator_agent.run(candidate_state)
#         candidate_state.update(validated_update)
#         return {
#             "market_analysis": candidate_state["market_analysis"],
#             "market_sources": candidate_state["market_sources"],
#             "market_evidence": candidate_state["market_evidence"],
#             "trend_signals": candidate_state["trend_signals"],
#             "business_model": candidate_state["business_model"],
#             "financial_assumptions": candidate_state["financial_assumptions"],
#             "scenario_analysis": candidate_state["scenario_analysis"],
#             "validation_report": candidate_state.get("validation_report"),
#             "validated_market_analysis": candidate_state.get("validated_market_analysis"),
#             "needs_revision": candidate_state.get("needs_revision", False),
#             "tool_audit": candidate_state["tool_audit"],
#             "token_usage": candidate_state.get("token_usage", token_usage),
#             "decomposition_depth_realized": 0,
#         }


# class BusinessModelAgent:
#     def __init__(self, llm: ChatOpenAI, strict_tools: bool = True):
#         self.structured_llm = llm.with_structured_output(BusinessOutput, include_raw=True)
#         self.assumptions_llm = llm.with_structured_output(FinancialAssumptions, include_raw=True)
#         self.calc_tool = BusinessCalcTool()
#         self.scenario_tool = ScenarioAnalysisTool()
#         self.strict_tools = strict_tools
#         self.assumptions_prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "# Role\n"
#                     "You are a startup financial assumptions analyst.\n\n"
#                     "# Instructions\n"
#                     "- Generate realistic Year-1 assumptions from startup and market context.\n"
#                     "- Use conservative, explainable values.\n"
#                     "- Stay strictly within schema bounds.\n\n"
#                     "# Output Format\n"
#                     "- Return content that strictly matches the structured schema fields.",
#                 ),
#                 (
#                     "human",
#                     "Startup concept:\n{refined_idea}\n\n"
#                     "Validated market analysis:\n{market_analysis}\n\n"
#                     "Trend signals:\n{trend_signals}\n\n"
#                     "Validation score:\n{validation_score}\n\n"
#                     "Return assumptions for users_year1, arpu_monthly, gross_margin and rationale.",
#                 ),
#             ]
#         )
#         self.prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "# Role\n"
#                     "You are a startup business model strategist.\n\n"
#                     "# Instructions\n"
#                     "- Use supplied market analysis, assumptions, calculator output, and scenario output.\n"
#                     "- Produce practical revenue model, pricing, costs, unit economics, and projection narrative.\n"
#                     "- Keep recommendations coherent with the provided quantitative inputs.\n\n"
#                     "# Output Format\n"
#                     "- Return content that strictly matches the structured schema fields.",
#                 ),
#                 (
#                     "human",
#                     "Startup concept:\n{refined_idea}\n\n"
#                     "Market analysis:\n{market_analysis}\n\n"
#                     "Financial assumptions:\n{financial_assumptions}\n\n"
#                     "Financial calculator output:\n{calc_output}\n\n"
#                     "Scenario analysis output:\n{scenario_output}\n\n"
#                     "Provide revenue model, pricing, costs, and projection.",
#                 ),
#             ]
#         )

#     def run(self, state: PitchState) -> PitchState:
#         assumptions_model, assumptions_usage = _invoke_structured_with_usage(
#             self.assumptions_llm,
#             self.assumptions_prompt.format_messages(
#                 refined_idea=_sget(state, "refined_idea", ""),
#                 market_analysis=(
#                     _sget(state, "validated_market_analysis")
#                     or _sget(state, "market_analysis", "")
#                 ),
#                 trend_signals=json.dumps(_sget(state, "trend_signals", {}), indent=2),
#                 validation_score=_sget(state, "validation_report", {}).get(
#                     "reliability_score", 0
#                 ),
#             ),
#         )
#         total_usage = _merge_token_usage({"token_usage": _empty_token_usage()}, assumptions_usage)
#         assumptions = {
#             "users_year1": max(1000, min(500000, int(assumptions_model.users_year1))),
#             "arpu_monthly": round(max(2.0, min(300.0, float(assumptions_model.arpu_monthly))), 2),
#             "gross_margin": round(max(0.2, min(0.95, float(assumptions_model.gross_margin))), 3),
#             "rationale": assumptions_model.rationale,
#         }
#         calc_script = f"""
# users_year1 = {assumptions["users_year1"]}
# arpu_monthly = {assumptions["arpu_monthly"]}
# annual_revenue = users_year1 * arpu_monthly * 12
# gross_margin = {assumptions["gross_margin"]}
# gross_profit = annual_revenue * gross_margin
# print(f'Year1 Revenue: ${{annual_revenue:,.0f}}')
# print(f'Year1 Gross Profit: ${{gross_profit:,.0f}}')
# """
#         calc_output = self.calc_tool.run(calc_script)
#         if self.strict_tools and calc_output.startswith("Python calc failed:"):
#             raise RuntimeError(calc_output)
#         scenario_output = self.scenario_tool.run(
#             users_year1=assumptions["users_year1"],
#             arpu_monthly=assumptions["arpu_monthly"],
#             gross_margin=assumptions["gross_margin"],
#         )

#         result, business_usage = _invoke_structured_with_usage(
#             self.structured_llm,
#             self.prompt.format_messages(
#                 refined_idea=_sget(state, "refined_idea", ""),
#                 market_analysis=(
#                     _sget(state, "validated_market_analysis")
#                     or _sget(state, "market_analysis", "")
#                 ),
#                 financial_assumptions=json.dumps(assumptions, indent=2),
#                 calc_output=calc_output,
#                 scenario_output=json.dumps(scenario_output, indent=2),
#             ),
#         )
#         total_usage = _merge_token_usage({"token_usage": total_usage}, business_usage)
#         token_usage = _merge_token_usage(state, total_usage)
#         business_model = (
#             f"Revenue Streams: {result.revenue_streams}\n"
#             f"Pricing Strategy: {result.pricing_strategy}\n"
#             f"Cost Structure: {result.cost_structure}\n"
#             f"Unit Economics: {result.unit_economics}\n"
#             f"Financial Projection: {result.financial_projection}\n"
#             f"Financial Assumptions:\n{json.dumps(assumptions, indent=2)}\n"
#             f"Calculator Baseline:\n{calc_output}\n"
#             f"Scenario Analysis:\n{json.dumps(scenario_output, indent=2)}"
#         )
#         tool_audit = list(_sget(state, "tool_audit", []))
#         tool_audit.append(
#             {
#                 "agent": "business_model",
#                 "tool": "python_calc",
#                 "status": "ok" if not calc_output.startswith("Python calc failed:") else "error",
#                 "assumptions": assumptions,
#                 "error": calc_output if calc_output.startswith("Python calc failed:") else "",
#                 "prompt_tokens_assumptions": assumptions_usage.get("prompt_tokens", 0),
#                 "completion_tokens_assumptions": assumptions_usage.get("completion_tokens", 0),
#                 "total_tokens_assumptions": assumptions_usage.get("total_tokens", 0),
#                 "prompt_tokens_business": business_usage.get("prompt_tokens", 0),
#                 "completion_tokens_business": business_usage.get("completion_tokens", 0),
#                 "total_tokens_business": business_usage.get("total_tokens", 0),
#             }
#         )
#         tool_audit.append(
#             {
#                 "agent": "business_model",
#                 "tool": "scenario_analysis",
#                 "status": "ok",
#                 "scenario_count": len(scenario_output),
#                 "error": "",
#             }
#         )
#         return {
#             "business_model": business_model,
#             "financial_assumptions": assumptions,
#             "scenario_analysis": scenario_output,
#             "tool_audit": tool_audit,
#             "token_usage": token_usage,
#         }


# class PitchDeckGeneratorAgent:
#     def __init__(self, llm: ChatOpenAI, output_dir: str = "output"):
#         self.structured_llm = llm.with_structured_output(PitchSlides, include_raw=True)
#         self.output_dir = Path(output_dir)
#         self.prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "# Role\n"
#                     "You are an investor pitch deck strategist.\n\n"
#                     "# Instructions\n"
#                     "- Convert refined idea, market analysis, and business model into concise slide-ready content.\n"
#                     "- Keep language clear, specific, and investor-oriented.\n"
#                     "- Cover all required sections.\n\n"
#                     "# Output Format\n"
#                     "- Return content that strictly matches the structured schema fields.",
#                 ),
#                 (
#                     "human",
#                     "Refined idea:\n{refined_idea}\n\n"
#                     "Market analysis:\n{market_analysis}\n\n"
#                     "Business model:\n{business_model}\n\n"
#                     "Return all required slide sections.",
#                 ),
#             ]
#         )

#     def run(self, state: PitchState) -> PitchState:
#         slide_model, usage = _invoke_structured_with_usage(
#             self.structured_llm,
#             self.prompt.format_messages(
#                 refined_idea=_sget(state, "refined_idea", ""),
#                 market_analysis=(
#                     _sget(state, "validated_market_analysis")
#                     or _sget(state, "market_analysis", "")
#                 ),
#                 business_model=_sget(state, "business_model", ""),
#             ),
#         )
#         token_usage = _merge_token_usage(state, usage)
#         slide_dict: Dict[str, str] = json.loads(slide_model.model_dump_json())

#         summary_line = _sget(state, "refined_idea", "")
#         words = re.findall(r"[a-zA-Z0-9]+", summary_line.lower())
#         safe_name = "startup_pitch_deck"
#         if words:
#             safe_name = "pitch_" + "_".join(words[:6])
#         ppt_path = self.output_dir / f"{safe_name}.pptx"
#         saved = generate_pitch_deck(slide_dict, str(ppt_path))

#         tool_audit = list(_sget(state, "tool_audit", []))
#         tool_audit.append(
#             {
#                 "agent": "pitch_deck_generator",
#                 "tool": "python_pptx",
#                 "status": "ok",
#                 "output_path": saved,
#                 "prompt_tokens": usage.get("prompt_tokens", 0),
#                 "completion_tokens": usage.get("completion_tokens", 0),
#                 "total_tokens": usage.get("total_tokens", 0),
#             }
#         )
#         return {
#             "pitch_content": slide_dict,
#             "ppt_path": saved,
#             "tool_audit": tool_audit,
#             "token_usage": token_usage,
#         }














# # from __future__ import annotations

# # import json
# # import re
# # from pathlib import Path
# # from typing import Any, Dict, List, Literal, Optional

# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_openai import ChatOpenAI
# # from pydantic import BaseModel, Field

# # from startup_pitch_refinery.state import PitchState
# # from startup_pitch_refinery.tools import (
# #     BusinessCalcTool,
# #     GoogleTrendsTool,
# #     MarketSearchTool,
# #     ScenarioAnalysisTool,
# #     generate_pitch_deck,
# # )


# # def _sget(state: PitchState | Dict[str, Any], key: str, default: Any = None) -> Any:
# #     if isinstance(state, dict):
# #         return state.get(key, default)
# #     return getattr(state, key, default)


# # def _empty_token_usage() -> Dict[str, int]:
# #     return {
# #         "prompt_tokens": 0,
# #         "completion_tokens": 0,
# #         "total_tokens": 0,
# #     }


# # def _extract_token_usage(raw_message: Any) -> Dict[str, int]:
# #     usage = _empty_token_usage()
# #     if raw_message is None:
# #         return usage

# #     usage_meta = getattr(raw_message, "usage_metadata", None) or {}
# #     response_meta = getattr(raw_message, "response_metadata", None) or {}
# #     token_usage = response_meta.get("token_usage", {}) if isinstance(response_meta, dict) else {}

# #     prompt = (
# #         usage_meta.get("input_tokens")
# #         or usage_meta.get("prompt_tokens")
# #         or token_usage.get("prompt_tokens")
# #         or 0
# #     )
# #     completion = (
# #         usage_meta.get("output_tokens")
# #         or usage_meta.get("completion_tokens")
# #         or token_usage.get("completion_tokens")
# #         or 0
# #     )
# #     total = (
# #         usage_meta.get("total_tokens")
# #         or token_usage.get("total_tokens")
# #         or (int(prompt) + int(completion))
# #     )

# #     usage["prompt_tokens"] = max(0, int(prompt))
# #     usage["completion_tokens"] = max(0, int(completion))
# #     usage["total_tokens"] = max(0, int(total))
# #     return usage


# # def _invoke_structured_with_usage(runnable: Any, messages: Any) -> tuple[Any, Dict[str, int]]:
# #     payload = runnable.invoke(messages)
# #     if isinstance(payload, dict) and "parsed" in payload:
# #         parsed = payload.get("parsed")
# #         if parsed is None:
# #             raise ValueError(f"Structured output parsing failed: {payload.get('parsing_error')}")
# #         usage = _extract_token_usage(payload.get("raw"))
# #         return parsed, usage
# #     return payload, _empty_token_usage()


# # def _merge_token_usage(
# #     state: PitchState | Dict[str, Any], usage_delta: Dict[str, int]
# # ) -> Dict[str, int]:
# #     base = _sget(state, "token_usage", {}) or {}
# #     merged = {
# #         "prompt_tokens": int(base.get("prompt_tokens", 0)) + int(usage_delta.get("prompt_tokens", 0)),
# #         "completion_tokens": int(base.get("completion_tokens", 0))
# #         + int(usage_delta.get("completion_tokens", 0)),
# #         "total_tokens": int(base.get("total_tokens", 0)) + int(usage_delta.get("total_tokens", 0)),
# #     }
# #     return merged


# # def _deterministic_reliability_score(claims: List[Dict[str, Any]]) -> int:
# #     """
# #     Deterministic score from claim-level outputs.
# #     Formula:
# #       claim_score = verdict_weight * confidence
# #       reliability = average(claim_score) * 100
# #     """
# #     if not claims:
# #         return 0

# #     verdict_weights = {
# #         "supported": 1.0,
# #         "weakly_supported": 0.6,
# #         "needs_review": 0.3,
# #         "unsupported": 0.0,
# #     }

# #     total = 0.0
# #     for claim in claims:
# #         verdict = str(claim.get("verdict", "needs_review"))
# #         confidence = claim.get("confidence", 0.0)
# #         try:
# #             conf = float(confidence)
# #         except (TypeError, ValueError):
# #             conf = 0.0
# #         conf = max(0.0, min(1.0, conf))
# #         total += verdict_weights.get(verdict, 0.3) * conf

# #     return int(round((total / len(claims)) * 100))


# # def _fallback_keywords(text: str) -> List[str]:
# #     """Domain-agnostic fallback if LLM keyword extraction fails."""
# #     tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", text.lower())
# #     stop = {
# #         "problem",
# #         "solution",
# #         "summary",
# #         "value",
# #         "proposition",
# #         "startup",
# #         "idea",
# #         "with",
# #         "that",
# #         "this",
# #         "from",
# #         "into",
# #         "your",
# #         "for",
# #         "and",
# #         "the",
# #     }
# #     keywords: List[str] = []
# #     for token in tokens:
# #         if token in stop:
# #             continue
# #         if token not in keywords:
# #             keywords.append(token)
# #     if not keywords:
# #         return ["startup market", "industry trends", "competitor landscape"]
# #     return keywords[:5]


# # def _limit_evidence_items(
# #     evidence: List[Dict[str, Any]],
# #     *,
# #     max_items: int,
# #     content_chars: int,
# # ) -> List[Dict[str, Any]]:
# #     """Keep evidence prompts bounded while preserving titles and source URLs."""
# #     limited: List[Dict[str, Any]] = []
# #     seen_urls = set()
# #     for item in evidence:
# #         if len(limited) >= max_items:
# #             break
# #         if not isinstance(item, dict):
# #             continue
# #         url = str(item.get("url", "")).strip()
# #         if url and url in seen_urls:
# #             continue
# #         if url:
# #             seen_urls.add(url)
# #         limited.append(
# #             {
# #                 "title": str(item.get("title", ""))[:160],
# #                 "url": url,
# #                 "domain": str(item.get("domain", ""))[:120],
# #                 "content": str(item.get("content", ""))[:content_chars],
# #                 "quality_score": item.get("quality_score"),
# #                 "claim_id": item.get("claim_id"),
# #             }
# #         )
# #     return limited


# # def _limited_search_payload(
# #     payload: Dict[str, Any],
# #     *,
# #     max_items: int,
# #     content_chars: int,
# # ) -> Dict[str, Any]:
# #     """Return a compact search payload for LLM prompts and downstream validation."""
# #     evidence = _limit_evidence_items(
# #         payload.get("results", []) or [],
# #         max_items=max_items,
# #         content_chars=content_chars,
# #     )
# #     compact = dict(payload)
# #     compact["results"] = evidence
# #     compact["results_json"] = json.dumps(evidence, indent=2)
# #     compact["sources"] = [item["url"] for item in evidence if item.get("url")]
# #     compact["source_count_raw"] = len(payload.get("sources", []) or [])
# #     compact["evidence_limit"] = {
# #         "max_items": max_items,
# #         "content_chars": content_chars,
# #         "raw_items": len(payload.get("results", []) or []),
# #         "used_items": len(evidence),
# #     }
# #     return compact


# # def _merge_evidence_items(
# #     primary: List[Dict[str, Any]],
# #     secondary: List[Dict[str, Any]],
# # ) -> List[Dict[str, Any]]:
# #     """Merge evidence lists while preserving order and dropping duplicate URLs."""
# #     merged: List[Dict[str, Any]] = []
# #     seen_urls = set()
# #     for item in list(primary or []) + list(secondary or []):
# #         if not isinstance(item, dict):
# #             continue
# #         url = str(item.get("url", "")).strip()
# #         dedupe_key = url or f"{item.get('title', '')}:{item.get('content', '')[:80]}"
# #         if dedupe_key in seen_urls:
# #             continue
# #         seen_urls.add(dedupe_key)
# #         merged.append(item)
# #     return merged


# # def _claim_repair_category(claim_text: str) -> str:
# #     """Classify a validator claim into a repair-relevant category."""
# #     text = (claim_text or "").lower()
# #     numeric_markers = [
# #         "$",
# #         "%",
# #         "cagr",
# #         "billion",
# #         "million",
# #         "trillion",
# #         "market size",
# #         "projected",
# #         "forecast",
# #         "reach",
# #         "grow",
# #         "growth",
# #     ]
# #     gap_markers = [
# #         "gap",
# #         "opportunity",
# #         "underserved",
# #         "under-served",
# #         "tailored",
# #         "specifically",
# #         "specific",
# #         "lack",
# #         "limited",
# #         "differentiation",
# #         "white space",
# #         "niche",
# #         "not directly",
# #         "no direct",
# #     ]
# #     competitor_markers = [
# #         "competitor",
# #         "competitors",
# #         "platforms",
# #         "vendors",
# #         "solutions",
# #         "offerings",
# #         "players",
# #         "include",
# #         "focus on",
# #     ]
# #     trend_markers = [
# #         "trend",
# #         "adoption",
# #         "increasing",
# #         "integrating",
# #         "demand",
# #         "shift",
# #         "moving toward",
# #         "real-time",
# #         "predictive",
# #         "automation",
# #     ]
# #     if any(marker in text for marker in numeric_markers):
# #         return "market_size_or_numeric"
# #     if any(marker in text for marker in gap_markers):
# #         return "gap_or_opportunity"
# #     if any(marker in text for marker in competitor_markers):
# #         return "competitor_landscape"
# #     if any(marker in text for marker in trend_markers):
# #         return "trend_or_adoption"
# #     return "general_market_claim"


# # def _claim_materiality(claim_text: str, category: str | None = None) -> int:
# #     """Estimate business-criticality for claim-targeted repair routing."""
# #     text = (claim_text or "").lower()
# #     category = category or _claim_repair_category(text)
# #     materiality = 2
# #     if category == "market_size_or_numeric":
# #         materiality = 5
# #     elif category in {"competitor_landscape", "trend_or_adoption"}:
# #         materiality = 4
# #     elif category == "gap_or_opportunity":
# #         materiality = 3
# #     if any(marker in text for marker in ["regulatory", "compliance", "clinical", "healthcare", "risk"]):
# #         materiality = min(5, materiality + 1)
# #     if any(marker in text for marker in ["may", "could", "potential", "opportunity", "hypothesis"]):
# #         materiality = max(1, materiality - 1)
# #     return max(1, min(5, materiality))


# # def _claim_failure_type(claim: Dict[str, Any], evidence_gaps: str = "") -> str:
# #     """Map validator feedback to a small repair-oriented failure taxonomy."""
# #     text = str(claim.get("claim", "") or "").lower()
# #     verdict = str(claim.get("verdict", "") or "").strip().lower()
# #     rationale = str(claim.get("rationale", "") or "").lower()
# #     gaps = (evidence_gaps or "").lower()
# #     combined = " ".join([text, rationale, gaps])
# #     if verdict == "supported":
# #         return "none"
# #     if any(marker in combined for marker in ["contradict", "conflict", "inconsistent"]):
# #         return "contradictory_evidence"
# #     if any(marker in combined for marker in ["cagr", "percent", "%", "market size", "billion", "million", "forecast"]):
# #         return "speculative_number"
# #     if any(marker in combined for marker in ["outdated", "older", "old source"]):
# #         return "outdated"
# #     if any(marker in combined for marker in ["broad", "generic", "not specific", "indirect", "adjacent"]):
# #         return "too_broad"
# #     if any(marker in combined for marker in ["unclear", "ambiguous", "attribution"]):
# #         return "unclear_attribution"
# #     return "missing_evidence"


# # def _repair_action_for_claim(claim: Dict[str, Any]) -> str:
# #     """Choose the safest repair action for a weak validator claim."""
# #     text = str(claim.get("claim", "") or "")
# #     verdict = str(claim.get("verdict", "") or "").strip().lower()
# #     category = _claim_repair_category(text)
# #     supporting_sources = claim.get("supporting_sources", []) or []

# #     if category == "gap_or_opportunity":
# #         return "qualify_or_remove"
# #     if category in {"market_size_or_numeric", "competitor_landscape", "trend_or_adoption"}:
# #         return "search_and_replace"
# #     if verdict == "unsupported" and not supporting_sources:
# #         return "remove"
# #     if verdict == "needs_review":
# #         return "qualify_or_remove"
# #     return "search_and_replace" if supporting_sources else "qualify_or_remove"


# # def build_claim_repair_plan(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
# #     """
# #     Convert weak validator claims into explicit repair actions.

# #     This keeps recursive retry from becoming an unconstrained rewrite. Factual
# #     claims get evidence search; speculative gap/opportunity claims are narrowed
# #     or removed because searching often just produces adjacent evidence and a new
# #     weak claim.
# #     """
# #     plan: List[Dict[str, Any]] = []
# #     for claim in claims or []:
# #         if not isinstance(claim, dict):
# #             continue
# #         claim_text = str(claim.get("claim", "") or "").strip()
# #         if not claim_text:
# #             continue
# #         category = _claim_repair_category(claim_text)
# #         action = _repair_action_for_claim(claim)
# #         materiality = int(claim.get("materiality") or _claim_materiality(claim_text, category))
# #         failure_type = str(claim.get("failure_type") or _claim_failure_type(claim))
# #         if action == "search_and_replace":
# #             instruction = (
# #                 "Search for direct evidence. Replace the claim only if snippets "
# #                 "support the revised wording; otherwise qualify it conservatively."
# #             )
# #         elif action == "remove":
# #             instruction = (
# #                 "Remove this unsupported claim from the factual market analysis. "
# #                 "Do not replace it with a broader speculative claim."
# #             )
# #         else:
# #             instruction = (
# #                 "Do not try to prove this as a validated market fact. Narrow it to "
# #                 "a limitation/hypothesis or remove it if direct evidence is absent."
# #             )
# #         plan.append(
# #             {
# #                 "claim_id": str(claim.get("claim_id") or f"c{len(plan) + 1}"),
# #                 "claim": claim_text,
# #                 "claim_text": claim_text,
# #                 "verdict": str(claim.get("verdict", "") or "").strip().lower(),
# #                 "confidence": claim.get("confidence"),
# #                 "category": category,
# #                 "materiality": materiality,
# #                 "failure_type": failure_type,
# #                 "action": action,
# #                 "instruction": instruction,
# #                 "supporting_sources": claim.get("supporting_sources", []),
# #                 "rationale": str(claim.get("rationale", "") or "").strip()[:300],
# #             }
# #         )
# #     return plan


# # def _evidence_budget_for_mode(mode: str, remaining_total_tokens: Any = None) -> Dict[str, int]:
# #     """Mode- and budget-aware evidence limits for synthesis/validation prompts."""
# #     mode_l = str(mode or "").strip().lower()
# #     if mode_l == "direct":
# #         max_items, content_chars = 5, 220
# #     elif mode_l == "recursive":
# #         max_items, content_chars = 10, 280
# #     else:
# #         max_items, content_chars = 8, 260

# #     if isinstance(remaining_total_tokens, int):
# #         if remaining_total_tokens <= 3500:
# #             max_items = min(max_items, 3)
# #             content_chars = min(content_chars, 180)
# #         elif remaining_total_tokens <= 7000:
# #             max_items = min(max_items, 5)
# #             content_chars = min(content_chars, 220)

# #     return {
# #         "max_items": max_items,
# #         "content_chars": content_chars,
# #     }


# # class RefinedIdea(BaseModel):
# #     problem: str = Field(..., description="Core user/customer problem")
# #     solution: str = Field(..., description="Proposed startup solution")
# #     value_proposition: str = Field(..., description="Why this solution wins")
# #     refined_summary: str = Field(..., description="Short integrated startup concept")


# # class MarketOutput(BaseModel):
# #     target_market: str
# #     market_size: str
# #     trends: str
# #     competitors: str
# #     differentiation_gaps: str


# # class ClaimPatchOutput(BaseModel):
# #     original_claim: str
# #     action: Literal["search_and_replace", "qualify_or_remove", "remove", "preserve"]
# #     replacement: str
# #     expected_reliability_effect: Literal[
# #         "improve_to_supported",
# #         "reduce_to_conservative_claim",
# #         "remove_unsupported_claim",
# #         "no_change",
# #     ]
# #     evidence_urls: List[str] = Field(default_factory=list)
# #     rationale: str


# # class MarketRepairOutput(MarketOutput):
# #     patch_summary: str
# #     patches: List[ClaimPatchOutput] = Field(default_factory=list)


# # class BusinessOutput(BaseModel):
# #     revenue_streams: str
# #     pricing_strategy: str
# #     cost_structure: str
# #     unit_economics: str
# #     financial_projection: str


# # class DirectStrategyOutput(BaseModel):
# #     target_market: str
# #     market_size: str
# #     trends: str
# #     competitors: str
# #     differentiation_gaps: str
# #     revenue_streams: str
# #     pricing_strategy: str
# #     cost_structure: str
# #     unit_economics: str
# #     financial_projection: str
# #     users_year1: int = Field(..., ge=1000, le=500000)
# #     arpu_monthly: float = Field(..., ge=2.0, le=300.0)
# #     gross_margin: float = Field(..., ge=0.2, le=0.95)
# #     assumptions_rationale: str


# # class FinancialAssumptions(BaseModel):
# #     users_year1: int = Field(..., ge=1000, le=500000)
# #     arpu_monthly: float = Field(..., ge=2.0, le=300.0)
# #     gross_margin: float = Field(..., ge=0.2, le=0.95)
# #     rationale: str


# # class PitchSlides(BaseModel):
# #     title: str
# #     subtitle: str
# #     problem: str
# #     solution: str
# #     market: str
# #     business_model: str
# #     competitive_advantage: str
# #     financials: str


# # class VerifiedClaim(BaseModel):
# #     claim: str
# #     verdict: Literal["supported", "weakly_supported", "unsupported", "needs_review"]
# #     confidence: float = Field(..., ge=0.0, le=1.0)
# #     rationale: str
# #     supporting_sources: List[str]


# # class ValidationOutput(BaseModel):
# #     validated_summary: str
# #     reliability_score: int = Field(..., ge=0, le=100)
# #     evidence_gaps: str
# #     claims: List[VerifiedClaim]


# # class ValidationRubricScores(BaseModel):
# #     evidence_grounding: int = Field(..., ge=0, le=100)
# #     source_credibility: int = Field(..., ge=0, le=100)
# #     claim_specificity: int = Field(..., ge=0, le=100)
# #     internal_consistency: int = Field(..., ge=0, le=100)


# # class JudgeValidationOutput(BaseModel):
# #     validated_summary: str
# #     evidence_gaps: str
# #     overall_reliability: int = Field(..., ge=0, le=100)
# #     rubric_scores: ValidationRubricScores
# #     claims: List[VerifiedClaim]


# # class SecondJudgeOutput(BaseModel):
# #     evidence_gaps: str
# #     overall_reliability: int = Field(..., ge=0, le=100)
# #     rubric_scores: ValidationRubricScores
# #     claim_assessments: List[VerifiedClaim]


# # class RepairPatchValidationOutput(BaseModel):
# #     evidence_gaps: str
# #     patch_reliability: int = Field(..., ge=0, le=100)
# #     rubric_scores: ValidationRubricScores
# #     repaired_claim_assessments: List[VerifiedClaim]
# #     can_accept_patch: bool
# #     rationale: str


# # class TrendKeywords(BaseModel):
# #     keywords: List[str] = Field(
# #         ...,
# #         description="3 to 7 concise, domain-agnostic trend keywords for the startup idea.",
# #         min_length=3,
# #         max_length=7,
# #     )


# # class ControllerDecision(BaseModel):
# #     mode: Literal["direct", "shallow", "recursive"]
# #     confidence: float = Field(..., ge=0.0, le=1.0)
# #     rationale: str
# #     estimated_complexity: int = Field(..., ge=0, le=100)
# #     expected_tool_calls_delta: int = Field(..., ge=0, le=20)
# #     expected_token_proxy_delta: int = Field(..., ge=50, le=6000)
# #     expected_runtime_seconds_delta: float = Field(..., ge=0.1, le=600.0)
# #     triggers: List[str] = Field(default_factory=list)


# # class SupervisorAgent:
# #     def run(self, state: PitchState) -> PitchState:
# #         plan = [
# #             "1. Understand and refine the startup idea.",
# #             "2. Research market size, trends, and competitors.",
# #             "3. Validate claims against cited sources and score reliability.",
# #             "4. Build business model and basic financial projection.",
# #             "5. Structure investor-ready pitch content.",
# #             "6. Generate PowerPoint pitch deck (.pptx).",
# #         ]
# #         return {"task_plan": plan}


# # class IdeaRefinementAgent:
# #     def __init__(self, llm: ChatOpenAI):
# #         self.structured_llm = llm.with_structured_output(RefinedIdea, include_raw=True)
# #         self.prompt = ChatPromptTemplate.from_messages(
# #             [
# #                 (
# #                     "system",
# #                     "# Role\n"
# #                     "You are a startup concept refiner.\n\n"
# #                     "# Instructions\n"
# #                     "- Transform the raw startup idea into an investor-ready concept.\n"
# #                     "- Keep outputs concise, concrete, and specific.\n"
# #                     "- Cover problem, solution, value proposition, and a unified summary.\n\n"
# #                     "# Output Format\n"
# #                     "- Return content that strictly matches the structured schema fields.",
# #                 ),
# #                 (
# #                     "human",
# #                     "Raw idea: {idea}\n\n"
# #                     "Define: problem, solution, value proposition, and refined summary.",
# #                 ),
# #             ]
# #         )

# #     def run(self, state: PitchState) -> PitchState:
# #         idea = _sget(state, "idea", "")
# #         result, usage = _invoke_structured_with_usage(
# #             self.structured_llm, self.prompt.format_messages(idea=idea)
# #         )
# #         token_usage = _merge_token_usage(state, usage)
# #         refined = (
# #             f"Problem: {result.problem}\n"
# #             f"Solution: {result.solution}\n"
# #             f"Value Proposition: {result.value_proposition}\n"
# #             f"Summary: {result.refined_summary}"
# #         )
# #         return {"refined_idea": refined, "token_usage": token_usage}


# # class AdaptiveControllerAgent:
# #     MODE_ORDER = ["direct", "shallow", "recursive"]
# #     MODE_QUALITY_PRIORITY = ["recursive", "shallow", "direct"]
# #     MODE_PRIORS = {
# #         "direct": {
# #             "tool_calls": 4,
# #             "token_proxy": 1700,
# #             "runtime_seconds": 28.0,
# #         },
# #         "shallow": {
# #             "tool_calls": 6,
# #             "token_proxy": 3000,
# #             "runtime_seconds": 38.0,
# #         },
# #         "recursive": {
# #             "tool_calls": 8,
# #             "token_proxy": 3900,
# #             "runtime_seconds": 55.0,
# #         },
# #     }

# #     def __init__(self, llm: ChatOpenAI):
# #         self.structured_llm = llm.with_structured_output(ControllerDecision, include_raw=True)
# #         self.prompt = ChatPromptTemplate.from_messages(
# #             [
# #                 (
# #                     "system",
# #                     "# Role\n"
# #                     "You are an adaptive decomposition controller for a controller-executor startup workflow.\n\n"
# #                     "# Instructions\n"
# #                     "- Choose one execution mode: direct, shallow, or recursive.\n"
# #                     "- `direct`: lowest decomposition and lowest cost for straightforward ideas.\n"
# #                     "- `shallow`: default decomposed path; it can still escalate later if validation finds material failed claims.\n"
# #                     "- `recursive`: recursion-ready path for extreme uncertainty only; do not choose it for close calls.\n"
# #                     "- Prefer shallow over recursive unless recursive has a clear expected quality margin over its extra cost.\n"
# #                     "- Balance expected quality gains against remaining budget.\n"
# #                     "- Estimate complexity (0-100), confidence, concise rationale, and expected incremental costs.\n"
# #                     "- Include short trigger phrases that justify the choice.\n\n"
# #                     "# Output Format\n"
# #                     "- Return content that strictly matches the structured schema fields.",
# #                 ),
# #                 (
# #                     "human",
# #                     "Raw idea:\n{idea}\n\n"
# #                     "Refined idea:\n{refined_idea}\n\n"
# #                     "Validation threshold: {validation_threshold}\n"
# #                     "Max retries available: {max_validation_retries}\n\n"
# #                     "Budget state:\n{budget_snapshot}\n\n"
# #                     "Select mode and provide rationale.",
# #                 ),
# #             ]
# #         )

# #     @staticmethod
# #     def _fallback_decision(idea_text: str) -> ControllerDecision:
# #         text = idea_text.lower()
# #         tokens = re.findall(r"[a-zA-Z0-9]+", text)
# #         token_count = len(tokens)
# #         complexity_keywords = [
# #             "enterprise",
# #             "compliance",
# #             "risk",
# #             "workflow",
# #             "contract",
# #             "integration",
# #             "platform",
# #             "multi",
# #             "automation",
# #             "regulatory",
# #             "b2b",
# #             "agent",
# #             "marketplace",
# #             "prediction",
# #         ]
# #         match_count = sum(1 for key in complexity_keywords if key in text)
# #         score = min(100, int(token_count * 0.6 + match_count * 8))

# #         if score < 30:
# #             mode = "direct"
# #             confidence = 0.78
# #             rationale = "Idea appears relatively narrow and can be handled with low decomposition overhead."
# #             expected_tool_calls_delta = 3
# #             expected_token_proxy_delta = 1400
# #             expected_runtime_seconds_delta = 18.0
# #         elif score < 78:
# #             mode = "shallow"
# #             confidence = 0.72
# #             rationale = "Idea has moderate complexity, so one decomposition pass with validation is appropriate."
# #             expected_tool_calls_delta = 5
# #             expected_token_proxy_delta = 2600
# #             expected_runtime_seconds_delta = 32.0
# #         else:
# #             mode = "recursive"
# #             confidence = 0.69
# #             rationale = "Idea is high-complexity or high-uncertainty and benefits from validation-driven recursion."
# #             expected_tool_calls_delta = 7
# #             expected_token_proxy_delta = 3600
# #             expected_runtime_seconds_delta = 48.0

# #         triggers = []
# #         if token_count >= 35:
# #             triggers.append("long_problem_description")
# #         if match_count >= 3:
# #             triggers.append("multiple_complexity_markers")
# #         if "compliance" in text or "risk" in text:
# #             triggers.append("high_stakes_domain")
# #         if not triggers:
# #             triggers = ["low_complexity_signal"]

# #         return ControllerDecision(
# #             mode=mode,
# #             confidence=confidence,
# #             rationale=rationale,
# #             estimated_complexity=score,
# #             expected_tool_calls_delta=expected_tool_calls_delta,
# #             expected_token_proxy_delta=expected_token_proxy_delta,
# #             expected_runtime_seconds_delta=expected_runtime_seconds_delta,
# #             triggers=triggers,
# #         )

# #     @staticmethod
# #     def _compute_remaining_budget(state: PitchState | Dict[str, Any]) -> Dict[str, Any]:
# #         max_tool_calls = _sget(state, "max_tool_calls")
# #         max_token_proxy = _sget(state, "max_token_proxy")
# #         max_total_tokens = _sget(state, "max_total_tokens")
# #         max_runtime_seconds = _sget(state, "max_runtime_seconds")
# #         tool_calls_current = int(_sget(state, "tool_calls_current", 0) or 0)
# #         token_proxy_current = int(_sget(state, "token_proxy_current", 0) or 0)
# #         total_tokens_current = int(_sget(state, "total_tokens_current", 0) or 0)
# #         runtime_elapsed_seconds = float(_sget(state, "runtime_elapsed_seconds", 0.0) or 0.0)

# #         return {
# #             "max_tool_calls": max_tool_calls,
# #             "max_token_proxy": max_token_proxy,
# #             "max_total_tokens": max_total_tokens,
# #             "max_runtime_seconds": max_runtime_seconds,
# #             "tool_calls_current": tool_calls_current,
# #             "token_proxy_current": token_proxy_current,
# #             "total_tokens_current": total_tokens_current,
# #             "runtime_elapsed_seconds": round(runtime_elapsed_seconds, 3),
# #             "remaining_tool_calls": (
# #                 None
# #                 if max_tool_calls is None
# #                 else int(max_tool_calls) - tool_calls_current
# #             ),
# #             "remaining_token_proxy": (
# #                 None
# #                 if max_token_proxy is None
# #                 else int(max_token_proxy) - token_proxy_current
# #             ),
# #             "remaining_total_tokens": (
# #                 None
# #                 if max_total_tokens is None
# #                 else int(max_total_tokens) - total_tokens_current
# #             ),
# #             "remaining_runtime_seconds": (
# #                 None
# #                 if max_runtime_seconds is None
# #                 else round(float(max_runtime_seconds) - runtime_elapsed_seconds, 3)
# #             ),
# #         }

# #     @staticmethod
# #     def _budget_guardrail_mode(budget_snapshot: Dict[str, Any]) -> Optional[str]:
# #         rem_calls = budget_snapshot.get("remaining_tool_calls")
# #         rem_tokens = budget_snapshot.get("remaining_token_proxy")
# #         rem_total_tokens = budget_snapshot.get("remaining_total_tokens")
# #         rem_runtime = budget_snapshot.get("remaining_runtime_seconds")

# #         tight = (
# #             (isinstance(rem_calls, int) and rem_calls <= 2)
# #             or (isinstance(rem_tokens, int) and rem_tokens <= 900)
# #             or (isinstance(rem_total_tokens, int) and rem_total_tokens <= 900)
# #             or (isinstance(rem_runtime, (int, float)) and rem_runtime <= 10.0)
# #         )
# #         moderate = (
# #             (isinstance(rem_calls, int) and rem_calls <= 4)
# #             or (isinstance(rem_tokens, int) and rem_tokens <= 1900)
# #             or (isinstance(rem_total_tokens, int) and rem_total_tokens <= 1900)
# #             or (isinstance(rem_runtime, (int, float)) and rem_runtime <= 22.0)
# #         )

# #         if tight:
# #             return "direct"
# #         if moderate:
# #             return "shallow"
# #         return None

# #     @classmethod
# #     def _prior_cost_for_mode(cls, mode: str, complexity: int) -> Dict[str, Any]:
# #         base = cls.MODE_PRIORS.get(mode, cls.MODE_PRIORS["shallow"])
# #         # Scale priors mildly by complexity (0-100 -> 0.9x to 1.25x).
# #         scale = 0.9 + (max(0, min(100, complexity)) / 100.0) * 0.35
# #         return {
# #             "tool_calls": max(1, int(round(base["tool_calls"] * scale))),
# #             "token_proxy": max(100, int(round(base["token_proxy"] * scale))),
# #             "runtime_seconds": round(max(0.5, base["runtime_seconds"] * scale), 3),
# #         }

# #     @classmethod
# #     def _calibrate_selected_cost(
# #         cls,
# #         decision: ControllerDecision,
# #         mode_override: Optional[str] = None,
# #     ) -> Dict[str, Any]:
# #         mode_for_prior = mode_override or decision.mode
# #         prior = cls._prior_cost_for_mode(mode_for_prior, decision.estimated_complexity)
# #         raw = {
# #             "tool_calls": int(decision.expected_tool_calls_delta),
# #             "token_proxy": int(decision.expected_token_proxy_delta),
# #             "runtime_seconds": float(decision.expected_runtime_seconds_delta),
# #         }
# #         # Blend with prior and enforce non-trivial floor (80% of prior).
# #         tool_calls = max(int(round(prior["tool_calls"] * 0.8)), int(round(0.75 * prior["tool_calls"] + 0.25 * raw["tool_calls"])))
# #         token_proxy = max(int(round(prior["token_proxy"] * 0.8)), int(round(0.75 * prior["token_proxy"] + 0.25 * raw["token_proxy"])))
# #         runtime_seconds = max(round(prior["runtime_seconds"] * 0.8, 3), round(0.75 * prior["runtime_seconds"] + 0.25 * raw["runtime_seconds"], 3))
# #         return {
# #             "tool_calls": tool_calls,
# #             "token_proxy": token_proxy,
# #             "runtime_seconds": runtime_seconds,
# #             "prior": prior,
# #             "raw": raw,
# #         }

# #     @staticmethod
# #     def _complexity_features(text: str) -> Dict[str, Any]:
# #         text_l = (text or "").lower()
# #         tokens = re.findall(r"[a-zA-Z0-9]+", text_l)
# #         complexity_terms = [
# #             "recursive",
# #             "adversarial",
# #             "uncertainty",
# #             "uncertain",
# #             "cross-border",
# #             "jurisdiction",
# #             "compliance",
# #             "regulatory",
# #             "critical infrastructure",
# #             "rollback",
# #             "re-plan",
# #             "replanning",
# #             "conflicting",
# #             "safety",
# #             "failure",
# #             "multi-agent",
# #             "enterprise",
# #             "integration",
# #             "workflow",
# #             "customer support",
# #             "support",
# #             "tickets",
# #             "refund",
# #             "refund-risk",
# #             "no-shows",
# #             "no show",
# #             "scheduling",
# #             "outpatient",
# #             "marketplace",
# #             "prediction",
# #             "optimization",
# #             "supply chain",
# #             "healthcare",
# #             "finance",
# #             "legal",
# #             "cybersecurity",
# #             "security",
# #             "public sector",
# #             "government",
# #             "city",
# #             "manufacturing",
# #             "manufacturers",
# #             "logistics",
# #             "shipment",
# #             "customs",
# #             "vendor",
# #             "geopolitical",
# #             "audit",
# #             "traceability",
# #             "real-time",
# #             "multi-sided",
# #             "stakeholders",
# #             "hipaa",
# #             "insurance",
# #             "pre-authorization",
# #             "authorization",
# #             "clinical",
# #             "documentation",
# #             "specialty",
# #             "governance",
# #             "fraud",
# #             "banks",
# #             "transaction",
# #             "incident",
# #             "response",
# #             "alerts",
# #             "logs",
# #             "containment",
# #             "renewable",
# #             "energy",
# #             "grid",
# #             "equipment",
# #             "maintenance",
# #             "scientific",
# #             "literature",
# #             "token budgets",
# #         ]
# #         evidence_terms = [
# #             "market",
# #             "competitor",
# #             "competitors",
# #             "regulation",
# #             "regulatory",
# #             "compliance",
# #             "clinical",
# #             "hipaa",
# #             "insurance",
# #             "pre-authorization",
# #             "authorization",
# #             "patient",
# #             "governance",
# #             "legal",
# #             "financial",
# #             "bank",
# #             "banks",
# #             "fraud",
# #             "audit",
# #             "risk",
# #             "refund",
# #             "refund-risk",
# #             "no-shows",
# #             "no show",
# #             "outpatient",
# #             "scheduling",
# #             "safety",
# #             "security",
# #             "certification",
# #             "benchmark",
# #             "accuracy",
# #             "claims",
# #             "evidence",
# #             "scientific",
# #             "literature",
# #             "customs",
# #             "geopolitical",
# #         ]
# #         uncertainty_terms = [
# #             "uncertain",
# #             "uncertainty",
# #             "predict",
# #             "prediction",
# #             "risk",
# #             "refund-risk",
# #             "refund",
# #             "no-shows",
# #             "no show",
# #             "failure",
# #             "adversarial",
# #             "conflicting",
# #             "dynamic",
# #             "adaptive",
# #             "real-time",
# #             "optimize",
# #             "optimization",
# #             "trade-off",
# #             "tradeoff",
# #             "multi-step",
# #             "long-horizon",
# #             "fallback",
# #             "replan",
# #             "re-plan",
# #             "shipment",
# #             "delays",
# #             "fraud",
# #             "incident",
# #             "alerts",
# #             "failures",
# #             "emergency",
# #             "constraints",
# #             "coverage",
# #             "budget",
# #             "budgets",
# #         ]
# #         workflow_terms = [
# #             "platform",
# #             "workflow",
# #             "coordinates",
# #             "coordination",
# #             "management",
# #             "monitoring",
# #             "dashboard",
# #             "assistant",
# #             "copilot",
# #             "agent",
# #             "multi-agent",
# #             "integration",
# #             "api",
# #             "tool",
# #             "automation",
# #             "pipeline",
# #             "orchestration",
# #             "enterprise",
# #             "team",
# #             "teams",
# #             "case-management",
# #             "case management",
# #             "scheduling",
# #             "ticket",
# #             "tickets",
# #             "customer support",
# #             "support",
# #             "clinic",
# #             "clinics",
# #             "outpatient",
# #             "small ecommerce",
# #             "documentation",
# #             "reports",
# #             "reviews",
# #             "coverage",
# #         ]

# #         def count_terms(terms: List[str]) -> int:
# #             return sum(1 for term in terms if term in text_l)

# #         marker_count = count_terms(complexity_terms)
# #         evidence_count = count_terms(evidence_terms)
# #         uncertainty_count = count_terms(uncertainty_terms)
# #         workflow_count = count_terms(workflow_terms)

# #         token_count = len(tokens)
# #         length_score = min(1.0, token_count / 45.0)
# #         complexity_marker_score = min(1.0, marker_count / 6.0)
# #         evidence_need_score = min(1.0, evidence_count / 4.0)
# #         uncertainty_score = min(1.0, uncertainty_count / 4.0)
# #         workflow_score = min(1.0, workflow_count / 5.0)
# #         structural_complexity = round(
# #             100.0
# #             * (
# #                 0.30 * length_score
# #                 + 0.25 * complexity_marker_score
# #                 + 0.20 * evidence_need_score
# #                 + 0.15 * uncertainty_score
# #                 + 0.10 * workflow_score
# #             ),
# #             2,
# #         )
# #         uncertainty_need = round(
# #             100.0
# #             * (
# #                 0.45 * uncertainty_score
# #                 + 0.30 * evidence_need_score
# #                 + 0.25 * workflow_score
# #             ),
# #             2,
# #         )
# #         return {
# #             "token_count": token_count,
# #             "marker_count": marker_count,
# #             "evidence_count": evidence_count,
# #             "uncertainty_count": uncertainty_count,
# #             "workflow_count": workflow_count,
# #             "length_score": round(length_score, 4),
# #             "complexity_marker_score": round(complexity_marker_score, 4),
# #             "evidence_need_score": round(evidence_need_score, 4),
# #             "uncertainty_score": round(uncertainty_score, 4),
# #             "workflow_score": round(workflow_score, 4),
# #             "structural_complexity": structural_complexity,
# #             "uncertainty_need": uncertainty_need,
# #             "has_high_stakes": marker_count >= 4 or evidence_count >= 3,
# #         }

# #     @classmethod
# #     def _score_modes(
# #         cls,
# #         *,
# #         idea_text: str,
# #         llm_decision: ControllerDecision,
# #         budget_snapshot: Dict[str, Any],
# #         validation_threshold: int,
# #         max_validation_retries: int,
# #     ) -> Dict[str, Any]:
# #         """
# #         Math-backed adaptive controller.

# #         Utility(mode) = expected_quality(mode) - lambda_cost * normalized_cost(mode) * 100
# #         where expected quality is estimated from structural complexity, uncertainty,
# #         evidence need, and workflow coupling. Costs are deterministic mode priors
# #         scaled by complexity and checked against the current budget.
# #         """
# #         raw_idea_text = ""
# #         for line in str(idea_text or "").splitlines():
# #             stripped = line.strip()
# #             if stripped:
# #                 raw_idea_text = stripped
# #                 break
# #         raw_features = cls._complexity_features(raw_idea_text or idea_text)
# #         features = cls._complexity_features(idea_text)
# #         complexity = max(
# #             int(round(float(features["structural_complexity"]))),
# #             int(llm_decision.estimated_complexity * 0.25),
# #         )
# #         complexity = max(0, min(100, complexity))

# #         c = complexity / 100.0
# #         u = float(features["uncertainty_need"]) / 100.0
# #         e = float(features["evidence_need_score"])
# #         w = float(features["workflow_score"])
# #         complexity_marker = float(features["complexity_marker_score"])

# #         mode_costs = {mode: cls._prior_cost_for_mode(mode, complexity) for mode in cls.MODE_ORDER}

# #         max_cost_tokens = max(cost["token_proxy"] for cost in mode_costs.values()) or 1
# #         max_cost_calls = max(cost["tool_calls"] for cost in mode_costs.values()) or 1
# #         max_cost_runtime = max(cost["runtime_seconds"] for cost in mode_costs.values()) or 1.0

# #         remaining_total = budget_snapshot.get("remaining_total_tokens")
# #         remaining_proxy = budget_snapshot.get("remaining_token_proxy")
# #         remaining_tokens = (
# #             remaining_total if isinstance(remaining_total, int) else remaining_proxy
# #         )
# #         token_budget_pressure = 0.0
# #         if isinstance(remaining_tokens, int) and remaining_tokens > 0:
# #             token_budget_pressure = max(
# #                 0.0,
# #                 min(1.0, 1.0 - (remaining_tokens / max(remaining_tokens, max_cost_tokens * 2.5))),
# #             )
# #         elif isinstance(remaining_tokens, int) and remaining_tokens <= 0:
# #             token_budget_pressure = 1.0

# #         remaining_calls = budget_snapshot.get("remaining_tool_calls")
# #         call_budget_pressure = 0.0
# #         if isinstance(remaining_calls, int) and remaining_calls > 0:
# #             call_budget_pressure = max(
# #                 0.0,
# #                 min(1.0, 1.0 - (remaining_calls / max(remaining_calls, max_cost_calls * 2.0))),
# #             )
# #         elif isinstance(remaining_calls, int) and remaining_calls <= 0:
# #             call_budget_pressure = 1.0

# #         budget_pressure = round(max(token_budget_pressure, call_budget_pressure), 4)
# #         lambda_cost = round(0.12 + 0.28 * budget_pressure, 4)

# #         # Expected reliability is deliberately quality-first when the experiment
# #         # asks for high validation confidence. Direct mode is cheap, but prior
# #         # runs showed it is fragile under evidence-grounded judge scoring.
# #         expected_quality = {
# #             "direct": (
# #                 53.0
# #                 + 7.0 * (1.0 - c)
# #                 + 2.0 * (1.0 - u)
# #                 - 7.0 * e
# #                 - 4.0 * w
# #                 - 3.0 * complexity_marker
# #             ),
# #             "shallow": 66.0 + 10.0 * c + 8.0 * e + 5.0 * w + 3.0 * u,
# #             "recursive": (
# #                 65.0
# #                 + 14.0 * c
# #                 + 11.0 * u
# #                 + 6.0 * e
# #                 + 3.0 * complexity_marker
# #             ),
# #         }

# #         # Recursion only gives value if the graph is allowed to revise.
# #         if max_validation_retries <= 0:
# #             expected_quality["recursive"] -= 5.0
# #         elif validation_threshold >= 75:
# #             expected_quality["shallow"] += 2.0
# #             expected_quality["recursive"] += 3.0
# #             expected_quality["direct"] -= 6.0

# #         raw_complexity = float(raw_features["structural_complexity"])
# #         raw_uncertainty = float(raw_features["uncertainty_need"])
# #         raw_evidence_count = int(raw_features["evidence_count"])
# #         raw_workflow_count = int(raw_features["workflow_count"])
# #         raw_marker_count = int(raw_features["marker_count"])
# #         raw_token_count = int(raw_features["token_count"])
# #         evidence_sensitive_medium = (
# #             22.0 <= raw_complexity < 64.0
# #             and budget_pressure <= 0.55
# #             and (
# #                 (raw_evidence_count >= 1 and raw_workflow_count >= 1)
# #                 or (raw_uncertainty >= 25.0 and raw_workflow_count >= 1)
# #                 or (raw_evidence_count >= 2 and raw_uncertainty >= 20.0)
# #             )
# #         )

# #         # Direct is eligible from the raw idea, not the expanded refined summary.
# #         # Otherwise the refinement step itself can make simple ideas look too
# #         # verbose/complex and direct is never tested.
# #         direct_eligible = (
# #             raw_complexity <= 22
# #             and raw_uncertainty <= 12
# #             and raw_evidence_count == 0
# #             and raw_workflow_count <= 1
# #             and raw_marker_count == 0
# #             and raw_token_count <= 24
# #             and complexity <= 45
# #         ) or budget_pressure >= 0.72
# #         if direct_eligible:
# #             expected_quality["direct"] += 12.0
# #             if validation_threshold >= 75:
# #                 # Keep direct viable for simple tasks, but not unrealistically
# #                 # dominant under a high evidence-grounding threshold.
# #                 expected_quality["direct"] -= 2.0
# #             if budget_pressure >= 0.50:
# #                 expected_quality["direct"] += 2.0
# #         elif not direct_eligible:
# #             expected_quality["direct"] -= 12.0

# #         recursive_upfront_allowed = (
# #             complexity >= 82
# #             or float(features["uncertainty_need"]) >= 78
# #             or (
# #                 bool(features["has_high_stakes"])
# #                 and complexity >= 72
# #                 and (
# #                     float(features["uncertainty_need"]) >= 68
# #                     or (e >= 0.85 and w >= 0.75)
# #                 )
# #             )
# #             or (
# #                 complexity >= 74
# #                 and float(features["uncertainty_need"]) >= 72
# #                 and e >= 0.65
# #             )
# #         )

# #         # Hard, uncertain, evidence-heavy tasks may start recursion-ready, but
# #         # medium tasks should usually run shallow first and recurse only after a
# #         # concrete claim-level validation failure.
# #         if recursive_upfront_allowed:
# #             expected_quality["recursive"] += 4.0
# #         else:
# #             expected_quality["recursive"] -= 7.0
# #         if complexity >= 45 and e >= 0.50:
# #             expected_quality["shallow"] += 1.5
# #             expected_quality["recursive"] += 1.0
# #         if evidence_sensitive_medium:
# #             # Medium-complexity tasks in regulated, operational, or risk-heavy
# #             # domains often need repair readiness, not upfront recursion.
# #             expected_quality["recursive"] += 2.0
# #             expected_quality["shallow"] += 2.0

# #         # Keep expected quality interpretable as a reliability estimate. The
# #         # controller utility may still exceed alternatives through lower cost
# #         # or advisory bonuses, but the quality term itself stays on 0-100.
# #         expected_quality = {
# #             mode: max(0.0, min(100.0, value))
# #             for mode, value in expected_quality.items()
# #         }

# #         mode_scores: Dict[str, Dict[str, Any]] = {}
# #         selected_mode = "direct"
# #         selected_utility = -10_000.0
# #         for mode in cls.MODE_ORDER:
# #             cost = mode_costs[mode]
# #             token_norm = cost["token_proxy"] / max_cost_tokens
# #             call_norm = cost["tool_calls"] / max_cost_calls
# #             runtime_norm = cost["runtime_seconds"] / max_cost_runtime
# #             normalized_cost = round(
# #                 0.60 * token_norm + 0.25 * call_norm + 0.15 * runtime_norm,
# #                 4,
# #             )
# #             feasible = cls._is_mode_feasible(cost, budget_snapshot)
# #             advisory_bonus = (
# #                 round(1.5 * float(llm_decision.confidence), 4)
# #                 if mode == llm_decision.mode
# #                 else 0.0
# #             )
# #             utility = (
# #                 expected_quality[mode]
# #                 - lambda_cost * normalized_cost * 100.0
# #                 + advisory_bonus
# #             )
# #             if not feasible:
# #                 utility -= 100.0
# #             mode_scores[mode] = {
# #                 "expected_quality": round(expected_quality[mode], 4),
# #                 "expected_cost": cost,
# #                 "normalized_cost": normalized_cost,
# #                 "advisory_bonus": advisory_bonus,
# #                 "feasible": feasible,
# #                 "utility": round(utility, 4),
# #             }
# #             if utility > selected_utility:
# #                 selected_utility = utility
# #                 selected_mode = mode

# #         sorted_scores = sorted(
# #             mode_scores.items(),
# #             key=lambda kv: float(kv[1]["utility"]),
# #             reverse=True,
# #         )
# #         utility_margin = 0.0
# #         if len(sorted_scores) > 1:
# #             utility_margin = round(
# #                 float(sorted_scores[0][1]["utility"]) - float(sorted_scores[1][1]["utility"]),
# #                 4,
# #             )

# #         policy_adjustments: List[str] = []
# #         if selected_mode == "direct" and not direct_eligible:
# #             non_direct = [
# #                 item
# #                 for item in sorted_scores
# #                 if item[0] != "direct" and bool(item[1].get("feasible", False))
# #             ]
# #             if non_direct:
# #                 selected_mode = non_direct[0][0]
# #                 selected_utility = float(non_direct[0][1]["utility"])
# #                 policy_adjustments.append("direct_not_eligible_promoted_to_non_direct")

# #         direct_score = mode_scores["direct"]
# #         recursive_score = mode_scores["recursive"]
# #         shallow_score = mode_scores["shallow"]
# #         direct_expected_quality = float(direct_score.get("expected_quality", 0.0) or 0.0)
# #         shallow_expected_quality = float(shallow_score.get("expected_quality", 0.0) or 0.0)
# #         recursive_expected_quality = float(recursive_score.get("expected_quality", 0.0) or 0.0)
# #         direct_quality_floor = max(62.0, float(validation_threshold) - 5.0)
# #         direct_quality_ready = direct_expected_quality >= direct_quality_floor
# #         shallow_quality_advantage = shallow_expected_quality - direct_expected_quality
# #         recursive_quality_advantage = recursive_expected_quality - shallow_expected_quality
# #         recursive_extra_token_proxy = max(
# #             1,
# #             int(mode_costs["recursive"]["token_proxy"])
# #             - int(mode_costs["shallow"]["token_proxy"]),
# #         )
# #         recursive_marginal_quality_per_1k_token = (
# #             recursive_quality_advantage / (recursive_extra_token_proxy / 1000.0)
# #         )
# #         recursive_roi_floor = 0.90 + 1.60 * budget_pressure
# #         recursive_cost_efficient = (
# #             recursive_quality_advantage >= 3.0
# #             or recursive_marginal_quality_per_1k_token >= recursive_roi_floor
# #             or (complexity >= 90 and u >= 0.75)
# #             or (
# #                 evidence_sensitive_medium
# #                 and recursive_quality_advantage >= 3.0
# #                 and recursive_marginal_quality_per_1k_token >= 0.90
# #             )
# #         )
# #         if (
# #             selected_mode == "shallow"
# #             and direct_eligible
# #             and bool(direct_score.get("feasible", False))
# #             and direct_quality_ready
# #             and shallow_quality_advantage <= 4.0
# #             and float(direct_score["utility"]) >= float(shallow_score["utility"]) - 3.5
# #         ):
# #             selected_mode = "direct"
# #             selected_utility = float(direct_score["utility"])
# #             policy_adjustments.append("validation_ready_close_call_prefers_direct")
# #         elif (
# #             selected_mode == "direct"
# #             and (
# #                 not direct_quality_ready
# #                 or (
# #                     bool(shallow_score.get("feasible", False))
# #                     and shallow_quality_advantage > 4.0
# #                     and budget_pressure < 0.65
# #                 )
# #             )
# #         ):
# #             non_direct = [
# #                 item
# #                 for item in sorted_scores
# #                 if item[0] != "direct" and bool(item[1].get("feasible", False))
# #             ]
# #             if non_direct:
# #                 selected_mode = non_direct[0][0]
# #                 selected_utility = float(non_direct[0][1]["utility"])
# #                 policy_adjustments.append("direct_deferred_until_validation_ready")

# #         if (
# #             selected_mode == "shallow"
# #             and bool(recursive_score.get("feasible", False))
# #             and max_validation_retries > 0
# #             and validation_threshold >= 75
# #             and budget_pressure <= 0.45
# #             and recursive_upfront_allowed
# #             and float(recursive_score["utility"]) > float(shallow_score["utility"]) + 3.0
# #             and recursive_cost_efficient
# #         ):
# #             selected_mode = "recursive"
# #             selected_utility = float(recursive_score["utility"])
# #             policy_adjustments.append("recursive_margin_cleared_after_shallow_first_gate")
# #         elif (
# #             selected_mode == "recursive"
# #             and bool(shallow_score.get("feasible", False))
# #             and (
# #                 not recursive_cost_efficient
# #                 or float(recursive_score["utility"]) <= float(shallow_score["utility"]) + 3.0
# #             )
# #         ):
# #             selected_mode = "shallow"
# #             selected_utility = float(shallow_score["utility"])
# #             policy_adjustments.append("recursive_deferred_until_claim_failure")
# #         elif selected_mode == "recursive" and not recursive_upfront_allowed:
# #             if bool(shallow_score.get("feasible", False)):
# #                 selected_mode = "shallow"
# #                 selected_utility = float(shallow_score["utility"])
# #                 policy_adjustments.append("recursive_deferred_until_validation_failure")

# #         if policy_adjustments:
# #             best_alternative = max(
# #                 (
# #                     float(score["utility"])
# #                     for mode, score in mode_scores.items()
# #                     if mode != selected_mode
# #                 ),
# #                 default=float(selected_utility),
# #             )
# #             utility_margin = round(float(selected_utility) - best_alternative, 4)

# #         formula = (
# #             "utility(mode)=expected_quality(mode)-lambda_cost*normalized_cost(mode)*100"
# #             "+llm_advisory_bonus; recursive is selected upfront only when it clears "
# #             "a shallow-first margin and cost-efficiency gate; normalized_cost="
# #             "0.60*token_norm+0.25*tool_call_norm+0.15*runtime_norm"
# #         )
# #         return {
# #             "selected_mode": selected_mode,
# #             "selected_utility": round(selected_utility, 4),
# #             "utility_margin": utility_margin,
# #             "features": features,
# #             "raw_features": raw_features,
# #             "estimated_complexity": complexity,
# #             "budget_pressure": budget_pressure,
# #             "lambda_cost": lambda_cost,
# #             "direct_eligible": direct_eligible,
# #             "direct_quality_ready": direct_quality_ready,
# #             "direct_quality_floor": round(direct_quality_floor, 4),
# #             "shallow_quality_advantage": round(shallow_quality_advantage, 4),
# #             "recursive_upfront_allowed": recursive_upfront_allowed,
# #             "evidence_sensitive_medium": evidence_sensitive_medium,
# #             "recursive_quality_advantage": round(recursive_quality_advantage, 4),
# #             "recursive_extra_token_proxy": recursive_extra_token_proxy,
# #             "recursive_marginal_quality_per_1k_token": round(
# #                 recursive_marginal_quality_per_1k_token,
# #                 4,
# #             ),
# #             "recursive_roi_floor": round(recursive_roi_floor, 4),
# #             "recursive_cost_efficient": recursive_cost_efficient,
# #             "policy_adjustments": policy_adjustments,
# #             "mode_scores": mode_scores,
# #             "formula": formula,
# #             "llm_advisory_mode": llm_decision.mode,
# #             "llm_advisory_confidence": llm_decision.confidence,
# #             "llm_advisory_complexity": llm_decision.estimated_complexity,
# #         }

# #     @staticmethod
# #     def _build_decomposition_graph(mode: str) -> Dict[str, Any]:
# #         """
# #         Persist the decomposition object D=(V,E,tau,rho) used by the controller.
# #         This is intentionally compact and inspectable for methodology reporting.
# #         """
# #         if mode == "direct":
# #             nodes = [
# #                 {
# #                     "id": "direct_synthesis",
# #                     "task": "Generate startup analysis in one integrated pass",
# #                     "interface": "structured_llm_output",
# #                     "executor": "direct_strategy_agent",
# #                     "atomicity": "coarse",
# #                 }
# #             ]
# #             edges: List[Dict[str, str]] = []
# #         else:
# #             nodes = [
# #                 {
# #                     "id": "idea_refinement",
# #                     "task": "Refine raw idea into problem, solution, value proposition, and summary",
# #                     "interface": "RefinedIdea schema",
# #                     "executor": "idea_refinement_agent",
# #                     "atomicity": "atomic",
# #                 },
# #                 {
# #                     "id": "market_research",
# #                     "task": "Retrieve market sources and synthesize market, competitors, and trends",
# #                     "interface": "MarketOutput schema plus search/trends tools",
# #                     "executor": "market_research_agent",
# #                     "atomicity": "composite",
# #                 },
# #                 {
# #                     "id": "source_validation",
# #                     "task": "Evaluate evidence support and assign reliability score",
# #                     "interface": "JudgeValidationOutput schema",
# #                     "executor": "source_validator_agent",
# #                     "atomicity": "atomic",
# #                 },
# #                 {
# #                     "id": "business_model",
# #                     "task": "Generate bounded financial assumptions and deterministic scenarios",
# #                     "interface": "FinancialAssumptions schema plus calculator",
# #                     "executor": "business_model_agent",
# #                     "atomicity": "atomic",
# #                 },
# #                 {
# #                     "id": "pitch_content",
# #                     "task": "Create investor-ready pitch narrative and optional deck",
# #                     "interface": "PitchSlides schema",
# #                     "executor": "pitch_deck_generator_agent",
# #                     "atomicity": "atomic",
# #                 },
# #             ]
# #             edges = [
# #                 {"from": "idea_refinement", "to": "market_research"},
# #                 {"from": "market_research", "to": "source_validation"},
# #                 {"from": "source_validation", "to": "business_model"},
# #                 {"from": "business_model", "to": "pitch_content"},
# #             ]
# #             if mode == "recursive":
# #                 nodes.append(
# #                     {
# #                         "id": "market_revision",
# #                         "task": "Repair weak or unsupported market claims after validation failure",
# #                         "interface": "claim repair plan plus focused evidence snippets",
# #                         "executor": "claim_repair_agent",
# #                         "atomicity": "conditional",
# #                     }
# #                 )
# #                 edges.extend(
# #                     [
# #                         {"from": "source_validation", "to": "market_revision"},
# #                         {"from": "market_revision", "to": "source_validation"},
# #                     ]
# #                 )

# #         depth = {"direct": 0, "shallow": 1, "recursive": 2}.get(mode, 1)
# #         node_count = len(nodes)
# #         atomic_nodes = sum(1 for node in nodes if node.get("atomicity") == "atomic")
# #         atomicity_ratio = round(atomic_nodes / max(1, node_count), 4)
# #         return {
# #             "formalization": "D=(V,E,tau,rho), where V=subtasks, E=dependencies, tau=interfaces, rho=executors",
# #             "mode": mode,
# #             "depth_target": depth,
# #             "nodes": nodes,
# #             "edges": edges,
# #             "metrics": {
# #                 "node_count": node_count,
# #                 "edge_count": len(edges),
# #                 "atomicity_ratio": atomicity_ratio,
# #                 "branching_factor_proxy": round(len(edges) / max(1, node_count), 4),
# #             },
# #         }

# #     @classmethod
# #     def _should_promote_recursive(
# #         cls,
# #         *,
# #         decision: ControllerDecision,
# #         combined_text: str,
# #         validation_threshold: int,
# #         max_validation_retries: int,
# #         budget_snapshot: Dict[str, Any],
# #     ) -> Dict[str, Any]:
# #         # Deterministic promotion only in high-assurance setups.
# #         if validation_threshold < 85 or max_validation_retries < 2:
# #             return {"promote": False, "reason": None}

# #         features = cls._complexity_features(combined_text)
# #         complexity_signal = (
# #             decision.estimated_complexity >= 40
# #             or features["token_count"] >= 45
# #             or features["marker_count"] >= 3
# #             or features["has_high_stakes"]
# #         )
# #         confidence_signal = decision.confidence <= 0.86
# #         rationale_l = (decision.rationale or "").lower()
# #         uncertainty_signal = any(
# #             token in rationale_l
# #             for token in ["complex", "uncertain", "risk", "adversarial", "conflicting"]
# #         )
# #         if not complexity_signal:
# #             return {"promote": False, "reason": None}
# #         if not (confidence_signal or uncertainty_signal):
# #             return {"promote": False, "reason": None}

# #         recursive_cost = cls._prior_cost_for_mode("recursive", decision.estimated_complexity)
# #         if not cls._is_mode_feasible(recursive_cost, budget_snapshot):
# #             return {"promote": False, "reason": "recursive_not_budget_feasible"}

# #         return {
# #             "promote": True,
# #             "reason": "deterministic_recursive_promotion_high_threshold",
# #         }

# #     @classmethod
# #     def _is_mode_feasible(
# #         cls,
# #         mode_cost: Dict[str, Any],
# #         budget_snapshot: Dict[str, Any],
# #     ) -> bool:
# #         rem_calls = budget_snapshot.get("remaining_tool_calls")
# #         rem_proxy_tokens = budget_snapshot.get("remaining_token_proxy")
# #         rem_total_tokens = budget_snapshot.get("remaining_total_tokens")
# #         # Prefer true-token budget when available; otherwise fallback to proxy budget.
# #         rem_tokens = rem_total_tokens if isinstance(rem_total_tokens, int) else rem_proxy_tokens
# #         rem_runtime = budget_snapshot.get("remaining_runtime_seconds")
# #         # Keep a risk buffer to reduce over-budget finishes caused by run-time variance.
# #         calls_buffer = 0.9
# #         tokens_buffer = 0.9
# #         runtime_buffer = 0.8
# #         if isinstance(rem_runtime, (int, float)):
# #             if float(rem_runtime) <= 45.0:
# #                 runtime_buffer = 0.7
# #             elif float(rem_runtime) <= 60.0:
# #                 runtime_buffer = 0.75

# #         if isinstance(rem_calls, int) and mode_cost["tool_calls"] > max(0, int(rem_calls * calls_buffer)):
# #             return False
# #         if isinstance(rem_tokens, int) and mode_cost["token_proxy"] > max(0, int(rem_tokens * tokens_buffer)):
# #             return False
# #         if isinstance(rem_runtime, (int, float)) and mode_cost["runtime_seconds"] > max(0.0, float(rem_runtime) * runtime_buffer):
# #             return False
# #         return True

# #     @classmethod
# #     def _best_feasible_mode(
# #         cls,
# #         initial_mode: str,
# #         budget_snapshot: Dict[str, Any],
# #         complexity: int,
# #     ) -> Dict[str, Any]:
# #         costs_by_mode = {
# #             mode: cls._prior_cost_for_mode(mode, complexity) for mode in cls.MODE_ORDER
# #         }
# #         initial_cost = costs_by_mode.get(initial_mode, costs_by_mode["shallow"])
# #         if cls._is_mode_feasible(initial_cost, budget_snapshot):
# #             return {
# #                 "mode_final": initial_mode,
# #                 "override": False,
# #                 "override_reason": None,
# #                 "costs_by_mode": costs_by_mode,
# #             }

# #         for mode in cls.MODE_QUALITY_PRIORITY:
# #             if cls._is_mode_feasible(costs_by_mode[mode], budget_snapshot):
# #                 return {
# #                     "mode_final": mode,
# #                     "override": mode != initial_mode,
# #                     "override_reason": f"budget_feasible_mode_{mode}",
# #                     "costs_by_mode": costs_by_mode,
# #                 }
# #         return {
# #             "mode_final": "direct",
# #             "override": initial_mode != "direct",
# #             "override_reason": "no_mode_fits_budget_forced_direct",
# #             "costs_by_mode": costs_by_mode,
# #         }

# #     def run(self, state: PitchState) -> PitchState:
# #         idea = _sget(state, "idea", "")
# #         refined = _sget(state, "refined_idea", "")
# #         combined = f"{idea}\n{refined}".strip()
# #         validation_threshold = int(_sget(state, "validation_threshold", 70) or 70)
# #         max_validation_retries = int(_sget(state, "max_validation_retries", 1) or 1)
# #         budget_snapshot = self._compute_remaining_budget(state)
# #         usage = _empty_token_usage()
# #         forced_mode = str(_sget(state, "forced_controller_mode", "") or "").strip().lower()
# #         forced_mode_applied = forced_mode in self.MODE_ORDER
# #         if forced_mode_applied:
# #             fallback = self._fallback_decision(combined)
# #             decision = ControllerDecision(
# #                 mode=forced_mode,  # type: ignore[arg-type]
# #                 confidence=1.0,
# #                 rationale=f"Forced controller mode `{forced_mode}` for fixed-policy baseline.",
# #                 estimated_complexity=fallback.estimated_complexity,
# #                 expected_tool_calls_delta=fallback.expected_tool_calls_delta,
# #                 expected_token_proxy_delta=fallback.expected_token_proxy_delta,
# #                 expected_runtime_seconds_delta=fallback.expected_runtime_seconds_delta,
# #                 triggers=["forced_mode_baseline"],
# #             )
# #         else:
# #             try:
# #                 decision, usage = _invoke_structured_with_usage(
# #                     self.structured_llm,
# #                     self.prompt.format_messages(
# #                         idea=idea,
# #                         refined_idea=refined,
# #                         validation_threshold=validation_threshold,
# #                         max_validation_retries=max_validation_retries,
# #                         budget_snapshot=json.dumps(budget_snapshot, indent=2),
# #                     ),
# #                 )
# #             except Exception:  # noqa: BLE001
# #                 decision = self._fallback_decision(combined)
# #         token_usage = _merge_token_usage(state, usage)

# #         chosen_mode_initial = decision.mode
# #         chosen_mode_final = decision.mode
# #         budget_override = False
# #         deterministic_trigger = None
# #         guardrail_trigger = None
# #         scorecard: Dict[str, Any] = {}

# #         if forced_mode_applied:
# #             chosen_mode_initial = forced_mode  # type: ignore[assignment]
# #             chosen_mode_final = forced_mode  # type: ignore[assignment]
# #             calibrated_selected = self._calibrate_selected_cost(
# #                 decision,
# #                 mode_override=chosen_mode_initial,
# #             )
# #             feasible_pick = {
# #                 "mode_final": chosen_mode_final,
# #                 "override": False,
# #                 "override_reason": "forced_mode_no_override",
# #                 "costs_by_mode": {
# #                     mode: self._prior_cost_for_mode(mode, decision.estimated_complexity)
# #                     for mode in self.MODE_ORDER
# #                 },
# #             }
# #             scorecard = self._score_modes(
# #                 idea_text=combined,
# #                 llm_decision=decision,
# #                 budget_snapshot=budget_snapshot,
# #                 validation_threshold=validation_threshold,
# #                 max_validation_retries=max_validation_retries,
# #             )
# #             scorecard["forced_mode_applied"] = True
# #             scorecard["selected_mode_before_forcing"] = scorecard.get("selected_mode")
# #             scorecard["selected_mode"] = chosen_mode_final
# #         else:
# #             scorecard = self._score_modes(
# #                 idea_text=combined,
# #                 llm_decision=decision,
# #                 budget_snapshot=budget_snapshot,
# #                 validation_threshold=validation_threshold,
# #                 max_validation_retries=max_validation_retries,
# #             )
# #             chosen_mode_initial = scorecard["selected_mode"]
# #             if chosen_mode_initial != decision.mode:
# #                 deterministic_trigger = "utility_scorecard_selected_mode"

# #             calibrated_selected = self._calibrate_selected_cost(
# #                 decision,
# #                 mode_override=chosen_mode_initial,
# #             )
# #             feasible_pick = self._best_feasible_mode(
# #                 initial_mode=chosen_mode_initial,
# #                 budget_snapshot=budget_snapshot,
# #                 complexity=decision.estimated_complexity,
# #             )
# #             chosen_mode_final = feasible_pick["mode_final"]
# #             budget_override = feasible_pick["override"]
# #             budget_guard = self._budget_guardrail_mode(budget_snapshot)
# #             if budget_override and feasible_pick["override_reason"]:
# #                 guardrail_trigger = feasible_pick["override_reason"]
# #             if budget_guard is not None and budget_guard != chosen_mode_final:
# #                 chosen_mode_final = budget_guard
# #                 budget_override = True
# #                 guardrail_trigger = f"budget_guardrail_forced_{budget_guard}"

# #         depth_map = {"direct": 0, "shallow": 1, "recursive": 2}
# #         decisions = list(_sget(state, "controller_decisions", []))
# #         tool_audit = list(_sget(state, "tool_audit", []))
# #         trigger_chain = list(decision.triggers)
# #         if deterministic_trigger:
# #             trigger_chain.append(deterministic_trigger)
# #         if guardrail_trigger:
# #             trigger_chain.append(guardrail_trigger)
# #         feature_summary = scorecard.get("features", {}) if isinstance(scorecard, dict) else {}
# #         decomposition_graph = self._build_decomposition_graph(chosen_mode_final)
# #         decisions.append(
# #             {
# #                 "mode_initial": chosen_mode_initial,
# #                 "mode_final": chosen_mode_final,
# #                 "confidence": decision.confidence,
# #                 "estimated_complexity": scorecard.get(
# #                     "estimated_complexity",
# #                     decision.estimated_complexity,
# #                 ),
# #                 "llm_advisory_mode": decision.mode,
# #                 "llm_advisory_complexity": decision.estimated_complexity,
# #                 "structural_complexity": feature_summary.get("structural_complexity"),
# #                 "uncertainty_need": feature_summary.get("uncertainty_need"),
# #                 "utility_margin": scorecard.get("utility_margin"),
# #                 "triggers": trigger_chain,
# #                 "rationale": decision.rationale,
# #                 "expected_tool_calls_delta": calibrated_selected["tool_calls"],
# #                 "expected_token_proxy_delta": calibrated_selected["token_proxy"],
# #                 "expected_runtime_seconds_delta": calibrated_selected["runtime_seconds"],
# #                 "costs_by_mode": feasible_pick["costs_by_mode"],
# #                 "budget_override": budget_override,
# #                 "deterministic_recursive_promotion": bool(deterministic_trigger),
# #                 "forced_mode_applied": forced_mode_applied,
# #             }
# #         )
# #         tool_audit.append(
# #             {
# #                 "agent": "adaptive_controller",
# #                 "tool": "mode_selector",
# #                 "status": "ok",
# #                 "mode_initial": chosen_mode_initial,
# #                 "mode_final": chosen_mode_final,
# #                 "confidence": decision.confidence,
# #                 "estimated_complexity": scorecard.get(
# #                     "estimated_complexity",
# #                     decision.estimated_complexity,
# #                 ),
# #                 "llm_advisory_mode": decision.mode,
# #                 "llm_advisory_complexity": decision.estimated_complexity,
# #                 "structural_complexity": feature_summary.get("structural_complexity"),
# #                 "uncertainty_need": feature_summary.get("uncertainty_need"),
# #                 "utility_margin": scorecard.get("utility_margin"),
# #                 "triggers": trigger_chain,
# #                 "expected_tool_calls_delta": calibrated_selected["tool_calls"],
# #                 "expected_token_proxy_delta": calibrated_selected["token_proxy"],
# #                 "expected_runtime_seconds_delta": calibrated_selected["runtime_seconds"],
# #                 "costs_by_mode": feasible_pick["costs_by_mode"],
# #                 "budget_override": budget_override,
# #                 "deterministic_recursive_promotion": bool(deterministic_trigger),
# #                 "forced_mode_applied": forced_mode_applied,
# #             }
# #         )
# #         return {
# #             "controller_mode": chosen_mode_final,
# #             "controller_mode_initial": chosen_mode_initial,
# #             "controller_budget_override": budget_override,
# #             "controller_confidence": decision.confidence,
# #             "controller_rationale": decision.rationale,
# #             "controller_expected_cost": {
# #                 "expected_tool_calls_delta": calibrated_selected["tool_calls"],
# #                 "expected_token_proxy_delta": calibrated_selected["token_proxy"],
# #                 "expected_runtime_seconds_delta": calibrated_selected["runtime_seconds"],
# #                 "raw_model_estimate": calibrated_selected["raw"],
# #                 "prior_for_mode": calibrated_selected["prior"],
# #                 "costs_by_mode": feasible_pick["costs_by_mode"],
# #                 "mode_for_estimate": chosen_mode_initial,
# #             },
# #             "controller_budget_snapshot": budget_snapshot,
# #             "controller_scorecard": scorecard,
# #             "controller_decisions": decisions,
# #             "decomposition_graph": decomposition_graph,
# #             "decomposition_depth_target": depth_map.get(chosen_mode_final, 1),
# #             "tool_audit": tool_audit,
# #             "token_usage": token_usage,
# #         }


# # class MarketResearchAgent:
# #     def __init__(
# #         self,
# #         llm: ChatOpenAI,
# #         strict_tools: bool = True,
# #         enable_trends: bool = True,
# #     ):
# #         self.structured_llm = llm.with_structured_output(MarketOutput, include_raw=True)
# #         self.keyword_llm = llm.with_structured_output(TrendKeywords, include_raw=True)
# #         self.search_tool = MarketSearchTool()
# #         self.trends_tool = GoogleTrendsTool() if enable_trends else None
# #         self.strict_tools = strict_tools
# #         self.enable_trends = enable_trends
# #         self.keyword_prompt = ChatPromptTemplate.from_messages(
# #             [
# #                 (
# #                     "system",
# #                     "# Role\n"
# #                     "You are a market keyword extractor.\n\n"
# #                     "# Instructions\n"
# #                     "- Extract domain-agnostic trend keywords for startup market research.\n"
# #                     "- Focus on product type, buyer segment, core technology, problem space, and industry context.\n"
# #                     "- Keep keywords short and directly searchable.\n\n"
# #                     "# Output Format\n"
# #                     "- Return content that strictly matches the structured schema fields.",
# #                 ),
# #                 (
# #                     "human",
# #                     "Startup concept:\n{refined_idea}\n\n"
# #                     "Return keyword phrases covering product type, buyer segment, core technology, "
# #                     "problem space, and industry context.",
# #                 ),
# #             ]
# #         )
# #         self.prompt = ChatPromptTemplate.from_messages(
# #             [
# #                 (
# #                     "system",
# #                     "# Role\n"
# #                     "You are a market research analyst.\n\n"
# #                     "# Instructions\n"
# #                     "- Use the provided web research snippets and trend signals.\n"
# #                     "- Synthesize target market, market size, trends, competitors, and differentiation gaps.\n"
# #                     "- Prioritize evidence-grounded statements and avoid speculation.\n\n"
# #                     "# Retry Repair Rules\n"
# #                     "- If retry_count is greater than 0, do targeted claim repair instead of a broad rewrite.\n"
# #                     "- Preserve supported claims from the previous analysis unless new evidence contradicts them.\n"
# #                     "- Repair, narrow, or remove weak and unsupported claims listed in the repair context.\n"
# #                     "- Do not introduce new market-size, CAGR, adoption-rate, or competitor claims unless directly supported by snippets.\n"
# #                     "- Prefer specific, source-grounded wording over impressive but weakly supported numbers.\n\n"
# #                     "# Recursive Mode Rules\n"
# #                     "- If controller_mode is recursive, prioritize reliability over breadth.\n"
# #                     "- Use conservative wording when evidence is indirect or proxy-based.\n"
# #                     "- Avoid very large market-size estimates unless a snippet directly supports the exact market and number.\n"
# #                     "- Prefer narrower adjacent-market framing over unsupported category expansion.\n"
# #                     "- Name competitors only when they appear in snippets or are clearly established in the provided evidence.\n"
# #                     "- Make each section easy for a validator to trace back to the snippets.\n\n"
# #                     "# Output Format\n"
# #                     "- Return content that strictly matches the structured schema fields.",
# #                 ),
# #                 (
# #                     "human",
# #                     "Startup concept:\n{refined_idea}\n\n"
# #                     "Controller mode: {controller_mode}\n"
# #                     "Retry count: {retry_count}\n"
# #                     "Previous validation gaps, if any:\n{validation_gaps}\n\n"
# #                     "Previous market analysis, if this is a retry:\n{previous_market_analysis}\n\n"
# #                     "Targeted repair context, if this is a retry:\n{repair_context}\n\n"
# #                     "Web research snippets:\n{search_results}\n\n"
# #                     "Keyword trend signals:\n{trend_signals}\n\n"
# #                     "Produce market analysis for investors. If this is a retry, repair the previous "
# #                     "analysis directly: keep what is supported, fix what is weak, and avoid adding "
# #                     "new unsupported claims.",
# #                 ),
# #             ]
# #         )

# #     def run(self, state: PitchState) -> PitchState:
# #         refined_idea = _sget(state, "refined_idea", "")
# #         total_usage = _empty_token_usage()
# #         retry_count = _sget(state, "retry_count", 0)
# #         is_retry = int(retry_count or 0) > 0
# #         controller_mode = str(_sget(state, "controller_mode", "shallow") or "shallow")
# #         prior_validation = _sget(state, "validation_report", {}) or {}
# #         repair_context = _sget(state, "market_repair_context", {}) or {}
# #         previous_market_analysis = ""
# #         if isinstance(repair_context, dict):
# #             previous_market_analysis = str(
# #                 repair_context.get("previous_market_analysis", "") or ""
# #             )
# #         if not previous_market_analysis:
# #             previous_market_analysis = str(_sget(state, "market_analysis", "") or "")
# #         validation_gaps = ""
# #         if isinstance(prior_validation, dict):
# #             validation_gaps = str(prior_validation.get("evidence_gaps", "") or "")
# #             weak_claims = [
# #                 str(claim.get("claim", "")).strip()
# #                 for claim in prior_validation.get("claims", []) or []
# #                 if str(claim.get("verdict", "")).strip().lower()
# #                 in {"weakly_supported", "unsupported"}
# #             ]
# #             if weak_claims:
# #                 validation_gaps = (
# #                     f"{validation_gaps}\nWeak or unsupported claims to repair:\n"
# #                     + "\n".join(f"- {claim}" for claim in weak_claims[:5])
# #                 ).strip()
# #         if is_retry:
# #             weak_repair_claims = []
# #             repair_source_hints = []
# #             if isinstance(repair_context, dict):
# #                 weak_repair_claims = [
# #                     str(item.get("claim", "")).strip()
# #                     for item in repair_context.get("weak_or_unsupported_claims", []) or []
# #                     if isinstance(item, dict) and str(item.get("claim", "")).strip()
# #                 ]
# #                 repair_source_hints = [
# #                     str(url).strip()
# #                     for item in repair_context.get("weak_or_unsupported_claims", []) or []
# #                     if isinstance(item, dict)
# #                     for url in (item.get("supporting_sources", []) or [])
# #                     if str(url).strip()
# #                 ]
# #             repair_focus = "; ".join(weak_repair_claims[:3]) or validation_gaps[:300]
# #             query_suffix = (
# #                 " authoritative recent evidence to verify, narrow, or correct weak claims: "
# #                 f"{repair_focus[:420]} "
# #                 f"{' '.join(repair_source_hints[:2])[:220]}"
# #             )
# #         else:
# #             query_suffix = ""
# #         query = f"startup market size competitors trends for: {refined_idea}{query_suffix}"
# #         search_payload = self.search_tool.search(query)
# #         if self.strict_tools and search_payload["status"] != "ok":
# #             raise RuntimeError(f"Market search tool unavailable: {search_payload['error']}")
# #         evidence_budget = _evidence_budget_for_mode(
# #             _sget(state, "controller_mode", "shallow"),
# #             _sget(state, "budget_remaining", {}).get("total_tokens")
# #             if isinstance(_sget(state, "budget_remaining", {}), dict)
# #             else None,
# #         )
# #         if is_retry:
# #             previous_evidence = list(_sget(state, "market_evidence", []) or [])
# #             new_evidence = list(search_payload.get("results", []) or [])
# #             focused_payload: Dict[str, Any] = {
# #                 "status": "skipped",
# #                 "query": "",
# #                 "results": [],
# #                 "sources": [],
# #                 "dropped_sources": 0,
# #                 "error": "",
# #             }
# #             focused_query = ""
# #             if controller_mode.strip().lower() == "recursive" and (
# #                 weak_repair_claims or validation_gaps
# #             ):
# #                 focused_query = (
# #                     "authoritative market evidence for "
# #                     f"{refined_idea}; verify these weak claims or replace with supported facts: "
# #                     f"{('; '.join(weak_repair_claims[:2]) or validation_gaps)[:520]}"
# #                 )
# #                 focused_payload = self.search_tool.search(focused_query)
# #                 if focused_payload["status"] == "ok":
# #                     new_evidence = _merge_evidence_items(
# #                         list(focused_payload.get("results", []) or [])[:5],
# #                         new_evidence,
# #                     )
# #             # Prioritize focused repair evidence, while retaining first-pass
# #             # snippets so supported claims do not lose their source context.
# #             combined_results = _merge_evidence_items(
# #                 new_evidence[:5],
# #                 previous_evidence + new_evidence[5:],
# #             )
# #             search_payload = {
# #                 **search_payload,
# #                 "results": combined_results,
# #                 "sources": [
# #                     str(item.get("url", "")).strip()
# #                     for item in combined_results
# #                     if isinstance(item, dict) and str(item.get("url", "")).strip()
# #                 ],
# #                 "source_count_raw": len(combined_results),
# #                 "focused_repair_query": focused_query,
# #                 "focused_repair_status": focused_payload.get("status"),
# #                 "focused_repair_source_count": len(focused_payload.get("sources", []) or []),
# #                 "focused_repair_error": focused_payload.get("error", ""),
# #                 "focused_repair_dropped_sources": focused_payload.get("dropped_sources", 0),
# #             }
# #         search_payload = _limited_search_payload(search_payload, **evidence_budget)
# #         search_results = search_payload["results_json"]
# #         if is_retry:
# #             previous_trends = _sget(state, "trend_signals", {}) or {}
# #             extracted_keywords = list(previous_trends.get("keywords", []) or [])[:5]
# #             if not extracted_keywords:
# #                 extracted_keywords = _fallback_keywords(refined_idea)
# #             trend_payload = {
# #                 **previous_trends,
# #                 "status": "reused" if previous_trends else "skipped",
# #                 "keywords": extracted_keywords[:5],
# #                 "error": (
# #                     "Reused first-pass trend signals during targeted repair."
# #                     if previous_trends
# #                     else "Skipped trend lookup during targeted repair."
# #                 ),
# #             }
# #         else:
# #             try:
# #                 kw_model, kw_usage = _invoke_structured_with_usage(
# #                     self.keyword_llm,
# #                     self.keyword_prompt.format_messages(refined_idea=refined_idea),
# #                 )
# #                 total_usage = _merge_token_usage({"token_usage": total_usage}, kw_usage)
# #                 extracted_keywords = [k.strip() for k in kw_model.keywords if k.strip()]
# #             except Exception:  # noqa: BLE001
# #                 extracted_keywords = _fallback_keywords(refined_idea)
# #             trend_payload = {}

# #         if (not is_retry) and self.enable_trends and self.trends_tool is not None:
# #             trend_payload = self.trends_tool.fetch(extracted_keywords[:5])
# #         elif not is_retry:
# #             trend_payload = {
# #                 "status": "skipped",
# #                 "keywords": extracted_keywords[:5],
# #                 "data": {},
# #                 "error": "Google Trends disabled by configuration.",
# #             }

# #         result, market_usage = _invoke_structured_with_usage(
# #             self.structured_llm,
# #             self.prompt.format_messages(
# #                 refined_idea=refined_idea,
# #                 controller_mode=controller_mode,
# #                 retry_count=retry_count,
# #                 validation_gaps=validation_gaps or "None.",
# #                 previous_market_analysis=previous_market_analysis or "None.",
# #                 repair_context=(
# #                     json.dumps(repair_context, indent=2) if repair_context else "None."
# #                 ),
# #                 search_results=search_results,
# #                 trend_signals=json.dumps(trend_payload, indent=2),
# #             ),
# #         )
# #         total_usage = _merge_token_usage({"token_usage": total_usage}, market_usage)
# #         token_usage = _merge_token_usage(state, total_usage)
# #         market_analysis = (
# #             f"Target Market: {result.target_market}\n"
# #             f"Market Size: {result.market_size}\n"
# #             f"Trends: {result.trends}\n"
# #             f"Competitors: {result.competitors}\n"
# #             f"Differentiation Gaps: {result.differentiation_gaps}"
# #         )
# #         tool_audit = list(_sget(state, "tool_audit", []))
# #         tool_audit.append(
# #             {
# #                 "agent": "market_research",
# #                 "tool": "linkup_search",
# #                 "query": query,
# #                 "status": search_payload["status"],
# #                 "source_count": len(search_payload["sources"]),
# #                 "source_count_raw": search_payload.get("source_count_raw"),
# #                 "evidence_limit": search_payload.get("evidence_limit"),
# #                 "dropped_sources": search_payload.get("dropped_sources", 0),
# #                 "error": search_payload["error"],
# #                 "repair_mode": is_retry,
# #                 "repair_context": {
# #                     "weak_claim_count": len(
# #                         repair_context.get("weak_or_unsupported_claims", [])
# #                         if isinstance(repair_context, dict)
# #                         else []
# #                     ),
# #                     "supported_claim_count": len(
# #                         repair_context.get("supported_claims", [])
# #                         if isinstance(repair_context, dict)
# #                         else []
# #                     ),
# #                     "reused_previous_evidence": is_retry,
# #                     "focused_repair_status": search_payload.get("focused_repair_status"),
# #                     "focused_repair_source_count": search_payload.get(
# #                         "focused_repair_source_count",
# #                         0,
# #                     ),
# #                 },
# #             }
# #         )
# #         if is_retry and search_payload.get("focused_repair_query"):
# #             tool_audit.append(
# #                 {
# #                     "agent": "market_research",
# #                     "tool": "focused_repair_search",
# #                     "query": search_payload.get("focused_repair_query"),
# #                     "status": search_payload.get("focused_repair_status", "skipped"),
# #                     "source_count": search_payload.get("focused_repair_source_count", 0),
# #                     "dropped_sources": search_payload.get(
# #                         "focused_repair_dropped_sources",
# #                         0,
# #                     ),
# #                     "error": search_payload.get("focused_repair_error", ""),
# #                 }
# #             )
# #         tool_audit.append(
# #             {
# #                 "agent": "market_research",
# #                 "tool": "google_trends",
# #                 "status": trend_payload["status"],
# #                 "keyword_count": len(trend_payload.get("keywords", [])),
# #                 "keywords": trend_payload.get("keywords", []),
# #                 "error": trend_payload.get("error", ""),
# #             }
# #         )
# #         return {
# #             "market_analysis": market_analysis,
# #             "market_sources": search_payload["sources"],
# #             "market_evidence": search_payload["results"],
# #             "trend_signals": trend_payload,
# #             "tool_audit": tool_audit,
# #             "token_usage": token_usage,
# #         }


# # class ClaimRepairAgent:
# #     """Lightweight recursive repair focused only on weak validator claims."""

# #     def __init__(self, llm: ChatOpenAI, strict_tools: bool = True):
# #         self.structured_llm = llm.with_structured_output(MarketRepairOutput, include_raw=True)
# #         self.search_tool = MarketSearchTool()
# #         self.strict_tools = strict_tools
# #         self.prompt = ChatPromptTemplate.from_messages(
# #             [
# #                 (
# #                     "system",
# #                     "# Role\n"
# #                     "You are a claim-level market-analysis repair agent.\n\n"
# #                     "# Goal\n"
# #                     "Patch a previous market analysis using only focused evidence for weak "
# #                     "or unsupported claims. This is not a full rewrite. The objective is "
# #                     "higher validation reliability, not more persuasive wording.\n\n"
# #                     "# Instructions\n"
# #                     "- Preserve supported claims and the original section structure.\n"
# #                     "- Do not rewrite sections that are not connected to a weak/unsupported claim.\n"
# #                     "- Keep supported claims semantically unchanged unless focused evidence directly contradicts them.\n"
# #                     "- Follow the repair plan action for each weak/unsupported claim.\n"
# #                     "- For `search_and_replace`, use focused snippets to replace the claim with a directly supported claim.\n"
# #                     "- For `qualify_or_remove`, do not try to turn a speculative gap/opportunity claim into a broad trend claim.\n"
# #                     "- For `remove`, remove the claim from factual market analysis rather than rewriting it creatively.\n"
# #                     "- Prefer conservative source-grounded wording over larger unsupported numbers.\n"
# #                     "- Do not introduce new market-size, CAGR, adoption-rate, or competitor claims "
# #                     "unless a focused snippet directly supports them.\n"
# #                     "- If evidence is indirect, explicitly frame it as adjacent/proxy evidence or a hypothesis.\n"
# #                     "- Avoid phrases like `there is a market trend toward` unless the snippet directly states that trend.\n"
# #                     "- Make every revised claim easy for the validator to trace to the snippets.\n\n"
# #                     "# Output Format\n"
# #                     "- Return content that strictly matches the structured schema fields.",
# #                 ),
# #                 (
# #                     "human",
# #                     "Startup concept:\n{refined_idea}\n\n"
# #                     "Previous market analysis:\n{previous_market_analysis}\n\n"
# #                     "Repair context from validator:\n{repair_context}\n\n"
# #                     "Repair plan:\n{repair_plan}\n\n"
# #                     "Supported claims that should be preserved:\n{supported_claims}\n\n"
# #                     "Focused evidence snippets:\n{focused_evidence}\n\n"
# #                     "Return the minimally repaired market analysis and patch metadata. If the focused "
# #                     "evidence does not directly support a stronger replacement, narrow or remove the "
# #                     "weak claim rather than adding a new speculative claim.",
# #                 ),
# #             ]
# #         )

# #     @staticmethod
# #     def _weak_claim_texts(repair_context: Dict[str, Any]) -> List[str]:
# #         return [
# #             str(item.get("claim", "")).strip()
# #             for item in repair_context.get("weak_or_unsupported_claims", []) or []
# #             if isinstance(item, dict) and str(item.get("claim", "")).strip()
# #         ]

# #     def run(self, state: PitchState) -> PitchState:
# #         refined_idea = _sget(state, "refined_idea", "")
# #         repair_context = _sget(state, "market_repair_context", {}) or {}
# #         previous_market_analysis = ""
# #         if isinstance(repair_context, dict):
# #             previous_market_analysis = str(
# #                 repair_context.get("previous_market_analysis", "") or ""
# #             )
# #         if not previous_market_analysis:
# #             previous_market_analysis = str(_sget(state, "market_analysis", "") or "")

# #         weak_claims = (
# #             self._weak_claim_texts(repair_context)
# #             if isinstance(repair_context, dict)
# #             else []
# #         )
# #         repair_plan = []
# #         supported_claims = []
# #         if isinstance(repair_context, dict):
# #             raw_plan = repair_context.get("repair_plan", []) or []
# #             if isinstance(raw_plan, list):
# #                 repair_plan = [item for item in raw_plan if isinstance(item, dict)]
# #             supported_claims = [
# #                 item
# #                 for item in repair_context.get("supported_claims", []) or []
# #                 if isinstance(item, dict)
# #             ]
# #         if not repair_plan:
# #             repair_plan = build_claim_repair_plan(
# #                 repair_context.get("weak_or_unsupported_claims", [])
# #                 if isinstance(repair_context, dict)
# #                 else []
# #             )

# #         search_targets = [
# #             item for item in repair_plan if item.get("action") == "search_and_replace"
# #         ]
# #         qualify_targets = [
# #             item
# #             for item in repair_plan
# #             if item.get("action") in {"qualify_or_remove", "remove"}
# #         ]
# #         if not repair_plan:
# #             tool_audit = list(_sget(state, "tool_audit", []))
# #             tool_audit.append(
# #                 {
# #                     "agent": "claim_repair",
# #                     "tool": "claim_micro_repair",
# #                     "status": "skipped_no_target_claims",
# #                     "weak_claim_count": 0,
# #                     "source_count": 0,
# #                     "prompt_tokens": 0,
# #                     "completion_tokens": 0,
# #                     "total_tokens": 0,
# #                 }
# #             )
# #             return {
# #                 "tool_audit": tool_audit,
# #                 "market_analysis": previous_market_analysis,
# #                 "market_sources": list(_sget(state, "market_sources", []) or []),
# #                 "market_evidence": list(_sget(state, "market_evidence", []) or []),
# #                 "token_usage": _sget(state, "token_usage", {}) or _empty_token_usage(),
# #             }

# #         query = ""
# #         previous_evidence = list(_sget(state, "market_evidence", []) or [])
# #         claim_evidence: Dict[str, List[Dict[str, Any]]] = {}
# #         focus = "; ".join(str(item.get("claim", "")).strip() for item in search_targets[:3])
# #         if focus:
# #             focused_evidence_items: List[Dict[str, Any]] = []
# #             query_statuses: List[Dict[str, Any]] = []
# #             raw_source_count = 0
# #             dropped_sources = 0
# #             errors: List[str] = []
# #             for target in search_targets[:3]:
# #                 claim_id = str(target.get("claim_id") or f"claim_{len(claim_evidence) + 1}")
# #                 claim_text = str(target.get("claim") or target.get("claim_text") or "").strip()
# #                 queries = [
# #                     str(q).strip()
# #                     for q in (target.get("candidate_queries") or [])
# #                     if str(q).strip()
# #                 ][:2]
# #                 if not queries and claim_text:
# #                     queries = [
# #                         f"{claim_text} authoritative source",
# #                         f"{refined_idea} {claim_text} recent evidence",
# #                     ]
# #                 payload = self.search_tool.search_many(queries[:2], max_results_per_query=2)
# #                 if payload.get("status") != "ok" and payload.get("error"):
# #                     errors.append(str(payload.get("error")))
# #                 limited = _limited_search_payload(
# #                     payload,
# #                     max_items=3,
# #                     content_chars=260,
# #                 )
# #                 claim_rows = []
# #                 for row in limited["results"]:
# #                     row = dict(row)
# #                     row["claim_id"] = claim_id
# #                     claim_rows.append(row)
# #                     focused_evidence_items.append(row)
# #                 claim_evidence[claim_id] = claim_rows
# #                 query_statuses.extend(payload.get("query_statuses", []) or [])
# #                 raw_source_count += int(payload.get("source_count_raw", len(payload.get("sources", []) or [])) or 0)
# #                 dropped_sources += int(payload.get("dropped_sources", 0) or 0)
# #             focused_evidence_items = _limit_evidence_items(
# #                 focused_evidence_items,
# #                 max_items=6,
# #                 content_chars=260,
# #             )
# #             sources = [item["url"] for item in focused_evidence_items if item.get("url")]
# #             query = " | ".join(item.get("query", "") for item in query_statuses if item.get("query"))
# #             focused_payload = {
# #                 "status": "ok" if focused_evidence_items else "error",
# #                 "query": query,
# #                 "results": focused_evidence_items,
# #                 "sources": sources,
# #                 "source_count_raw": raw_source_count,
# #                 "evidence_limit": {
# #                     "max_items": 6,
# #                     "content_chars": 260,
# #                     "raw_items": raw_source_count,
# #                     "used_items": len(focused_evidence_items),
# #                 },
# #                 "dropped_sources": dropped_sources,
# #                 "error": "; ".join(errors) if errors else "",
# #                 "query_statuses": query_statuses,
# #             }
# #             if self.strict_tools and focused_payload["status"] != "ok":
# #                 raise RuntimeError(f"Claim repair search unavailable: {focused_payload['error']}")
# #         else:
# #             # Gap/opportunity repairs usually fail when the model tries to find
# #             # proof for an underserved niche. Use the existing evidence corpus
# #             # and force a conservative qualify/remove patch instead.
# #             focused_payload = {
# #                 "status": "skipped_qualify_or_remove",
# #                 "query": "",
# #                 "results": _limit_evidence_items(
# #                     previous_evidence,
# #                     max_items=6,
# #                     content_chars=260,
# #                 ),
# #                 "sources": [],
# #                 "source_count_raw": 0,
# #                 "evidence_limit": {
# #                     "max_items": 6,
# #                     "content_chars": 260,
# #                     "raw_items": len(previous_evidence),
# #                     "used_items": min(6, len(previous_evidence)),
# #                 },
# #                 "dropped_sources": 0,
# #                 "error": "",
# #             }
# #             focused_evidence_items = focused_payload["results"]
# #             claim_evidence = {
# #                 str(item.get("claim_id") or f"claim_{idx}"): focused_evidence_items
# #                 for idx, item in enumerate(repair_plan, start=1)
# #             }
# #         focused_evidence = json.dumps(focused_evidence_items, indent=2)
# #         result, repair_usage = _invoke_structured_with_usage(
# #             self.structured_llm,
# #             self.prompt.format_messages(
# #                 refined_idea=refined_idea,
# #                 previous_market_analysis=previous_market_analysis or "None.",
# #                 repair_context=(
# #                     json.dumps(repair_context, indent=2) if repair_context else "None."
# #                 ),
# #                 focused_evidence=focused_evidence,
# #                 repair_plan=json.dumps(repair_plan, indent=2),
# #                 supported_claims=json.dumps(supported_claims[:8], indent=2),
# #             ),
# #         )
# #         token_usage = _merge_token_usage(state, repair_usage)
# #         market_analysis = (
# #             f"Target Market: {result.target_market}\n"
# #             f"Market Size: {result.market_size}\n"
# #             f"Trends: {result.trends}\n"
# #             f"Competitors: {result.competitors}\n"
# #             f"Differentiation Gaps: {result.differentiation_gaps}"
# #         )

# #         combined_evidence = _limit_evidence_items(
# #             _merge_evidence_items(focused_payload["results"], previous_evidence),
# #             max_items=8,
# #             content_chars=260,
# #         )
# #         market_sources = [
# #             str(item.get("url", "")).strip()
# #             for item in combined_evidence
# #             if isinstance(item, dict) and str(item.get("url", "")).strip()
# #         ]

# #         tool_audit = list(_sget(state, "tool_audit", []))
# #         tool_audit.append(
# #             {
# #                 "agent": "claim_repair",
# #                 "tool": "focused_repair_search",
# #                 "query": query,
# #                 "status": focused_payload["status"],
# #                 "source_count": len(focused_payload["sources"]),
# #                 "source_count_raw": focused_payload.get("source_count_raw"),
# #                 "evidence_limit": focused_payload.get("evidence_limit"),
# #                 "dropped_sources": focused_payload.get("dropped_sources", 0),
# #                 "error": focused_payload["error"],
# #                 "search_target_count": len(search_targets),
# #                 "qualify_or_remove_target_count": len(qualify_targets),
# #             }
# #         )
# #         tool_audit.append(
# #             {
# #                 "agent": "claim_repair",
# #                 "tool": "claim_micro_repair",
# #                 "status": "ok",
# #                 "weak_claim_count": len(weak_claims),
# #                 "source_count": len(focused_payload["sources"]),
# #                 "combined_source_count": len(market_sources),
# #                 "repair_action_counts": {
# #                     "search_and_replace": len(search_targets),
# #                     "qualify_or_remove": sum(
# #                         1 for item in repair_plan if item.get("action") == "qualify_or_remove"
# #                     ),
# #                     "remove": sum(1 for item in repair_plan if item.get("action") == "remove"),
# #                 },
# #                 "patch_summary": result.patch_summary,
# #                 "patches": [patch.model_dump() for patch in result.patches],
# #                 "prompt_tokens": repair_usage.get("prompt_tokens", 0),
# #                 "completion_tokens": repair_usage.get("completion_tokens", 0),
# #                 "total_tokens": repair_usage.get("total_tokens", 0),
# #             }
# #         )
# #         patch_rows = [patch.model_dump() for patch in result.patches]

# #         return {
# #             "market_analysis": market_analysis,
# #             "market_sources": market_sources,
# #             "market_evidence": combined_evidence,
# #             "claim_evidence": claim_evidence,
# #             "repair_plan": {"actions": repair_plan},
# #             "repair_patches": patch_rows,
# #             "tool_audit": tool_audit,
# #             "token_usage": token_usage,
# #         }


# # class SourceValidatorAgent:
# #     MIN_VALIDATION_CLAIMS = 3
# #     LOW_CLAIM_PENALTY_PER_MISSING_CLAIM = 5

# #     RUBRIC_WEIGHTS = {
# #         "evidence_grounding": 0.35,
# #         "source_credibility": 0.35,
# #         "claim_specificity": 0.15,
# #         "internal_consistency": 0.15,
# #     }

# #     def __init__(
# #         self,
# #         llm: ChatOpenAI,
# #         secondary_judge_llm: Optional[ChatOpenAI] = None,
# #     ):
# #         judge_b_llm = secondary_judge_llm or llm
# #         self.judge_a_model = self._model_name(llm)
# #         self.judge_b_model = self._model_name(judge_b_llm)
# #         self.cross_model_judging = self.judge_a_model != self.judge_b_model
# #         self.primary_judge_llm = llm.with_structured_output(
# #             JudgeValidationOutput, include_raw=True
# #         )
# #         self.secondary_judge_llm = judge_b_llm.with_structured_output(
# #             SecondJudgeOutput, include_raw=True
# #         )
# #         self.repair_judge_llm = llm.with_structured_output(
# #             RepairPatchValidationOutput, include_raw=True
# #         )
# #         self.primary_prompt = ChatPromptTemplate.from_messages(
# #             [
# #                 (
# #                     "system",
# #                     "# Role\n"
# #                     "You are Judge A, a strict source validation analyst.\n\n"
# #                     "# Instructions\n"
# #                     "- Verify material market claims only against provided evidence snippets and URLs.\n"
# #                     "- Do not invent sources or unsupported claims.\n"
# #                     "- Validate at least 3 distinct material claims when possible.\n"
# #                     "- Prefer claims covering target market, market size/trends, competitors, and differentiation.\n"
# #                     "- Provide claim-level verdicts, confidence, rationale, and supporting sources.\n"
# #                     "- Score rubric dimensions (0-100): evidence grounding, source credibility, claim specificity, internal consistency.\n"
# #                     "- Treat evidence grounding and source credibility as the most important rubric dimensions.\n"
# #                     "- Provide overall reliability (0-100).\n\n"
# #                     "# Output Format\n"
# #                     "- Return content that strictly matches the structured schema fields.",
# #                 ),
# #                 (
# #                     "human",
# #                     "Startup concept:\n{refined_idea}\n\n"
# #                     "Market analysis draft:\n{market_analysis}\n\n"
# #                     "Evidence snippets (title/url/content):\n{market_evidence}\n\n"
# #                     "Return a claim-by-claim validation report with confidence and source links.",
# #                 ),
# #             ]
# #         )
# #         self.secondary_prompt = ChatPromptTemplate.from_messages(
# #             [
# #                 (
# #                     "system",
# #                     "# Role\n"
# #                     "You are Judge B, an independent methodology-focused reviewer.\n\n"
# #                     "# Instructions\n"
# #                     "- Independently evaluate each provided claim against the same evidence.\n"
# #                     "- Do not copy Judge A labels blindly; reassess verdict and confidence per claim.\n"
# #                     "- If fewer than 3 claims are provided, treat claim coverage as weak and reflect that in the rubric.\n"
# #                     "- Score rubric dimensions (0-100): evidence grounding, source credibility, claim specificity, internal consistency.\n"
# #                     "- Treat evidence grounding and source credibility as the most important rubric dimensions.\n"
# #                     "- Provide overall reliability (0-100) and evidence gaps.\n\n"
# #                     "# Output Format\n"
# #                     "- Return content that strictly matches the structured schema fields.",
# #                 ),
# #                 (
# #                     "human",
# #                     "Startup concept:\n{refined_idea}\n\n"
# #                     "Market analysis draft:\n{market_analysis}\n\n"
# #                     "Evidence snippets (title/url/content):\n{market_evidence}\n\n"
# #                     "Claims to evaluate:\n{claims_to_review}\n\n"
# #                     "Return independent claim assessments and rubric scores.",
# #                 ),
# #             ]
# #         )
# #         self.repair_prompt = ChatPromptTemplate.from_messages(
# #             [
# #                 (
# #                     "system",
# #                     "# Role\n"
# #                     "You are a focused repair validator.\n\n"
# #                     "# Instructions\n"
# #                     "- Validate only the repaired claims produced by the claim-repair agent.\n"
# #                     "- Do not rejudge supported claims that were preserved from the prior validation.\n"
# #                     "- Treat qualified/removal patches favorably only if they reduce unsupported factual risk.\n"
# #                     "- If a replacement is framed as a hypothesis, limitation, or adjacent-evidence claim, do not mark it as fully supported unless snippets directly support it.\n"
# #                     "- Return claim-level verdicts, confidence, rationale, and source links.\n"
# #                     "- Keep the assessment strict but cheaper than full dual-judge validation.\n\n"
# #                     "# Output Format\n"
# #                     "- Return content that strictly matches the structured schema fields.",
# #                 ),
# #                 (
# #                     "human",
# #                     "Startup concept:\n{refined_idea}\n\n"
# #                     "Previous market analysis:\n{previous_market_analysis}\n\n"
# #                     "Repaired market analysis:\n{market_analysis}\n\n"
# #                     "Repair plan:\n{repair_plan}\n\n"
# #                     "Patch metadata from repair agent:\n{patches}\n\n"
# #                     "Evidence snippets:\n{market_evidence}\n\n"
# #                     "Validate only the repaired or replacement claims. If a weak claim was removed, "
# #                     "return a conservative assessment explaining that unsupported factual risk was removed.",
# #                 ),
# #             ]
# #         )

# #     @staticmethod
# #     def _model_name(llm: ChatOpenAI) -> str:
# #         params = getattr(llm, "_default_params", {}) or {}
# #         return str(
# #             getattr(llm, "model_name", None)
# #             or getattr(llm, "model", None)
# #             or params.get("model")
# #             or "unknown"
# #         )

# #     @staticmethod
# #     def _rubric_mean(rubric: Dict[str, Any]) -> float:
# #         """Evidence-weighted rubric score for source-validation reliability."""
# #         weighted_total = 0.0
# #         weight_total = 0.0
# #         for key, weight in SourceValidatorAgent.RUBRIC_WEIGHTS.items():
# #             try:
# #                 value = float(rubric.get(key, 0))
# #             except (TypeError, ValueError):
# #                 value = 0.0
# #             weighted_total += max(0.0, min(100.0, value)) * weight
# #             weight_total += weight
# #         if weight_total <= 0:
# #             return 0.0
# #         return weighted_total / weight_total

# #     @staticmethod
# #     def _clamp_score(value: float) -> int:
# #         return int(max(0, min(100, round(value))))

# #     @staticmethod
# #     def _judge_summary(
# #         report: Dict[str, Any],
# #         det_score: int,
# #         usage: Dict[str, int],
# #     ) -> Dict[str, Any]:
# #         return {
# #             "overall_reliability": int(report.get("overall_reliability", 0) or 0),
# #             "deterministic_claim_score": int(det_score),
# #             "rubric_scores": dict(report.get("rubric_scores", {}) or {}),
# #             "rubric_mean": round(
# #                 SourceValidatorAgent._rubric_mean(report.get("rubric_scores", {}) or {}),
# #                 3,
# #             ),
# #             "claim_count": len(report.get("claims", []) or report.get("claim_assessments", [])),
# #             "token_usage": usage,
# #         }

# #     @staticmethod
# #     def _claim_ledger(
# #         claims: List[Dict[str, Any]],
# #         evidence_gaps: str = "",
# #     ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
# #         """Create stable, repairable claim units from judge output."""
# #         claim_units: List[Dict[str, Any]] = []
# #         failing_claims: List[Dict[str, Any]] = []
# #         failure_counts: Dict[str, int] = {}
# #         material_unsupported = 0
# #         verifiable_count = 0
# #         for idx, claim in enumerate(claims or [], start=1):
# #             if not isinstance(claim, dict):
# #                 continue
# #             text = str(claim.get("claim", "") or "").strip()
# #             if not text:
# #                 continue
# #             category = _claim_repair_category(text)
# #             verdict = str(claim.get("verdict", "") or "").strip().lower()
# #             failure_type = _claim_failure_type(claim, evidence_gaps)
# #             materiality = _claim_materiality(text, category)
# #             unit = {
# #                 "claim_id": f"c{idx}",
# #                 "claim_text": text,
# #                 "claim": text,
# #                 "span_text": text,
# #                 "span_start": None,
# #                 "span_end": None,
# #                 "category": category,
# #                 "verdict": verdict,
# #                 "confidence": claim.get("confidence"),
# #                 "materiality": materiality,
# #                 "verifiable": verdict != "needs_review",
# #                 "failure_type": failure_type,
# #                 "failure_type_hint": failure_type,
# #                 "supporting_sources": claim.get("supporting_sources", []),
# #                 "rationale": str(claim.get("rationale", "") or "").strip()[:300],
# #                 "candidate_queries": [
# #                     f"{text} authoritative source",
# #                     f"{text} recent evidence",
# #                 ],
# #             }
# #             claim_units.append(unit)
# #             if unit["verifiable"]:
# #                 verifiable_count += 1
# #             if verdict in {"weakly_supported", "unsupported", "needs_review"}:
# #                 failing_claims.append(unit)
# #                 failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
# #                 if materiality >= 4:
# #                     material_unsupported += 1
# #         summary = {
# #             "verifiable_claim_count": verifiable_count,
# #             "failing_claim_count": len(failing_claims),
# #             "unsupported_material_claim_count": material_unsupported,
# #             "failure_type_counts": failure_counts,
# #         }
# #         return claim_units, failing_claims, summary

# #     @staticmethod
# #     def _agreement_stats(
# #         primary_claims: List[Dict[str, Any]],
# #         secondary_claims: List[Dict[str, Any]],
# #         rubric_a: Dict[str, Any],
# #         rubric_b: Dict[str, Any],
# #         score_a: int,
# #         score_b: int,
# #     ) -> Dict[str, Any]:
# #         rubric_dims = [
# #             "evidence_grounding",
# #             "source_credibility",
# #             "claim_specificity",
# #             "internal_consistency",
# #         ]
# #         rubric_diffs: Dict[str, float] = {}
# #         for dim in rubric_dims:
# #             va = float(rubric_a.get(dim, 0) or 0)
# #             vb = float(rubric_b.get(dim, 0) or 0)
# #             rubric_diffs[dim] = round(abs(va - vb), 3)

# #         primary_map = {
# #             str(c.get("claim", "")).strip().lower(): str(c.get("verdict", "")).strip().lower()
# #             for c in primary_claims
# #             if str(c.get("claim", "")).strip()
# #         }
# #         secondary_map = {
# #             str(c.get("claim", "")).strip().lower(): str(c.get("verdict", "")).strip().lower()
# #             for c in secondary_claims
# #             if str(c.get("claim", "")).strip()
# #         }
# #         shared = sorted(set(primary_map.keys()) & set(secondary_map.keys()))
# #         verdict_matches = 0
# #         for key in shared:
# #             if primary_map.get(key) == secondary_map.get(key):
# #                 verdict_matches += 1
# #         verdict_agreement = (
# #             round(verdict_matches / len(shared), 4) if shared else 0.0
# #         )

# #         score_delta = abs(int(score_a) - int(score_b))
# #         score_agreement = round(max(0.0, 1.0 - (score_delta / 100.0)), 4)
# #         rubric_mae = (
# #             round(sum(rubric_diffs.values()) / max(1, len(rubric_diffs)), 4)
# #             if rubric_diffs
# #             else 0.0
# #         )
# #         rubric_agreement = round(max(0.0, 1.0 - (rubric_mae / 100.0)), 4)
# #         overall_agreement = round(
# #             (score_agreement + rubric_agreement + verdict_agreement) / 3.0, 4
# #         )
# #         return {
# #             "score_delta_abs": score_delta,
# #             "score_agreement": score_agreement,
# #             "rubric_mae": rubric_mae,
# #             "rubric_diffs": rubric_diffs,
# #             "rubric_agreement": rubric_agreement,
# #             "shared_claims": len(shared),
# #             "verdict_agreement_rate": verdict_agreement,
# #             "overall_agreement": overall_agreement,
# #         }

# #     def run(self, state: PitchState) -> PitchState:
# #         token_usage_current = _sget(state, "token_usage", {}) or {}
# #         max_total_tokens = _sget(state, "max_total_tokens")
# #         total_tokens_current = int(token_usage_current.get("total_tokens", 0) or 0)
# #         remaining_total_tokens = (
# #             int(max_total_tokens) - total_tokens_current
# #             if isinstance(max_total_tokens, int)
# #             else None
# #         )
# #         evidence_budget = _evidence_budget_for_mode(
# #             _sget(state, "controller_mode", "shallow"),
# #             remaining_total_tokens,
# #         )
# #         raw_evidence = _sget(state, "market_evidence", []) or []
# #         validation_evidence = _limit_evidence_items(raw_evidence, **evidence_budget)
# #         evidence_json = json.dumps(validation_evidence, indent=2)
# #         primary_result, usage_primary = _invoke_structured_with_usage(
# #             self.primary_judge_llm,
# #             self.primary_prompt.format_messages(
# #                 refined_idea=_sget(state, "refined_idea", ""),
# #                 market_analysis=_sget(state, "market_analysis", ""),
# #                 market_evidence=evidence_json,
# #             ),
# #         )
# #         primary_report: Dict[str, Any] = json.loads(primary_result.model_dump_json())
# #         primary_claims = primary_report.get("claims", [])
# #         primary_det_score = _deterministic_reliability_score(primary_claims)
# #         claims_for_secondary = json.dumps(
# #             [{"claim": c.get("claim", "")} for c in primary_claims], indent=2
# #         )

# #         secondary_failed = False
# #         usage_secondary = _empty_token_usage()
# #         secondary_report: Dict[str, Any] = {}
# #         try:
# #             secondary_result, usage_secondary = _invoke_structured_with_usage(
# #                 self.secondary_judge_llm,
# #                 self.secondary_prompt.format_messages(
# #                     refined_idea=_sget(state, "refined_idea", ""),
# #                     market_analysis=_sget(state, "market_analysis", ""),
# #                     market_evidence=evidence_json,
# #                     claims_to_review=claims_for_secondary,
# #                 ),
# #             )
# #             secondary_report = json.loads(secondary_result.model_dump_json())
# #         except Exception:  # noqa: BLE001
# #             secondary_failed = True
# #             # Fallback to primary output so pipeline remains robust.
# #             secondary_report = {
# #                 "evidence_gaps": primary_report.get("evidence_gaps", ""),
# #                 "overall_reliability": int(primary_report.get("overall_reliability", 0) or 0),
# #                 "rubric_scores": dict(primary_report.get("rubric_scores", {}) or {}),
# #                 "claim_assessments": list(primary_claims),
# #             }

# #         secondary_claims = secondary_report.get("claim_assessments", [])
# #         secondary_det_score = _deterministic_reliability_score(secondary_claims)
# #         score_a = int(primary_report.get("overall_reliability", 0) or 0)
# #         score_b = int(secondary_report.get("overall_reliability", 0) or 0)
# #         rubric_a = dict(primary_report.get("rubric_scores", {}) or {})
# #         rubric_b = dict(secondary_report.get("rubric_scores", {}) or {})

# #         agreement = self._agreement_stats(
# #             primary_claims=primary_claims,
# #             secondary_claims=secondary_claims,
# #             rubric_a=rubric_a,
# #             rubric_b=rubric_b,
# #             score_a=score_a,
# #             score_b=score_b,
# #         )
# #         rubric_mean_a = self._rubric_mean(rubric_a)
# #         rubric_mean_b = self._rubric_mean(rubric_b)
# #         blended_score = (
# #             0.35 * ((score_a + score_b) / 2.0)
# #             + 0.35 * ((primary_det_score + secondary_det_score) / 2.0)
# #             + 0.30 * ((rubric_mean_a + rubric_mean_b) / 2.0)
# #         )
# #         claims_total = len(primary_claims)
# #         missing_claims = max(0, self.MIN_VALIDATION_CLAIMS - claims_total)
# #         low_claim_count_flag = missing_claims > 0
# #         claim_count_penalty = missing_claims * self.LOW_CLAIM_PENALTY_PER_MISSING_CLAIM
# #         final_score = self._clamp_score(blended_score - claim_count_penalty)
# #         claim_units, failing_claims, claim_failure_summary = self._claim_ledger(
# #             primary_claims,
# #             primary_report.get("evidence_gaps", ""),
# #         )

# #         # Use primary claim list as canonical for downstream compatibility.
# #         report: Dict[str, Any] = {
# #             "validated_summary": primary_report.get("validated_summary", ""),
# #             "evidence_gaps": primary_report.get("evidence_gaps", ""),
# #             "claims": primary_claims,
# #             "claim_units": claim_units,
# #             "decontextualized_claims": claim_units,
# #             "failing_claims": failing_claims,
# #             "verifiable_claim_count": claim_failure_summary["verifiable_claim_count"],
# #             "unsupported_material_claim_count": claim_failure_summary[
# #                 "unsupported_material_claim_count"
# #             ],
# #             "claim_failure_summary": claim_failure_summary,
# #             "reliability_score": final_score,
# #             "judge_scores": {
# #                 "judge_a": self._judge_summary(primary_report, primary_det_score, usage_primary),
# #                 "judge_b": self._judge_summary(
# #                     {
# #                         "overall_reliability": score_b,
# #                         "rubric_scores": rubric_b,
# #                         "claim_assessments": secondary_claims,
# #                     },
# #                     secondary_det_score,
# #                     usage_secondary,
# #                 ),
# #                 "aggregated": {
# #                     "final_reliability_score": final_score,
# #                     "final_formula": (
# #                         "0.35*avg(judge_overall) + 0.35*avg(deterministic_claim_score) + "
# #                         "0.30*avg(weighted_rubric_score)"
# #                     ),
# #                     "rubric_weights": dict(self.RUBRIC_WEIGHTS),
# #                     "minimum_claims_required": self.MIN_VALIDATION_CLAIMS,
# #                     "claims_total": claims_total,
# #                     "low_claim_count_flag": low_claim_count_flag,
# #                     "claim_count_penalty": claim_count_penalty,
# #                     "secondary_judge_fallback": secondary_failed,
# #                     "judge_a_model": self.judge_a_model,
# #                     "judge_b_model": self.judge_b_model,
# #                     "cross_model_judging": self.cross_model_judging,
# #                 },
# #             },
# #             "agreement_stats": agreement,
# #             "evaluation_primary": {
# #                 "used_for_decision": True,
# #                 "primary_reliability_score": final_score,
# #                 "primary_judge_agreement": agreement.get("overall_agreement"),
# #                 "decision_rule": (
# #                     "needs_revision = primary_reliability_score < validation_threshold"
# #                 ),
# #             },
# #         }
# #         total_usage = _merge_token_usage({"token_usage": usage_primary}, usage_secondary)
# #         token_usage = _merge_token_usage(state, total_usage)
# #         validated = (
# #             f"{_sget(state, 'market_analysis', '')}\n\n"
# #             f"Validation Score: {final_score}/100\n"
# #             f"Validated Summary: {report.get('validated_summary', '')}\n"
# #             f"Evidence Gaps: {report.get('evidence_gaps', '')}\n"
# #             f"Judge Agreement: {agreement.get('overall_agreement')}"
# #         )
# #         tool_audit = list(_sget(state, "tool_audit", []))
# #         tool_audit.append(
# #             {
# #                 "agent": "source_validator",
# #                 "tool": "llm_claim_verifier_dual_judge",
# #                 "status": "ok",
# #                 "claims_checked": len(report.get("claims", [])),
# #                 "reliability_score": final_score,
# #                 "judge_a_score": score_a,
# #                 "judge_b_score": score_b,
# #                 "judge_agreement": agreement.get("overall_agreement"),
# #                 "secondary_judge_fallback": secondary_failed,
# #                 "minimum_claims_required": self.MIN_VALIDATION_CLAIMS,
# #                 "claims_total": claims_total,
# #                 "low_claim_count_flag": low_claim_count_flag,
# #                 "claim_count_penalty": claim_count_penalty,
# #                 "unsupported_material_claim_count": claim_failure_summary[
# #                     "unsupported_material_claim_count"
# #                 ],
# #                 "judge_a_model": self.judge_a_model,
# #                 "judge_b_model": self.judge_b_model,
# #                 "cross_model_judging": self.cross_model_judging,
# #                 "evidence_items_raw": len(raw_evidence),
# #                 "evidence_items_used": len(validation_evidence),
# #                 "evidence_limit": evidence_budget,
# #                 "prompt_tokens": total_usage.get("prompt_tokens", 0),
# #                 "completion_tokens": total_usage.get("completion_tokens", 0),
# #                 "total_tokens": total_usage.get("total_tokens", 0),
# #             }
# #         )
# #         return {
# #             "validation_report": report,
# #             "validated_market_analysis": validated,
# #             "claim_units": claim_units,
# #             "failing_claims": failing_claims,
# #             "needs_revision": final_score < _sget(state, "validation_threshold", 70),
# #             "tool_audit": tool_audit,
# #             "token_usage": token_usage,
# #         }

# #     def run_repair_only(self, state: PitchState) -> PitchState:
# #         """Cheaper validation for claim-level repair continuations.

# #         Full dual-judge validation remains the primary evaluator. This method is
# #         only used after an explicit micro-repair when the repair plan indicates
# #         no factual search-and-replace claims. It evaluates only the repaired
# #         claims, then merges them with the previous validation's supported claims.
# #         """
# #         repair_context = _sget(state, "market_repair_context", {}) or {}
# #         previous_validation = _sget(state, "validation_report", {}) or {}
# #         if not isinstance(repair_context, dict) or not isinstance(previous_validation, dict):
# #             return self.run(state)

# #         repair_plan = [
# #             item
# #             for item in repair_context.get("repair_plan", []) or []
# #             if isinstance(item, dict)
# #         ]
# #         action_counts = repair_context.get("repair_action_counts", {}) or {}
# #         if int(action_counts.get("search_and_replace", 0) or 0) > 0:
# #             return self.run(state)

# #         raw_evidence = _sget(state, "market_evidence", []) or []
# #         validation_evidence = _limit_evidence_items(
# #             raw_evidence,
# #             max_items=6,
# #             content_chars=220,
# #         )
# #         micro_repair_audits = [
# #             audit
# #             for audit in (_sget(state, "tool_audit", []) or [])
# #             if isinstance(audit, dict) and audit.get("tool") == "claim_micro_repair"
# #         ]
# #         latest_repair_audit = micro_repair_audits[-1] if micro_repair_audits else {}
# #         patch_payload = {
# #             "patch_summary": latest_repair_audit.get("patch_summary", ""),
# #             "patches": latest_repair_audit.get("patches", []),
# #             "repair_action_counts": latest_repair_audit.get("repair_action_counts", {}),
# #         }

# #         result, repair_validation_usage = _invoke_structured_with_usage(
# #             self.repair_judge_llm,
# #             self.repair_prompt.format_messages(
# #                 refined_idea=_sget(state, "refined_idea", ""),
# #                 previous_market_analysis=repair_context.get("previous_market_analysis", ""),
# #                 market_analysis=_sget(state, "market_analysis", ""),
# #                 repair_plan=json.dumps(repair_plan, indent=2),
# #                 patches=json.dumps(patch_payload, indent=2),
# #                 market_evidence=json.dumps(validation_evidence, indent=2),
# #             ),
# #         )
# #         repair_report: Dict[str, Any] = json.loads(result.model_dump_json())
# #         repaired_claims = repair_report.get("repaired_claim_assessments", []) or []
# #         previous_claims = previous_validation.get("claims", []) or []
# #         weak_claim_keys = {
# #             str(item.get("claim", "") or "").strip().lower()
# #             for item in repair_plan
# #             if str(item.get("claim", "") or "").strip()
# #         }
# #         preserved_claims = [
# #             claim
# #             for claim in previous_claims
# #             if isinstance(claim, dict)
# #             and str(claim.get("claim", "") or "").strip().lower() not in weak_claim_keys
# #         ]
# #         merged_claims = preserved_claims + [
# #             claim for claim in repaired_claims if isinstance(claim, dict)
# #         ]
# #         if len(merged_claims) < self.MIN_VALIDATION_CLAIMS:
# #             merged_claims = list(previous_claims)

# #         previous_score = int(previous_validation.get("reliability_score", 0) or 0)
# #         deterministic_score = _deterministic_reliability_score(merged_claims)
# #         patch_score = int(repair_report.get("patch_reliability", deterministic_score) or 0)
# #         rubric_mean = self._rubric_mean(repair_report.get("rubric_scores", {}) or {})
# #         candidate_score = self._clamp_score(
# #             0.50 * deterministic_score + 0.25 * patch_score + 0.25 * rubric_mean
# #         )
# #         final_score = max(previous_score, candidate_score)
# #         accepted = final_score > previous_score and bool(repair_report.get("can_accept_patch", False))
# #         if not accepted:
# #             # Keep the previous score if the focused judge cannot confidently
# #             # accept the patch; checkpoint selection can still compare snapshots.
# #             final_score = previous_score

# #         previous_agreement = (
# #             previous_validation.get("agreement_stats", {}).get("overall_agreement", 0.0)
# #             if isinstance(previous_validation.get("agreement_stats", {}), dict)
# #             else 0.0
# #         )
# #         claim_units, failing_claims, claim_failure_summary = self._claim_ledger(
# #             merged_claims if accepted else previous_claims,
# #             repair_report.get("evidence_gaps", ""),
# #         )
# #         report = {
# #             "validated_summary": previous_validation.get("validated_summary", ""),
# #             "evidence_gaps": repair_report.get("evidence_gaps", ""),
# #             "claims": merged_claims if accepted else previous_claims,
# #             "claim_units": claim_units,
# #             "decontextualized_claims": claim_units,
# #             "failing_claims": failing_claims,
# #             "verifiable_claim_count": claim_failure_summary["verifiable_claim_count"],
# #             "unsupported_material_claim_count": claim_failure_summary[
# #                 "unsupported_material_claim_count"
# #             ],
# #             "claim_failure_summary": claim_failure_summary,
# #             "reliability_score": final_score,
# #             "judge_scores": {
# #                 "repair_judge": {
# #                     "patch_reliability": patch_score,
# #                     "deterministic_claim_score": deterministic_score,
# #                     "rubric_scores": repair_report.get("rubric_scores", {}),
# #                     "rubric_mean": round(rubric_mean, 3),
# #                     "claim_count": len(repaired_claims),
# #                     "token_usage": repair_validation_usage,
# #                 },
# #                 "aggregated": {
# #                     "final_reliability_score": final_score,
# #                     "final_formula": (
# #                         "lightweight repair validation: max(previous_score, "
# #                         "0.50*merged_deterministic_claim_score + "
# #                         "0.25*patch_reliability + 0.25*weighted_patch_rubric)"
# #                     ),
# #                     "rubric_weights": dict(self.RUBRIC_WEIGHTS),
# #                     "minimum_claims_required": self.MIN_VALIDATION_CLAIMS,
# #                     "claims_total": len(merged_claims if accepted else previous_claims),
# #                     "low_claim_count_flag": len(merged_claims) < self.MIN_VALIDATION_CLAIMS,
# #                     "claim_count_penalty": 0,
# #                     "secondary_judge_fallback": False,
# #                     "judge_a_model": self.judge_a_model,
# #                     "judge_b_model": "not_used_lightweight_repair_validation",
# #                     "cross_model_judging": False,
# #                     "lightweight_repair_validation": True,
# #                     "repair_patch_accepted": accepted,
# #                     "previous_reliability_score": previous_score,
# #                     "candidate_reliability_score": candidate_score,
# #                 },
# #             },
# #             "agreement_stats": {
# #                 "score_delta_abs": abs(final_score - previous_score),
# #                 "score_agreement": round(max(0.0, 1.0 - abs(final_score - previous_score) / 100.0), 4),
# #                 "rubric_mae": 0.0,
# #                 "rubric_diffs": {},
# #                 "rubric_agreement": 1.0,
# #                 "shared_claims": len(preserved_claims),
# #                 "verdict_agreement_rate": 1.0,
# #                 "overall_agreement": round(float(previous_agreement or 0.0), 4),
# #                 "lightweight_repair_validation": True,
# #             },
# #             "evaluation_primary": {
# #                 "used_for_decision": True,
# #                 "primary_reliability_score": final_score,
# #                 "primary_judge_agreement": previous_agreement,
# #                 "decision_rule": (
# #                     "needs_revision = primary_reliability_score < validation_threshold"
# #                 ),
# #             },
# #             "repair_validation": {
# #                 "accepted": accepted,
# #                 "previous_reliability_score": previous_score,
# #                 "candidate_reliability_score": candidate_score,
# #                 "patch_reliability": patch_score,
# #                 "can_accept_patch": bool(repair_report.get("can_accept_patch", False)),
# #                 "rationale": repair_report.get("rationale", ""),
# #             },
# #         }
# #         token_usage = _merge_token_usage(state, repair_validation_usage)
# #         validated = (
# #             f"{_sget(state, 'market_analysis', '')}\n\n"
# #             f"Validation Score: {final_score}/100\n"
# #             "Validated Summary: Lightweight repair-only validation after claim-level patch.\n"
# #             f"Evidence Gaps: {report.get('evidence_gaps', '')}\n"
# #             f"Judge Agreement: {previous_agreement}"
# #         )
# #         tool_audit = list(_sget(state, "tool_audit", []))
# #         tool_audit.append(
# #             {
# #                 "agent": "source_validator",
# #                 "tool": "llm_claim_repair_validator",
# #                 "status": "ok",
# #                 "claims_checked": len(repaired_claims),
# #                 "reliability_score": final_score,
# #                 "previous_reliability_score": previous_score,
# #                 "candidate_reliability_score": candidate_score,
# #                 "repair_patch_accepted": accepted,
# #                 "lightweight_repair_validation": True,
# #                 "minimum_claims_required": self.MIN_VALIDATION_CLAIMS,
# #                 "claims_total": len(merged_claims if accepted else previous_claims),
# #                 "low_claim_count_flag": len(merged_claims) < self.MIN_VALIDATION_CLAIMS,
# #                 "claim_count_penalty": 0,
# #                 "unsupported_material_claim_count": claim_failure_summary[
# #                     "unsupported_material_claim_count"
# #                 ],
# #                 "secondary_judge_fallback": False,
# #                 "judge_a_model": self.judge_a_model,
# #                 "judge_b_model": "not_used_lightweight_repair_validation",
# #                 "cross_model_judging": False,
# #                 "evidence_items_raw": len(raw_evidence),
# #                 "evidence_items_used": len(validation_evidence),
# #                 "prompt_tokens": repair_validation_usage.get("prompt_tokens", 0),
# #                 "completion_tokens": repair_validation_usage.get("completion_tokens", 0),
# #                 "total_tokens": repair_validation_usage.get("total_tokens", 0),
# #             }
# #         )
# #         return {
# #             "validation_report": report,
# #             "validated_market_analysis": validated,
# #             "claim_units": claim_units,
# #             "failing_claims": failing_claims,
# #             "micro_validation": report.get("repair_validation", {}),
# #             "accepted_patch_count": int(_sget(state, "accepted_patch_count", 0) or 0)
# #             + (1 if accepted else 0),
# #             "rejected_patch_count": int(_sget(state, "rejected_patch_count", 0) or 0)
# #             + (0 if accepted else 1),
# #             "needs_revision": final_score < _sget(state, "validation_threshold", 70),
# #             "tool_audit": tool_audit,
# #             "token_usage": token_usage,
# #         }


# # class DirectStrategyAgent:
# #     def __init__(
# #         self,
# #         llm: ChatOpenAI,
# #         strict_tools: bool = True,
# #         enable_trends: bool = True,
# #         secondary_judge_llm: Optional[ChatOpenAI] = None,
# #     ):
# #         self.structured_llm = llm.with_structured_output(DirectStrategyOutput, include_raw=True)
# #         self.validator_agent = SourceValidatorAgent(
# #             llm,
# #             secondary_judge_llm=secondary_judge_llm,
# #         )
# #         self.search_tool = MarketSearchTool()
# #         self.trends_tool = GoogleTrendsTool() if enable_trends else None
# #         self.calc_tool = BusinessCalcTool()
# #         self.scenario_tool = ScenarioAnalysisTool()
# #         self.strict_tools = strict_tools
# #         self.enable_trends = enable_trends
# #         self.prompt = ChatPromptTemplate.from_messages(
# #             [
# #                 (
# #                     "system",
# #                     "# Role\n"
# #                     "You are a direct (low-decomposition) startup strategist.\n\n"
# #                     "# Instructions\n"
# #                     "- Produce market analysis and business model in one synthesis pass.\n"
# #                     "- Also propose bounded Year-1 assumptions (users/arpu/gross margin).\n"
# #                     "- Keep outputs concise, evidence-aware, and investor-ready.\n\n"
# #                     "# Output Format\n"
# #                     "- Return content that strictly matches the structured schema fields.",
# #                 ),
# #                 (
# #                     "human",
# #                     "Startup concept:\n{refined_idea}\n\n"
# #                     "Web snippets:\n{search_results}\n\n"
# #                     "Trend signals:\n{trend_signals}\n\n"
# #                     "Provide integrated market + business + assumptions output.",
# #                 ),
# #             ]
# #         )

# #     def run(self, state: PitchState) -> PitchState:
# #         refined_idea = _sget(state, "refined_idea", "")
# #         total_usage = _empty_token_usage()
# #         query = f"startup market size competitors trends for: {refined_idea}"
# #         search_payload = self.search_tool.search(query)
# #         if self.strict_tools and search_payload["status"] != "ok":
# #             raise RuntimeError(f"Market search tool unavailable: {search_payload['error']}")
# #         evidence_budget = _evidence_budget_for_mode(
# #             "direct",
# #             _sget(state, "budget_remaining", {}).get("total_tokens")
# #             if isinstance(_sget(state, "budget_remaining", {}), dict)
# #             else None,
# #         )
# #         search_payload = _limited_search_payload(search_payload, **evidence_budget)

# #         extracted_keywords = _fallback_keywords(refined_idea)
# #         if self.enable_trends and self.trends_tool is not None:
# #             trend_payload = self.trends_tool.fetch(extracted_keywords[:5])
# #         else:
# #             trend_payload = {
# #                 "status": "skipped",
# #                 "keywords": extracted_keywords[:5],
# #                 "data": {},
# #                 "error": "Google Trends disabled by configuration.",
# #             }

# #         result, direct_usage = _invoke_structured_with_usage(
# #             self.structured_llm,
# #             self.prompt.format_messages(
# #                 refined_idea=refined_idea,
# #                 search_results=search_payload["results_json"],
# #                 trend_signals=json.dumps(trend_payload, indent=2),
# #             ),
# #         )
# #         total_usage = _merge_token_usage({"token_usage": total_usage}, direct_usage)
# #         token_usage = _merge_token_usage(state, total_usage)
# #         market_analysis = (
# #             f"Target Market: {result.target_market}\n"
# #             f"Market Size: {result.market_size}\n"
# #             f"Trends: {result.trends}\n"
# #             f"Competitors: {result.competitors}\n"
# #             f"Differentiation Gaps: {result.differentiation_gaps}"
# #         )
# #         assumptions = {
# #             "users_year1": max(1000, min(500000, int(result.users_year1))),
# #             "arpu_monthly": round(max(2.0, min(300.0, float(result.arpu_monthly))), 2),
# #             "gross_margin": round(max(0.2, min(0.95, float(result.gross_margin))), 3),
# #             "rationale": result.assumptions_rationale,
# #         }
# #         calc_script = f"""
# # users_year1 = {assumptions["users_year1"]}
# # arpu_monthly = {assumptions["arpu_monthly"]}
# # annual_revenue = users_year1 * arpu_monthly * 12
# # gross_margin = {assumptions["gross_margin"]}
# # gross_profit = annual_revenue * gross_margin
# # print(f'Year1 Revenue: ${{annual_revenue:,.0f}}')
# # print(f'Year1 Gross Profit: ${{gross_profit:,.0f}}')
# # """
# #         calc_output = self.calc_tool.run(calc_script)
# #         if self.strict_tools and calc_output.startswith("Python calc failed:"):
# #             raise RuntimeError(calc_output)
# #         scenario_output = self.scenario_tool.run(
# #             users_year1=assumptions["users_year1"],
# #             arpu_monthly=assumptions["arpu_monthly"],
# #             gross_margin=assumptions["gross_margin"],
# #         )
# #         business_model = (
# #             f"Revenue Streams: {result.revenue_streams}\n"
# #             f"Pricing Strategy: {result.pricing_strategy}\n"
# #             f"Cost Structure: {result.cost_structure}\n"
# #             f"Unit Economics: {result.unit_economics}\n"
# #             f"Financial Projection: {result.financial_projection}\n"
# #             f"Financial Assumptions:\n{json.dumps(assumptions, indent=2)}\n"
# #             f"Calculator Baseline:\n{calc_output}\n"
# #             f"Scenario Analysis:\n{json.dumps(scenario_output, indent=2)}"
# #         )
# #         tool_audit = list(_sget(state, "tool_audit", []))
# #         tool_audit.append(
# #             {
# #                 "agent": "adaptive_direct_strategy",
# #                 "tool": "llm_direct_synthesis",
# #                 "status": "ok",
# #                 "prompt_tokens": direct_usage.get("prompt_tokens", 0),
# #                 "completion_tokens": direct_usage.get("completion_tokens", 0),
# #                 "total_tokens": direct_usage.get("total_tokens", 0),
# #             }
# #         )
# #         tool_audit.append(
# #             {
# #                 "agent": "adaptive_direct_strategy",
# #                 "tool": "linkup_search",
# #                 "query": query,
# #                 "status": search_payload["status"],
# #                 "source_count": len(search_payload["sources"]),
# #                 "source_count_raw": search_payload.get("source_count_raw"),
# #                 "evidence_limit": search_payload.get("evidence_limit"),
# #                 "dropped_sources": search_payload.get("dropped_sources", 0),
# #                 "error": search_payload["error"],
# #             }
# #         )
# #         tool_audit.append(
# #             {
# #                 "agent": "adaptive_direct_strategy",
# #                 "tool": "google_trends",
# #                 "status": trend_payload["status"],
# #                 "keyword_count": len(trend_payload.get("keywords", [])),
# #                 "keywords": trend_payload.get("keywords", []),
# #                 "error": trend_payload.get("error", ""),
# #             }
# #         )
# #         tool_audit.append(
# #             {
# #                 "agent": "adaptive_direct_strategy",
# #                 "tool": "python_calc",
# #                 "status": "ok" if not calc_output.startswith("Python calc failed:") else "error",
# #                 "assumptions": assumptions,
# #                 "error": calc_output if calc_output.startswith("Python calc failed:") else "",
# #             }
# #         )
# #         tool_audit.append(
# #             {
# #                 "agent": "adaptive_direct_strategy",
# #                 "tool": "scenario_analysis",
# #                 "status": "ok",
# #                 "scenario_count": len(scenario_output),
# #                 "error": "",
# #             }
# #         )
# #         candidate_state = {
# #             **(state if isinstance(state, dict) else state.model_dump()),
# #             "market_analysis": market_analysis,
# #             "market_sources": search_payload["sources"],
# #             "market_evidence": search_payload["results"],
# #             "trend_signals": trend_payload,
# #             "business_model": business_model,
# #             "financial_assumptions": assumptions,
# #             "scenario_analysis": scenario_output,
# #             "tool_audit": tool_audit,
# #             "token_usage": token_usage,
# #             "decomposition_depth_realized": 0,
# #         }
# #         validated_update = self.validator_agent.run(candidate_state)
# #         candidate_state.update(validated_update)
# #         return {
# #             "market_analysis": candidate_state["market_analysis"],
# #             "market_sources": candidate_state["market_sources"],
# #             "market_evidence": candidate_state["market_evidence"],
# #             "trend_signals": candidate_state["trend_signals"],
# #             "business_model": candidate_state["business_model"],
# #             "financial_assumptions": candidate_state["financial_assumptions"],
# #             "scenario_analysis": candidate_state["scenario_analysis"],
# #             "validation_report": candidate_state.get("validation_report"),
# #             "validated_market_analysis": candidate_state.get("validated_market_analysis"),
# #             "needs_revision": candidate_state.get("needs_revision", False),
# #             "tool_audit": candidate_state["tool_audit"],
# #             "token_usage": candidate_state.get("token_usage", token_usage),
# #             "decomposition_depth_realized": 0,
# #         }


# # class BusinessModelAgent:
# #     def __init__(self, llm: ChatOpenAI, strict_tools: bool = True):
# #         self.structured_llm = llm.with_structured_output(BusinessOutput, include_raw=True)
# #         self.assumptions_llm = llm.with_structured_output(FinancialAssumptions, include_raw=True)
# #         self.calc_tool = BusinessCalcTool()
# #         self.scenario_tool = ScenarioAnalysisTool()
# #         self.strict_tools = strict_tools
# #         self.assumptions_prompt = ChatPromptTemplate.from_messages(
# #             [
# #                 (
# #                     "system",
# #                     "# Role\n"
# #                     "You are a startup financial assumptions analyst.\n\n"
# #                     "# Instructions\n"
# #                     "- Generate realistic Year-1 assumptions from startup and market context.\n"
# #                     "- Use conservative, explainable values.\n"
# #                     "- Stay strictly within schema bounds.\n\n"
# #                     "# Output Format\n"
# #                     "- Return content that strictly matches the structured schema fields.",
# #                 ),
# #                 (
# #                     "human",
# #                     "Startup concept:\n{refined_idea}\n\n"
# #                     "Validated market analysis:\n{market_analysis}\n\n"
# #                     "Trend signals:\n{trend_signals}\n\n"
# #                     "Validation score:\n{validation_score}\n\n"
# #                     "Return assumptions for users_year1, arpu_monthly, gross_margin and rationale.",
# #                 ),
# #             ]
# #         )
# #         self.prompt = ChatPromptTemplate.from_messages(
# #             [
# #                 (
# #                     "system",
# #                     "# Role\n"
# #                     "You are a startup business model strategist.\n\n"
# #                     "# Instructions\n"
# #                     "- Use supplied market analysis, assumptions, calculator output, and scenario output.\n"
# #                     "- Produce practical revenue model, pricing, costs, unit economics, and projection narrative.\n"
# #                     "- Keep recommendations coherent with the provided quantitative inputs.\n\n"
# #                     "# Output Format\n"
# #                     "- Return content that strictly matches the structured schema fields.",
# #                 ),
# #                 (
# #                     "human",
# #                     "Startup concept:\n{refined_idea}\n\n"
# #                     "Market analysis:\n{market_analysis}\n\n"
# #                     "Financial assumptions:\n{financial_assumptions}\n\n"
# #                     "Financial calculator output:\n{calc_output}\n\n"
# #                     "Scenario analysis output:\n{scenario_output}\n\n"
# #                     "Provide revenue model, pricing, costs, and projection.",
# #                 ),
# #             ]
# #         )

# #     def run(self, state: PitchState) -> PitchState:
# #         assumptions_model, assumptions_usage = _invoke_structured_with_usage(
# #             self.assumptions_llm,
# #             self.assumptions_prompt.format_messages(
# #                 refined_idea=_sget(state, "refined_idea", ""),
# #                 market_analysis=(
# #                     _sget(state, "validated_market_analysis")
# #                     or _sget(state, "market_analysis", "")
# #                 ),
# #                 trend_signals=json.dumps(_sget(state, "trend_signals", {}), indent=2),
# #                 validation_score=_sget(state, "validation_report", {}).get(
# #                     "reliability_score", 0
# #                 ),
# #             ),
# #         )
# #         total_usage = _merge_token_usage({"token_usage": _empty_token_usage()}, assumptions_usage)
# #         assumptions = {
# #             "users_year1": max(1000, min(500000, int(assumptions_model.users_year1))),
# #             "arpu_monthly": round(max(2.0, min(300.0, float(assumptions_model.arpu_monthly))), 2),
# #             "gross_margin": round(max(0.2, min(0.95, float(assumptions_model.gross_margin))), 3),
# #             "rationale": assumptions_model.rationale,
# #         }
# #         calc_script = f"""
# # users_year1 = {assumptions["users_year1"]}
# # arpu_monthly = {assumptions["arpu_monthly"]}
# # annual_revenue = users_year1 * arpu_monthly * 12
# # gross_margin = {assumptions["gross_margin"]}
# # gross_profit = annual_revenue * gross_margin
# # print(f'Year1 Revenue: ${{annual_revenue:,.0f}}')
# # print(f'Year1 Gross Profit: ${{gross_profit:,.0f}}')
# # """
# #         calc_output = self.calc_tool.run(calc_script)
# #         if self.strict_tools and calc_output.startswith("Python calc failed:"):
# #             raise RuntimeError(calc_output)
# #         scenario_output = self.scenario_tool.run(
# #             users_year1=assumptions["users_year1"],
# #             arpu_monthly=assumptions["arpu_monthly"],
# #             gross_margin=assumptions["gross_margin"],
# #         )

# #         result, business_usage = _invoke_structured_with_usage(
# #             self.structured_llm,
# #             self.prompt.format_messages(
# #                 refined_idea=_sget(state, "refined_idea", ""),
# #                 market_analysis=(
# #                     _sget(state, "validated_market_analysis")
# #                     or _sget(state, "market_analysis", "")
# #                 ),
# #                 financial_assumptions=json.dumps(assumptions, indent=2),
# #                 calc_output=calc_output,
# #                 scenario_output=json.dumps(scenario_output, indent=2),
# #             ),
# #         )
# #         total_usage = _merge_token_usage({"token_usage": total_usage}, business_usage)
# #         token_usage = _merge_token_usage(state, total_usage)
# #         business_model = (
# #             f"Revenue Streams: {result.revenue_streams}\n"
# #             f"Pricing Strategy: {result.pricing_strategy}\n"
# #             f"Cost Structure: {result.cost_structure}\n"
# #             f"Unit Economics: {result.unit_economics}\n"
# #             f"Financial Projection: {result.financial_projection}\n"
# #             f"Financial Assumptions:\n{json.dumps(assumptions, indent=2)}\n"
# #             f"Calculator Baseline:\n{calc_output}\n"
# #             f"Scenario Analysis:\n{json.dumps(scenario_output, indent=2)}"
# #         )
# #         tool_audit = list(_sget(state, "tool_audit", []))
# #         tool_audit.append(
# #             {
# #                 "agent": "business_model",
# #                 "tool": "python_calc",
# #                 "status": "ok" if not calc_output.startswith("Python calc failed:") else "error",
# #                 "assumptions": assumptions,
# #                 "error": calc_output if calc_output.startswith("Python calc failed:") else "",
# #                 "prompt_tokens_assumptions": assumptions_usage.get("prompt_tokens", 0),
# #                 "completion_tokens_assumptions": assumptions_usage.get("completion_tokens", 0),
# #                 "total_tokens_assumptions": assumptions_usage.get("total_tokens", 0),
# #                 "prompt_tokens_business": business_usage.get("prompt_tokens", 0),
# #                 "completion_tokens_business": business_usage.get("completion_tokens", 0),
# #                 "total_tokens_business": business_usage.get("total_tokens", 0),
# #             }
# #         )
# #         tool_audit.append(
# #             {
# #                 "agent": "business_model",
# #                 "tool": "scenario_analysis",
# #                 "status": "ok",
# #                 "scenario_count": len(scenario_output),
# #                 "error": "",
# #             }
# #         )
# #         return {
# #             "business_model": business_model,
# #             "financial_assumptions": assumptions,
# #             "scenario_analysis": scenario_output,
# #             "tool_audit": tool_audit,
# #             "token_usage": token_usage,
# #         }


# # class PitchDeckGeneratorAgent:
# #     def __init__(self, llm: ChatOpenAI, output_dir: str = "output"):
# #         self.structured_llm = llm.with_structured_output(PitchSlides, include_raw=True)
# #         self.output_dir = Path(output_dir)
# #         self.prompt = ChatPromptTemplate.from_messages(
# #             [
# #                 (
# #                     "system",
# #                     "# Role\n"
# #                     "You are an investor pitch deck strategist.\n\n"
# #                     "# Instructions\n"
# #                     "- Convert refined idea, market analysis, and business model into concise slide-ready content.\n"
# #                     "- Keep language clear, specific, and investor-oriented.\n"
# #                     "- Cover all required sections.\n\n"
# #                     "# Output Format\n"
# #                     "- Return content that strictly matches the structured schema fields.",
# #                 ),
# #                 (
# #                     "human",
# #                     "Refined idea:\n{refined_idea}\n\n"
# #                     "Market analysis:\n{market_analysis}\n\n"
# #                     "Business model:\n{business_model}\n\n"
# #                     "Return all required slide sections.",
# #                 ),
# #             ]
# #         )

# #     def run(self, state: PitchState) -> PitchState:
# #         slide_model, usage = _invoke_structured_with_usage(
# #             self.structured_llm,
# #             self.prompt.format_messages(
# #                 refined_idea=_sget(state, "refined_idea", ""),
# #                 market_analysis=(
# #                     _sget(state, "validated_market_analysis")
# #                     or _sget(state, "market_analysis", "")
# #                 ),
# #                 business_model=_sget(state, "business_model", ""),
# #             ),
# #         )
# #         token_usage = _merge_token_usage(state, usage)
# #         slide_dict: Dict[str, str] = json.loads(slide_model.model_dump_json())

# #         summary_line = _sget(state, "refined_idea", "")
# #         words = re.findall(r"[a-zA-Z0-9]+", summary_line.lower())
# #         safe_name = "startup_pitch_deck"
# #         if words:
# #             safe_name = "pitch_" + "_".join(words[:6])
# #         ppt_path = self.output_dir / f"{safe_name}.pptx"
# #         saved = generate_pitch_deck(slide_dict, str(ppt_path))

# #         tool_audit = list(_sget(state, "tool_audit", []))
# #         tool_audit.append(
# #             {
# #                 "agent": "pitch_deck_generator",
# #                 "tool": "python_pptx",
# #                 "status": "ok",
# #                 "output_path": saved,
# #                 "prompt_tokens": usage.get("prompt_tokens", 0),
# #                 "completion_tokens": usage.get("completion_tokens", 0),
# #                 "total_tokens": usage.get("total_tokens", 0),
# #             }
# #         )
# #         return {
# #             "pitch_content": slide_dict,
# #             "ppt_path": saved,
# #             "tool_audit": tool_audit,
# #             "token_usage": token_usage,
# #         }
