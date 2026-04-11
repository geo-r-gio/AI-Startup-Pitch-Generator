from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal

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
        self.structured_llm = llm.with_structured_output(RefinedIdea)
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
        result: RefinedIdea = self.structured_llm.invoke(
            self.prompt.format_messages(idea=idea)
        )
        refined = (
            f"Problem: {result.problem}\n"
            f"Solution: {result.solution}\n"
            f"Value Proposition: {result.value_proposition}\n"
            f"Summary: {result.refined_summary}"
        )
        return {"refined_idea": refined}


class AdaptiveControllerAgent:
    def __init__(self, llm: ChatOpenAI):
        self.structured_llm = llm.with_structured_output(ControllerDecision)
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
                    "- Estimate complexity (0-100), confidence, and concise rationale.\n"
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
        elif score < 60:
            mode = "shallow"
            confidence = 0.72
            rationale = "Idea has moderate complexity, so one decomposition pass with validation is appropriate."
        else:
            mode = "recursive"
            confidence = 0.69
            rationale = "Idea is high-complexity or high-uncertainty and benefits from validation-driven recursion."

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
            triggers=triggers,
        )

    def run(self, state: PitchState) -> PitchState:
        idea = _sget(state, "idea", "")
        refined = _sget(state, "refined_idea", "")
        combined = f"{idea}\n{refined}".strip()
        try:
            decision: ControllerDecision = self.structured_llm.invoke(
                self.prompt.format_messages(
                    idea=idea,
                    refined_idea=refined,
                    validation_threshold=_sget(state, "validation_threshold", 70),
                    max_validation_retries=_sget(state, "max_validation_retries", 1),
                )
            )
        except Exception:  # noqa: BLE001
            decision = self._fallback_decision(combined)

        depth_map = {"direct": 0, "shallow": 1, "recursive": 2}
        decisions = list(_sget(state, "controller_decisions", []))
        tool_audit = list(_sget(state, "tool_audit", []))
        decisions.append(
            {
                "mode": decision.mode,
                "confidence": decision.confidence,
                "estimated_complexity": decision.estimated_complexity,
                "triggers": decision.triggers,
                "rationale": decision.rationale,
            }
        )
        tool_audit.append(
            {
                "agent": "adaptive_controller",
                "tool": "mode_selector",
                "status": "ok",
                "mode": decision.mode,
                "confidence": decision.confidence,
                "estimated_complexity": decision.estimated_complexity,
                "triggers": decision.triggers,
            }
        )
        return {
            "controller_mode": decision.mode,
            "controller_confidence": decision.confidence,
            "controller_rationale": decision.rationale,
            "controller_decisions": decisions,
            "decomposition_depth_target": depth_map.get(decision.mode, 1),
            "tool_audit": tool_audit,
        }


class MarketResearchAgent:
    def __init__(
        self,
        llm: ChatOpenAI,
        strict_tools: bool = True,
        enable_trends: bool = True,
    ):
        self.structured_llm = llm.with_structured_output(MarketOutput)
        self.keyword_llm = llm.with_structured_output(TrendKeywords)
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
            kw_model: TrendKeywords = self.keyword_llm.invoke(
                self.keyword_prompt.format_messages(refined_idea=refined_idea)
            )
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

        result: MarketOutput = self.structured_llm.invoke(
            self.prompt.format_messages(
                refined_idea=refined_idea,
                search_results=search_results,
                trend_signals=json.dumps(trend_payload, indent=2),
            )
        )
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
        }


class SourceValidatorAgent:
    def __init__(self, llm: ChatOpenAI):
        self.structured_llm = llm.with_structured_output(ValidationOutput)
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
        result: ValidationOutput = self.structured_llm.invoke(
            self.prompt.format_messages(
                refined_idea=_sget(state, "refined_idea", ""),
                market_analysis=_sget(state, "market_analysis", ""),
                market_evidence=evidence_json,
            )
        )
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
            }
        )
        return {
            "validation_report": report,
            "validated_market_analysis": validated,
            "needs_revision": deterministic_score < _sget(state, "validation_threshold", 70),
            "tool_audit": tool_audit,
        }


class DirectStrategyAgent:
    def __init__(
        self,
        llm: ChatOpenAI,
        strict_tools: bool = True,
        enable_trends: bool = True,
    ):
        self.structured_llm = llm.with_structured_output(DirectStrategyOutput)
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

        result: DirectStrategyOutput = self.structured_llm.invoke(
            self.prompt.format_messages(
                refined_idea=refined_idea,
                search_results=search_payload["results_json"],
                trend_signals=json.dumps(trend_payload, indent=2),
            )
        )
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
            "decomposition_depth_realized": 0,
        }


class BusinessModelAgent:
    def __init__(self, llm: ChatOpenAI, strict_tools: bool = True):
        self.structured_llm = llm.with_structured_output(BusinessOutput)
        self.assumptions_llm = llm.with_structured_output(FinancialAssumptions)
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
        assumptions_model: FinancialAssumptions = self.assumptions_llm.invoke(
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
            )
        )
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

        result: BusinessOutput = self.structured_llm.invoke(
            self.prompt.format_messages(
                refined_idea=_sget(state, "refined_idea", ""),
                market_analysis=(
                    _sget(state, "validated_market_analysis")
                    or _sget(state, "market_analysis", "")
                ),
                financial_assumptions=json.dumps(assumptions, indent=2),
                calc_output=calc_output,
                scenario_output=json.dumps(scenario_output, indent=2),
            )
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
                "agent": "business_model",
                "tool": "python_calc",
                "status": "ok" if not calc_output.startswith("Python calc failed:") else "error",
                "assumptions": assumptions,
                "error": calc_output if calc_output.startswith("Python calc failed:") else "",
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
        }


class PitchDeckGeneratorAgent:
    def __init__(self, llm: ChatOpenAI, output_dir: str = "output"):
        self.structured_llm = llm.with_structured_output(PitchSlides)
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
        slide_model: PitchSlides = self.structured_llm.invoke(
            self.prompt.format_messages(
                refined_idea=_sget(state, "refined_idea", ""),
                market_analysis=(
                    _sget(state, "validated_market_analysis")
                    or _sget(state, "market_analysis", "")
                ),
                business_model=_sget(state, "business_model", ""),
            )
        )
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
            }
        )
        return {"pitch_content": slide_dict, "ppt_path": saved, "tool_audit": tool_audit}
