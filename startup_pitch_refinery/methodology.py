from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from startup_pitch_refinery.agents import PitchDeckGeneratorAgent, SourceValidatorAgent
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
    metrics = {
        "strategy": strategy,
        "controller_mode": state.get("controller_mode", "n/a"),
        "runtime_seconds": round(runtime_seconds, 3),
        "reliability_score": _safe_int(validation.get("reliability_score", 0), 0),
        "claims_total": len(validation.get("claims", []) if validation else []),
        "supported_ratio": ratios["supported_ratio"],
        "weak_or_better_ratio": ratios["weak_or_better_ratio"],
        "market_sources_count": len(state.get("market_sources", [])),
        "tool_calls": len(state.get("tool_audit", [])),
        "retry_count": _safe_int(state.get("retry_count", 0), 0),
        "decomposition_depth_realized": _safe_int(state.get("decomposition_depth_realized", 0), 0),
        "needs_revision": bool(state.get("needs_revision", False)),
        "trend_status": (state.get("trend_signals") or {}).get("status", "unknown"),
        "token_proxy_total": _estimate_tokens_proxy(state),
    }
    return metrics


def _aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(rec["strategy"], []).append(rec)

    numeric_fields = [
        "runtime_seconds",
        "reliability_score",
        "supported_ratio",
        "weak_or_better_ratio",
        "market_sources_count",
        "tool_calls",
        "retry_count",
        "decomposition_depth_realized",
        "token_proxy_total",
    ]
    aggregate: Dict[str, Dict[str, float]] = {}
    for strategy, rows in grouped.items():
        strat_summary: Dict[str, float] = {"runs": float(len(rows))}
        for field in numeric_fields:
            values = [float(r[field]) for r in rows]
            strat_summary[f"{field}_mean"] = round(statistics.fmean(values), 4)
            strat_summary[f"{field}_min"] = round(min(values), 4)
            strat_summary[f"{field}_max"] = round(max(values), 4)
        aggregate[strategy] = strat_summary
    return aggregate


class SingleAgentPitchRunner:
    """Single-pass baseline for methodology comparisons."""

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        temperature: float = 0.0,
        seed: int = 42,
        enable_trends: bool = True,
    ) -> None:
        llm = ChatOpenAI(model=model, temperature=temperature, seed=seed)
        self.llm = llm.with_structured_output(SingleAgentOutput)
        self.validator = SourceValidatorAgent(ChatOpenAI(model=model, temperature=temperature, seed=seed))
        self.pitch = PitchDeckGeneratorAgent(ChatOpenAI(model=model, temperature=temperature, seed=seed))
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
        generate_ppt: bool = False,
    ) -> Dict[str, Any]:
        result: SingleAgentOutput = self.llm.invoke(self.prompt.format_messages(idea=idea))

        query = f"startup market size competitors trends for: {result.refined_idea}"
        search_payload = self.search_tool.search(query)

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
            "tool_audit": [
                {
                    "agent": "single_agent",
                    "tool": "llm_single_pass",
                    "status": "ok",
                },
                {
                    "agent": "single_agent",
                    "tool": "linkup_search",
                    "status": search_payload.get("status", "unknown"),
                    "source_count": len(search_payload.get("sources", [])),
                    "error": search_payload.get("error", ""),
                },
                {
                    "agent": "single_agent",
                    "tool": "google_trends",
                    "status": trends_payload.get("status", "unknown"),
                    "keyword_count": len(trends_payload.get("keywords", [])),
                    "error": trends_payload.get("error", ""),
                },
            ],
        }

        # Reuse the same validator logic for fair quality scoring across strategies.
        validated_update = self.validator.run(baseline_state)
        baseline_state.update(validated_update)

        if generate_ppt:
            pitch_update = self.pitch.run(baseline_state)
            baseline_state.update(pitch_update)

        return baseline_state


def run_methodology_comparison(
    idea: str,
    model: str = "gpt-4.1-nano",
    temperature: float = 0.0,
    seed: int = 42,
    strict_tools: bool = True,
    enable_trends: bool = True,
    compare_runs: int = 1,
    strategies: List[str] | None = None,
    validation_threshold: int = 70,
    max_validation_retries: int = 1,
    generate_ppt: bool = False,
    thread_prefix: str = "methodology",
) -> Dict[str, Any]:
    strategies = strategies or ["single_agent", "multi_agent", "adaptive_controller"]
    normalized = [s.strip().lower() for s in strategies if s.strip()]

    allowed = {"single_agent", "multi_agent", "adaptive_controller"}
    invalid = [s for s in normalized if s not in allowed]
    if invalid:
        raise ValueError(f"Unsupported strategies: {invalid}. Allowed: {sorted(allowed)}")

    single_runner = None
    if "single_agent" in normalized:
        single_runner = SingleAgentPitchRunner(
            model=model,
            temperature=temperature,
            seed=seed,
            enable_trends=enable_trends,
        )

    multi_runner = None
    if "multi_agent" in normalized:
        multi_runner = StartupPitchRefinery(
            model=model,
            temperature=temperature,
            seed=seed,
            strict_tools=strict_tools,
            enable_trends=enable_trends,
            generate_pitch=generate_ppt,
        )

    adaptive_runner = None
    if "adaptive_controller" in normalized:
        adaptive_runner = StartupPitchRefinery(
            model=model,
            temperature=temperature,
            seed=seed,
            strict_tools=strict_tools,
            enable_trends=enable_trends,
            generate_pitch=generate_ppt,
            controller_policy="adaptive",
        )

    run_rows: List[Dict[str, Any]] = []
    states: Dict[str, Dict[str, Any]] = {}

    for run_idx in range(compare_runs):
        for strategy in normalized:
            start = time.perf_counter()
            if strategy == "single_agent":
                assert single_runner is not None
                state = single_runner.run(
                    idea=idea,
                    validation_threshold=validation_threshold,
                    generate_ppt=generate_ppt,
                )
            elif strategy == "multi_agent":
                assert multi_runner is not None
                state = multi_runner.run(
                    idea=idea,
                    thread_id=f"{thread_prefix}-{strategy}-{run_idx}",
                    max_validation_retries=max_validation_retries,
                    validation_threshold=validation_threshold,
                )
            else:
                assert adaptive_runner is not None
                state = adaptive_runner.run(
                    idea=idea,
                    thread_id=f"{thread_prefix}-{strategy}-{run_idx}",
                    max_validation_retries=max_validation_retries,
                    validation_threshold=validation_threshold,
                )

            runtime = time.perf_counter() - start
            metrics = _extract_metrics(state, strategy, runtime)
            run_key = f"{strategy}_run_{run_idx}"
            states[run_key] = state
            run_rows.append(
                {
                    "run_index": run_idx,
                    "strategy": strategy,
                    "metrics": metrics,
                }
            )

    aggregate = _aggregate_metrics([row["metrics"] for row in run_rows])

    recommendation = None
    if aggregate:
        ranked = sorted(
            aggregate.items(),
            key=lambda kv: (
                -kv[1].get("reliability_score_mean", 0.0),
                kv[1].get("runtime_seconds_mean", float("inf")),
                kv[1].get("token_proxy_total_mean", float("inf")),
            ),
        )
        best_name, best_stats = ranked[0]
        recommendation = {
            "best_strategy": best_name,
            "selection_rule": "highest mean reliability score, then lower runtime and token proxy",
            "stats": best_stats,
        }

    return {
        "metadata": {
            "idea": idea,
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "strategies": normalized,
            "compare_runs": compare_runs,
            "validation_threshold": validation_threshold,
            "max_validation_retries": max_validation_retries,
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
