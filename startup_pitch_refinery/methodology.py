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
    token_proxy_total = _estimate_tokens_proxy(state)
    token_usage = state.get("token_usage", {}) or {}
    prompt_tokens_total = _safe_int(token_usage.get("prompt_tokens", 0), 0)
    completion_tokens_total = _safe_int(token_usage.get("completion_tokens", 0), 0)
    actual_total_tokens = _safe_int(token_usage.get("total_tokens", 0), 0)
    reliability_per_1k_token = round(
        (reliability_score * 1000.0) / max(1, token_proxy_total), 4
    )
    reliability_per_1k_actual_token = round(
        (reliability_score * 1000.0) / max(1, actual_total_tokens), 4
    )
    reliability_per_second = round(
        reliability_score / max(0.001, float(runtime_seconds)), 4
    )
    validator_scores = [
        _safe_int(entry.get("reliability_score", 0), 0)
        for entry in (state.get("tool_audit", []) or [])
        if entry.get("agent") == "source_validator"
        and entry.get("tool") == "llm_claim_verifier"
    ]
    retry_score_delta = 0
    retry_effective = 0
    if len(validator_scores) >= 2:
        retry_score_delta = validator_scores[-1] - validator_scores[0]
        retry_effective = 1 if retry_score_delta > 0 else 0

    metrics = {
        "strategy": strategy,
        "controller_mode": state.get("controller_mode", "n/a"),
        "controller_budget_override": 1 if state.get("controller_budget_override", False) else 0,
        "runtime_seconds": round(runtime_seconds, 3),
        "reliability_score": reliability_score,
        "claims_total": len(validation.get("claims", []) if validation else []),
        "supported_ratio": ratios["supported_ratio"],
        "weak_or_better_ratio": ratios["weak_or_better_ratio"],
        "market_sources_count": len(state.get("market_sources", [])),
        "tool_calls": len(state.get("tool_audit", [])),
        "retry_count": _safe_int(state.get("retry_count", 0), 0),
        "decomposition_depth_realized": _safe_int(state.get("decomposition_depth_realized", 0), 0),
        "needs_revision": bool(state.get("needs_revision", False)),
        "trend_status": (state.get("trend_signals") or {}).get("status", "unknown"),
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "actual_total_tokens": actual_total_tokens,
        "token_proxy_total": token_proxy_total,
        "reliability_per_1k_token": reliability_per_1k_token,
        "reliability_per_1k_actual_token": reliability_per_1k_actual_token,
        "reliability_per_second": reliability_per_second,
        "validator_passes": len(validator_scores),
        "retry_score_delta": retry_score_delta,
        "retry_effective": retry_effective,
        "budget_hit": 1 if budget_hit else 0,
        "budget_violation_count": budget_violation_count,
        "finished_under_budget": 0 if budget_hit else 1,
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
        "supported_ratio",
        "weak_or_better_ratio",
        "market_sources_count",
        "tool_calls",
        "retry_count",
        "decomposition_depth_realized",
        "token_proxy_total",
        "prompt_tokens_total",
        "completion_tokens_total",
        "actual_total_tokens",
        "reliability_per_1k_token",
        "reliability_per_1k_actual_token",
        "reliability_per_second",
        "validator_passes",
        "retry_score_delta",
        "retry_effective",
        "reliability_gain_vs_single",
        "depth_delta_vs_single",
        "depth_vs_reliability_gain",
        "controller_budget_override",
        "budget_hit",
        "budget_violation_count",
        "finished_under_budget",
    ]
    aggregate: Dict[str, Dict[str, float]] = {}
    for strategy, rows in grouped.items():
        strat_summary: Dict[str, float] = {"runs": float(len(rows))}
        for field in numeric_fields:
            values = [float(r[field]) for r in rows]
            strat_summary[f"{field}_mean"] = round(statistics.fmean(values), 4)
            strat_summary[f"{field}_min"] = round(min(values), 4)
            strat_summary[f"{field}_max"] = round(max(values), 4)

        mode_counts: Dict[str, int] = {}
        for row in rows:
            mode = str(row.get("controller_mode", "n/a")).strip().lower() or "n/a"
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        strat_summary["mode_distribution"] = mode_counts
        if rows:
            strat_summary["mode_distribution_pct"] = {
                mode: round(count / len(rows), 4) for mode, count in mode_counts.items()
            }
        aggregate[strategy] = strat_summary
    return aggregate


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
        temperature: float = 0.0,
        seed: int = 42,
        enable_trends: bool = True,
        output_dir: str = "output",
    ) -> None:
        llm = ChatOpenAI(model=model, temperature=temperature, seed=seed)
        self.llm = llm.with_structured_output(SingleAgentOutput, include_raw=True)
        self.validator = SourceValidatorAgent(ChatOpenAI(model=model, temperature=temperature, seed=seed))
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
    max_tool_calls: int | None = None,
    max_token_proxy: int | None = None,
    max_total_tokens: int | None = None,
    max_runtime_seconds: float | None = None,
    generate_ppt: bool = False,
    thread_prefix: str = "methodology",
    output_dir: str = "output",
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
            output_dir=output_dir,
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
            output_dir=output_dir,
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
            output_dir=output_dir,
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
                )
            else:
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

    recommendation = None
    if aggregate:
        ranked = sorted(
            aggregate.items(),
            key=lambda kv: (
                kv[1].get("budget_violation_count_mean", float("inf")),
                -kv[1].get("reliability_score_mean", 0.0),
                kv[1].get("runtime_seconds_mean", float("inf")),
                kv[1].get("token_proxy_total_mean", float("inf")),
            ),
        )
        best_name, best_stats = ranked[0]
        recommendation = {
            "best_strategy": best_name,
            "selection_rule": "lowest budget violations, then highest mean reliability score, then lower runtime and token proxy",
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
        "idea",
        "run_index",
        "strategy",
        "controller_mode",
        "reliability_score",
        "supported_ratio",
        "runtime_seconds",
        "actual_total_tokens",
        "token_proxy_total",
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
        "runtime_seconds_mean",
        "actual_total_tokens_mean",
        "token_proxy_total_mean",
        "supported_ratio_mean",
        "decomposition_depth_realized_mean",
        "reliability_gain_vs_single_mean",
        "budget_hit_mean",
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


def run_methodology_batch_comparison(
    ideas: List[str],
    model: str = "gpt-4.1-nano",
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
    cleaned_ideas = [i.strip() for i in ideas if i and i.strip()]
    if not cleaned_ideas:
        raise ValueError("No ideas provided for batch comparison.")

    per_idea_reports: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    all_states: Dict[str, Dict[str, Any]] = {}

    for idx, idea in enumerate(cleaned_ideas):
        report = run_methodology_comparison(
            idea=idea,
            model=model,
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
                "idea": idea,
                "aggregate": report.get("aggregate", {}),
                "recommendation": report.get("recommendation"),
            }
        )
        for row in report.get("runs", []):
            copied = dict(row)
            copied["idea_index"] = idx
            copied["idea"] = idea
            all_rows.append(copied)
        for key, state in report.get("states", {}).items():
            all_states[f"idea_{idx}:{key}"] = state

    _attach_relative_metrics(all_rows)
    aggregate = _aggregate_metrics([row["metrics"] for row in all_rows])
    recommendation = None
    if aggregate:
        ranked = sorted(
            aggregate.items(),
            key=lambda kv: (
                kv[1].get("budget_violation_count_mean", float("inf")),
                -kv[1].get("reliability_score_mean", 0.0),
                kv[1].get("runtime_seconds_mean", float("inf")),
                kv[1].get("token_proxy_total_mean", float("inf")),
            ),
        )
        best_name, best_stats = ranked[0]
        recommendation = {
            "best_strategy": best_name,
            "selection_rule": "lowest budget violations, then highest mean reliability score, then lower runtime and token proxy",
            "stats": best_stats,
        }

    return {
        "metadata": {
            "ideas_count": len(cleaned_ideas),
            "ideas": cleaned_ideas,
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "strategies": (strategies or ["single_agent", "multi_agent", "adaptive_controller"]),
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
