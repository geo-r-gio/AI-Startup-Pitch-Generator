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
    DirectStrategyAgent,
    IdeaRefinementAgent,
    MarketResearchAgent,
    PitchDeckGeneratorAgent,
    SourceValidatorAgent,
    SupervisorAgent,
)
from startup_pitch_refinery.state import PitchState


class StartupPitchRefinery:
    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        temperature: float = 0.0,
        seed: int = 42,
        strict_tools: bool = True,
        enable_trends: bool = True,
        generate_pitch: bool = True,
        controller_policy: str = "fixed",
        output_dir: str = "output",
    ):
        llm = ChatOpenAI(model=model, temperature=temperature, seed=seed)
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
        self.validator_agent = SourceValidatorAgent(llm)
        self.direct_agent = (
            DirectStrategyAgent(llm, strict_tools=strict_tools, enable_trends=enable_trends)
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
        retry_state = {**base_state, "retry_count": current_retry + 1}
        updates = self._run_node_with_budget(retry_state, self.market_agent.run, "market_retry")
        updates["retry_count"] = current_retry + 1
        return updates

    def _run_supervisor(self, state: PitchState):
        return self._run_node_with_budget(state, self.supervisor.run, "supervisor")

    def _run_idea(self, state: PitchState):
        return self._run_node_with_budget(state, self.idea_agent.run, "idea")

    def _run_controller(self, state: PitchState):
        if self.controller_agent is None:
            return {}
        return self._run_node_with_budget(state, self.controller_agent.run, "controller")

    def _run_market(self, state: PitchState):
        return self._run_node_with_budget(state, self.market_agent.run, "market")

    def _run_validator(self, state: PitchState):
        return self._run_node_with_budget(state, self.validator_agent.run, "validator")

    def _run_direct(self, state: PitchState):
        if self.direct_agent is None:
            return {}
        return self._run_node_with_budget(state, self.direct_agent.run, "direct")

    def _run_pitch(self, state: PitchState):
        if self.pitch_agent is None:
            return {}
        return self._run_node_with_budget(state, self.pitch_agent.run, "pitch")

    def _route_after_validation(self, state: PitchState) -> str:
        if bool(self._sget(state, "budget_hit", False)):
            return "stop"
        controller_mode = str(self._sget(state, "controller_mode", "")).strip().lower()
        if controller_mode == "shallow":
            return "continue"
        if controller_mode == "direct":
            return "continue"

        needs_revision = bool(self._sget(state, "needs_revision", False))
        retry_count = int(self._sget(state, "retry_count", 0))
        max_retries = int(self._sget(state, "max_validation_retries", 1))
        if needs_revision and retry_count < max_retries:
            return "retry_market"
        return "continue"

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
            depth = 1
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
        workflow.add_node("market_retry", self._market_retry)
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
                "retry_market": "market_retry",
                "continue": "business",
            },
        )
        workflow.add_conditional_edges(
            "market_retry",
            self._route_by_budget,
            {
                "stop": END,
                "continue": "validator",
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

        return self.graph.invoke(
            PitchState(
                idea=idea,
                controller_policy=self.controller_policy,
                max_validation_retries=max_validation_retries,
                validation_threshold=validation_threshold,
                max_tool_calls=normalized_max_tool_calls,
                max_token_proxy=normalized_max_token_proxy,
                max_total_tokens=normalized_max_total_tokens,
                max_runtime_seconds=normalized_max_runtime_seconds,
                runtime_started_at=time.time(),
            ).model_dump(),
            config={"configurable": {"thread_id": thread_id}},
        )
