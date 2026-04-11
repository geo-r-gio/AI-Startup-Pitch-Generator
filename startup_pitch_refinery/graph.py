from __future__ import annotations

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
        self.pitch_agent = PitchDeckGeneratorAgent(llm) if self.generate_pitch else None
        self.checkpointer = MemorySaver()

        self.graph = self._build_graph()

    @staticmethod
    def _sget(state: PitchState | Dict[str, Any], key: str, default: Any = None) -> Any:
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)

    def _market_retry(self, state: PitchState):
        current_retry = self._sget(state, "retry_count", 0)
        base_state = state if isinstance(state, dict) else state.model_dump()
        retry_state = {**base_state, "retry_count": current_retry + 1}
        updates = self.market_agent.run(retry_state)
        updates["retry_count"] = current_retry + 1
        return updates

    def _route_after_validation(self, state: PitchState) -> str:
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
        mode = str(self._sget(state, "controller_mode", "shallow")).strip().lower()
        if mode not in {"direct", "shallow", "recursive"}:
            return "shallow"
        return mode

    def _run_business_with_depth(self, state: PitchState):
        updates = self.business_agent.run(state)
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

        workflow.add_node("supervisor", self.supervisor.run)
        workflow.add_node("idea", self.idea_agent.run)
        if self.controller_policy == "adaptive" and self.controller_agent is not None:
            workflow.add_node("controller", self.controller_agent.run)
        workflow.add_node("market", self.market_agent.run)
        workflow.add_node("validator", self.validator_agent.run)
        workflow.add_node("market_retry", self._market_retry)
        workflow.add_node("business", self._run_business_with_depth)
        if self.controller_policy == "adaptive" and self.direct_agent is not None:
            workflow.add_node("direct", self.direct_agent.run)
        if self.generate_pitch and self.pitch_agent is not None:
            workflow.add_node("pitch", self.pitch_agent.run)

        workflow.add_edge(START, "supervisor")
        workflow.add_edge("supervisor", "idea")
        if self.controller_policy == "adaptive" and self.controller_agent is not None:
            workflow.add_edge("idea", "controller")
            workflow.add_conditional_edges(
                "controller",
                self._route_after_controller,
                {
                    "direct": "direct",
                    "shallow": "market",
                    "recursive": "market",
                },
            )
        else:
            workflow.add_edge("idea", "market")
        workflow.add_edge("market", "validator")
        workflow.add_conditional_edges(
            "validator",
            self._route_after_validation,
            {
                "retry_market": "market_retry",
                "continue": "business",
            },
        )
        workflow.add_edge("market_retry", "validator")
        if self.controller_policy == "adaptive" and self.direct_agent is not None:
            if self.generate_pitch and self.pitch_agent is not None:
                workflow.add_edge("direct", "pitch")
            else:
                workflow.add_edge("direct", END)
        if self.generate_pitch and self.pitch_agent is not None:
            workflow.add_edge("business", "pitch")
            workflow.add_edge("pitch", END)
        else:
            workflow.add_edge("business", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def run(
        self,
        idea: str,
        thread_id: str = "default-thread",
        max_validation_retries: int = 1,
        validation_threshold: int = 70,
    ) -> PitchState:
        return self.graph.invoke(
            PitchState(
                idea=idea,
                controller_policy=self.controller_policy,
                max_validation_retries=max_validation_retries,
                validation_threshold=validation_threshold,
            ).model_dump(),
            config={"configurable": {"thread_id": thread_id}},
        )
