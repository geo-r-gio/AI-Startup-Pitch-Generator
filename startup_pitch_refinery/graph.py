from __future__ import annotations

from typing import Any, Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI

from startup_pitch_refinery.agents import (
    BusinessModelAgent,
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
    ):
        llm = ChatOpenAI(model=model, temperature=temperature, seed=seed)

        self.supervisor = SupervisorAgent()
        self.idea_agent = IdeaRefinementAgent(llm)
        self.market_agent = MarketResearchAgent(
            llm,
            strict_tools=strict_tools,
            enable_trends=enable_trends,
        )
        self.validator_agent = SourceValidatorAgent(llm)
        self.business_agent = BusinessModelAgent(llm, strict_tools=strict_tools)
        self.pitch_agent = PitchDeckGeneratorAgent(llm)
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
        needs_revision = bool(self._sget(state, "needs_revision", False))
        retry_count = int(self._sget(state, "retry_count", 0))
        max_retries = int(self._sget(state, "max_validation_retries", 1))
        if needs_revision and retry_count < max_retries:
            return "retry_market"
        return "continue"

    def _build_graph(self):
        workflow = StateGraph(PitchState)

        workflow.add_node("supervisor", self.supervisor.run)
        workflow.add_node("idea", self.idea_agent.run)
        workflow.add_node("market", self.market_agent.run)
        workflow.add_node("validator", self.validator_agent.run)
        workflow.add_node("market_retry", self._market_retry)
        workflow.add_node("business", self.business_agent.run)
        workflow.add_node("pitch", self.pitch_agent.run)

        workflow.add_edge(START, "supervisor")
        workflow.add_edge("supervisor", "idea")
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
        workflow.add_edge("business", "pitch")
        workflow.add_edge("pitch", END)

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
                max_validation_retries=max_validation_retries,
                validation_threshold=validation_threshold,
            ).model_dump(),
            config={"configurable": {"thread_id": thread_id}},
        )
