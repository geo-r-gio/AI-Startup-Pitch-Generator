from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PitchState(BaseModel):
    idea: str
    controller_policy: str = "fixed"
    controller_mode: Optional[str] = None
    controller_mode_initial: Optional[str] = None
    controller_budget_override: bool = False
    controller_confidence: Optional[float] = None
    controller_rationale: Optional[str] = None
    controller_expected_cost: Dict[str, Any] = Field(default_factory=dict)
    controller_budget_snapshot: Dict[str, Any] = Field(default_factory=dict)
    controller_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    decomposition_depth_target: Optional[int] = None
    decomposition_depth_realized: Optional[int] = None
    task_plan: Optional[List[str]] = None
    refined_idea: Optional[str] = None
    market_analysis: Optional[str] = None
    market_sources: List[str] = Field(default_factory=list)
    market_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    trend_signals: Dict[str, Any] = Field(default_factory=dict)
    validation_report: Optional[Dict[str, Any]] = None
    validated_market_analysis: Optional[str] = None
    business_model: Optional[str] = None
    financial_assumptions: Dict[str, Any] = Field(default_factory=dict)
    scenario_analysis: Dict[str, Any] = Field(default_factory=dict)
    pitch_content: Optional[Dict[str, Any]] = None
    ppt_path: Optional[str] = None
    tool_audit: List[Dict[str, Any]] = Field(default_factory=list)
    retry_count: int = 0
    max_validation_retries: int = 1
    validation_threshold: int = 70
    needs_revision: bool = False
    max_tool_calls: Optional[int] = None
    max_token_proxy: Optional[int] = None
    max_total_tokens: Optional[int] = None
    max_runtime_seconds: Optional[float] = None
    runtime_started_at: Optional[float] = None
    runtime_elapsed_seconds: float = 0.0
    tool_calls_current: int = 0
    token_proxy_current: int = 0
    token_usage: Dict[str, int] = Field(default_factory=dict)
    prompt_tokens_current: int = 0
    completion_tokens_current: int = 0
    total_tokens_current: int = 0
    budget_hit: bool = False
    budget_hit_reasons: List[str] = Field(default_factory=list)
    budget_remaining: Dict[str, Any] = Field(default_factory=dict)
