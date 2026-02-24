from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PitchState(BaseModel):
    idea: str
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
