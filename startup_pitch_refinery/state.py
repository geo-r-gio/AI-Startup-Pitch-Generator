from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PitchState(BaseModel):
    idea: str

    # Controller / routing
    controller_policy: str = "fixed"
    forced_controller_mode: Optional[str] = None
    controller_mode: Optional[str] = None
    controller_mode_initial: Optional[str] = None
    controller_mode_realized: Optional[str] = None
    controller_escalated: bool = False
    controller_escalation_reason: Optional[str] = None
    direct_precheck_allowed: bool = False
    direct_precheck_skipped: bool = False
    direct_precheck_escalated: bool = False
    direct_precheck_accepted: bool = False
    direct_precheck_score: int = 0
    direct_precheck_incremental_tokens: int = 0
    direct_precheck_gate_bucket: Optional[str] = None
    direct_precheck_gate_reason: Optional[str] = None
    direct_precheck_gate_block_reasons: List[str] = Field(default_factory=list)
    direct_precheck_gate_probability: float = 0.0
    direct_precheck_gate_threshold: float = 0.0
    direct_precheck_gate_expected_net_tokens: float = 0.0
    controller_budget_override: bool = False
    controller_confidence: Optional[float] = None
    controller_rationale: Optional[str] = None
    controller_expected_cost: Dict[str, Any] = Field(default_factory=dict)
    controller_budget_snapshot: Dict[str, Any] = Field(default_factory=dict)
    controller_scorecard: Dict[str, Any] = Field(default_factory=dict)
    controller_decisions: List[Dict[str, Any]] = Field(default_factory=list)

    # Adaptive ablations / policy configuration
    adaptive_retry_enabled: bool = True
    adaptive_checkpoint_enabled: bool = True
    adaptive_policy: str = "bayes"
    policy_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "posterior_accept_min": 0.35,
            "retrieval_quality_min": 0.55,
            "roi_min": 1.25,
            "eu_margin": 0.5,
            "max_bundle_claims": 1.0,
            "max_repair_rounds": 1.0,
            "tail_token_reserve": 3500.0,
            "min_remaining_tokens_for_repair": 4000.0,
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
        }
    )
    utility_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "lambda_tok": 0.5,
            "lambda_time": 0.03,
            "lambda_crit": 4.0,
            "lambda_low": 6.0,
            "z_uncertainty": 1.0,
            "repair_success_margin": 1.0,
            "gain_kappa": 2.0,
            "token_kappa": 3.0,
            "seconds_kappa": 5.0,
        }
    )

    # Decomposition tracking
    decomposition_graph: Dict[str, Any] = Field(default_factory=dict)
    retry_budget_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    decomposition_depth_target: Optional[int] = None
    decomposition_depth_realized: Optional[int] = None

    # Planning / idea refinement
    task_plan: Optional[List[str]] = None
    refined_idea: Optional[str] = None
    shared_refinement_locked: bool = False

    # Market research
    market_analysis: Optional[str] = None
    market_sources: List[str] = Field(default_factory=list)
    market_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    trend_signals: Dict[str, Any] = Field(default_factory=dict)

    # Claim-level repair state
    market_repair_context: Dict[str, Any] = Field(default_factory=dict)
    market_repair_history: List[Dict[str, Any]] = Field(default_factory=list)
    claim_units: List[Dict[str, Any]] = Field(default_factory=list)
    failing_claims: List[Dict[str, Any]] = Field(default_factory=list)
    claim_evidence: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    repair_plan: List[Dict[str, Any]] = Field(default_factory=list)
    repair_patches: List[Dict[str, Any]] = Field(default_factory=list)
    micro_validation: Dict[str, Any] = Field(default_factory=dict)
    repair_validation_history: List[Dict[str, Any]] = Field(default_factory=list)
    accepted_patch_count: int = 0
    rejected_patch_count: int = 0
    policy_success_count: int = 0
    policy_failure_count: int = 0
    repair_rounds: int = 0

    # Bayesian retrieval / retry policy state
    retrieval_diagnostics: List[Dict[str, Any]] = Field(default_factory=list)
    action_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    retry_policy_decision: Dict[str, Any] = Field(default_factory=dict)
    selected_action: Dict[str, Any] = Field(default_factory=dict)
    selected_action_history: List[Dict[str, Any]] = Field(default_factory=list)
    claim_posteriors: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    policy_stats: Dict[str, Any] = Field(default_factory=dict)
    retry_expected_utility: float = 0.0
    retry_expected_gain: float = 0.0
    retry_expected_cost_tokens: int = 0
    retry_expected_cost_seconds: float = 0.0
    posterior_acceptance_probability: float = 0.0

    # Validation / checkpointing
    validation_report: Optional[Dict[str, Any]] = None
    validation_snapshots: List[Dict[str, Any]] = Field(default_factory=list)
    checkpoint_history: List[Dict[str, Any]] = Field(default_factory=list)
    baseline_checkpoint: Dict[str, Any] = Field(default_factory=dict)
    best_checkpoint: Dict[str, Any] = Field(default_factory=dict)
    selected_validation_checkpoint: Dict[str, Any] = Field(default_factory=dict)
    selected_checkpoint: Dict[str, Any] = Field(default_factory=dict)
    adaptive_checkpoint_selection: Dict[str, Any] = Field(default_factory=dict)
    checkpoint_utilities: List[Dict[str, Any]] = Field(default_factory=list)
    validated_market_analysis: Optional[str] = None
    needs_revision: bool = False
    retry_count: int = 0
    max_validation_retries: int = 1
    validation_threshold: int = 70

    # Controller diagnostics / analysis
    historical_mode_stats: Dict[str, Any] = Field(default_factory=dict)
    controller_features_bucket: Dict[str, Any] = Field(default_factory=dict)
    retry_roi_estimate: float = 0.0
    over_decomposition_flag: bool = False

    # Business model / pitch output
    business_model: Optional[str] = None
    financial_assumptions: Dict[str, Any] = Field(default_factory=dict)
    scenario_analysis: Dict[str, Any] = Field(default_factory=dict)
    pitch_content: Optional[Dict[str, Any]] = None
    ppt_path: Optional[str] = None

    # Tool audit / metrics
    tool_audit: List[Dict[str, Any]] = Field(default_factory=list)
    token_usage: Dict[str, int] = Field(default_factory=dict)

    # Budget controls
    max_tool_calls: Optional[int] = None
    max_token_proxy: Optional[int] = None
    max_total_tokens: Optional[int] = None
    max_runtime_seconds: Optional[float] = None
    runtime_started_at: Optional[float] = None
    runtime_elapsed_seconds: float = 0.0
    tool_calls_current: int = 0
    token_proxy_current: int = 0
    prompt_tokens_current: int = 0
    completion_tokens_current: int = 0
    total_tokens_current: int = 0
    budget_hit: bool = False
    budget_hit_reasons: List[str] = Field(default_factory=list)
    budget_remaining: Dict[str, Any] = Field(default_factory=dict)



















