"""OmniForge Phase 11 — Empirical Model Intelligence.

The performance package provides an immutable, append-only performance-event
ledger and deterministic derived statistics. It consumes normalized outcomes
from execution, recovery, review, context construction, and integration
advancement without modifying authority semantics.
"""

from src.performance.attribution import (
    PerformanceAttribution,
    affects_model_quality,
    attribution_from_failure_classification,
    attribution_from_task_outcome,
    is_infrastructure_attribution,
)
from src.performance.builder import StatisticsBuilder
from src.performance.event import (
    AcceptanceStatus,
    AuthorityAdherenceStatus,
    Cost,
    CostState,
    FindingDisposition,
    OutcomeCategory,
    PerformanceEvent,
    PerformanceEventType,
    RepairMetadata,
    Usage,
    event_identity,
    performance_event_fingerprint,
)
from src.performance.ledger import PerformanceLedger
from src.performance.persistence import (
    CURRENT_PERFORMANCE_SCHEMA_VERSION,
    performance_state_from_dict,
    performance_state_to_dict,
)
from src.performance.statistics import (
    ContextStrategyStatistics,
    LanguageFrameworkStatistics,
    ModelRoleDimensionalStatistics,
    ModelRoleStatistics,
    PerformanceStatisticsBundle,
    ProjectPerformanceStatistics,
    ReviewerStatistics,
    RiskDifficultyStatistics,
    RouteStatistics,
    TaskLifecycleStatistics,
    safe_rate,
)

__all__ = [
    "PerformanceAttribution",
    "affects_model_quality",
    "attribution_from_failure_classification",
    "attribution_from_task_outcome",
    "is_infrastructure_attribution",
    "StatisticsBuilder",
    "AcceptanceStatus",
    "AuthorityAdherenceStatus",
    "Cost",
    "CostState",
    "FindingDisposition",
    "OutcomeCategory",
    "PerformanceEvent",
    "PerformanceEventType",
    "RepairMetadata",
    "Usage",
    "event_identity",
    "performance_event_fingerprint",
    "PerformanceLedger",
    "CURRENT_PERFORMANCE_SCHEMA_VERSION",
    "performance_state_from_dict",
    "performance_state_to_dict",
    "ContextStrategyStatistics",
    "LanguageFrameworkStatistics",
    "ModelRoleDimensionalStatistics",
    "ModelRoleStatistics",
    "PerformanceStatisticsBundle",
    "ProjectPerformanceStatistics",
    "ReviewerStatistics",
    "RiskDifficultyStatistics",
    "RouteStatistics",
    "TaskLifecycleStatistics",
    "safe_rate",
]
