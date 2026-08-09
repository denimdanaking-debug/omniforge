"""OmniForge Phase 6 and Phase 10 recovery engine.

Public API surface for provider health state machine, failure-domain propagation,
reserve capacity, quota-aware balancing, outage survival, failure-type-aware
retry/escalation, and telemetry.
"""

from src.recovery.authority_recovery import (
    AuthorityRecoveryResult,
    apply_authority_violation_recovery,
)
from src.recovery.backoff import BackoffPolicy, HotLoopPolicy, RetryBudget
from src.recovery.clock import Clock, FixedClock, ManualClock, SystemClock
from src.recovery.context_recovery import (
    ContextRebuildResult,
    build_context_overflow_metadata,
    context_rebuild_attempt_exceeds_budget,
    context_recovery_evidence,
)
from src.recovery.evidence import (
    ImplementationFailureEvidence,
    PlanningRejectionEvidence,
    implementation_failure_signature,
    planning_failure_signature,
)
from src.recovery.failure_classification import (
    AuthorityViolationData,
    ContextOverflowMetadata,
    FailureCategory,
    FailureClassification,
    FailureClassifier,
    FailureClassifierInput,
    FailureDomain,
    FailureSubtype,
    PlanningValidationResult,
    Retryability,
    StructuredOutputValidationResult,
    ValidationResultSummary,
    failure_classification_fingerprint,
)
from src.recovery.failure_domain import FailureDomainIndex
from src.recovery.quota_balance import QuotaBalancer, QuotaCandidate, QuotaPressure
from src.recovery.recovery_coordinator import (
    RecoveryAction,
    RecoveryCandidate,
    RecoveryCoordinator,
    RecoveryCoordinatorInput,
    RecoveryDecision,
)
from src.recovery.reserve import ReserveCapacityPolicy, evaluate_reserve_eligibility
from src.recovery.retry_policy import FailureRecoveryPolicy
from src.recovery.retry_state import (
    FailureAttemptRecord,
    RetryLedger,
    RetryType,
    WaitState,
)
from src.recovery.scheduler import RecheckPolicy, RecoveryScheduler
from src.recovery.signals import (
    ProviderSignal,
    SignalKind,
    signal_from_error,
    signal_from_health_check,
    signal_from_quota,
    signal_from_response,
)
from src.recovery.state_machine import (
    HealthStateMachine,
    HealthTransition,
    RouteRecoveryState,
    StateMachineConfig,
)
from src.recovery.survival import (
    DispatchChoice,
    DispatchDecision,
    OutageSurvivalEngine,
    PersistedWait,
    SurvivalCandidate,
    WaitReason,
)
from src.recovery.telemetry import RecoveryEvent, RecoveryEventType, RecoveryTelemetryBuffer

__all__ = [
    # Phase 6
    "Clock",
    "FixedClock",
    "ManualClock",
    "SystemClock",
    "FailureDomainIndex",
    "QuotaBalancer",
    "QuotaCandidate",
    "QuotaPressure",
    "ReserveCapacityPolicy",
    "evaluate_reserve_eligibility",
    "RecheckPolicy",
    "RecoveryScheduler",
    "ProviderSignal",
    "SignalKind",
    "signal_from_error",
    "signal_from_health_check",
    "signal_from_quota",
    "signal_from_response",
    "HealthStateMachine",
    "HealthTransition",
    "RouteRecoveryState",
    "StateMachineConfig",
    "DispatchChoice",
    "DispatchDecision",
    "OutageSurvivalEngine",
    "PersistedWait",
    "SurvivalCandidate",
    "WaitReason",
    "RecoveryEvent",
    "RecoveryEventType",
    "RecoveryTelemetryBuffer",
    "BackoffPolicy",
    "HotLoopPolicy",
    "RetryBudget",
    # Phase 10
    "AuthorityRecoveryResult",
    "apply_authority_violation_recovery",
    "ContextRebuildResult",
    "build_context_overflow_metadata",
    "context_recovery_evidence",
    "context_rebuild_attempt_exceeds_budget",
    "ImplementationFailureEvidence",
    "PlanningRejectionEvidence",
    "implementation_failure_signature",
    "planning_failure_signature",
    "AuthorityViolationData",
    "ContextOverflowMetadata",
    "FailureCategory",
    "FailureClassification",
    "FailureClassifier",
    "FailureClassifierInput",
    "FailureDomain",
    "FailureSubtype",
    "PlanningValidationResult",
    "Retryability",
    "StructuredOutputValidationResult",
    "ValidationResultSummary",
    "failure_classification_fingerprint",
    "RecoveryAction",
    "RecoveryCandidate",
    "RecoveryCoordinator",
    "RecoveryCoordinatorInput",
    "RecoveryDecision",
    "FailureRecoveryPolicy",
    "FailureAttemptRecord",
    "RetryLedger",
    "RetryType",
    "WaitState",
]
