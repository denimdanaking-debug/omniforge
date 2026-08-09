"""OmniForge Phase 6 provider recovery engine.

Public API surface for health state machine, failure-domain propagation,
reserve capacity, quota-aware balancing, outage survival, and telemetry.
"""

from src.recovery.clock import Clock, FixedClock, ManualClock, SystemClock
from src.recovery.failure_domain import FailureDomainIndex
from src.recovery.quota_balance import QuotaBalancer, QuotaCandidate, QuotaPressure
from src.recovery.reserve import ReserveCapacityPolicy, evaluate_reserve_eligibility
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
]
