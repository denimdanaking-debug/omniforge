"""Tests for Phase 10 recovery coordinator decisions."""

from __future__ import annotations

import datetime

import pytest

from src.policy.risk import RiskLevel
from src.providers.errors import ProviderError, ProviderErrorCode
from src.providers.identity import ProviderHealth, ProviderQuotaState, QuotaSignal
from src.recovery import FixedClock
from src.recovery.failure_classification import (
    AuthorityViolationData,
    ContextOverflowMetadata,
    FailureClassifierInput,
    PlanningValidationResult,
    StructuredOutputValidationResult,
    ValidationResultSummary,
)
from src.recovery.recovery_coordinator import (
    RecoveryAction,
    RecoveryCandidate,
    RecoveryCoordinator,
    RecoveryCoordinatorInput,
)
from src.recovery.retry_policy import FailureRecoveryPolicy
from src.recovery.retry_state import RetryLedger, RetryType
from src.recovery.state_machine import RouteRecoveryState
from src.routing.capabilities import ModelCapabilities
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity
from src.routing.policy import RoutingPin
from src.routing.roles import ExecutionRole


@pytest.fixture
def base_time() -> datetime.datetime:
    return datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)


@pytest.fixture
def clock(base_time: datetime.datetime) -> FixedClock:
    return FixedClock(timestamp=base_time)


@pytest.fixture
def policy() -> FailureRecoveryPolicy:
    return FailureRecoveryPolicy()


def _candidate(
    provider_id: str,
    model_id: str,
    route_id: str,
    *,
    health: ProviderHealth = ProviderHealth.HEALTHY,
    context_tokens: int = 1000,
    failure_domain: str = "",
    quota: ProviderQuotaState | None = None,
    supported_roles: frozenset[str] | None = None,
) -> RecoveryCandidate:
    if supported_roles is None:
        supported_roles = frozenset({ExecutionRole.CODING.value})
    return RecoveryCandidate(
        provider_id=provider_id,
        model_id=model_id,
        route_id=route_id,
        model_identity=ModelIdentity(model_id=model_id, family=model_id),
        route_identity=InferenceRouteIdentity(
            route_id=route_id,
            provider_id=provider_id,
            route_type=RouteType.DIRECT,
            endpoint_key=route_id,
            failure_domain=failure_domain or provider_id,
        ),
        capabilities=ModelCapabilities(
            context_tokens=context_tokens,
            supported_roles=supported_roles,
        ),
        recovery_state=RouteRecoveryState(health=health),
        quota=quota,
        failure_domain=failure_domain or provider_id,
    )


class TestTransientFailures:
    def test_transient_failure_does_not_penalize_model_quality(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct", failure_domain="anthropic"),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT, message="timeout"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
            failure_domain="openai",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert not decision.classification.model_quality_effect
        assert decision.action in {
            RecoveryAction.RETRY_SAME_ROUTE,
            RecoveryAction.RETRY_ALTERNATE_ROUTE,
        }

    def test_alternate_route_reroute_works(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct", failure_domain="openai"),
            _candidate("anthropic", "claude", "anthropic-direct", failure_domain="anthropic"),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT, message="timeout"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
            failure_domain="openai",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.RETRY_ALTERNATE_ROUTE
        assert decision.selected_candidate is not None
        assert decision.selected_candidate.provider_id == "anthropic"

    def test_no_capacity_enters_wait(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (
            _candidate(
                "openai",
                "gpt-4o",
                "openai-direct",
                health=ProviderHealth.UNAVAILABLE,
                failure_domain="openai",
            ),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.PROVIDER_UNAVAILABLE, message="down"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
            failure_domain="openai",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.WAIT_FOR_PROVIDER
        assert decision.wait_reason == "no_alternate_route_available"

    def test_transient_retry_count_preserved_after_restart(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        ledger = RetryLedger("task-1")
        for _ in range(2):
            ledger.record(
                failure_category="INFRASTRUCTURE_TRANSIENT",
                failure_subtype="TRANSIENT_TRANSPORT",
                failure_signature="sig-transient",
                provider_id="openai",
                model_id="gpt-4o",
                route_id="openai-direct",
                action_taken="RETRY_SAME_ROUTE",
                retry_type=RetryType.TRANSIENT_RETRY,
                timestamp=clock.now(),
            )
        candidates = (_candidate("openai", "gpt-4o", "openai-direct"),)
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT, message="timeout"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=ledger,
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.attempt_counters["transient_retry_count"] == 2


class TestQuota:
    def test_quota_exhausted_reroutes_when_alternative_exists(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (
            _candidate(
                "openai",
                "gpt-4o",
                "openai-direct",
                quota=ProviderQuotaState(provider_signal=QuotaSignal.EXHAUSTED),
            ),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.QUOTA_EXHAUSTED, message="quota exhausted"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.REROUTE_PROVIDER
        assert decision.selected_candidate is not None
        assert decision.selected_candidate.provider_id == "anthropic"

    def test_quota_exhausted_waits_when_no_alternative(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (
            _candidate(
                "openai",
                "gpt-4o",
                "openai-direct",
                quota=ProviderQuotaState(provider_signal=QuotaSignal.EXHAUSTED),
            ),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.QUOTA_EXHAUSTED, message="quota exhausted"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.WAIT_FOR_PROVIDER
        assert "no_eligible_capacity" in decision.wait_reason

    def test_quota_failure_does_not_penalize_model_quality(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (_candidate("openai", "gpt-4o", "openai-direct"),)
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(code=ProviderErrorCode.QUOTA_EXHAUSTED, message="quota"),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert not decision.classification.model_quality_effect

    def test_pinned_exhausted_target_not_silently_bypassed(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        pin = RoutingPin(provider_id="openai", model_id="gpt-4o", route_id="openai-direct")
        candidates = (
            _candidate(
                "openai",
                "gpt-4o",
                "openai-direct",
                quota=ProviderQuotaState(provider_signal=QuotaSignal.EXHAUSTED),
            ),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(code=ProviderErrorCode.QUOTA_EXHAUSTED, message="quota"),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            pin=pin,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.WAIT_FOR_PROVIDER
        assert "PINNED_CAPACITY_UNAVAILABLE" in decision.wait_reason


class TestAuth:
    def test_auth_failure_does_not_hot_loop(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (_candidate("openai", "gpt-4o", "openai-direct"),)
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(code=ProviderErrorCode.AUTH_FAILURE, message="bad key"),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.BLOCK
        assert not decision.retry_allowed

    def test_auth_failure_may_use_alternate_configured_route(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("openai", "gpt-4o", "openai-fallback"),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(code=ProviderErrorCode.AUTH_FAILURE, message="bad key"),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.REROUTE_PROVIDER


class TestStructuredOutput:
    def test_first_invalid_structured_output_permits_constrained_retry(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (_candidate("openai", "gpt-4o", "openai-direct"),)
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            structured_output_validation=StructuredOutputValidationResult(
                missing_required_fields=("risk",)
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.CONSTRAINED_OUTPUT_RETRY
        assert decision.evidence_packet.get("missing_required_fields") == ["risk"]

    def test_structured_output_retries_bounded(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        ledger = RetryLedger("task-1")
        for i in range(policy.max_structured_output_retries):
            ledger.record(
                failure_category="STRUCTURED_OUTPUT_INVALID",
                failure_subtype="MISSING_REQUIRED_FIELDS",
                failure_signature="sig-structured",
                provider_id="openai",
                model_id="gpt-4o",
                route_id="openai-direct",
                action_taken="CONSTRAINED_OUTPUT_RETRY",
                retry_type=RetryType.CONSTRAINED_OUTPUT_RETRY,
                timestamp=clock.now() + datetime.timedelta(seconds=i),
            )
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            structured_output_validation=StructuredOutputValidationResult(
                missing_required_fields=("risk",)
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=ledger,
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.REROUTE_MODEL


class TestPlanning:
    def test_invalid_plan_preserves_rejection_evidence(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (
            _candidate(
                "openai",
                "gpt-4o",
                "openai-direct",
                supported_roles=frozenset({ExecutionRole.PLANNING.value}),
            ),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.PLANNING,
            planning_validation=PlanningValidationResult(
                missing_steps=("validate",),
                schema_errors=("missing objective",),
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.PLANNING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.REPLAN
        assert decision.evidence_packet.get("missing_steps") == ["validate"]

    def test_repeated_plan_failure_switches_planner(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        ledger = RetryLedger("task-1")
        for i in range(policy.max_planning_retries):
            ledger.record(
                failure_category="PLANNING_OUTPUT_INVALID",
                failure_subtype="MISSING_PLAN_STEPS",
                failure_signature="sig-plan",
                provider_id="openai",
                model_id="gpt-4o",
                route_id="openai-direct",
                action_taken="REPLAN",
                retry_type=RetryType.REPLAN,
                timestamp=clock.now() + datetime.timedelta(seconds=i),
            )
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate(
                "anthropic",
                "claude",
                "anthropic-direct",
                supported_roles=frozenset({ExecutionRole.PLANNING.value}),
            ),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.PLANNING,
            planning_validation=PlanningValidationResult(missing_steps=("validate",)),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=ledger,
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.PLANNING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.REROUTE_MODEL
        assert decision.selected_candidate is not None
        assert decision.selected_candidate.model_id == "claude"


class TestImplementation:
    def test_deterministic_failure_receives_evidence(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (_candidate("openai", "gpt-4o", "openai-direct"),)
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            deterministic_validation=ValidationResultSummary(
                validator="pytest",
                passed=False,
                failing_check_names=("test_foo",),
                exit_status=1,
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.REPAIR_WITH_EVIDENCE
        assert decision.evidence_packet.get("failing_check_names") == ["test_foo"]

    def test_repeated_identical_signature_triggers_cross_model_escalation(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        ledger = RetryLedger("task-1")
        for i in range(policy.require_cross_provider_after_same_signature):
            ledger.record(
                failure_category="IMPLEMENTATION_DETERMINISTIC",
                failure_subtype="TEST_FAILURE",
                failure_signature="sig-test-x",
                provider_id="openai",
                model_id="gpt-4o",
                route_id="openai-direct",
                action_taken="REPAIR_WITH_EVIDENCE",
                retry_type=RetryType.REPAIR,
                timestamp=clock.now() + datetime.timedelta(seconds=i),
            )
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            deterministic_validation=ValidationResultSummary(
                validator="pytest",
                passed=False,
                failing_check_names=("test_foo",),
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=ledger,
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.CROSS_MODEL_REPAIR


class TestContextOverflow:
    def test_context_overflow_rebuilds_context(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (_candidate("openai", "gpt-4o", "openai-direct", context_tokens=1000),)
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            context_overflow=ContextOverflowMetadata(
                estimated_input_chars=5000,
                model_context_tokens=1000,
                authority_required=True,
                authority_items_present=2,
                authority_items_raw=2,
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.REBUILD_CONTEXT
        assert decision.require_context_rebuild

    def test_context_overflow_chooses_larger_context_model(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        ledger = RetryLedger("task-1")
        for i in range(policy.max_context_rebuilds):
            ledger.record(
                failure_category="CONTEXT_CAPACITY",
                failure_subtype="CONTEXT_OVERFLOW",
                failure_signature="sig-context",
                provider_id="openai",
                model_id="gpt-4o",
                route_id="openai-direct",
                action_taken="REBUILD_CONTEXT",
                retry_type=RetryType.REBUILD_CONTEXT,
                timestamp=clock.now() + datetime.timedelta(seconds=i),
            )
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct", context_tokens=1000),
            _candidate(
                "anthropic",
                "claude",
                "anthropic-direct",
                context_tokens=10000,
                supported_roles=frozenset({ExecutionRole.CODING.value}),
            ),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            context_overflow=ContextOverflowMetadata(
                estimated_input_chars=5000,
                model_context_tokens=1000,
                authority_required=True,
                authority_items_raw=2,
                rebuild_attempts=policy.max_context_rebuilds,
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=ledger,
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.REROUTE_MODEL
        assert decision.selected_candidate is not None
        assert decision.selected_candidate.model_id == "claude"


class TestAuthorityViolation:
    def test_authority_violation_blocks_and_escalates_risk(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            authority_violation=AuthorityViolationData(
                touched_authority_paths=("docs/PROJECT_STATE.json",),
                attempted_state_advancement=True,
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action in {RecoveryAction.BLOCK, RecoveryAction.CROSS_MODEL_REPAIR}
        assert decision.require_risk_escalation


class TestRetryStormPrevention:
    def test_total_attempt_limit_blocks(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        ledger = RetryLedger("task-1")
        for i in range(policy.max_total_attempts):
            ledger.record(
                failure_category="UNKNOWN_FAILURE",
                failure_subtype="UNKNOWN",
                failure_signature=f"sig-{i}",
                provider_id="openai",
                model_id="gpt-4o",
                route_id="openai-direct",
                action_taken="RETRY",
                retry_type=RetryType.TRANSIENT_RETRY,
                timestamp=clock.now() + datetime.timedelta(seconds=i),
            )
        candidates = (_candidate("openai", "gpt-4o", "openai-direct"),)
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT, message="timeout"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=ledger,
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.BLOCK
        assert decision.terminal

    def test_same_signature_threshold_prevents_infinite_loop(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        ledger = RetryLedger("task-1")
        for i in range(policy.max_same_signature_attempts):
            ledger.record(
                failure_category="IMPLEMENTATION_DETERMINISTIC",
                failure_subtype="TEST_FAILURE",
                failure_signature="sig-same",
                provider_id="openai",
                model_id="gpt-4o",
                route_id="openai-direct",
                action_taken="REPAIR",
                retry_type=RetryType.REPAIR,
                timestamp=clock.now() + datetime.timedelta(seconds=i),
            )
        candidates = (_candidate("openai", "gpt-4o", "openai-direct"),)
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            deterministic_validation=ValidationResultSummary(
                validator="pytest",
                passed=False,
                failing_check_names=("test_foo",),
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=ledger,
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.BLOCK

    def test_cancelled_is_terminal(self, clock: FixedClock, policy: FailureRecoveryPolicy) -> None:
        candidates = (_candidate("openai", "gpt-4o", "openai-direct"),)
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            cancelled=True,
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide(coord_input)
        assert decision.action is RecoveryAction.CANCEL
        assert decision.terminal

    def test_decision_deterministic_under_repetition(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT, message="timeout"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decisions = [coordinator.decide(coord_input) for _ in range(100)]
        assert all(d.action == decisions[0].action for d in decisions)
        assert all(d.failure_signature == decisions[0].failure_signature for d in decisions)

    def test_candidate_ordering_does_not_affect_decision(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        c1 = _candidate("openai", "gpt-4o", "openai-direct")
        c2 = _candidate("anthropic", "claude", "anthropic-direct")
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT, message="timeout"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input1 = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=(c1, c2),
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coord_input2 = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=(c2, c1),
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        d1 = coordinator.decide(coord_input1)
        d2 = coordinator.decide(coord_input2)
        assert d1.action == d2.action
        assert d1.selected_candidate is not None
        assert d2.selected_candidate is not None
        assert d1.selected_candidate.key == d2.selected_candidate.key


class TestCurrentCandidateEligibility:
    """Same-candidate retries must survive canonical Phase 8 eligibility."""

    def test_transient_same_route_not_selected_when_route_disabled(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT, message="timeout"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            route_enabled={"openai-direct": False},
            role=ExecutionRole.CODING,
        )
        decision = RecoveryCoordinator(clock=clock).decide(coord_input)
        assert (
            decision.selected_candidate is None
            or decision.selected_candidate.route_id != "openai-direct"
        )

    def test_constrained_output_retry_not_selected_when_model_disabled(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            structured_output_validation=StructuredOutputValidationResult(
                missing_required_fields=("risk",)
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            model_enabled={"gpt-4o": False},
            role=ExecutionRole.CODING,
        )
        decision = RecoveryCoordinator(clock=clock).decide(coord_input)
        assert (
            decision.selected_candidate is None or decision.selected_candidate.model_id != "gpt-4o"
        )

    def test_replan_not_selected_when_route_unhealthy(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (
            _candidate(
                "openai",
                "gpt-4o",
                "openai-direct",
                supported_roles=frozenset({ExecutionRole.PLANNING.value}),
                health=ProviderHealth.UNAVAILABLE,
            ),
            _candidate(
                "anthropic",
                "claude",
                "anthropic-direct",
                supported_roles=frozenset({ExecutionRole.PLANNING.value}),
            ),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.PLANNING,
            planning_validation=PlanningValidationResult(missing_steps=("validate",)),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.PLANNING,
        )
        decision = RecoveryCoordinator(clock=clock).decide(coord_input)
        assert (
            decision.selected_candidate is None
            or decision.selected_candidate.route_id != "openai-direct"
        )

    def test_repair_not_selected_when_provider_quota_exhausted(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (
            _candidate(
                "openai",
                "gpt-4o",
                "openai-direct",
                quota=ProviderQuotaState(provider_signal=QuotaSignal.EXHAUSTED),
            ),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            deterministic_validation=ValidationResultSummary(
                validator="pytest",
                passed=False,
                failing_check_names=("test_foo",),
                exit_status=1,
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        decision = RecoveryCoordinator(clock=clock).decide(coord_input)
        assert (
            decision.selected_candidate is None
            or decision.selected_candidate.provider_id != "openai"
        )

    def test_unknown_retry_not_selected_when_current_route_rate_limited(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        candidates = (
            _candidate(
                "openai",
                "gpt-4o",
                "openai-direct",
                health=ProviderHealth.RATE_LIMITED,
            ),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=policy,
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        decision = RecoveryCoordinator(clock=clock).decide(coord_input)
        assert (
            decision.selected_candidate is None
            or decision.selected_candidate.route_id != "openai-direct"
        )
