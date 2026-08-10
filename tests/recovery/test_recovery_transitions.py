"""Tests for exactly-once retry-state transitions and restart safety."""

from __future__ import annotations

import datetime

import pytest

from src.context.schema import AuthorityContextItem, AuthorityPresence, ContextPacket, ProvenanceRef
from src.policy.risk import RiskLevel
from src.providers.errors import ProviderError, ProviderErrorCode
from src.providers.identity import ProviderQuotaState, QuotaSignal
from src.recovery import FixedClock
from src.recovery.failure_classification import (
    FailureClassifierInput,
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
from src.risk.context_policy import RiskContextRequirements
from src.routing.capabilities import ModelCapabilities
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity
from src.routing.roles import ExecutionRole


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(timestamp=datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.UTC))


def _candidate(
    provider_id: str,
    model_id: str,
    route_id: str,
    *,
    quota: ProviderQuotaState | None = None,
) -> RecoveryCandidate:
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
            failure_domain=provider_id,
        ),
        capabilities=ModelCapabilities(
            context_tokens=1000,
            supported_roles=frozenset({ExecutionRole.CODING.value}),
        ),
        recovery_state=RouteRecoveryState(),
        quota=quota,
        failure_domain=provider_id,
    )


def _valid_raw_authority_packet() -> ContextPacket:
    provenance = ProvenanceRef(
        source_type="authority",
        path="docs/PROJECT_STATE.json",
        revision="abc123",
        content_hash="hash1",
        authority_level="project",
    )
    authority = AuthorityContextItem(
        authority_id="state",
        provenance_id="state-prov",
        full_source_ref="docs/PROJECT_STATE.json",
        revision="abc123",
        content_hash="hash1",
        content="{}",
        raw_included=True,
    )
    return ContextPacket(
        authority=(authority,),
        provenance_index={"state-prov": provenance},
        authority_presence=AuthorityPresence.RAW_INCLUDED,
        raw_item_count=1,
    )


def _r2_raw_authority_requirements() -> RiskContextRequirements:
    return RiskContextRequirements(
        strategy_preference="hybrid",
        authority_required=True,
        require_raw_authority=True,
        include_test_evidence=False,
        include_historical_findings=False,
        budget_multiplier=1.0,
        rationale="test",
    )


def _make_input(
    clock: FixedClock,
    ledger: RetryLedger,
    **kwargs: object,
) -> RecoveryCoordinatorInput:
    classifier_input = FailureClassifierInput(
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
    defaults: dict[str, object] = {
        "classifier_input": classifier_input,
        "candidates": (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct"),
        ),
        "ledger": ledger,
        "policy": FailureRecoveryPolicy(),
        "current_risk": RiskLevel.R2_NORMAL,
        "role": ExecutionRole.CODING,
    }
    defaults.update(kwargs)
    return RecoveryCoordinatorInput(**defaults)  # type: ignore[arg-type]


class TestRetryStateTransitions:
    def test_repair_decision_records_and_restarts(self, clock: FixedClock) -> None:
        ledger = RetryLedger("task-1")
        inputs = _make_input(clock, ledger)
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide_and_record(inputs)
        assert decision.action is RecoveryAction.REPAIR_WITH_EVIDENCE
        assert ledger.attempt_count == 1
        assert ledger.repair_count() == 1

        # Simulate restart by serializing and reloading the ledger.
        reloaded = RetryLedger.from_dict(ledger.to_dict())
        inputs2 = _make_input(clock, reloaded)
        decision2 = coordinator.decide_and_record(inputs2)
        assert decision2.action is RecoveryAction.REPAIR_WITH_EVIDENCE
        assert reloaded.attempt_count == 2
        assert reloaded.repair_count() == 2

    def test_exhausted_path_survives_restart(self, clock: FixedClock) -> None:
        ledger = RetryLedger("task-1")
        policy = FailureRecoveryPolicy(max_same_model_repairs=1)
        inputs = _make_input(clock, ledger, policy=policy)
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide_and_record(inputs)
        assert decision.action is RecoveryAction.REPAIR_WITH_EVIDENCE

        # After one repair the same-model path is exhausted for this signature.
        reloaded = RetryLedger.from_dict(ledger.to_dict())
        inputs2 = _make_input(clock, reloaded, policy=policy)
        decision2 = coordinator.decide(inputs2)
        assert decision2.action is RecoveryAction.CROSS_MODEL_REPAIR
        assert decision2.selected_candidate is not None
        assert decision2.selected_candidate.model_id == "claude"

    def test_wait_state_survives_restart(self, clock: FixedClock) -> None:
        ledger = RetryLedger("task-1")
        classifier_input = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(code=ProviderErrorCode.QUOTA_EXHAUSTED, message="quota"),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        inputs = RecoveryCoordinatorInput(
            classifier_input=classifier_input,
            candidates=(
                _candidate(
                    "openai",
                    "gpt-4o",
                    "openai-direct",
                    quota=ProviderQuotaState(provider_signal=QuotaSignal.EXHAUSTED),
                ),
            ),
            ledger=ledger,
            policy=FailureRecoveryPolicy(),
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide_and_record(inputs)
        assert decision.action is RecoveryAction.WAIT_FOR_PROVIDER
        assert ledger.current_wait is not None

        reloaded = RetryLedger.from_dict(ledger.to_dict())
        assert reloaded.current_wait is not None
        assert reloaded.current_wait.reason == ledger.current_wait.reason

    def test_context_rebuild_state_survives_restart(self, clock: FixedClock) -> None:
        ledger = RetryLedger("task-1")
        from src.recovery.failure_classification import ContextOverflowMetadata

        classifier_input = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            context_overflow=ContextOverflowMetadata(
                estimated_input_chars=5000,
                model_context_tokens=1000,
                authority_required=True,
                authority_items_raw=2,
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        inputs = RecoveryCoordinatorInput(
            classifier_input=classifier_input,
            candidates=(_candidate("openai", "gpt-4o", "openai-direct"),),
            ledger=ledger,
            policy=FailureRecoveryPolicy(),
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
            context_packet=_valid_raw_authority_packet(),
            risk_context_requirements=_r2_raw_authority_requirements(),
        )
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide_and_record(inputs)
        assert decision.action is RecoveryAction.REBUILD_CONTEXT
        assert ledger.context_rebuild_count() == 1
        assert ledger.current_context_rebuild

        reloaded = RetryLedger.from_dict(ledger.to_dict())
        assert reloaded.context_rebuild_count() == 1
        assert reloaded.current_context_rebuild

    def test_replay_is_idempotent(self, clock: FixedClock) -> None:
        ledger = RetryLedger("task-1")
        inputs = _make_input(clock, ledger)
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide_and_record(inputs)
        assert ledger.attempt_count == 1

        # Re-applying the exact same decision must not add another record.
        coordinator.apply_decision(inputs, decision)
        assert ledger.attempt_count == 1

    def test_action_maps_to_retry_type(self, clock: FixedClock) -> None:
        ledger = RetryLedger("task-1")
        inputs = _make_input(clock, ledger)
        coordinator = RecoveryCoordinator(clock=clock)
        decision = coordinator.decide_and_record(inputs)
        record = ledger.last_record()
        assert record is not None
        assert record.action_taken == decision.action.value
        assert record.retry_type is RetryType.REPAIR
