"""Tests for deterministic recovery-decision input fingerprint."""

from __future__ import annotations

import datetime

import pytest

from src.policy.risk import RiskLevel
from src.providers.errors import ProviderError, ProviderErrorCode
from src.providers.identity import ProviderHealth, ProviderQuotaState, QuotaSignal
from src.recovery import FixedClock
from src.recovery.failure_classification import FailureClassifierInput
from src.recovery.fingerprint import recovery_input_fingerprint
from src.recovery.recovery_coordinator import (
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
    health: ProviderHealth = ProviderHealth.HEALTHY,
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
        recovery_state=RouteRecoveryState(health=health),
        quota=quota,
        failure_domain=provider_id,
    )


def _base_inputs(
    clock: FixedClock,
    ledger: RetryLedger | None = None,
    policy: FailureRecoveryPolicy | None = None,
    candidates: tuple[RecoveryCandidate, ...] | None = None,
    current_risk: RiskLevel = RiskLevel.R2_NORMAL,
) -> RecoveryCoordinatorInput:
    return RecoveryCoordinatorInput(
        classifier_input=FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT, message="timeout"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        ),
        candidates=candidates
        or (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct"),
        ),
        ledger=ledger or RetryLedger("task-1"),
        policy=policy or FailureRecoveryPolicy(),
        current_risk=current_risk,
        role=ExecutionRole.CODING,
    )


class TestRecoveryFingerprint:
    def test_failure_signature_separate_from_recovery_fingerprint(self, clock: FixedClock) -> None:
        inputs = _base_inputs(clock)
        decision = RecoveryCoordinator(clock=clock).decide(inputs)
        assert decision.failure_signature
        assert decision.deterministic_input_fingerprint
        assert decision.failure_signature != decision.deterministic_input_fingerprint

    def test_exhausted_ledger_changes_recovery_fingerprint(self, clock: FixedClock) -> None:
        empty_ledger = RetryLedger("task-1")
        exhausted_ledger = RetryLedger("task-1")
        for i in range(FailureRecoveryPolicy().max_total_attempts):
            exhausted_ledger.record(
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
        inputs_empty = _base_inputs(clock, ledger=empty_ledger)
        inputs_exhausted = _base_inputs(clock, ledger=exhausted_ledger)
        d_empty = RecoveryCoordinator(clock=clock).decide(inputs_empty)
        d_exhausted = RecoveryCoordinator(clock=clock).decide(inputs_exhausted)
        assert (
            d_empty.deterministic_input_fingerprint != d_exhausted.deterministic_input_fingerprint
        )
        assert d_empty.failure_signature == d_exhausted.failure_signature

    def test_candidate_quota_change_changes_fingerprint(self, clock: FixedClock) -> None:
        inputs1 = _base_inputs(
            clock,
            candidates=(
                _candidate("openai", "gpt-4o", "openai-direct"),
                _candidate("anthropic", "claude", "anthropic-direct"),
            ),
        )
        inputs2 = _base_inputs(
            clock,
            candidates=(
                _candidate("openai", "gpt-4o", "openai-direct"),
                _candidate(
                    "anthropic",
                    "claude",
                    "anthropic-direct",
                    quota=ProviderQuotaState(provider_signal=QuotaSignal.EXHAUSTED),
                ),
            ),
        )
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint

    def test_candidate_health_change_changes_fingerprint(self, clock: FixedClock) -> None:
        inputs1 = _base_inputs(
            clock,
            candidates=(
                _candidate("openai", "gpt-4o", "openai-direct"),
                _candidate("anthropic", "claude", "anthropic-direct"),
            ),
        )
        inputs2 = _base_inputs(
            clock,
            candidates=(
                _candidate("openai", "gpt-4o", "openai-direct"),
                _candidate(
                    "anthropic",
                    "claude",
                    "anthropic-direct",
                    health=ProviderHealth.UNAVAILABLE,
                ),
            ),
        )
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint

    def test_current_risk_change_changes_fingerprint(self, clock: FixedClock) -> None:
        inputs1 = _base_inputs(clock, current_risk=RiskLevel.R2_NORMAL)
        inputs2 = _base_inputs(clock, current_risk=RiskLevel.R4_CRITICAL_AUTHORITY)
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint

    def test_policy_change_changes_fingerprint(self, clock: FixedClock) -> None:
        inputs1 = _base_inputs(clock, policy=FailureRecoveryPolicy(max_total_attempts=10))
        inputs2 = _base_inputs(clock, policy=FailureRecoveryPolicy(max_total_attempts=5))
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint

    def test_candidate_ordering_does_not_change_fingerprint(self, clock: FixedClock) -> None:
        c1 = _candidate("openai", "gpt-4o", "openai-direct")
        c2 = _candidate("anthropic", "claude", "anthropic-direct")
        inputs1 = _base_inputs(clock, candidates=(c1, c2))
        inputs2 = _base_inputs(clock, candidates=(c2, c1))
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint == d2.deterministic_input_fingerprint
        assert d1.action == d2.action
        assert d1.selected_candidate is not None
        assert d2.selected_candidate is not None
        assert d1.selected_candidate.key == d2.selected_candidate.key

    def test_repeated_identical_input_same_action_and_fingerprint(self, clock: FixedClock) -> None:
        inputs = _base_inputs(clock)
        coordinator = RecoveryCoordinator(clock=clock)
        decisions = [coordinator.decide(inputs) for _ in range(100)]
        assert all(
            d.deterministic_input_fingerprint == decisions[0].deterministic_input_fingerprint
            for d in decisions
        )
        assert all(d.action == decisions[0].action for d in decisions)

    def test_recovery_input_fingerprint_excludes_secrets(self, clock: FixedClock) -> None:
        inputs = _base_inputs(clock)
        fp = recovery_input_fingerprint(
            inputs,
            RecoveryCoordinator(clock=clock)._classifier.classify(inputs.classifier_input),
        )
        assert "secret" not in fp.lower()
        assert "password" not in fp.lower()
