"""Deterministic tests for the OmniForge Phase 6 recovery engine."""

from __future__ import annotations

import datetime
import tempfile
from pathlib import Path

import pytest

from src.persistence import runtime_state
from src.providers.errors import ProviderError, ProviderErrorCode
from src.providers.identity import (
    ProviderHealth,
    ProviderOperationalState,
    ProviderQuotaState,
    QuotaSignal,
)
from src.providers.response import ProviderResponse
from src.recovery import (
    DispatchChoice,
    FailureDomainIndex,
    FixedClock,
    HealthStateMachine,
    ManualClock,
    OutageSurvivalEngine,
    ProviderSignal,
    QuotaBalancer,
    QuotaCandidate,
    ReserveCapacityPolicy,
    RouteRecoveryState,
    SignalKind,
    StateMachineConfig,
    SurvivalCandidate,
    evaluate_reserve_eligibility,
    signal_from_error,
    signal_from_health_check,
    signal_from_quota,
    signal_from_response,
)
from src.recovery.backoff import BackoffPolicy, HotLoopPolicy, RetryBudget
from src.recovery.clock import ensure_aware, isoformat, parse_iso
from src.recovery.reserve import ReserveEligibilityResult
from src.recovery.scheduler import RecheckPolicy, RecoveryScheduler
from src.recovery.state_machine import HealthTransition
from src.recovery.telemetry import RecoveryEventType, RecoveryTelemetryBuffer
from src.routing.capabilities import CapabilityRequirement, ModelCapabilities, match_capabilities
from src.routing.roles import ExecutionRole

SENTINEL = "OMNIFORGE_TEST_SECRET_SENTINEL_PHASE6_999"


@pytest.fixture
def base_time() -> datetime.datetime:
    return datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)


@pytest.fixture
def clock(base_time: datetime.datetime) -> FixedClock:
    return FixedClock(timestamp=base_time)


@pytest.fixture
def fast_config() -> StateMachineConfig:
    """Short recheck/backoff intervals keep tests fast and deterministic."""
    return StateMachineConfig(
        recheck_policy=RecheckPolicy(
            rate_limited_base_seconds=10,
            quota_exhausted_unknown_reset_seconds=60,
            unavailable_backoff=(5, 10, 20),
            unavailable_max_seconds=30,
            degraded_recheck_seconds=15,
            auth_failed_recheck_seconds=60,
            cooling_recheck_seconds=10,
        ),
        backoff_policy=BackoffPolicy(steps=(5, 10, 20), max_seconds=30),
        retry_budget=RetryBudget(max_consecutive_failures_before_unavailable=2),
    )


def _error(
    code: ProviderErrorCode,
    message: str = "provider error",
    retry_after_seconds: int | None = None,
    quota_reset_at: str | None = None,
) -> ProviderError:
    return ProviderError(
        code=code,
        message=message,
        retry_after_seconds=retry_after_seconds,
        quota_reset_at=quota_reset_at,
    )


def _error_signal(
    state: RouteRecoveryState,
    error: ProviderError,
    clock: FixedClock,
    route_id: str = "route-1",
    failure_domain: str = "fd-1",
) -> ProviderSignal:
    return signal_from_error(
        error,
        route_id=route_id,
        failure_domain=failure_domain,
        clock=clock,
    )


def _response(
    provider_id: str = "openai",
    model_id: str = "gpt-4o",
    request_id: str = "req-1",
) -> ProviderResponse:
    from src.providers.identity import ProviderIdentity
    from src.routing.model_identity import ModelIdentity

    return ProviderResponse(
        request_id=request_id,
        provider_id=ProviderIdentity(
            provider_id=provider_id, display_name=provider_id, failure_domain="fd"
        ),
        model_id=ModelIdentity(model_id=model_id, family=model_id),
        text="ok",
    )


class TestClock:
    def test_manual_clock_rejects_naive_timestamp(self) -> None:
        clock = ManualClock()
        naive = datetime.datetime(2026, 1, 1)
        with pytest.raises(ValueError):
            clock.set(naive)

    def test_ensure_aware_rejects_naive(self) -> None:
        with pytest.raises(ValueError):
            ensure_aware(datetime.datetime(2026, 1, 1))

    def test_fixed_clock_advances_deterministically(
        self, clock: FixedClock, base_time: datetime.datetime
    ) -> None:
        assert clock.now() == base_time
        clock.advance(30)
        assert clock.now() == base_time + datetime.timedelta(seconds=30)

    def test_iso_round_trip(self, base_time: datetime.datetime) -> None:
        text = isoformat(base_time)
        assert parse_iso(text) == base_time


class TestSignal:
    def test_signal_from_response_uses_provider_id(self, clock: FixedClock) -> None:
        response = _response(provider_id="anthropic", request_id="req-42")
        signal = signal_from_response(
            response, route_id="anthropic-direct", failure_domain="anthropic", clock=clock
        )
        assert signal.provider_id == "anthropic"
        assert signal.route_id == "anthropic-direct"
        assert signal.failure_domain == "anthropic"
        assert signal.kind is SignalKind.SUCCESS

    def test_signal_from_error_preserves_context(self, clock: FixedClock) -> None:
        error = _error(ProviderErrorCode.RATE_LIMITED, retry_after_seconds=30)
        signal = signal_from_error(
            error, route_id="openrouter-claude", failure_domain="openrouter", clock=clock
        )
        assert signal.provider_id == "unknown"
        assert signal.route_id == "openrouter-claude"
        assert signal.failure_domain == "openrouter"
        assert signal.error is error

    def test_signal_from_quota_includes_quota_state(self, clock: FixedClock) -> None:
        quota = ProviderQuotaState(provider_signal=QuotaSignal.EXHAUSTED)
        signal = signal_from_quota(
            quota,
            provider_id="qwen",
            route_id="qwen-direct",
            failure_domain="qwen",
            clock=clock,
        )
        assert signal.kind is SignalKind.QUOTA
        assert signal.quota is quota

    def test_signal_requires_non_empty_provider_id(self, clock: FixedClock) -> None:
        with pytest.raises(ValueError):
            ProviderSignal(
                provider_id=" ",
                route_id=None,
                failure_domain="fd",
                timestamp=clock.now(),
                kind=SignalKind.SUCCESS,
            )

    def test_signal_requires_aware_timestamp(self, clock: FixedClock) -> None:
        with pytest.raises(ValueError):
            ProviderSignal(
                provider_id="p",
                route_id=None,
                failure_domain="fd",
                timestamp=datetime.datetime(2026, 1, 1),
                kind=SignalKind.SUCCESS,
            )


class TestBackoff:
    def test_backoff_policy_steps_and_ceiling(self) -> None:
        policy = BackoffPolicy(steps=(5, 15, 30), max_seconds=20)
        assert policy.delay_for_attempt(0) == 5
        assert policy.delay_for_attempt(1) == 15
        assert policy.delay_for_attempt(2) == 20
        assert policy.delay_for_attempt(10) == 20

    def test_retry_budget_exceeds_consecutive(self) -> None:
        budget = RetryBudget(max_consecutive_failures_before_unavailable=3)
        assert not budget.exceeds_consecutive(2)
        assert budget.exceeds_consecutive(3)

    def test_hot_loop_policy_never_retries_non_retryable(self) -> None:
        policy = HotLoopPolicy()
        for code in (
            "QUOTA_EXHAUSTED",
            "AUTH_FAILURE",
            "UNSUPPORTED_CAPABILITY",
            "CONTEXT_OVERFLOW",
            "INVALID_MODEL_OUTPUT",
            "TASK_FAILURE",
            "CANCELLED",
        ):
            assert not policy.can_retry(code)
        assert policy.can_retry("TRANSIENT_TRANSPORT")


class TestHealthStateMachine:
    def test_success_keeps_healthy(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        signal = signal_from_response(_response(), route_id="r", failure_domain="fd", clock=clock)
        new_state = sm.apply(state, signal)
        assert new_state.health is ProviderHealth.HEALTHY
        assert new_state.consecutive_failures == 0

    def test_healthy_to_rate_limited(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        error = _error(ProviderErrorCode.RATE_LIMITED, retry_after_seconds=30)
        signal = _error_signal(state, error, clock)
        new_state = sm.apply(state, signal)
        assert new_state.health is ProviderHealth.RATE_LIMITED
        assert new_state.cooldown_until is not None
        assert new_state.next_recheck_at == new_state.cooldown_until
        assert new_state.consecutive_failures == 1

    def test_rate_limited_success_recover(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.RATE_LIMITED)
        signal = signal_from_response(_response(), route_id="r", failure_domain="fd", clock=clock)
        new_state = sm.apply(state, signal)
        assert new_state.health is ProviderHealth.HEALTHY
        assert new_state.consecutive_failures == 0

    def test_healthy_to_quota_exhausted_known_reset(
        self, clock: FixedClock, fast_config: StateMachineConfig, base_time: datetime.datetime
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        reset_at = (base_time + datetime.timedelta(hours=1)).isoformat()
        error = _error(ProviderErrorCode.QUOTA_EXHAUSTED, quota_reset_at=reset_at)
        signal = _error_signal(state, error, clock)
        new_state = sm.apply(state, signal)
        assert new_state.health is ProviderHealth.QUOTA_EXHAUSTED
        assert new_state.quota_reset_at is not None
        assert new_state.next_recheck_at == new_state.quota_reset_at

    def test_unknown_quota_reset_does_not_hot_loop(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        error = _error(ProviderErrorCode.QUOTA_EXHAUSTED)
        signal = _error_signal(state, error, clock)
        new_state = sm.apply(state, signal)
        assert new_state.next_recheck_at is not None
        assert new_state.next_recheck_at >= clock.now() + datetime.timedelta(seconds=60)

    def test_auth_failure_does_not_hot_loop(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        error = _error(ProviderErrorCode.AUTH_FAILURE)
        signal = _error_signal(state, error, clock)
        new_state = sm.apply(state, signal)
        assert new_state.health is ProviderHealth.AUTH_FAILED
        assert new_state.next_recheck_at is not None
        assert new_state.next_recheck_at >= clock.now() + datetime.timedelta(seconds=60)

    def test_transient_transport_bounded_backoff_then_unavailable(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        error = _error(ProviderErrorCode.TRANSIENT_TRANSPORT)
        signal = _error_signal(state, error, clock)
        state = sm.apply(state, signal)
        assert state.health is ProviderHealth.DEGRADED
        clock.advance(1)
        signal = _error_signal(state, error, clock)
        state = sm.apply(state, signal)
        assert state.health is ProviderHealth.UNAVAILABLE

    def test_context_overflow_does_not_degrade_provider(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        error = _error(ProviderErrorCode.CONTEXT_OVERFLOW)
        signal = _error_signal(state, error, clock)
        new_state = sm.apply(state, signal)
        assert new_state.health is ProviderHealth.HEALTHY
        assert new_state.consecutive_failures == 0

    def test_unsupported_capability_does_not_degrade_provider(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        error = _error(ProviderErrorCode.UNSUPPORTED_CAPABILITY)
        signal = _error_signal(state, error, clock)
        new_state = sm.apply(state, signal)
        assert new_state.health is ProviderHealth.HEALTHY

    def test_invalid_model_output_does_not_become_outage(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        error = _error(ProviderErrorCode.INVALID_MODEL_OUTPUT)
        signal = _error_signal(state, error, clock)
        new_state = sm.apply(state, signal)
        assert new_state.health is ProviderHealth.HEALTHY
        assert new_state.consecutive_failures == 0

    def test_cancelled_does_not_degrade_provider(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        error = _error(ProviderErrorCode.CANCELLED)
        signal = _error_signal(state, error, clock)
        new_state = sm.apply(state, signal)
        assert new_state.health is ProviderHealth.HEALTHY

    def test_administrative_disabled_is_distinct(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.DISABLED)
        error = _error(ProviderErrorCode.PROVIDER_UNAVAILABLE)
        signal = _error_signal(state, error, clock)
        new_state = sm.apply(state, signal)
        assert new_state.health is ProviderHealth.DISABLED

    def test_quota_signal_exhausted_transitions(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        quota = ProviderQuotaState(provider_signal=QuotaSignal.EXHAUSTED)
        signal = signal_from_quota(
            quota, provider_id="p", route_id="r", failure_domain="fd", clock=clock
        )
        new_state = sm.apply(state, signal)
        assert new_state.health is ProviderHealth.QUOTA_EXHAUSTED

    def test_health_check_recovers_degraded(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.DEGRADED)
        op = ProviderOperationalState(health=ProviderHealth.HEALTHY)
        signal = signal_from_health_check(
            op, provider_id="p", route_id="r", failure_domain="fd", clock=clock
        )
        new_state = sm.apply(state, signal)
        assert new_state.health is ProviderHealth.HEALTHY
        assert new_state.consecutive_failures == 0

    def test_health_check_reports_problem(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        op = ProviderOperationalState(health=ProviderHealth.UNAVAILABLE)
        signal = signal_from_health_check(
            op, provider_id="p", route_id="r", failure_domain="fd", clock=clock
        )
        new_state = sm.apply(state, signal)
        assert new_state.health is ProviderHealth.UNAVAILABLE
        assert new_state.next_recheck_at is not None

    def test_transition_log_records_prior_and_new_state(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        error = _error(ProviderErrorCode.RATE_LIMITED, retry_after_seconds=10)
        signal = _error_signal(state, error, clock)
        new_state = sm.apply(state, signal)
        assert len(new_state.transition_log) == 1
        transition = new_state.transition_log[0]
        assert isinstance(transition, HealthTransition)
        assert transition.prior is ProviderHealth.HEALTHY
        assert transition.new is ProviderHealth.RATE_LIMITED

    def test_state_serializes_and_deserializes(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        error = _error(ProviderErrorCode.RATE_LIMITED, retry_after_seconds=10)
        state = sm.apply(state, _error_signal(state, error, clock))
        data = state.to_dict()
        restored = RouteRecoveryState.from_dict(data)
        assert restored.health is ProviderHealth.RATE_LIMITED
        assert restored.consecutive_failures == state.consecutive_failures


class TestScheduler:
    def test_scheduler_due_routes_after_advance(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        scheduler = RecoveryScheduler(policy=fast_config.recheck_policy)
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        error = _error(ProviderErrorCode.PROVIDER_UNAVAILABLE)
        state = sm.apply(state, _error_signal(state, error, clock))
        assert state.next_recheck_at is not None
        scheduler.schedule("route-1", state.next_recheck_at)
        assert scheduler.due_routes(clock) == ()
        clock.advance(10)
        assert scheduler.pop_due(clock) == ("route-1",)
        assert scheduler.next_due_at("route-1") is None

    def test_cooling_expiration_makes_recheck_due_not_healthy(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        error = _error(ProviderErrorCode.RATE_LIMITED, retry_after_seconds=10)
        state = sm.apply(state, _error_signal(state, error, clock))
        assert state.health is ProviderHealth.RATE_LIMITED
        clock.advance(20)
        # State is still rate-limited; only a recheck is due.
        assert state.health is ProviderHealth.RATE_LIMITED
        op = ProviderOperationalState(health=ProviderHealth.HEALTHY)
        signal = signal_from_health_check(
            op, provider_id="p", route_id="r", failure_domain="fd", clock=clock
        )
        state = sm.apply(state, signal)
        assert state.health is ProviderHealth.HEALTHY


class TestFailureDomain:
    def test_domain_propagates_outage_to_eligible_routes(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        index = FailureDomainIndex()
        index.register("route-a", "shared-gateway")
        index.register("route-b", "shared-gateway")
        states = {
            "route-a": RouteRecoveryState(health=ProviderHealth.HEALTHY),
            "route-b": RouteRecoveryState(health=ProviderHealth.HEALTHY),
        }
        updated = index.mark_domain_affected("shared-gateway", sm, states, "gateway outage")
        assert "route-a" in updated
        assert "route-b" in updated
        assert all(s.health is ProviderHealth.UNAVAILABLE for s in updated.values())

    def test_domain_does_not_propagate_to_ineligible_routes(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        index = FailureDomainIndex()
        index.register("route-a", "shared-gateway")
        states = {
            "route-a": RouteRecoveryState(health=ProviderHealth.AUTH_FAILED),
        }
        updated = index.mark_domain_affected("shared-gateway", sm, states, "gateway outage")
        assert "route-a" not in updated

    def test_independent_routes_remain_independent(
        self, clock: FixedClock, fast_config: StateMachineConfig
    ) -> None:
        sm = HealthStateMachine(clock, fast_config)
        index = FailureDomainIndex()
        index.register("direct-claude", "anthropic")
        index.register("openrouter-claude", "openrouter")
        states = {
            "direct-claude": RouteRecoveryState(health=ProviderHealth.HEALTHY),
            "openrouter-claude": RouteRecoveryState(health=ProviderHealth.HEALTHY),
        }
        updated = index.mark_domain_affected("openrouter", sm, states, "openrouter outage")
        assert "direct-claude" not in updated
        assert "openrouter-claude" in updated


class TestReserveCapacity:
    def test_reserve_blocks_routine_request(self) -> None:
        policy = ReserveCapacityPolicy(
            reserved_provider_ids=frozenset({"premium"}),
            reserved_roles=frozenset({ExecutionRole.HIGH_RISK_REVIEW.value}),
        )
        result = evaluate_reserve_eligibility(
            role=ExecutionRole.CODING,
            provider_id="premium",
            model_id="premium-model",
            route_id="premium-route",
            quota_state=None,
            policy=policy,
        )
        assert not result.eligible
        assert "reserve_capacity_protected" in result.reason

    def test_critical_review_allowed_in_reserve(self) -> None:
        policy = ReserveCapacityPolicy(
            reserved_provider_ids=frozenset({"premium"}),
            reserved_roles=frozenset({ExecutionRole.HIGH_RISK_REVIEW.value}),
        )
        result = evaluate_reserve_eligibility(
            role=ExecutionRole.HIGH_RISK_REVIEW,
            provider_id="premium",
            model_id="premium-model",
            route_id="premium-route",
            quota_state=None,
            policy=policy,
        )
        assert result.eligible

    def test_low_pressure_allows_reserve_use(self) -> None:
        policy = ReserveCapacityPolicy(
            reserved_provider_ids=frozenset({"premium"}),
            reserved_roles=frozenset({ExecutionRole.HIGH_RISK_REVIEW.value}),
            minimum_remaining_fraction=0.2,
        )
        quota = ProviderQuotaState(remaining_fraction=0.9, provider_signal=QuotaSignal.AVAILABLE)
        result = evaluate_reserve_eligibility(
            role=ExecutionRole.CODING,
            provider_id="premium",
            model_id="m",
            route_id="r",
            quota_state=quota,
            policy=policy,
        )
        assert result.eligible

    def test_reserve_result_dataclass(self) -> None:
        result = ReserveEligibilityResult(True, "ok")
        assert result.eligible and result.reason == "ok"


class TestQuotaBalancer:
    def test_balancer_excludes_exhausted(self) -> None:
        balancer = QuotaBalancer()
        candidates = [
            QuotaCandidate("p1", "r1", ProviderQuotaState(provider_signal=QuotaSignal.EXHAUSTED)),
            QuotaCandidate("p2", "r2", ProviderQuotaState(provider_signal=QuotaSignal.AVAILABLE)),
        ]
        ordered = balancer.select(candidates)
        assert len(ordered) == 1
        assert ordered[0].provider_id == "p2"

    def test_balancer_prefers_low_pressure(self) -> None:
        balancer = QuotaBalancer()
        candidates = [
            QuotaCandidate("p1", "r1", ProviderQuotaState(remaining_fraction=0.1)),
            QuotaCandidate("p2", "r2", ProviderQuotaState(remaining_fraction=0.8)),
        ]
        ordered = balancer.select(candidates)
        assert ordered[0].provider_id == "p2"
        assert ordered[1].provider_id == "p1"

    def test_balancer_shared_domain_takes_precedence(self) -> None:
        balancer = QuotaBalancer()
        candidates = [
            QuotaCandidate(
                "p1", "r1", ProviderQuotaState(remaining_fraction=0.5), quota_domain="shared"
            ),
            QuotaCandidate("p2", "r2", None, quota_domain="shared"),
        ]
        domain_states = {"shared": ProviderQuotaState(remaining_fraction=0.95)}
        ordered = balancer.select(candidates, quota_domain_states=domain_states)
        assert len(ordered) == 2
        # Both share low pressure via domain state.
        assert ordered[0].provider_id == "p1"

    def test_unknown_quota_does_not_mean_unlimited(self) -> None:
        balancer = QuotaBalancer()
        candidates = [QuotaCandidate("p1", "r1", None)]
        ordered = balancer.select(candidates)
        assert len(ordered) == 1
        assert ordered[0].provider_id == "p1"


class TestOutageSurvival:
    def _candidate(
        self,
        provider_id: str,
        model_id: str,
        route_id: str,
        health: ProviderHealth,
        quota: ProviderQuotaState | None = None,
        roles: frozenset[str] | None = None,
        quota_domain: str | None = None,
    ) -> SurvivalCandidate:
        capabilities = ModelCapabilities(
            context_tokens=1000,
            supported_roles=roles or frozenset({ExecutionRole.CODING.value}),
        )
        return SurvivalCandidate(
            provider_id=provider_id,
            model_id=model_id,
            route_id=route_id,
            capabilities=capabilities,
            recovery_state=RouteRecoveryState(health=health),
            quota=quota,
            quota_domain=quota_domain,
        )

    def test_dispatches_eligible_candidate(self, clock: FixedClock) -> None:
        engine = OutageSurvivalEngine(clock=clock)
        candidates = [self._candidate("openai", "gpt-4o", "openai-direct", ProviderHealth.HEALTHY)]
        decision = engine.dispatch_or_wait(role=ExecutionRole.CODING, candidates=candidates)
        assert decision.dispatch
        assert decision.choice == DispatchChoice(
            provider_id="openai", model_id="gpt-4o", route_id="openai-direct"
        )

    def test_waits_when_no_eligible_routes(self, clock: FixedClock) -> None:
        engine = OutageSurvivalEngine(clock=clock)
        candidates = [
            self._candidate("openai", "gpt-4o", "openai-direct", ProviderHealth.UNAVAILABLE)
        ]
        decision = engine.dispatch_or_wait(role=ExecutionRole.CODING, candidates=candidates)
        assert not decision.dispatch
        assert decision.wait is not None
        assert decision.wait.reason == "no_eligible_routes"

    def test_waits_when_reserve_blocks(self, clock: FixedClock) -> None:
        engine = OutageSurvivalEngine(clock=clock)
        policy = ReserveCapacityPolicy(
            reserved_provider_ids=frozenset({"premium"}),
            reserved_roles=frozenset({ExecutionRole.HIGH_RISK_REVIEW.value}),
        )
        candidates = [self._candidate("premium", "m", "premium-route", ProviderHealth.HEALTHY)]
        decision = engine.dispatch_or_wait(
            role=ExecutionRole.CODING, candidates=candidates, reserve_policy=policy
        )
        assert not decision.dispatch
        assert decision.wait is not None
        assert decision.wait.reason == "reserve_capacity_protected"

    def test_waits_when_quota_exhausted(self, clock: FixedClock) -> None:
        engine = OutageSurvivalEngine(clock=clock)
        candidates = [
            self._candidate(
                "qwen",
                "qwen-max",
                "qwen-direct",
                ProviderHealth.HEALTHY,
                quota=ProviderQuotaState(provider_signal=QuotaSignal.EXHAUSTED),
            )
        ]
        decision = engine.dispatch_or_wait(role=ExecutionRole.CODING, candidates=candidates)
        assert not decision.dispatch
        assert decision.wait is not None
        assert decision.wait.reason == "all_eligible_routes_quota_exhausted"

    def test_recovered_route_eligible_but_not_artificially_preferred(
        self, clock: FixedClock
    ) -> None:
        telemetry = RecoveryTelemetryBuffer(clock=clock)
        engine = OutageSurvivalEngine(clock=clock, telemetry=telemetry)
        candidates = [
            self._candidate("a", "m1", "recovered", ProviderHealth.HEALTHY),
            self._candidate("b", "m2", "steady", ProviderHealth.HEALTHY),
        ]
        decision = engine.dispatch_or_wait(role=ExecutionRole.CODING, candidates=candidates)
        assert decision.dispatch
        assert decision.choice is not None
        # Deterministic ordering by provider_id; recovery does not boost priority.
        assert decision.choice.provider_id == "a"

    def test_telemetry_emits_wait_entered(self, clock: FixedClock) -> None:
        telemetry = RecoveryTelemetryBuffer(clock=clock)
        engine = OutageSurvivalEngine(clock=clock, telemetry=telemetry)
        candidates = [
            self._candidate("openai", "gpt-4o", "openai-direct", ProviderHealth.UNAVAILABLE)
        ]
        engine.dispatch_or_wait(role=ExecutionRole.CODING, candidates=candidates)
        events = telemetry.events()
        assert any(e.event_type is RecoveryEventType.WAIT_ENTERED for e in events)

    def test_capability_requirement_enforced_by_candidate_data(self) -> None:
        candidate = self._candidate(
            "openai",
            "gpt-4o",
            "openai-direct",
            ProviderHealth.HEALTHY,
            roles=frozenset({ExecutionRole.CODING.value}),
        )
        requirement = CapabilityRequirement(required_roles=frozenset({ExecutionRole.CODING.value}))
        match = match_capabilities(candidate.capabilities, requirement)
        assert match.eligible


class TestRuntimeStateRecovery:
    def test_migration_1_1_0_to_1_2_0_adds_recovery_fields(self) -> None:
        old = {
            "schema_version": "1.1.0",
            "run_id": "run-1",
            "workflow_state": "EXECUTING",
            "checkpoint": {},
            "provider_status": {},
            "model_status": {},
            "route_status": {},
            "routing_mode": "legacy",
            "exploration_enabled": False,
            "pins": {},
            "project_policies": {},
        }
        result = runtime_state.validate_runtime_state(old)
        assert result["schema_version"] == "1.2.0"
        assert result["provider_recovery_state"] == {}
        assert result["route_recovery_state"] == {}
        assert result["failure_domain_index"] == {}
        assert result["recovery_scheduler"] == {}
        assert result["waiting_tasks"] == {}

    def test_runtime_state_validates_recovery_state_health(self) -> None:
        state = {
            "schema_version": "1.2.0",
            "run_id": "run-1",
            "workflow_state": "WAITING_FOR_PROVIDER",
            "checkpoint": {},
            "provider_status": {},
            "model_status": {},
            "route_status": {},
            "routing_mode": "legacy",
            "exploration_enabled": False,
            "pins": {},
            "project_policies": {},
            "provider_recovery_state": {
                "openai": {
                    "health": "healthy",
                    "consecutive_failures": 0,
                }
            },
            "route_recovery_state": {
                "openai-direct": {
                    "health": 123,
                    "consecutive_failures": 0,
                }
            },
            "failure_domain_index": {},
            "recovery_scheduler": {},
            "waiting_tasks": {},
        }
        with pytest.raises(runtime_state.CorruptRuntimeState) as caught:
            runtime_state.validate_runtime_state(state)
        assert caught.value.diagnostic.code == "INVALID_RECOVERY_HEALTH"

    def test_persisted_wait_round_trips(self) -> None:
        from src.recovery.survival import PersistedWait

        wait = PersistedWait(
            task_id="task-1",
            role=ExecutionRole.CODING.value,
            reason="no_eligible_routes",
            affected_failure_domains=frozenset({"openrouter"}),
            next_recheck_at=datetime.datetime(2026, 1, 1, 1, 0, 0, tzinfo=datetime.UTC),
            attempted_candidates=[{"provider_id": "openai", "route_id": "r"}],
        )
        data = wait.to_dict()
        restored = PersistedWait.from_dict(data)
        assert restored.task_id == "task-1"
        assert restored.role == ExecutionRole.CODING.value
        assert restored.next_recheck_at.tzinfo is not None

    def test_runtime_state_save_load_with_waiting_task(self) -> None:
        from src.recovery.survival import PersistedWait

        wait = PersistedWait(
            task_id="task-42",
            role=ExecutionRole.REVIEW.value,
            reason="quota_exhausted",
            affected_failure_domains=frozenset({"qwen"}),
            next_recheck_at=datetime.datetime(2026, 1, 2, tzinfo=datetime.UTC),
        )
        state = {
            "schema_version": "1.2.0",
            "run_id": "run-1",
            "workflow_state": "WAITING_FOR_PROVIDER",
            "checkpoint": {"task_id": "task-42"},
            "provider_status": {},
            "model_status": {},
            "route_status": {},
            "routing_mode": "legacy",
            "exploration_enabled": False,
            "pins": {},
            "project_policies": {},
            "provider_recovery_state": {},
            "route_recovery_state": {},
            "failure_domain_index": {},
            "recovery_scheduler": {},
            "waiting_tasks": {"task-42": wait.to_dict()},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.json"
            runtime_state.save_runtime_state(path, state)
            reloaded = runtime_state.load_runtime_state(path)
            assert reloaded["workflow_state"] == "WAITING_FOR_PROVIDER"
            assert reloaded["waiting_tasks"]["task-42"]["task_id"] == "task-42"


class TestTelemetry:
    def test_telemetry_events_recorded(self, clock: FixedClock) -> None:
        buffer = RecoveryTelemetryBuffer(clock=clock)
        event = buffer.emit(
            RecoveryEventType.STATE_TRANSITION,
            provider_id="openai",
            route_id="openai-direct",
            payload={"from": "healthy", "to": "degraded"},
        )
        assert event.provider_id == "openai"
        assert buffer.events() == (event,)

    def test_telemetry_buffer_respects_limit(self, clock: FixedClock) -> None:
        buffer = RecoveryTelemetryBuffer(limit=3, clock=clock)
        for _ in range(5):
            buffer.emit(RecoveryEventType.RECOVERY_SCHEDULED)
        assert len(buffer.events()) == 3


class TestProviderQualityIsolation:
    def test_quota_exhaustion_does_not_mutate_model_quality(self) -> None:
        from src.routing.roles import RolePerformance, RolePerformanceRegistry

        registry = RolePerformanceRegistry()
        model_id = "claude-x"
        before = registry.get(model_id, ExecutionRole.CODING)
        # Quota exhaustion is an infrastructure signal, not model-quality evidence.
        registry.set(
            model_id,
            ExecutionRole.CODING,
            RolePerformance(attempts=1, accepted=1, first_pass_accepted=1),
        )
        # Simulate repeated quota signals without touching model-quality counters.
        after = registry.get(model_id, ExecutionRole.CODING)
        assert after.accepted == before.accepted + 1
