"""Architectural invariants for OmniForge Phase 10.

These tests guard against the anti-patterns listed in the Phase 10 spec:
- one generic retry loop for all failures
- blind identical retries
- provider/quota outages treated as model-quality failures
- route failures contaminating underlying model reputation
- provider-specific retry branches scattered across adapters
- infinite or reset-on-restart retry counters
- context overflow silently dropping authority
- authority violations proceeding to integration
- permanent model penalties before Phase 11
- random or brand-based recovery decisions
- secret leakage in persisted recovery evidence
- Phase 11 performance ledger implementation
- authority-state mutation
"""

from __future__ import annotations

import datetime
import math
import tempfile
from pathlib import Path

import pytest

from src.context.strategy import ContextBuildRequest
from src.persistence import configuration, runtime_state
from src.policy.risk import RiskLevel
from src.providers.errors import ProviderError, ProviderErrorCode
from src.providers.identity import ProviderHealth, ProviderQuotaState
from src.recovery import FixedClock
from src.recovery.authority_recovery import (
    AuthorityRecoveryResult,
    apply_authority_violation_recovery,
)
from src.recovery.clock import ManualClock
from src.recovery.context_recovery import (
    ContextRebuildResult,
    build_context_overflow_metadata,
    context_recovery_evidence,
)
from src.recovery.evidence import (
    ImplementationFailureEvidence,
    implementation_failure_signature,
)
from src.recovery.failure_classification import (
    AuthorityViolationData,
    FailureCategory,
    FailureClassifierInput,
    PlanningValidationResult,
    StructuredOutputValidationResult,
    ValidationResultSummary,
    failure_classification_fingerprint,
)
from src.recovery.recovery_coordinator import (
    RecoveryAction,
    RecoveryCandidate,
    RecoveryCoordinator,
    RecoveryCoordinatorInput,
    RecoveryDecision,
)
from src.recovery.retry_policy import FailureRecoveryPolicy
from src.recovery.retry_state import RetryLedger, RetryType, WaitState
from src.recovery.signals import ProviderSignal, SignalKind
from src.recovery.state_machine import HealthStateMachine, RouteRecoveryState
from src.risk.runtime import RiskRuntimeEvent, RiskRuntimeEventType, RuntimeRiskEscalator
from src.routing.capabilities import ModelCapabilities
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity, ModelLifecycle
from src.routing.roles import ExecutionRole
from src.security.redaction import contains_secret

SENTINEL = "OMNIFORGE_PHASE10_ARCH_SECRET_SENTINEL_999"


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
    lifecycle: ModelLifecycle = ModelLifecycle.NORMAL,
    roles: frozenset[str] | None = None,
) -> RecoveryCandidate:
    return RecoveryCandidate(
        provider_id=provider_id,
        model_id=model_id,
        route_id=route_id,
        model_identity=ModelIdentity(
            model_id=model_id,
            family=model_id,
            lifecycle=lifecycle,
        ),
        route_identity=InferenceRouteIdentity(
            route_id=route_id,
            provider_id=provider_id,
            route_type=RouteType.DIRECT,
            endpoint_key=route_id,
            failure_domain=failure_domain or provider_id,
        ),
        capabilities=ModelCapabilities(
            context_tokens=context_tokens,
            supported_roles=roles or frozenset({ExecutionRole.CODING.value}),
        ),
        recovery_state=RouteRecoveryState(health=health),
        quota=quota,
        failure_domain=failure_domain or provider_id,
    )


def _coordinator_input(
    classifier_input: FailureClassifierInput,
    candidates: tuple[RecoveryCandidate, ...],
    ledger: RetryLedger | None = None,
    policy: FailureRecoveryPolicy | None = None,
    current_risk: RiskLevel = RiskLevel.R2_NORMAL,
    role: ExecutionRole = ExecutionRole.CODING,
) -> RecoveryCoordinatorInput:
    return RecoveryCoordinatorInput(
        classifier_input=classifier_input,
        candidates=candidates,
        ledger=ledger or RetryLedger(classifier_input.task_id),
        policy=policy or FailureRecoveryPolicy(),
        current_risk=current_risk,
        role=role,
    )


class TestNoGenericRetryLoop:
    def test_coordinator_has_typed_handler_per_category(self) -> None:
        handlers = {
            name
            for name in dir(RecoveryCoordinator)
            if name.startswith("_handle_") and name != "_handle_unknown"
        }
        expected_categories = {
            "infrastructure_transient",
            "quota_exhaustion",
            "auth_failure",
            "capability_mismatch",
            "context_overflow",
            "structured_output_invalid",
            "planning_output_invalid",
            "deterministic_implementation",
            "conceptual_implementation",
            "authority_violation",
        }
        for category in expected_categories:
            assert any(category in handler for handler in handlers), category

    def test_distinct_recovery_actions_per_failure_category(self, clock: FixedClock) -> None:
        base = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            provider_id="provider",
            model_id="model",
            route_id="route",
        )
        candidates = (_candidate("provider", "model", "route"),)
        coordinator = RecoveryCoordinator(clock=clock)

        actions: dict[FailureCategory, RecoveryAction] = {}
        for category, build_input in [
            (
                FailureCategory.INFRASTRUCTURE_TRANSIENT,
                base.with_provider_error(
                    ProviderError(code=ProviderErrorCode.TRANSIENT_TRANSPORT, message="x")
                ),
            ),
            (
                FailureCategory.INFRASTRUCTURE_QUOTA,
                base.with_provider_error(
                    ProviderError(code=ProviderErrorCode.QUOTA_EXHAUSTED, message="x")
                ),
            ),
            (
                FailureCategory.STRUCTURED_OUTPUT_INVALID,
                base.with_structured_output_validation(
                    StructuredOutputValidationResult(missing_required_fields=("risk",))
                ),
            ),
            (
                FailureCategory.PLANNING_OUTPUT_INVALID,
                base.with_planning_validation(
                    PlanningValidationResult(missing_steps=("validate",))
                ),
            ),
            (
                FailureCategory.IMPLEMENTATION_DETERMINISTIC,
                base.with_deterministic_validation(
                    ValidationResultSummary(validator="pytest", passed=False)
                ),
            ),
        ]:
            inputs = _coordinator_input(build_input, candidates)
            actions[category] = coordinator.decide(inputs).action

        # No two categories above share the exact same action; this proves the
        # coordinator does not collapse everything to a single generic retry.
        assert len(set(actions.values())) == len(actions)


class TestNoBlindIdenticalRetry:
    def test_transient_retry_carries_backoff_metadata(self, clock: FixedClock) -> None:
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT, message="timeout"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = _coordinator_input(inputs, (_candidate("openai", "gpt-4o", "openai-direct"),))
        decision = RecoveryCoordinator(clock=clock).decide(coord_input)
        assert decision.action is RecoveryAction.RETRY_SAME_ROUTE
        assert decision.retry_after is not None
        assert decision.retry_after > clock.now()

    def test_unknown_failure_bounded_then_blocks(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        ledger = RetryLedger("t")
        threshold = policy.max_total_attempts // 2
        for i in range(threshold):
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
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = _coordinator_input(
            inputs, (_candidate("openai", "gpt-4o", "openai-direct"),), ledger=ledger
        )
        decision = RecoveryCoordinator(clock=clock).decide(coord_input)
        assert decision.terminal
        assert decision.action is RecoveryAction.BLOCK


class TestModelQualitySeparation:
    def test_provider_unavailable_does_not_penalize_model_quality(self, clock: FixedClock) -> None:
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.PROVIDER_UNAVAILABLE, message="down"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        decision = RecoveryCoordinator(clock=clock).decide(
            _coordinator_input(inputs, (_candidate("openai", "gpt-4o", "openai-direct"),))
        )
        assert not decision.classification.model_quality_effect

    def test_quota_exhausted_does_not_penalize_model_quality(self, clock: FixedClock) -> None:
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(code=ProviderErrorCode.QUOTA_EXHAUSTED, message="quota"),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        decision = RecoveryCoordinator(clock=clock).decide(
            _coordinator_input(inputs, (_candidate("openai", "gpt-4o", "openai-direct"),))
        )
        assert not decision.classification.model_quality_effect

    def test_route_failure_does_not_contaminate_underlying_model_quality(
        self, clock: FixedClock
    ) -> None:
        # A gateway route fails for model "gpt-4o"; the failure is attributed to
        # the route, not the model's output quality.
        candidates = (
            _candidate(
                "gateway",
                "gpt-4o",
                "openrouter",
                failure_domain="openrouter",
                health=ProviderHealth.UNAVAILABLE,
            ),
            _candidate("openai", "gpt-4o", "openai-direct", failure_domain="openai"),
        )
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.PROVIDER_UNAVAILABLE, message="gateway down"
            ),
            provider_id="gateway",
            model_id="gpt-4o",
            route_id="openrouter",
            failure_domain="openrouter",
        )
        decision = RecoveryCoordinator(clock=clock).decide(_coordinator_input(inputs, candidates))
        assert decision.classification.route_health_effect
        assert not decision.classification.model_quality_effect
        assert decision.selected_candidate is not None
        assert decision.selected_candidate.route_id == "openai-direct"


class TestRetryBoundsAndStormPrevention:
    def test_policy_rejects_infinite_values(self) -> None:
        with pytest.raises(ValueError):
            FailureRecoveryPolicy(max_total_attempts=math.inf)  # type: ignore[arg-type]

    def test_provider_ping_pong_is_bounded(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        ledger = RetryLedger("t")
        for i in range(policy.max_provider_switches):
            ledger.record(
                failure_category="INFRASTRUCTURE_TRANSIENT",
                failure_subtype="TRANSIENT_TRANSPORT",
                failure_signature="sig-ping-pong",
                provider_id="openai" if i % 2 == 0 else "anthropic",
                model_id="model",
                route_id="route",
                action_taken="REROUTE_PROVIDER",
                retry_type=RetryType.REROUTE_PROVIDER,
                timestamp=clock.now() + datetime.timedelta(seconds=i),
            )
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT, message="timeout"
            ),
            provider_id="openai",
            model_id="model",
            route_id="route",
        )
        candidates = (
            _candidate("openai", "model", "route", failure_domain="openai"),
            _candidate("anthropic", "model", "route", failure_domain="anthropic"),
        )
        decision = RecoveryCoordinator(clock=clock).decide(
            _coordinator_input(inputs, candidates, ledger=ledger, policy=policy)
        )
        assert decision.action is not RecoveryAction.REROUTE_PROVIDER

    def test_model_switch_is_bounded(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        ledger = RetryLedger("t")
        for i in range(policy.max_model_switches):
            ledger.record(
                failure_category="STRUCTURED_OUTPUT_INVALID",
                failure_subtype="SCHEMA_MISMATCH",
                failure_signature="sig-model-switch",
                provider_id="openai",
                model_id="gpt-4o" if i % 2 == 0 else "claude",
                route_id="route",
                action_taken="REROUTE_MODEL",
                retry_type=RetryType.REROUTE_MODEL,
                timestamp=clock.now() + datetime.timedelta(seconds=i),
            )
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            structured_output_validation=StructuredOutputValidationResult(
                missing_required_fields=("risk",)
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="route",
        )
        candidates = (
            _candidate("openai", "gpt-4o", "route"),
            _candidate("openai", "claude", "route"),
        )
        decision = RecoveryCoordinator(clock=clock).decide(
            _coordinator_input(inputs, candidates, ledger=ledger, policy=policy)
        )
        assert decision.action is not RecoveryAction.REROUTE_MODEL

    def test_same_signature_threshold_blocks_loop(
        self, clock: FixedClock, policy: FailureRecoveryPolicy
    ) -> None:
        ledger = RetryLedger("t")
        for i in range(policy.max_same_signature_attempts):
            ledger.record(
                failure_category="IMPLEMENTATION_DETERMINISTIC",
                failure_subtype="TEST_FAILURE",
                failure_signature="sig-same",
                provider_id="openai",
                model_id="gpt-4o",
                route_id="route",
                action_taken="REPAIR",
                retry_type=RetryType.REPAIR,
                timestamp=clock.now() + datetime.timedelta(seconds=i),
            )
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            deterministic_validation=ValidationResultSummary(
                validator="pytest", passed=False, failing_check_names=("test_x",)
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="route",
        )
        decision = RecoveryCoordinator(clock=clock).decide(
            _coordinator_input(inputs, (_candidate("openai", "gpt-4o", "route"),), ledger=ledger)
        )
        assert decision.terminal
        assert decision.action is RecoveryAction.BLOCK


class TestEvidenceAndSignatures:
    def test_repair_evidence_includes_validation_artifact_fields(self, clock: FixedClock) -> None:
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            deterministic_validation=ValidationResultSummary(
                validator="pytest",
                passed=False,
                failing_check_names=("test_foo",),
                exit_status=1,
                affected_files=("src/a.py",),
                error_excerpts=("assert 1 == 2",),
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="route",
        )
        decision = RecoveryCoordinator(clock=clock).decide(
            _coordinator_input(inputs, (_candidate("openai", "gpt-4o", "route"),))
        )
        assert decision.action is RecoveryAction.REPAIR_WITH_EVIDENCE
        assert decision.evidence_packet["command"] == "pytest"
        assert decision.evidence_packet["exit_status"] == 1
        assert decision.evidence_packet["failing_check_names"] == ["test_foo"]

    def test_same_model_may_repair_first_bounded_failure(self, clock: FixedClock) -> None:
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            deterministic_validation=ValidationResultSummary(
                validator="pytest", passed=False, failing_check_names=("test_foo",)
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="route",
        )
        decision = RecoveryCoordinator(clock=clock).decide(
            _coordinator_input(inputs, (_candidate("openai", "gpt-4o", "route"),))
        )
        assert decision.action is RecoveryAction.REPAIR_WITH_EVIDENCE
        assert decision.selected_candidate is not None
        assert decision.selected_candidate.model_id == "gpt-4o"

    def test_different_textual_noise_same_underlying_signature(
        self,
    ) -> None:
        base = ImplementationFailureEvidence(
            command="pytest",
            exit_status=1,
            failing_check_names=("test_foo",),
            error_excerpts=("noise A",),
            affected_files=("src/a.py",),
            validation_artifact_refs=(),
            prior_implementation_fingerprint=None,
        )
        sig_a = implementation_failure_signature(base)
        sig_b = implementation_failure_signature(
            ImplementationFailureEvidence(
                command="pytest",
                exit_status=1,
                failing_check_names=("test_foo",),
                error_excerpts=("noise B",),
                affected_files=("src/a.py",),
                validation_artifact_refs=(),
                prior_implementation_fingerprint=None,
            )
        )
        assert sig_a == sig_b

    def test_material_validation_change_changes_signature(self) -> None:
        a = ImplementationFailureEvidence(
            command="pytest",
            exit_status=1,
            failing_check_names=("test_foo",),
            error_excerpts=(),
            affected_files=(),
            validation_artifact_refs=(),
            prior_implementation_fingerprint=None,
        )
        b = ImplementationFailureEvidence(
            command="pytest",
            exit_status=1,
            failing_check_names=("test_bar",),
            error_excerpts=(),
            affected_files=(),
            validation_artifact_refs=(),
            prior_implementation_fingerprint=None,
        )
        assert implementation_failure_signature(a) != implementation_failure_signature(b)


class TestConceptualEscalation:
    def test_conceptual_failure_prefers_different_model(self, clock: FixedClock) -> None:
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            review_findings=("repeated architecture violation",),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="route",
        )
        candidates = (
            _candidate("openai", "gpt-4o", "route"),
            _candidate("anthropic", "claude", "route"),
        )
        decision = RecoveryCoordinator(clock=clock).decide(_coordinator_input(inputs, candidates))
        assert decision.action is RecoveryAction.CROSS_MODEL_REPAIR
        assert decision.selected_candidate is not None
        assert decision.selected_candidate.model_id == "claude"
        assert decision.require_risk_escalation

    def test_cross_model_escalation_is_task_local_not_reputation(self, clock: FixedClock) -> None:
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            review_findings=("conceptual",),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="route",
        )
        decision = RecoveryCoordinator(clock=clock).decide(
            _coordinator_input(
                inputs,
                (
                    _candidate("openai", "gpt-4o", "route"),
                    _candidate("anthropic", "claude", "route"),
                ),
            )
        )
        # No long-term quality score is produced; only the reroute decision.
        assert not hasattr(decision, "model_quality_score")
        assert "score" not in decision.evidence_packet


class TestContextRecovery:
    def test_context_overflow_authority_preserved_in_metadata(self) -> None:
        request = ContextBuildRequest(
            task_id="t",
            role=ExecutionRole.CODING,
            risk=RiskLevel.R4_CRITICAL_AUTHORITY,
            authority_refs=("docs/PROJECT_STATE.json",),
        )
        rebuild = ContextRebuildResult(
            success=False,
            strategy_name="compact",
            authority_presence="raw",
            estimated_input_chars=5000,
            estimated_input_tokens=1250,
            required_context_tokens=1250,
            excluded_material=(),
            authority_items_present=1,
            authority_items_raw=1,
            rebuild_attempt=1,
        )
        meta = build_context_overflow_metadata(request, rebuild, model_context_tokens=1000)
        assert meta.authority_required
        assert meta.authority_items_present == 1
        assert meta.authority_items_raw == 1
        evidence = context_recovery_evidence(meta, rebuild)
        assert evidence["authority_required"] is True

    def test_context_rebuild_state_survives_serialization(
        self, base_time: datetime.datetime
    ) -> None:
        ledger = RetryLedger("t")
        ledger.current_context_rebuild = {
            "strategy": "compact",
            "authority_items_raw": 2,
            "rebuild_attempt": 1,
        }
        ledger.set_wait(
            WaitState(
                reason="no_capacity",
                next_recheck_at=base_time + datetime.timedelta(minutes=5),
                entered_at=base_time,
            )
        )
        data = ledger.to_dict()
        restored = RetryLedger.from_dict(data)
        assert restored.current_context_rebuild["strategy"] == "compact"
        assert restored.current_wait is not None

    def test_exhausted_path_not_retried_after_restart(self, base_time: datetime.datetime) -> None:
        ledger = RetryLedger("t")
        ledger.mark_exhausted_path("sig-x", "openai", "gpt-4o")
        state = {
            "schema_version": "1.4.0",
            "run_id": "run-1",
            "workflow_state": "WAITING_FOR_RETRY",
            "checkpoint": {"task_id": "t"},
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
            "waiting_tasks": {},
            "task_risk_state": {},
            "task_retry_state": {"t": ledger.to_dict()},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.json"
            runtime_state.save_runtime_state(path, state)
            reloaded = runtime_state.load_runtime_state(path)
        restored = RetryLedger.from_dict(reloaded["task_retry_state"]["t"])
        assert restored.is_exhausted_path("sig-x", "openai", "gpt-4o")


class TestAuthorityViolation:
    def test_authority_violation_escalates_to_r4(self) -> None:
        result = apply_authority_violation_recovery(
            RiskLevel.R2_NORMAL,
            AuthorityViolationData(
                touched_authority_paths=("docs/PROJECT_STATE.json",),
                attempted_state_advancement=True,
            ),
        )
        assert result.escalated_risk is RiskLevel.R4_CRITICAL_AUTHORITY

    def test_authority_violation_blocks_integration(self) -> None:
        result = apply_authority_violation_recovery(
            RiskLevel.R2_NORMAL,
            AuthorityViolationData(touched_authority_paths=("docs/PROJECT_STATE.json",)),
        )
        assert result.block_integration is True
        assert isinstance(result, AuthorityRecoveryResult)

    def test_authority_violation_evidence_preserved(self) -> None:
        data = AuthorityViolationData(
            touched_authority_paths=("docs/PROJECT_STATE.json", "docs/ROADMAP_AUTHORITY.json"),
            ignored_immutable_authority=True,
        )
        result = apply_authority_violation_recovery(RiskLevel.R2_NORMAL, data)
        assert "docs/PROJECT_STATE.json" in result.evidence["touched_authority_paths"]
        assert result.evidence["ignored_immutable_authority"] is True

    def test_same_offending_model_not_reused_when_alternative_exists(
        self, clock: FixedClock
    ) -> None:
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            authority_violation=AuthorityViolationData(
                touched_authority_paths=("docs/PROJECT_STATE.json",)
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="route",
        )
        candidates = (
            _candidate("openai", "gpt-4o", "route"),
            _candidate("anthropic", "claude", "route"),
        )
        decision = RecoveryCoordinator(clock=clock).decide(_coordinator_input(inputs, candidates))
        assert decision.action is RecoveryAction.CROSS_MODEL_REPAIR
        assert decision.selected_candidate is not None
        assert decision.selected_candidate.model_id != "gpt-4o"


class TestRiskIntegration:
    def test_provider_outage_does_not_raise_risk_alone(self, clock: FixedClock) -> None:
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.PROVIDER_UNAVAILABLE, message="down"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="route",
        )
        decision = RecoveryCoordinator(clock=clock).decide(
            _coordinator_input(inputs, (_candidate("openai", "gpt-4o", "route"),))
        )
        assert not decision.require_risk_escalation

    def test_repeated_deterministic_failure_can_raise_risk(self) -> None:
        escalator = RuntimeRiskEscalator(repair_loop_threshold=3)
        event = RiskRuntimeEvent(
            event_type=RiskRuntimeEventType.REPAIR_LOOP,
            material=True,
            evidence="repeated identical test failure",
            count=3,
        )
        new_risk, _ = escalator.escalate(RiskLevel.R2_NORMAL, event)
        assert new_risk is RiskLevel.R3_HIGH

    def test_risk_escalation_reconsiders_candidate_eligibility(self, clock: FixedClock) -> None:
        # At R4, a NORMAL-lifecycle candidate is ineligible; only HIGH_RISK may be chosen.
        normal = _candidate("openai", "gpt-4o", "route", lifecycle=ModelLifecycle.NORMAL)
        high_risk = _candidate("anthropic", "claude", "route", lifecycle=ModelLifecycle.HIGH_RISK)
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            deterministic_validation=ValidationResultSummary(
                validator="pytest", passed=False, failing_check_names=("test_x",)
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="route",
        )
        ledger = RetryLedger("t")
        for i in range(FailureRecoveryPolicy().require_cross_provider_after_same_signature):
            ledger.record(
                failure_category="IMPLEMENTATION_DETERMINISTIC",
                failure_subtype="TEST_FAILURE",
                failure_signature="sig-x",
                provider_id="openai",
                model_id="gpt-4o",
                route_id="route",
                action_taken="REPAIR",
                retry_type=RetryType.REPAIR,
                timestamp=clock.now() + datetime.timedelta(seconds=i),
            )
        decision = RecoveryCoordinator(clock=clock).decide(
            RecoveryCoordinatorInput(
                classifier_input=inputs,
                candidates=(normal, high_risk),
                ledger=ledger,
                policy=FailureRecoveryPolicy(),
                current_risk=RiskLevel.R4_CRITICAL_AUTHORITY,
                role=ExecutionRole.CODING,
            )
        )
        # The only eligible cross-model candidate is the high-risk one.
        assert decision.action is RecoveryAction.CROSS_MODEL_REPAIR
        assert decision.selected_candidate is not None
        assert decision.selected_candidate.model_id == "claude"


class TestPhase6And8And9Preservation:
    def test_recovered_provider_reenters_eligibility(self) -> None:
        clock = ManualClock()
        state = RouteRecoveryState(health=ProviderHealth.UNAVAILABLE)
        hsm = HealthStateMachine(clock=clock)
        recovered = hsm.apply(
            state,
            ProviderSignal(
                provider_id="openai",
                route_id="route",
                failure_domain="openai",
                timestamp=clock.now(),
                kind=SignalKind.SUCCESS,
            ),
        )
        assert recovered.health is ProviderHealth.HEALTHY

    def test_rate_limited_follows_canonical_cooling(self) -> None:
        clock = ManualClock()
        state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
        hsm = HealthStateMachine(clock=clock)
        new_state = hsm.apply(
            state,
            ProviderSignal(
                provider_id="openai",
                route_id="route",
                failure_domain="openai",
                timestamp=clock.now(),
                kind=SignalKind.ERROR,
                error=ProviderError(
                    code=ProviderErrorCode.RATE_LIMITED,
                    message="rate limited",
                    retry_after_seconds=60,
                ),
            ),
        )
        assert new_state.health is ProviderHealth.RATE_LIMITED
        assert new_state.cooldown_until is not None
        assert new_state.cooldown_until > clock.now()

    def test_legacy_routing_remains_default(self) -> None:
        migrated = configuration.migrate_config({"schema_version": "1.0.0"})
        assert migrated["routing_mode"] == "legacy"

    def test_dynamic_routing_and_exploration_disabled_by_default(self) -> None:
        migrated = configuration.migrate_config({"schema_version": "1.0.0"})
        assert migrated["dynamic_routing_enabled"] is False
        assert migrated["exploration_enabled"] is False
        assert migrated["router_config"]["exploration_enabled"] is False

    def test_phase6_recovery_state_preserved_in_admin_state(self) -> None:
        admin = configuration.extract_administrative_state(
            {"schema_version": "1.0.0", "providers": {}}
        )
        assert admin["routing_mode"] == "legacy"
        assert admin["dynamic_routing_enabled"] is False
        assert admin["exploration_enabled"] is False
        assert "recovery_policy" in admin

    def test_phase9_risk_policy_preserved_through_migration(self) -> None:
        config = {
            "schema_version": "1.3.0",
            "providers": {},
            "risk_policy": {"default_risk": "R2_NORMAL"},
        }
        migrated = configuration.migrate_config(config)
        assert "risk_policy" in migrated
        assert "recovery_policy" in migrated


class TestNoPhase11Ledger:
    def test_no_phase11_performance_ledger_in_recovery(self) -> None:
        root = Path(__file__).parent.parent / "src" / "recovery"
        names = {p.stem for p in root.glob("*.py")}
        forbidden = {"performance_ledger", "model_reputation", "quality_ledger"}
        assert not (forbidden & names)

    def test_recovery_decision_has_no_quality_penalty_field(self) -> None:
        fields = {f.name for f in RecoveryDecision.__dataclass_fields__.values()}
        assert "quality_penalty" not in fields
        assert "model_quality_score" not in fields


class TestProviderAdapterIsolation:
    def test_provider_adapters_do_not_embed_retry_logic(self) -> None:
        root = Path(__file__).parent.parent / "src" / "providers"
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "RecoveryAction" not in text, f"{path} imports RecoveryAction"
            assert "RETRY_SAME_ROUTE" not in text, f"{path} embeds retry logic"


class TestSecurityAndRedaction:
    def test_secret_sentinel_absent_from_decision_evidence(self, clock: FixedClock) -> None:
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.AUTH_FAILURE,
                message=f"Authorization: Bearer {SENTINEL}",
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="route",
        )
        decision = RecoveryCoordinator(clock=clock).decide(
            _coordinator_input(inputs, (_candidate("openai", "gpt-4o", "route"),))
        )
        serialized = str(decision.evidence_packet)
        assert not contains_secret(serialized, SENTINEL)

    def test_secret_sentinel_absent_from_failure_fingerprint(self) -> None:
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT,
                message=f"timeout {SENTINEL}",
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="route",
        )
        fingerprint = failure_classification_fingerprint(inputs)
        assert not contains_secret(fingerprint, SENTINEL)


class TestExplainability:
    def test_decision_explanation_is_structured_and_secret_free(self, clock: FixedClock) -> None:
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT, message="timeout"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="route",
        )
        decision = RecoveryCoordinator(clock=clock).decide(
            _coordinator_input(inputs, (_candidate("openai", "gpt-4o", "route"),))
        )
        assert decision.explanation
        assert "infrastructure" in decision.explanation.lower()
        # No hidden reasoning markers.
        assert "<thinking>" not in decision.explanation
        assert "chain-of-thought" not in decision.explanation.lower()


class TestDeterminism:
    def test_recovery_decision_independent_of_candidate_input_ordering(
        self, clock: FixedClock
    ) -> None:
        c1 = _candidate("openai", "gpt-4o", "route")
        c2 = _candidate("anthropic", "claude", "route")
        inputs = FailureClassifierInput(
            task_id="t",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT, message="timeout"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="route",
        )
        d1 = RecoveryCoordinator(clock=clock).decide(_coordinator_input(inputs, (c1, c2)))
        d2 = RecoveryCoordinator(clock=clock).decide(_coordinator_input(inputs, (c2, c1)))
        assert d1.action == d2.action
        assert d1.selected_candidate is not None
        assert d2.selected_candidate is not None
        assert d1.selected_candidate.key == d2.selected_candidate.key
