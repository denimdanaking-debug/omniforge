"""Tests for deterministic recovery-decision input fingerprint."""

from __future__ import annotations

import datetime

import pytest

from src.context.schema import AuthorityContextItem, AuthorityPresence, ContextPacket, ProvenanceRef
from src.policy.risk import RiskLevel
from src.providers.errors import ProviderError, ProviderErrorCode
from src.providers.identity import ProviderHealth, ProviderQuotaState, QuotaSignal
from src.recovery import FixedClock
from src.recovery.failure_classification import ContextOverflowMetadata, FailureClassifierInput
from src.recovery.failure_domain import FailureDomainIndex
from src.recovery.fingerprint import recovery_input_fingerprint
from src.recovery.recovery_coordinator import (
    RecoveryCandidate,
    RecoveryCoordinator,
    RecoveryCoordinatorInput,
)
from src.recovery.reserve import ReserveCapacityPolicy
from src.recovery.retry_policy import FailureRecoveryPolicy
from src.recovery.retry_state import RetryLedger, RetryType
from src.recovery.state_machine import RouteRecoveryState
from src.risk.context_policy import RiskContextPolicy, RiskContextRequirements
from src.routing.capabilities import ModelCapabilities
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity
from src.routing.policy import ProjectRoutingPolicy
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
    **kwargs: object,
) -> RecoveryCoordinatorInput:
    defaults: dict[str, object] = {
        "classifier_input": FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            provider_error=ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT, message="timeout"
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        ),
        "candidates": candidates
        or (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct"),
        ),
        "ledger": ledger or RetryLedger("task-1"),
        "policy": policy or FailureRecoveryPolicy(),
        "current_risk": current_risk,
        "role": ExecutionRole.CODING,
    }
    defaults.update(kwargs)
    return RecoveryCoordinatorInput(**defaults)  # type: ignore[arg-type]


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


def _invalid_raw_missing_hash_packet() -> ContextPacket:
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
        revision="",
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


def _referenced_authority_packet() -> ContextPacket:
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
        content=None,
        raw_included=False,
    )
    return ContextPacket(
        authority=(authority,),
        provenance_index={"state-prov": provenance},
        authority_presence=AuthorityPresence.RAW_REFERENCED,
        raw_item_count=0,
    )


def _r4_raw_requirements() -> RiskContextRequirements:
    return RiskContextRequirements(
        strategy_preference="large_context",
        authority_required=True,
        require_raw_authority=True,
        include_test_evidence=False,
        include_historical_findings=False,
        budget_multiplier=2.0,
        rationale="test",
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


class TestRecoveryFingerprintCompleteness:
    def test_multiple_exhausted_paths_fingerprint_succeeds(self, clock: FixedClock) -> None:
        ledger = RetryLedger("task-1")
        ledger.mark_exhausted_path("sig-a", "openai", "gpt-4o")
        ledger.mark_exhausted_path("sig-b", "anthropic", "claude")
        inputs = _base_inputs(clock, ledger=ledger)
        RecoveryCoordinator(clock=clock).decide(inputs)

    def test_exhausted_path_ordering_invariant(self, clock: FixedClock) -> None:
        ledger1 = RetryLedger("task-1")
        ledger1.mark_exhausted_path("sig-a", "openai", "gpt-4o")
        ledger1.mark_exhausted_path("sig-b", "anthropic", "claude")
        ledger2 = RetryLedger("task-1")
        ledger2.mark_exhausted_path("sig-b", "anthropic", "claude")
        ledger2.mark_exhausted_path("sig-a", "openai", "gpt-4o")
        inputs1 = _base_inputs(clock, ledger=ledger1)
        inputs2 = _base_inputs(clock, ledger=ledger2)
        coordinator = RecoveryCoordinator(clock=clock)
        d1 = coordinator.decide(inputs1)
        d2 = coordinator.decide(inputs2)
        assert d1.deterministic_input_fingerprint == d2.deterministic_input_fingerprint

    def test_provider_enabled_change_changes_fingerprint(self, clock: FixedClock) -> None:
        inputs1 = _base_inputs(clock)
        inputs2 = _base_inputs(clock, provider_enabled={"openai": False})
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint

    def test_project_prohibition_change_changes_fingerprint(self, clock: FixedClock) -> None:
        inputs1 = _base_inputs(clock)
        inputs2 = _base_inputs(
            clock,
            project_policy=ProjectRoutingPolicy(prohibited_provider_ids=frozenset({"openai"})),
        )
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint

    def test_reserve_state_change_changes_fingerprint(self, clock: FixedClock) -> None:
        inputs1 = _base_inputs(clock)
        reserve = ReserveCapacityPolicy(
            reserved_provider_ids=frozenset({"anthropic"}),
            reserved_roles=frozenset({"review"}),
        )
        inputs2 = _base_inputs(clock, reserve_policy=reserve)
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint

    def test_quota_domain_state_change_changes_fingerprint(self, clock: FixedClock) -> None:
        inputs1 = _base_inputs(clock)
        inputs2 = _base_inputs(
            clock,
            quota_domain_states={
                "shared": ProviderQuotaState(provider_signal=QuotaSignal.EXHAUSTED)
            },
        )
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint

    def test_failure_domain_index_change_changes_fingerprint(self, clock: FixedClock) -> None:
        inputs1 = _base_inputs(clock)
        index = FailureDomainIndex()
        index.register("openai-direct", "shared-domain")
        index.register("anthropic-direct", "shared-domain")
        inputs2 = _base_inputs(clock, failure_domain_index=index)
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint

    def test_exhausted_paths_with_optional_ids_sortable(self, clock: FixedClock) -> None:
        ledger = RetryLedger("task-1")
        ledger.mark_exhausted_path("sig-a", None, "model-a")
        ledger.mark_exhausted_path("sig-b", "provider-b", "model-b")
        ledger.mark_exhausted_path("sig-c", "provider-c", None)
        inputs = _base_inputs(clock, ledger=ledger)
        # Must not raise TypeError from comparing None to str.
        RecoveryCoordinator(clock=clock).decide(inputs)

    def test_exhausted_paths_with_optional_ids_order_invariant(self, clock: FixedClock) -> None:
        ledger1 = RetryLedger("task-1")
        ledger1.mark_exhausted_path("sig-a", None, "model-a")
        ledger1.mark_exhausted_path("sig-b", "provider-b", "model-b")
        ledger2 = RetryLedger("task-1")
        ledger2.mark_exhausted_path("sig-b", "provider-b", "model-b")
        ledger2.mark_exhausted_path("sig-a", None, "model-a")
        inputs1 = _base_inputs(clock, ledger=ledger1)
        inputs2 = _base_inputs(clock, ledger=ledger2)
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint == d2.deterministic_input_fingerprint

    def test_exhausted_path_identity_change_changes_fingerprint(self, clock: FixedClock) -> None:
        ledger1 = RetryLedger("task-1")
        ledger1.mark_exhausted_path("sig-a", "provider-a", "model-a")
        ledger2 = RetryLedger("task-1")
        ledger2.mark_exhausted_path("sig-a", "provider-a", "model-b")
        inputs1 = _base_inputs(clock, ledger=ledger1)
        inputs2 = _base_inputs(clock, ledger=ledger2)
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint


class TestContextAuthorityFingerprint:
    def test_valid_vs_invalid_raw_authority_changes_fingerprint(self, clock: FixedClock) -> None:
        valid = _valid_raw_authority_packet()
        invalid = _invalid_raw_missing_hash_packet()
        inputs1 = _base_inputs(
            clock,
            context_packet=valid,
            risk_context_requirements=_r4_raw_requirements(),
        )
        inputs2 = _base_inputs(
            clock,
            context_packet=invalid,
            risk_context_requirements=_r4_raw_requirements(),
        )
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint

    def test_r4_raw_included_vs_referenced_changes_fingerprint(self, clock: FixedClock) -> None:
        included = _valid_raw_authority_packet()
        referenced = _referenced_authority_packet()
        inputs1 = _base_inputs(
            clock,
            context_packet=included,
            risk_context_requirements=_r4_raw_requirements(),
        )
        inputs2 = _base_inputs(
            clock,
            context_packet=referenced,
            risk_context_requirements=_r4_raw_requirements(),
        )
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint

    def test_roundtripped_packet_same_fingerprint(self, clock: FixedClock) -> None:
        packet = _valid_raw_authority_packet()
        roundtripped = ContextPacket.from_dict(packet.to_dict())
        inputs1 = _base_inputs(
            clock,
            context_packet=packet,
            risk_context_requirements=_r4_raw_requirements(),
        )
        inputs2 = _base_inputs(
            clock,
            context_packet=roundtripped,
            risk_context_requirements=_r4_raw_requirements(),
        )
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint == d2.deterministic_input_fingerprint

    def test_risk_context_requirement_change_changes_fingerprint(self, clock: FixedClock) -> None:
        packet = _valid_raw_authority_packet()
        req1 = _r4_raw_requirements()
        req2 = RiskContextRequirements(
            strategy_preference="hybrid",
            authority_required=True,
            require_raw_authority=False,
            include_test_evidence=False,
            include_historical_findings=False,
            budget_multiplier=1.0,
            rationale="test",
        )
        inputs1 = _base_inputs(
            clock,
            context_packet=packet,
            risk_context_requirements=req1,
        )
        inputs2 = _base_inputs(
            clock,
            context_packet=packet,
            risk_context_requirements=req2,
        )
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint


class TestEffectiveContextRequirementsFingerprint:
    def _overflow_inputs(
        self,
        clock: FixedClock,
        risk_context_requirements: RiskContextRequirements | None = None,
        risk_context_policy: RiskContextPolicy | None = None,
        current_risk: RiskLevel = RiskLevel.R3_HIGH,
    ) -> RecoveryCoordinatorInput:
        return _base_inputs(
            clock,
            classifier_input=FailureClassifierInput(
                task_id="task-1",
                role=ExecutionRole.CODING,
                context_overflow=ContextOverflowMetadata(
                    estimated_input_chars=160_000,
                    model_context_tokens=16_000,
                    authority_required=True,
                    authority_items_raw=1,
                ),
                provider_id="openai",
                model_id="gpt-4o",
                route_id="openai-direct",
            ),
            context_packet=_valid_raw_authority_packet(),
            risk_context_requirements=risk_context_requirements,
            risk_context_policy=risk_context_policy,
            current_risk=current_risk,
        )

    def test_same_override_key_different_require_raw_authority_changes_fingerprint(
        self, clock: FixedClock
    ) -> None:
        # R2 base has require_raw_authority=False; override can strengthen it.
        policy1 = RiskContextPolicy(
            overrides={
                "R2_NORMAL": {
                    "strategy_preference": "large_context",
                    "authority_required": True,
                    "require_raw_authority": True,
                    "include_test_evidence": False,
                    "include_historical_findings": False,
                    "budget_multiplier": 2.0,
                    "rationale": "a",
                }
            }
        )
        policy2 = RiskContextPolicy(
            overrides={
                "R2_NORMAL": {
                    "strategy_preference": "large_context",
                    "authority_required": True,
                    "require_raw_authority": False,
                    "include_test_evidence": False,
                    "include_historical_findings": False,
                    "budget_multiplier": 2.0,
                    "rationale": "a",
                }
            }
        )
        inputs1 = self._overflow_inputs(
            clock, risk_context_policy=policy1, current_risk=RiskLevel.R2_NORMAL
        )
        inputs2 = self._overflow_inputs(
            clock, risk_context_policy=policy2, current_risk=RiskLevel.R2_NORMAL
        )
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint

    def test_same_override_key_different_authority_required_changes_fingerprint(
        self, clock: FixedClock
    ) -> None:
        # R1 base has authority_required=False; override can strengthen it.
        policy1 = RiskContextPolicy(
            overrides={
                "R1_LOW": {
                    "strategy_preference": "hybrid",
                    "authority_required": True,
                    "require_raw_authority": False,
                    "budget_multiplier": 1.0,
                    "rationale": "a",
                }
            }
        )
        policy2 = RiskContextPolicy(
            overrides={
                "R1_LOW": {
                    "strategy_preference": "hybrid",
                    "authority_required": False,
                    "require_raw_authority": False,
                    "budget_multiplier": 1.0,
                    "rationale": "a",
                }
            }
        )
        inputs1 = self._overflow_inputs(
            clock, risk_context_policy=policy1, current_risk=RiskLevel.R1_LOW
        )
        inputs2 = self._overflow_inputs(
            clock, risk_context_policy=policy2, current_risk=RiskLevel.R1_LOW
        )
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint

    def test_same_override_key_different_strategy_budget_changes_fingerprint(
        self, clock: FixedClock
    ) -> None:
        policy1 = RiskContextPolicy(
            overrides={
                "R2_NORMAL": {
                    "strategy_preference": "large_context",
                    "authority_required": True,
                    "require_raw_authority": True,
                    "budget_multiplier": 2.0,
                    "rationale": "a",
                }
            }
        )
        policy2 = RiskContextPolicy(
            overrides={
                "R2_NORMAL": {
                    "strategy_preference": "targeted",
                    "authority_required": True,
                    "require_raw_authority": True,
                    "budget_multiplier": 1.0,
                    "rationale": "b",
                }
            }
        )
        inputs1 = self._overflow_inputs(
            clock, risk_context_policy=policy1, current_risk=RiskLevel.R2_NORMAL
        )
        inputs2 = self._overflow_inputs(
            clock, risk_context_policy=policy2, current_risk=RiskLevel.R2_NORMAL
        )
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint != d2.deterministic_input_fingerprint

    def test_default_derived_equals_explicit_equivalent_requirements(
        self, clock: FixedClock
    ) -> None:
        policy = RiskContextPolicy.default()
        explicit = policy.requirements_for(RiskLevel.R3_HIGH)
        inputs1 = self._overflow_inputs(
            clock, risk_context_policy=policy, current_risk=RiskLevel.R3_HIGH
        )
        inputs2 = self._overflow_inputs(
            clock, risk_context_requirements=explicit, current_risk=RiskLevel.R3_HIGH
        )
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint == d2.deterministic_input_fingerprint

    def test_policy_dict_ordering_does_not_change_fingerprint(self, clock: FixedClock) -> None:
        policy1 = RiskContextPolicy(
            overrides={
                "R2_NORMAL": {
                    "strategy_preference": "large_context",
                    "authority_required": True,
                    "require_raw_authority": True,
                    "budget_multiplier": 2.0,
                    "rationale": "a",
                }
            }
        )
        policy2 = RiskContextPolicy(
            overrides={
                "R2_NORMAL": {
                    "budget_multiplier": 2.0,
                    "require_raw_authority": True,
                    "strategy_preference": "large_context",
                    "authority_required": True,
                    "rationale": "a",
                }
            }
        )
        inputs1 = self._overflow_inputs(
            clock, risk_context_policy=policy1, current_risk=RiskLevel.R2_NORMAL
        )
        inputs2 = self._overflow_inputs(
            clock, risk_context_policy=policy2, current_risk=RiskLevel.R2_NORMAL
        )
        d1 = RecoveryCoordinator(clock=clock).decide(inputs1)
        d2 = RecoveryCoordinator(clock=clock).decide(inputs2)
        assert d1.deterministic_input_fingerprint == d2.deterministic_input_fingerprint
