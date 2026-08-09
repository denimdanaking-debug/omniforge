"""Tests for context-overflow capacity semantics and authority safety."""

from __future__ import annotations

import datetime

import pytest

from src.context.budget import estimate_tokens
from src.context.schema import (
    AuthorityContextItem,
    AuthorityPresence,
    ContextPacket,
    ProvenanceRef,
)
from src.context.strategy import ContextBuildRequest
from src.policy.risk import RiskLevel
from src.recovery import FixedClock
from src.recovery.context_recovery import (
    ContextRebuildResult,
    build_context_overflow_metadata,
    context_rebuild_attempt_exceeds_budget,
)
from src.recovery.failure_classification import (
    ContextOverflowMetadata,
    FailureCategory,
    FailureClassifier,
    FailureClassifierInput,
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
    context_tokens: int = 1000,
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
            context_tokens=context_tokens,
            supported_roles=frozenset({ExecutionRole.CODING.value}),
        ),
        recovery_state=RouteRecoveryState(),
        failure_domain=provider_id,
    )


class TestContextCapacityConversion:
    def test_chars_converted_to_tokens_not_compared_directly(self) -> None:
        metadata = ContextOverflowMetadata(
            estimated_input_chars=30_000,
            model_context_tokens=16_000,
        )
        # Direct char comparison would overflow; canonical token estimate does not.
        assert not context_rebuild_attempt_exceeds_budget(metadata)

    def test_required_tokens_drive_overflow(self) -> None:
        metadata = ContextOverflowMetadata(
            estimated_input_chars=30_000,
            required_context_tokens=40_000,
            model_context_tokens=16_000,
        )
        assert context_rebuild_attempt_exceeds_budget(metadata)

    def test_unknown_capacity_is_conservative(self) -> None:
        metadata = ContextOverflowMetadata(
            estimated_input_chars=30_000,
            model_context_tokens=None,
        )
        assert not context_rebuild_attempt_exceeds_budget(metadata)


class TestContextRebuildMetadata:
    def test_build_metadata_uses_token_unit(self) -> None:
        request = ContextBuildRequest(
            task_id="task-1",
            role=ExecutionRole.CODING,
            risk=RiskLevel.R2_NORMAL,
        )
        result = ContextRebuildResult(
            success=True,
            strategy_name="compact",
            authority_presence="raw",
            estimated_input_chars=30_000,
            estimated_input_tokens=estimate_tokens(30_000),
            required_context_tokens=estimate_tokens(30_000),
            excluded_material=(),
            authority_items_present=2,
            authority_items_raw=2,
            rebuild_attempt=1,
        )
        metadata = build_context_overflow_metadata(request, result, model_context_tokens=16_000)
        assert metadata.estimated_input_tokens == estimate_tokens(30_000)
        assert metadata.required_context_tokens == estimate_tokens(30_000)


class TestContextOverflowRecovery:
    def test_small_context_candidate_not_selected_for_large_requirement(
        self, clock: FixedClock
    ) -> None:
        ledger = RetryLedger("task-1")
        for i in range(FailureRecoveryPolicy().max_context_rebuilds):
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
            _candidate("openai", "gpt-4o", "openai-direct", context_tokens=16_000),
            _candidate("anthropic", "claude", "anthropic-direct", context_tokens=64_000),
        )
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            context_overflow=ContextOverflowMetadata(
                estimated_input_chars=160_000,
                model_context_tokens=16_000,
                authority_required=True,
                authority_items_raw=2,
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=ledger,
            policy=FailureRecoveryPolicy(),
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        decision = RecoveryCoordinator(clock=clock).decide(coord_input)
        assert decision.action is RecoveryAction.REROUTE_MODEL
        assert decision.selected_candidate is not None
        assert decision.selected_candidate.model_id == "claude"

    def test_required_raw_authority_blocks_when_missing(self, clock: FixedClock) -> None:
        candidates = (_candidate("openai", "gpt-4o", "openai-direct", context_tokens=16_000),)
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            context_overflow=ContextOverflowMetadata(
                estimated_input_chars=160_000,
                model_context_tokens=16_000,
                authority_required=True,
                authority_items_raw=0,
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=RetryLedger("task-1"),
            policy=FailureRecoveryPolicy(),
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        decision = RecoveryCoordinator(clock=clock).decide(coord_input)
        assert decision.action is RecoveryAction.BLOCK
        assert "authority validation failed" in decision.explanation.lower()

    def test_rebuild_context_is_bounded(self, clock: FixedClock) -> None:
        ledger = RetryLedger("task-1")
        for i in range(FailureRecoveryPolicy().max_context_rebuilds):
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
        candidates = (_candidate("openai", "gpt-4o", "openai-direct", context_tokens=16_000),)
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            context_overflow=ContextOverflowMetadata(
                estimated_input_chars=160_000,
                model_context_tokens=16_000,
                authority_required=True,
                authority_items_raw=2,
            ),
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        coord_input = RecoveryCoordinatorInput(
            classifier_input=inputs,
            candidates=candidates,
            ledger=ledger,
            policy=FailureRecoveryPolicy(),
            current_risk=RiskLevel.R2_NORMAL,
            role=ExecutionRole.CODING,
        )
        decision = RecoveryCoordinator(clock=clock).decide(coord_input)
        assert decision.action is RecoveryAction.BLOCK


class TestFailureClassifierContextOverflow:
    def test_classifier_uses_token_conversion(self) -> None:
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            context_overflow=ContextOverflowMetadata(
                estimated_input_chars=30_000,
                model_context_tokens=16_000,
            ),
        )
        classification = FailureClassifier().classify(inputs)
        assert classification.category is not FailureCategory.CONTEXT_CAPACITY

    def test_classifier_overflow_when_required_tokens_exceed_capacity(self) -> None:
        inputs = FailureClassifierInput(
            task_id="task-1",
            role=ExecutionRole.CODING,
            context_overflow=ContextOverflowMetadata(
                estimated_input_chars=30_000,
                required_context_tokens=40_000,
                model_context_tokens=16_000,
            ),
        )
        classification = FailureClassifier().classify(inputs)
        assert classification.category is FailureCategory.CONTEXT_CAPACITY


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


def _invalid_revision_mismatch_packet() -> ContextPacket:
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
        revision="def456",
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


def _summary_only_authority_packet() -> ContextPacket:
    from src.context.schema import ContextSummary

    provenance = ProvenanceRef(
        source_type="summary",
        path="docs/PROJECT_STATE.json",
        revision="abc123",
        content_hash="hash1",
    )
    summary = ContextSummary(
        text="summary of state",
        source_provenance_ids=("summary-prov",),
        source_hashes=("hash1",),
        level="high",
        lossy=True,
    )
    return ContextPacket(
        summaries=(summary,),
        provenance_index={"summary-prov": provenance},
        authority_presence=AuthorityPresence.NOT_REQUIRED,
        summary_count=1,
    )


def _make_overflow_input_with_packet(
    clock: FixedClock,
    packet: ContextPacket | None,
    requirements: RiskContextRequirements | None = None,
    ledger: RetryLedger | None = None,
    candidates: tuple[RecoveryCandidate, ...] | None = None,
) -> RecoveryCoordinatorInput:
    return RecoveryCoordinatorInput(
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
        candidates=candidates
        or (
            RecoveryCandidate(
                provider_id="openai",
                model_id="gpt-4o",
                route_id="openai-direct",
                model_identity=ModelIdentity(model_id="gpt-4o", family="gpt-4o"),
                route_identity=InferenceRouteIdentity(
                    route_id="openai-direct",
                    provider_id="openai",
                    route_type=RouteType.DIRECT,
                    endpoint_key="openai-direct",
                    failure_domain="openai",
                ),
                capabilities=ModelCapabilities(
                    context_tokens=16_000,
                    supported_roles=frozenset({ExecutionRole.CODING.value}),
                ),
                recovery_state=RouteRecoveryState(),
                failure_domain="openai",
            ),
        ),
        ledger=ledger or RetryLedger("task-1"),
        policy=FailureRecoveryPolicy(),
        current_risk=RiskLevel.R2_NORMAL,
        role=ExecutionRole.CODING,
        context_packet=packet,
        risk_context_requirements=requirements,
    )


class TestContextAuthorityValidation:
    def test_valid_raw_authority_allowed_when_it_fits(self, clock: FixedClock) -> None:
        inputs = _make_overflow_input_with_packet(
            clock,
            _valid_raw_authority_packet(),
            RiskContextRequirements(
                strategy_preference="large_context",
                authority_required=True,
                require_raw_authority=True,
                include_test_evidence=False,
                include_historical_findings=False,
                budget_multiplier=2.0,
                rationale="test",
            ),
        )
        decision = RecoveryCoordinator(clock=clock).decide(inputs)
        assert decision.action is RecoveryAction.REBUILD_CONTEXT

    def test_invalid_raw_missing_hash_blocked(self, clock: FixedClock) -> None:
        inputs = _make_overflow_input_with_packet(
            clock,
            _invalid_raw_missing_hash_packet(),
            RiskContextRequirements(
                strategy_preference="large_context",
                authority_required=True,
                require_raw_authority=True,
                include_test_evidence=False,
                include_historical_findings=False,
                budget_multiplier=2.0,
                rationale="test",
            ),
        )
        decision = RecoveryCoordinator(clock=clock).decide(inputs)
        assert decision.action is RecoveryAction.BLOCK

    def test_invalid_revision_mismatch_blocked(self, clock: FixedClock) -> None:
        inputs = _make_overflow_input_with_packet(
            clock,
            _invalid_revision_mismatch_packet(),
            RiskContextRequirements(
                strategy_preference="large_context",
                authority_required=True,
                require_raw_authority=True,
                include_test_evidence=False,
                include_historical_findings=False,
                budget_multiplier=2.0,
                rationale="test",
            ),
        )
        decision = RecoveryCoordinator(clock=clock).decide(inputs)
        assert decision.action is RecoveryAction.BLOCK

    def test_referenced_authority_allowed_when_policy_permits(self, clock: FixedClock) -> None:
        inputs = _make_overflow_input_with_packet(
            clock,
            _referenced_authority_packet(),
            RiskContextRequirements(
                strategy_preference="hybrid",
                authority_required=True,
                require_raw_authority=False,
                include_test_evidence=False,
                include_historical_findings=False,
                budget_multiplier=1.0,
                rationale="test",
            ),
        )
        decision = RecoveryCoordinator(clock=clock).decide(inputs)
        assert decision.action is RecoveryAction.REBUILD_CONTEXT

    def test_referenced_authority_blocked_when_raw_required(self, clock: FixedClock) -> None:
        inputs = _make_overflow_input_with_packet(
            clock,
            _referenced_authority_packet(),
            RiskContextRequirements(
                strategy_preference="large_context",
                authority_required=True,
                require_raw_authority=True,
                include_test_evidence=False,
                include_historical_findings=False,
                budget_multiplier=2.0,
                rationale="test",
            ),
        )
        decision = RecoveryCoordinator(clock=clock).decide(inputs)
        assert decision.action is RecoveryAction.BLOCK

    def test_summary_only_authority_blocked(self, clock: FixedClock) -> None:
        inputs = _make_overflow_input_with_packet(
            clock,
            _summary_only_authority_packet(),
            RiskContextRequirements(
                strategy_preference="hybrid",
                authority_required=True,
                require_raw_authority=False,
                include_test_evidence=False,
                include_historical_findings=False,
                budget_multiplier=1.0,
                rationale="test",
            ),
        )
        decision = RecoveryCoordinator(clock=clock).decide(inputs)
        assert decision.action is RecoveryAction.BLOCK

    def test_valid_r4_packet_allows_larger_model_reroute(self, clock: FixedClock) -> None:
        ledger = RetryLedger("task-1")
        for i in range(FailureRecoveryPolicy().max_context_rebuilds):
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
            RecoveryCandidate(
                provider_id="openai",
                model_id="gpt-4o",
                route_id="openai-direct",
                model_identity=ModelIdentity(model_id="gpt-4o", family="gpt-4o"),
                route_identity=InferenceRouteIdentity(
                    route_id="openai-direct",
                    provider_id="openai",
                    route_type=RouteType.DIRECT,
                    endpoint_key="openai-direct",
                    failure_domain="openai",
                ),
                capabilities=ModelCapabilities(
                    context_tokens=16_000,
                    supported_roles=frozenset({ExecutionRole.CODING.value}),
                ),
                recovery_state=RouteRecoveryState(),
                failure_domain="openai",
            ),
            RecoveryCandidate(
                provider_id="anthropic",
                model_id="claude",
                route_id="anthropic-direct",
                model_identity=ModelIdentity(model_id="claude", family="claude"),
                route_identity=InferenceRouteIdentity(
                    route_id="anthropic-direct",
                    provider_id="anthropic",
                    route_type=RouteType.DIRECT,
                    endpoint_key="anthropic-direct",
                    failure_domain="anthropic",
                ),
                capabilities=ModelCapabilities(
                    context_tokens=128_000,
                    supported_roles=frozenset({ExecutionRole.CODING.value}),
                ),
                recovery_state=RouteRecoveryState(),
                failure_domain="anthropic",
            ),
        )
        inputs = _make_overflow_input_with_packet(
            clock,
            _valid_raw_authority_packet(),
            RiskContextRequirements(
                strategy_preference="large_context",
                authority_required=True,
                require_raw_authority=True,
                include_test_evidence=False,
                include_historical_findings=False,
                budget_multiplier=2.0,
                rationale="test",
            ),
            ledger=ledger,
            candidates=candidates,
        )
        decision = RecoveryCoordinator(clock=clock).decide(inputs)
        assert decision.action is RecoveryAction.REROUTE_MODEL
        assert decision.selected_candidate is not None
        assert decision.selected_candidate.model_id == "claude"
