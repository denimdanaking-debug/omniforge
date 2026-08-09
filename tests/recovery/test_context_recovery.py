"""Tests for context-overflow capacity semantics and authority safety."""

from __future__ import annotations

import datetime

import pytest

from src.context.budget import estimate_tokens
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
        assert "raw authority cannot be preserved" in decision.explanation.lower()

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
