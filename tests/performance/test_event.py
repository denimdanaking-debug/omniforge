"""Tests for the immutable performance-event model."""

from __future__ import annotations

import datetime

import pytest

from src.performance import (
    AcceptanceStatus,
    Cost,
    CostState,
    OutcomeCategory,
    PerformanceEvent,
    PerformanceEventType,
    Usage,
    event_identity,
)
from src.security.redaction import contains_secret


@pytest.fixture
def base_time() -> datetime.datetime:
    return datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)


def _event(
    *,
    event_id: str = "ev1",
    timestamp: datetime.datetime,
    event_type: PerformanceEventType = PerformanceEventType.TASK_OUTCOME,
    outcome_category: OutcomeCategory = OutcomeCategory.SUCCESS,
    first_pass: bool = True,
    acceptance_status: AcceptanceStatus = AcceptanceStatus.ACCEPTED,
    model_id: str = "gpt-4o",
    provider_id: str = "openai",
    route_id: str = "openai-direct",
    usage: Usage | None = None,
    direct_cost: Cost | None = None,
) -> PerformanceEvent:
    return PerformanceEvent(
        event_id=event_id,
        schema_version="1.0.0",
        timestamp=timestamp,
        event_type=event_type,
        project_id="omniforge",
        task_id="task-1",
        run_id="run-1",
        execution_role="coding",
        task_class="feature",
        risk="R2_NORMAL",
        provider_id=provider_id,
        model_id=model_id,
        route_id=route_id,
        outcome_category=outcome_category,
        acceptance_status=acceptance_status,
        first_pass=first_pass,
        usage=usage or Usage(input_tokens=100, output_tokens=50),
        direct_cost=direct_cost or Cost(0.005, "USD", CostState.ACTUAL),
    )


class TestPerformanceEventImmutability:
    def test_event_is_frozen(self, base_time: datetime.datetime) -> None:
        event = _event(timestamp=base_time)
        with pytest.raises(AttributeError):
            event.model_id = "other"  # type: ignore[misc]

    def test_event_round_trip(self, base_time: datetime.datetime) -> None:
        event = _event(timestamp=base_time)
        restored = PerformanceEvent.from_dict(event.to_dict())
        assert restored == event

    def test_event_safe_dict_redacts_secrets(self, base_time: datetime.datetime) -> None:
        event = _event(timestamp=base_time)
        event = PerformanceEvent.from_dict(
            {
                **event.to_dict(),
                "originating_ids": {"api_key": "sk-live-abcdef"},
            }
        )
        safe = event.to_safe_dict()
        assert not contains_secret(safe, "sk-live-abcdef")


class TestEventIdentity:
    def test_same_logical_event_same_id(self, base_time: datetime.datetime) -> None:
        eid1 = event_identity(
            event_type=PerformanceEventType.TASK_OUTCOME,
            task_id="task-1",
            run_id="run-1",
            timestamp=base_time,
            model_id="gpt-4o",
            provider_id="openai",
            route_id="openai-direct",
            outcome_category=OutcomeCategory.SUCCESS,
            sequence=0,
        )
        eid2 = event_identity(
            event_type=PerformanceEventType.TASK_OUTCOME,
            task_id="task-1",
            run_id="run-1",
            timestamp=base_time,
            model_id="gpt-4o",
            provider_id="openai",
            route_id="openai-direct",
            outcome_category=OutcomeCategory.SUCCESS,
            sequence=0,
        )
        assert eid1 == eid2

    def test_material_difference_changes_id(self, base_time: datetime.datetime) -> None:
        eid1 = event_identity(
            event_type=PerformanceEventType.TASK_OUTCOME,
            task_id="task-1",
            run_id="run-1",
            timestamp=base_time,
            model_id="gpt-4o",
            provider_id="openai",
            route_id="openai-direct",
            outcome_category=OutcomeCategory.SUCCESS,
        )
        eid2 = event_identity(
            event_type=PerformanceEventType.TASK_OUTCOME,
            task_id="task-1",
            run_id="run-1",
            timestamp=base_time,
            model_id="claude",
            provider_id="anthropic",
            route_id="anthropic-direct",
            outcome_category=OutcomeCategory.SUCCESS,
        )
        assert eid1 != eid2

    def test_sequence_disambiguates_same_task_events(self, base_time: datetime.datetime) -> None:
        eid1 = event_identity(
            event_type=PerformanceEventType.TASK_OUTCOME,
            task_id="task-1",
            run_id="run-1",
            timestamp=base_time,
            model_id="gpt-4o",
            provider_id="openai",
            route_id="openai-direct",
            outcome_category=OutcomeCategory.SUCCESS,
            sequence=0,
        )
        eid2 = event_identity(
            event_type=PerformanceEventType.TASK_OUTCOME,
            task_id="task-1",
            run_id="run-1",
            timestamp=base_time,
            model_id="gpt-4o",
            provider_id="openai",
            route_id="openai-direct",
            outcome_category=OutcomeCategory.SUCCESS,
            sequence=1,
        )
        assert eid1 != eid2


class TestCostAndUsage:
    def test_unknown_cost_preserved(self, base_time: datetime.datetime) -> None:
        event = _event(timestamp=base_time, direct_cost=Cost(None, "USD", CostState.UNKNOWN))
        assert event.direct_cost.state is CostState.UNKNOWN
        assert event.direct_cost.amount is None

    def test_negative_cost_rejected(self, base_time: datetime.datetime) -> None:
        with pytest.raises(ValueError):
            Cost(-1.0, "USD", CostState.ACTUAL)

    def test_unknown_token_usage_preserved(self, base_time: datetime.datetime) -> None:
        event = _event(timestamp=base_time, usage=Usage())
        assert event.usage.input_tokens is None

    def test_negative_tokens_rejected(self, base_time: datetime.datetime) -> None:
        with pytest.raises(ValueError):
            Usage(input_tokens=-1)
