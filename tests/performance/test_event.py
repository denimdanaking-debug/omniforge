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
    performance_event_fingerprint,
)
from src.performance.event import thaw_json_value
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

    def test_validation_result_summary_cannot_be_mutated(
        self, base_time: datetime.datetime
    ) -> None:
        event = PerformanceEvent.from_dict(
            {
                **_event(timestamp=base_time).to_dict(),
                "validation_result_summary": {"exit_status": 1},
            }
        )
        with pytest.raises(TypeError):
            event.validation_result_summary["exit_status"] = 2  # type: ignore[index]

    def test_review_finding_dispositions_cannot_be_mutated(
        self, base_time: datetime.datetime
    ) -> None:
        from src.performance import FindingDisposition

        event = PerformanceEvent.from_dict(
            {
                **_event(timestamp=base_time).to_dict(),
                "review_finding_dispositions": {"f1": FindingDisposition.SUPPORTED},
            }
        )
        with pytest.raises(TypeError):
            event.review_finding_dispositions["f1"] = FindingDisposition.UNSUPPORTED  # type: ignore[index]

    def test_originating_ids_cannot_be_mutated(self, base_time: datetime.datetime) -> None:
        event = _event(timestamp=base_time)
        with pytest.raises(TypeError):
            event.originating_ids["extra"] = "x"  # type: ignore[index]

    def test_mutating_input_dict_after_creation_does_not_affect_event(
        self, base_time: datetime.datetime
    ) -> None:
        summary = {"exit_status": 1}
        event = PerformanceEvent.from_dict(
            {**_event(timestamp=base_time).to_dict(), "validation_result_summary": summary}
        )
        summary["exit_status"] = 99
        assert event.validation_result_summary["exit_status"] == 1

    def test_nested_mutation_is_blocked(self, base_time: datetime.datetime) -> None:
        summary = {"validator": {"failures": ["a"]}}
        event = PerformanceEvent.from_dict(
            {**_event(timestamp=base_time).to_dict(), "validation_result_summary": summary}
        )
        with pytest.raises((TypeError, AttributeError)):
            event.validation_result_summary["validator"]["failures"].append("b")

    def test_mutating_original_nested_list_after_creation_does_not_affect_event(
        self, base_time: datetime.datetime
    ) -> None:
        summary = {"validator": {"failures": ["a"]}}
        event = PerformanceEvent.from_dict(
            {**_event(timestamp=base_time).to_dict(), "validation_result_summary": summary}
        )
        summary["validator"]["failures"].append("b")
        assert list(event.validation_result_summary["validator"]["failures"]) == ["a"]

    def test_deeply_nested_structures_frozen(self, base_time: datetime.datetime) -> None:
        summary = {"a": {"b": [{"c": [1, 2, {"d": "e"}]}]}}
        event = PerformanceEvent.from_dict(
            {**_event(timestamp=base_time).to_dict(), "validation_result_summary": summary}
        )
        assert thaw_json_value(event.validation_result_summary) == summary
        with pytest.raises((TypeError, AttributeError)):
            event.validation_result_summary["a"]["b"][0]["c"].append(3)

    def test_round_trip_preserves_nested_content(self, base_time: datetime.datetime) -> None:
        summary = {"top": {"list": [1, 2, {"nested": "value"}]}}
        event = PerformanceEvent.from_dict(
            {**_event(timestamp=base_time).to_dict(), "validation_result_summary": summary}
        )
        restored = PerformanceEvent.from_dict(event.to_dict())
        assert event.to_dict()["validation_result_summary"] == summary
        assert restored.to_dict()["validation_result_summary"] == summary

    def test_fingerprint_invariant_to_nested_dict_insertion_order(
        self, base_time: datetime.datetime
    ) -> None:
        summary_a = {"z": 1, "a": {"y": 2, "x": 3}}
        summary_b = {"a": {"x": 3, "y": 2}, "z": 1}
        base = _event(event_id="e1", timestamp=base_time).to_dict()
        e1 = PerformanceEvent.from_dict(
            {**base, "validation_result_summary": summary_a, "event_fingerprint": ""}
        )
        e2 = PerformanceEvent.from_dict(
            {**base, "validation_result_summary": summary_b, "event_fingerprint": ""}
        )
        assert e1.event_fingerprint == e2.event_fingerprint


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


class TestEventFingerprint:
    def test_fingerprint_covers_usage(self, base_time: datetime.datetime) -> None:
        e1 = _event(event_id="e1", timestamp=base_time, usage=Usage(input_tokens=1))
        e2 = _event(event_id="e2", timestamp=base_time, usage=Usage(input_tokens=2))
        assert e1.event_fingerprint != e2.event_fingerprint

    def test_fingerprint_covers_cost(self, base_time: datetime.datetime) -> None:
        e1 = _event(
            event_id="e1", timestamp=base_time, direct_cost=Cost(0.1, "USD", CostState.ACTUAL)
        )
        e2 = _event(
            event_id="e2", timestamp=base_time, direct_cost=Cost(0.2, "USD", CostState.ACTUAL)
        )
        assert e1.event_fingerprint != e2.event_fingerprint

    def test_fingerprint_covers_context_strategy(self, base_time: datetime.datetime) -> None:
        e1 = PerformanceEvent.from_dict(
            {**_event(event_id="e1", timestamp=base_time).to_dict(), "context_strategy": "targeted"}
        )
        e2 = PerformanceEvent.from_dict(
            {**_event(event_id="e2", timestamp=base_time).to_dict(), "context_strategy": "hybrid"}
        )
        assert e1.event_fingerprint != e2.event_fingerprint

    def test_fingerprint_covers_authority_adherence(self, base_time: datetime.datetime) -> None:
        from src.performance import AuthorityAdherenceStatus

        e1 = PerformanceEvent.from_dict(
            {
                **_event(event_id="e1", timestamp=base_time).to_dict(),
                "authority_adherence": AuthorityAdherenceStatus.COMPLIANT,
            }
        )
        e2 = PerformanceEvent.from_dict(
            {
                **_event(event_id="e2", timestamp=base_time).to_dict(),
                "authority_adherence": AuthorityAdherenceStatus.ATTEMPTED_MUTATION,
            }
        )
        assert e1.event_fingerprint != e2.event_fingerprint

    def test_fingerprint_is_stable_for_identical_event(self, base_time: datetime.datetime) -> None:
        payload = _event(event_id="e1", timestamp=base_time).to_dict()
        e1 = PerformanceEvent.from_dict(payload)
        e2 = PerformanceEvent.from_dict(payload)
        assert e1.event_fingerprint == e2.event_fingerprint
        assert performance_event_fingerprint(e1) == e1.event_fingerprint


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
