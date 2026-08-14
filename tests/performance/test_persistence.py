"""Tests for runtime-state persistence integration."""

from __future__ import annotations

import datetime

import pytest

from src.performance import (
    AcceptanceStatus,
    Cost,
    CostState,
    OutcomeCategory,
    PerformanceAttribution,
    PerformanceEvent,
    PerformanceEventType,
    PerformanceLedger,
    performance_state_from_dict,
    performance_state_to_dict,
)
from src.persistence import runtime_state


@pytest.fixture
def base_time() -> datetime.datetime:
    return datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)


def _event(
    event_id: str,
    timestamp: datetime.datetime,
    attribution: str = PerformanceAttribution.MODEL_QUALITY.value,
) -> PerformanceEvent:
    return PerformanceEvent(
        event_id=event_id,
        schema_version="1.0.0",
        timestamp=timestamp,
        event_type=PerformanceEventType.TASK_OUTCOME,
        project_id="omniforge",
        task_id="task-1",
        run_id="run-1",
        execution_role="coding",
        task_class="feature",
        risk="R2_NORMAL",
        provider_id="openai",
        model_id="gpt-4o",
        route_id="openai-direct",
        outcome_category=OutcomeCategory.SUCCESS,
        acceptance_status=AcceptanceStatus.ACCEPTED,
        first_pass=True,
        direct_cost=Cost(0.01, "USD", CostState.ACTUAL),
        attribution=attribution,
    )


class TestRuntimeStateMigration:
    def test_migration_adds_performance_statistics(self) -> None:
        old = {
            "schema_version": "1.4.0",
            "run_id": "r1",
            "workflow_state": "STOPPED",
            "checkpoint": {},
        }
        migrated = runtime_state.migrate_runtime_state(old)
        assert migrated["schema_version"] == runtime_state.CURRENT_RUNTIME_STATE_VERSION
        assert "performance_statistics" in migrated
        assert isinstance(migrated["performance_statistics"], dict)

    def test_phase_10_task_retry_state_preserved(self) -> None:
        old = {
            "schema_version": "1.4.0",
            "run_id": "r1",
            "workflow_state": "STOPPED",
            "checkpoint": {},
            "task_retry_state": {"task-1": {"records": []}},
        }
        migrated = runtime_state.migrate_runtime_state(old)
        assert migrated["task_retry_state"]["task-1"]["records"] == []

    def test_validation_accepts_performance_statistics(self) -> None:
        state = {
            "schema_version": runtime_state.CURRENT_RUNTIME_STATE_VERSION,
            "run_id": "r1",
            "workflow_state": "STOPPED",
            "checkpoint": {},
            "performance_statistics": {"ledger": {"schema_version": "1.0.0", "events": []}},
        }
        validated = runtime_state.validate_runtime_state(state)
        assert "performance_statistics" in validated


class TestPerformanceStateSerialization:
    def test_round_trip(self, base_time: datetime.datetime) -> None:
        event = _event("e1", base_time)
        ledger = PerformanceLedger().append(event)
        state = performance_state_to_dict(ledger)
        restored_ledger, bundle = performance_state_from_dict(state)
        assert restored_ledger == ledger
        assert bundle.total_events == 1

    def test_rebuild_when_statistics_absent(self, base_time: datetime.datetime) -> None:
        event = _event("e1", base_time)
        ledger = PerformanceLedger().append(event)
        state = {"ledger": ledger.to_dict()}
        restored_ledger, bundle = performance_state_from_dict(state)
        assert restored_ledger == ledger
        assert bundle.total_events == 1

    def test_no_secrets_in_state(self, base_time: datetime.datetime) -> None:
        event = _event("e1", base_time)
        event = PerformanceEvent.from_dict(
            {
                **event.to_dict(),
                "originating_ids": {"api_key": "sk-live-12345"},
                "event_fingerprint": "",
            }
        )
        ledger = PerformanceLedger().append(event)
        state = performance_state_to_dict(ledger)
        text = str(state)
        assert "sk-live-12345" not in text

    def test_stale_statistics_rebuilt_from_ledger(self, base_time: datetime.datetime) -> None:
        event = _event("e1", base_time)
        ledger = PerformanceLedger().append(event)
        state = performance_state_to_dict(ledger)
        # Corrupt the derived statistics cache.
        state["statistics"]["total_events"] = 999
        state["statistics"]["model_role"]["gpt-4o:coding"]["accepted"] = 999
        restored_ledger, bundle = performance_state_from_dict(state)
        assert restored_ledger == ledger
        assert bundle.total_events == 1
        assert bundle.model_role[("gpt-4o", "coding")].accepted == 1

    def test_missing_statistics_rebuilt_from_ledger(self, base_time: datetime.datetime) -> None:
        event = _event("e1", base_time)
        ledger = PerformanceLedger().append(event)
        state = {"ledger": ledger.to_dict()}
        restored_ledger, bundle = performance_state_from_dict(state)
        assert restored_ledger == ledger
        assert bundle.total_events == 1
