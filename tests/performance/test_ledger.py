"""Tests for the append-only performance-event ledger."""

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
    PerformanceLedger,
    Usage,
)
from src.security.redaction import contains_secret


@pytest.fixture
def base_time() -> datetime.datetime:
    return datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)


def _event(
    *,
    event_id: str,
    timestamp: datetime.datetime,
    model_id: str = "gpt-4o",
    provider_id: str = "openai",
    route_id: str = "openai-direct",
    outcome_category: OutcomeCategory = OutcomeCategory.SUCCESS,
    first_pass: bool = True,
    usage: Usage | None = None,
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
        provider_id=provider_id,
        model_id=model_id,
        route_id=route_id,
        outcome_category=outcome_category,
        acceptance_status=AcceptanceStatus.ACCEPTED,
        first_pass=first_pass,
        usage=usage or Usage(input_tokens=10, output_tokens=5),
        direct_cost=Cost(0.001, "USD", CostState.ACTUAL),
    )


class TestPerformanceLedger:
    def test_append_only(self, base_time: datetime.datetime) -> None:
        ledger = PerformanceLedger()
        event = _event(event_id="e1", timestamp=base_time)
        ledger2 = ledger.append(event)
        assert len(ledger.events) == 0
        assert len(ledger2.events) == 1

    def test_exact_duplicate_is_idempotent(self, base_time: datetime.datetime) -> None:
        event = _event(event_id="e1", timestamp=base_time)
        ledger = PerformanceLedger().append(event)
        ledger2 = ledger.append(event)
        assert ledger2 is ledger
        assert len(ledger2.events) == 1

    def test_same_id_different_fingerprint_fails_closed(self, base_time: datetime.datetime) -> None:
        e1 = _event(event_id="e1", timestamp=base_time, usage=Usage(input_tokens=1))
        e2 = _event(event_id="e1", timestamp=base_time, usage=Usage(input_tokens=2))
        ledger = PerformanceLedger().append(e1)
        with pytest.raises(ValueError, match="collision"):
            ledger.append(e2)

    def test_append_all_exact_duplicates_are_idempotent(self, base_time: datetime.datetime) -> None:
        event = _event(event_id="e1", timestamp=base_time)
        ledger = PerformanceLedger().append_all((event, event))
        assert len(ledger.events) == 1

    def test_append_all_conflicting_intra_batch_duplicate_fails(
        self, base_time: datetime.datetime
    ) -> None:
        e1 = _event(event_id="e1", timestamp=base_time, usage=Usage(input_tokens=1))
        e2 = _event(event_id="e1", timestamp=base_time, usage=Usage(input_tokens=2))
        with pytest.raises(ValueError, match="collision"):
            PerformanceLedger().append_all((e1, e2))

    def test_append_rejects_forged_fingerprint(self, base_time: datetime.datetime) -> None:
        event = PerformanceEvent.from_dict(
            {
                **_event(event_id="e1", timestamp=base_time).to_dict(),
                "event_fingerprint": "forged",
            }
        )
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            PerformanceLedger().append(event)

    def test_append_all_exact_duplicate_plus_new_event(self, base_time: datetime.datetime) -> None:
        e1 = _event(event_id="e1", timestamp=base_time)
        e2 = _event(event_id="e2", timestamp=base_time)
        ledger = PerformanceLedger().append(e1).append_all((e1, e2))
        assert len(ledger.events) == 2
        assert ledger.has_event("e1")
        assert ledger.has_event("e2")

    def test_append_all_collision_leaves_ledger_unchanged(
        self, base_time: datetime.datetime
    ) -> None:
        e1 = _event(event_id="e1", timestamp=base_time, usage=Usage(input_tokens=1))
        e2 = _event(event_id="e2", timestamp=base_time)
        e1_modified = _event(event_id="e1", timestamp=base_time, usage=Usage(input_tokens=2))
        ledger = PerformanceLedger().append(e1)
        with pytest.raises(ValueError, match="collision"):
            ledger.append_all((e2, e1_modified))
        assert len(ledger.events) == 1
        assert ledger.events[0].event_id == "e1"
        assert ledger.events[0].usage.input_tokens == 1

    def test_serialize_restart_replay_no_duplicate(self, base_time: datetime.datetime) -> None:
        event = _event(event_id="e1", timestamp=base_time)
        ledger = PerformanceLedger().append(event)
        reloaded = PerformanceLedger.from_dict(ledger.to_dict())
        replayed = reloaded.append(event)
        assert replayed == reloaded
        assert len(replayed.events) == 1

    def test_has_event(self, base_time: datetime.datetime) -> None:
        event = _event(event_id="e1", timestamp=base_time)
        ledger = PerformanceLedger().append(event)
        assert ledger.has_event("e1")
        assert not ledger.has_event("e2")

    def test_events_for_filters(self, base_time: datetime.datetime) -> None:
        e1 = _event(event_id="e1", timestamp=base_time, model_id="gpt-4o")
        e2 = _event(
            event_id="e2",
            timestamp=base_time,
            model_id="claude",
            provider_id="anthropic",
            route_id="anthropic-direct",
        )
        ledger = PerformanceLedger().append_all((e1, e2))
        assert ledger.events_for_model("gpt-4o") == (e1,)
        assert ledger.events_for_role("coding") == (e1, e2)
        assert ledger.events_for_project("omniforge") == (e1, e2)

    def test_serialization_survives_restart(self, base_time: datetime.datetime) -> None:
        event = _event(event_id="e1", timestamp=base_time)
        ledger = PerformanceLedger().append(event)
        reloaded = PerformanceLedger.from_dict(ledger.to_dict())
        assert reloaded == ledger

    def test_safe_dict_no_secret_sentinel(self, base_time: datetime.datetime) -> None:
        sentinel = "OMNIFORGE_PHASE11_LEDGER_SECRET_999"
        event = PerformanceEvent.from_dict(
            {
                **_event(event_id="e1", timestamp=base_time).to_dict(),
                "originating_ids": {"api_key": sentinel},
                "event_fingerprint": "",
            }
        )
        ledger = PerformanceLedger().append(event)
        safe = ledger.to_safe_dict()
        assert not contains_secret(safe, sentinel)
        assert sentinel not in str(safe)

    def test_from_dict_rejects_forged_fingerprint(self, base_time: datetime.datetime) -> None:
        event = _event(event_id="e1", timestamp=base_time)
        payload = event.to_dict()
        payload["event_fingerprint"] = "forged"
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            PerformanceLedger.from_dict({"schema_version": "1.0.0", "events": [payload]})

    def test_from_dict_rejects_duplicate_event_ids(self, base_time: datetime.datetime) -> None:
        payload = _event(event_id="e1", timestamp=base_time).to_dict()
        with pytest.raises(ValueError, match="duplicate event_id"):
            PerformanceLedger.from_dict({"schema_version": "1.0.0", "events": [payload, payload]})

    def test_replay_with_same_fingerprint_is_idempotent(self, base_time: datetime.datetime) -> None:
        event = _event(event_id="e1", timestamp=base_time)
        ledger = PerformanceLedger().append(event)
        reloaded = PerformanceLedger.from_dict(ledger.to_dict())
        assert reloaded == ledger
