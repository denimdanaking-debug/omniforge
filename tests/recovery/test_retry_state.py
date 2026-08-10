"""Tests for Phase 10 retry ledger and state persistence."""

from __future__ import annotations

import datetime
import tempfile
from pathlib import Path

import pytest

from src.persistence import runtime_state
from src.recovery.retry_state import (
    RetryLedger,
    RetryType,
    WaitState,
)
from src.security.redaction import contains_secret

SENTINEL = "OMNIFORGE_TEST_SECRET_SENTINEL_PHASE10_RETRY_555"


@pytest.fixture
def base_time() -> datetime.datetime:
    return datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)


@pytest.fixture
def ledger(base_time: datetime.datetime) -> RetryLedger:
    return RetryLedger(task_id="task-1")


class TestRetryLedger:
    def test_record_increments_attempt_index(
        self, ledger: RetryLedger, base_time: datetime.datetime
    ) -> None:
        record = ledger.record(
            failure_category="INFRASTRUCTURE_TRANSIENT",
            failure_subtype="TRANSIENT_TRANSPORT",
            failure_signature="sig-1",
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
            action_taken="RETRY_SAME_ROUTE",
            retry_type=RetryType.TRANSIENT_RETRY,
            timestamp=base_time,
        )
        assert record.attempt_index == 0
        record2 = ledger.record(
            failure_category="INFRASTRUCTURE_TRANSIENT",
            failure_subtype="TRANSIENT_TRANSPORT",
            failure_signature="sig-1",
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
            action_taken="RETRY_SAME_ROUTE",
            retry_type=RetryType.TRANSIENT_RETRY,
            timestamp=base_time + datetime.timedelta(seconds=1),
        )
        assert record2.attempt_index == 1

    def test_signature_count_tracks_repeated_failures(
        self, ledger: RetryLedger, base_time: datetime.datetime
    ) -> None:
        for _ in range(3):
            ledger.record(
                failure_category="IMPLEMENTATION_DETERMINISTIC",
                failure_subtype="TEST_FAILURE",
                failure_signature="sig-test-x",
                provider_id="openai",
                model_id="gpt-4o",
                route_id="openai-direct",
                action_taken="REPAIR",
                retry_type=RetryType.REPAIR,
                timestamp=base_time,
            )
        assert ledger.signature_count("sig-test-x") == 3

    def test_provider_switch_count(self, ledger: RetryLedger, base_time: datetime.datetime) -> None:
        for provider in ("openai", "anthropic", "openai"):
            ledger.record(
                failure_category="INFRASTRUCTURE_TRANSIENT",
                failure_subtype="TRANSIENT_TRANSPORT",
                failure_signature="sig-1",
                provider_id=provider,
                model_id="model",
                route_id="route",
                action_taken="REROUTE",
                retry_type=RetryType.REROUTE_PROVIDER,
                timestamp=base_time,
            )
        assert ledger.provider_switch_count() == 2

    def test_model_switch_count(self, ledger: RetryLedger, base_time: datetime.datetime) -> None:
        for model in ("gpt-4o", "claude", "gpt-4o"):
            ledger.record(
                failure_category="INFRASTRUCTURE_TRANSIENT",
                failure_subtype="TRANSIENT_TRANSPORT",
                failure_signature="sig-1",
                provider_id="openai",
                model_id=model,
                route_id="route",
                action_taken="REROUTE",
                retry_type=RetryType.REROUTE_MODEL,
                timestamp=base_time,
            )
        assert ledger.model_switch_count() == 2

    def test_exhausted_path_tracked(
        self, ledger: RetryLedger, base_time: datetime.datetime
    ) -> None:
        ledger.mark_exhausted_path("sig-1", "openai", "gpt-4o")
        assert ledger.is_exhausted_path("sig-1", "openai", "gpt-4o")
        assert not ledger.is_exhausted_path("sig-1", "anthropic", "claude")

    def test_ledger_serializes_and_deserializes(
        self, ledger: RetryLedger, base_time: datetime.datetime
    ) -> None:
        ledger.record(
            failure_category="INFRASTRUCTURE_TRANSIENT",
            failure_subtype="TRANSIENT_TRANSPORT",
            failure_signature="sig-1",
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
            action_taken="RETRY_SAME_ROUTE",
            retry_type=RetryType.TRANSIENT_RETRY,
            timestamp=base_time,
        )
        data = ledger.to_dict()
        restored = RetryLedger.from_dict(data)
        assert restored.attempt_count == ledger.attempt_count
        assert restored.signature_count("sig-1") == 1

    def test_wait_state_serializes_and_deserializes(
        self, ledger: RetryLedger, base_time: datetime.datetime
    ) -> None:
        wait = WaitState(
            reason="no_eligible_routes",
            next_recheck_at=base_time + datetime.timedelta(minutes=5),
            entered_at=base_time,
        )
        ledger.set_wait(wait)
        data = ledger.to_dict()
        restored = RetryLedger.from_dict(data)
        assert restored.current_wait is not None
        assert restored.current_wait.reason == "no_eligible_routes"

    def test_retry_state_survives_runtime_state_save_load(
        self, ledger: RetryLedger, base_time: datetime.datetime
    ) -> None:
        ledger.record(
            failure_category="INFRASTRUCTURE_TRANSIENT",
            failure_subtype="TRANSIENT_TRANSPORT",
            failure_signature="sig-1",
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
            action_taken="RETRY_SAME_ROUTE",
            retry_type=RetryType.TRANSIENT_RETRY,
            timestamp=base_time,
            retry_after=base_time + datetime.timedelta(seconds=30),
        )
        state = {
            "schema_version": "1.4.0",
            "run_id": "run-1",
            "workflow_state": "WAITING_FOR_RETRY",
            "checkpoint": {"task_id": "task-1"},
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
            "task_retry_state": {"task-1": ledger.to_dict()},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.json"
            runtime_state.save_runtime_state(path, state)
            reloaded = runtime_state.load_runtime_state(path)
            retry_state = reloaded["task_retry_state"]["task-1"]
            restored_ledger = RetryLedger.from_dict(retry_state)
            assert restored_ledger.attempt_count == 1
            last = restored_ledger.last_record()
            assert last is not None
            assert last.failure_signature == "sig-1"

    def test_secret_sentinel_absent_from_serialized_ledger(
        self, ledger: RetryLedger, base_time: datetime.datetime
    ) -> None:
        ledger.record(
            failure_category="INFRASTRUCTURE_AUTH",
            failure_subtype="AUTH_FAILURE",
            failure_signature="sig-auth",
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
            action_taken=f"auth failed bearer {SENTINEL}",
            retry_type=RetryType.BLOCK,
            timestamp=base_time,
        )
        data = ledger.to_dict()
        text = str(data)
        assert not contains_secret(text, SENTINEL)
        assert "***" in text
