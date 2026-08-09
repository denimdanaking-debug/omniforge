"""Retry ledger and persisted retry state for restart-safe recovery.

The ledger tracks per-active-task attempt history, failure signatures, and
bounded counters. It is designed to serialize cleanly into runtime state and
survive process restart without losing progress.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.recovery.clock import ensure_aware, isoformat
from src.security.redaction import redact


class RetryType(StrEnum):
    """Canonical retry action types recorded in the ledger."""

    TRANSIENT_RETRY = "transient_retry"
    REROUTE_PROVIDER = "reroute_provider"
    REROUTE_ROUTE = "reroute_route"
    REROUTE_MODEL = "reroute_model"
    CONSTRAINED_OUTPUT_RETRY = "constrained_output_retry"
    REPLAN = "replan"
    REPAIR = "repair"
    CROSS_MODEL_REPAIR = "cross_model_repair"
    REBUILD_CONTEXT = "rebuild_context"
    WAIT_FOR_PROVIDER = "wait_for_provider"
    BLOCK = "block"
    CANCEL = "cancel"


@dataclass(frozen=True)
class FailureAttemptRecord:
    """One recorded recovery attempt."""

    attempt_index: int
    failure_category: str
    failure_subtype: str
    failure_signature: str
    provider_id: str | None
    model_id: str | None
    route_id: str | None
    action_taken: str
    retry_type: RetryType
    timestamp: datetime.datetime
    retry_after: datetime.datetime | None = None
    context_rebuild_number: int = 0
    repair_number: int = 0

    def __post_init__(self) -> None:
        if self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.retry_after is not None and self.retry_after.tzinfo is None:
            raise ValueError("retry_after must be timezone-aware")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "failure_category": self.failure_category,
            "failure_subtype": self.failure_subtype,
            "failure_signature": self.failure_signature,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "route_id": self.route_id,
            "action_taken": self.action_taken,
            "retry_type": self.retry_type.value,
            "timestamp": isoformat(self.timestamp),
            "retry_after": isoformat(self.retry_after) if self.retry_after else None,
            "context_rebuild_number": self.context_rebuild_number,
            "repair_number": self.repair_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureAttemptRecord:
        timestamp = datetime.datetime.fromisoformat(data["timestamp"])
        ensure_aware(timestamp)
        retry_after = data.get("retry_after")
        if retry_after is not None:
            retry_after = datetime.datetime.fromisoformat(retry_after)
            ensure_aware(retry_after)
        return cls(
            attempt_index=int(data["attempt_index"]),
            failure_category=str(data["failure_category"]),
            failure_subtype=str(data["failure_subtype"]),
            failure_signature=str(data["failure_signature"]),
            provider_id=data.get("provider_id"),
            model_id=data.get("model_id"),
            route_id=data.get("route_id"),
            action_taken=str(data["action_taken"]),
            retry_type=RetryType(data["retry_type"]),
            timestamp=timestamp,
            retry_after=retry_after,
            context_rebuild_number=int(data.get("context_rebuild_number", 0)),
            repair_number=int(data.get("repair_number", 0)),
        )


@dataclass(frozen=True)
class WaitState:
    """Persisted wait state for a task."""

    reason: str
    next_recheck_at: datetime.datetime
    entered_at: datetime.datetime
    affected_failure_domains: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.next_recheck_at.tzinfo is None:
            raise ValueError("next_recheck_at must be timezone-aware")
        if self.entered_at.tzinfo is None:
            raise ValueError("entered_at must be timezone-aware")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "next_recheck_at": isoformat(self.next_recheck_at),
            "entered_at": isoformat(self.entered_at),
            "affected_failure_domains": sorted(self.affected_failure_domains),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WaitState:
        next_recheck_at = datetime.datetime.fromisoformat(data["next_recheck_at"])
        entered_at = datetime.datetime.fromisoformat(data["entered_at"])
        ensure_aware(next_recheck_at)
        ensure_aware(entered_at)
        return cls(
            reason=str(data["reason"]),
            next_recheck_at=next_recheck_at,
            entered_at=entered_at,
            affected_failure_domains=frozenset(data.get("affected_failure_domains", [])),
        )


@dataclass
class RetryLedger:
    """Canonical retry history and state for one active task."""

    task_id: str
    records: list[FailureAttemptRecord] = field(default_factory=list)
    current_wait: WaitState | None = None
    current_context_rebuild: dict[str, Any] = field(default_factory=dict)
    exhausted_paths: list[tuple[str, str | None, str | None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")

    @property
    def attempt_count(self) -> int:
        return len(self.records)

    @property
    def total_attempt_index(self) -> int:
        if not self.records:
            return 0
        return max(r.attempt_index for r in self.records) + 1

    def signature_count(self, signature: str) -> int:
        return sum(1 for r in self.records if r.failure_signature == signature)

    def provider_switch_count(self) -> int:
        switches = 0
        previous: str | None = None
        for r in self.records:
            if r.provider_id != previous and previous is not None:
                switches += 1
            previous = r.provider_id
        return switches

    def model_switch_count(self) -> int:
        switches = 0
        previous: str | None = None
        for r in self.records:
            if r.model_id != previous and previous is not None:
                switches += 1
            previous = r.model_id
        return switches

    def context_rebuild_count(self) -> int:
        return sum(1 for r in self.records if r.retry_type is RetryType.REBUILD_CONTEXT)

    def constrained_output_retry_count(self) -> int:
        return sum(1 for r in self.records if r.retry_type is RetryType.CONSTRAINED_OUTPUT_RETRY)

    def repair_count(self) -> int:
        return sum(
            1
            for r in self.records
            if r.retry_type in {RetryType.REPAIR, RetryType.CROSS_MODEL_REPAIR}
        )

    def transient_retry_count(self) -> int:
        return sum(1 for r in self.records if r.retry_type is RetryType.TRANSIENT_RETRY)

    def planning_retry_count(self) -> int:
        return sum(1 for r in self.records if r.retry_type is RetryType.REPLAN)

    def last_record(self) -> FailureAttemptRecord | None:
        return self.records[-1] if self.records else None

    def record(
        self,
        *,
        failure_category: str,
        failure_subtype: str,
        failure_signature: str,
        provider_id: str | None,
        model_id: str | None,
        route_id: str | None,
        action_taken: str,
        retry_type: RetryType,
        timestamp: datetime.datetime,
        retry_after: datetime.datetime | None = None,
    ) -> FailureAttemptRecord:
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        record = FailureAttemptRecord(
            attempt_index=self.total_attempt_index,
            failure_category=failure_category,
            failure_subtype=failure_subtype,
            failure_signature=failure_signature,
            provider_id=provider_id,
            model_id=model_id,
            route_id=route_id,
            action_taken=action_taken,
            retry_type=retry_type,
            timestamp=timestamp,
            retry_after=retry_after,
            context_rebuild_number=self.context_rebuild_count(),
            repair_number=self.repair_count(),
        )
        self.records.append(record)
        return record

    def set_wait(self, wait: WaitState) -> None:
        self.current_wait = wait

    def clear_wait(self) -> None:
        self.current_wait = None

    def mark_exhausted_path(
        self,
        failure_signature: str,
        provider_id: str | None,
        model_id: str | None,
    ) -> None:
        path = (failure_signature, provider_id, model_id)
        if path not in self.exhausted_paths:
            self.exhausted_paths.append(path)

    def is_exhausted_path(
        self,
        failure_signature: str,
        provider_id: str | None,
        model_id: str | None,
    ) -> bool:
        return (failure_signature, provider_id, model_id) in self.exhausted_paths

    def to_dict(self) -> dict[str, Any]:
        return (
            redact(
                {
                    "task_id": self.task_id,
                    "records": [r.to_dict() for r in self.records],
                    "current_wait": self.current_wait.to_dict() if self.current_wait else None,
                    "current_context_rebuild": dict(self.current_context_rebuild),
                    "exhausted_paths": [
                        {"signature": s, "provider_id": p, "model_id": m}
                        for s, p, m in self.exhausted_paths
                    ],
                }
            )
            or {}
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetryLedger:
        ledger = cls(
            task_id=str(data["task_id"]),
            records=[FailureAttemptRecord.from_dict(r) for r in data.get("records", [])],
            current_wait=(
                WaitState.from_dict(data["current_wait"]) if data.get("current_wait") else None
            ),
            current_context_rebuild=dict(data.get("current_context_rebuild", {})),
        )
        for path in data.get("exhausted_paths", []):
            ledger.exhausted_paths.append(
                (
                    str(path["signature"]),
                    path.get("provider_id"),
                    path.get("model_id"),
                )
            )
        return ledger
