"""Immutable performance-event model for Phase 11 empirical intelligence."""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

from src.recovery.clock import ensure_aware, isoformat
from src.security.redaction import redact


class PerformanceEventType(StrEnum):
    """Canonical performance event types."""

    TASK_OUTCOME = "task_outcome"
    RECOVERY_DECISION = "recovery_decision"
    REVIEWER_FINDING = "reviewer_finding"
    REVIEWER_FALSE_NEGATIVE = "reviewer_false_negative"
    CONTEXT_OUTCOME = "context_outcome"
    INTEGRATION_ACCEPTED = "integration_accepted"
    INTEGRATION_REJECTED = "integration_rejected"
    CORRECTION = "correction"


class OutcomeCategory(StrEnum):
    """High-level outcome category for an event."""

    SUCCESS = "success"
    FIRST_PASS_SUCCESS = "first_pass_success"
    PLAN_INVALID = "plan_invalid"
    STRUCTURED_OUTPUT_INVALID = "structured_output_invalid"
    DETERMINISTIC_VALIDATION_FAILURE = "deterministic_validation_failure"
    CONCEPTUAL_FAILURE = "conceptual_failure"
    AUTHORITY_VIOLATION = "authority_violation"
    PROVIDER_FAILURE = "provider_failure"
    QUOTA_EXHAUSTED = "quota_exhausted"
    AUTH_FAILURE = "auth_failure"
    ROUTE_FAILURE = "route_failure"
    INFRASTRUCTURE_TRANSIENT = "infrastructure_transient"
    CAPABILITY_MISMATCH = "capability_mismatch"
    CONTEXT_CAPACITY = "context_capacity"
    INTEGRATION_FAILURE = "integration_failure"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class AcceptanceStatus(StrEnum):
    """Task acceptance state."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ABANDONED = "abandoned"
    PENDING = "pending"


class FindingDisposition(StrEnum):
    """Lifecycle disposition for a reviewer finding."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    STALE = "stale"
    DUPLICATE = "duplicate"
    MIS_SEVERITY = "mis_severity"
    PENDING = "pending"


class AuthorityAdherenceStatus(StrEnum):
    """Authority-adherence classification."""

    COMPLIANT = "compliant"
    ATTEMPTED_MUTATION = "attempted_mutation"
    IGNORED_IMMUTABLE = "ignored_immutable"
    INVALID_STATE_ADVANCEMENT = "invalid_state_advancement"
    SUMMARY_SUBSTITUTED_FOR_RAW = "summary_substituted_for_raw"
    INTEGRATION_STATE_MISMATCH = "integration_state_mismatch"


class CostState(StrEnum):
    """Cost reporting state."""

    ACTUAL = "actual"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Usage:
    """Normalized token usage for a model call."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("cached_tokens", self.cached_tokens),
            ("reasoning_tokens", self.reasoning_tokens),
            ("total_tokens", self.total_tokens),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Usage:
        return cls(
            input_tokens=_optional_int(data.get("input_tokens")),
            output_tokens=_optional_int(data.get("output_tokens")),
            cached_tokens=_optional_int(data.get("cached_tokens")),
            reasoning_tokens=_optional_int(data.get("reasoning_tokens")),
            total_tokens=_optional_int(data.get("total_tokens")),
        )


@dataclass(frozen=True)
class Cost:
    """Cost amount with state."""

    amount: float | None
    currency: str
    state: CostState

    def __post_init__(self) -> None:
        if self.amount is not None and self.amount < 0:
            raise ValueError("cost amount must be non-negative")
        if not self.currency.strip():
            raise ValueError("currency must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "amount": self.amount,
            "currency": self.currency,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cost:
        return cls(
            amount=_optional_float(data.get("amount")),
            currency=str(data.get("currency", "USD")),
            state=CostState(data.get("state", CostState.UNKNOWN.value)),
        )


@dataclass(frozen=True)
class RepairMetadata:
    """Metadata about a repair attempt and its outcome."""

    repair_model_id: str | None = None
    original_model_id: str | None = None
    repair_number: int = 0
    resolved: bool | None = None
    introduced_new_failure: bool = False
    required_cross_model_escalation: bool = False

    def __post_init__(self) -> None:
        if self.repair_number < 0:
            raise ValueError("repair_number must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_model_id": self.repair_model_id,
            "original_model_id": self.original_model_id,
            "repair_number": self.repair_number,
            "resolved": self.resolved,
            "introduced_new_failure": self.introduced_new_failure,
            "required_cross_model_escalation": self.required_cross_model_escalation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepairMetadata:
        return cls(
            repair_model_id=data.get("repair_model_id"),
            original_model_id=data.get("original_model_id"),
            repair_number=int(data.get("repair_number", 0)),
            resolved=data.get("resolved"),
            introduced_new_failure=bool(data.get("introduced_new_failure", False)),
            required_cross_model_escalation=bool(
                data.get("required_cross_model_escalation", False)
            ),
        )


@dataclass(frozen=True)
class PerformanceEvent:
    """One immutable empirical observation."""

    event_id: str
    schema_version: str
    timestamp: datetime.datetime
    event_type: PerformanceEventType
    project_id: str
    task_id: str
    run_id: str
    execution_role: str
    task_class: str
    risk: str
    provider_id: str | None
    model_id: str | None
    route_id: str | None
    failure_domain: str | None = None
    context_strategy: str | None = None
    language_framework: str | None = None
    outcome_category: OutcomeCategory = OutcomeCategory.UNKNOWN
    acceptance_status: AcceptanceStatus = AcceptanceStatus.PENDING
    first_pass: bool | None = None
    plan_valid: bool | None = None
    validation_result_summary: dict[str, Any] = field(default_factory=dict)
    repair_metadata: RepairMetadata | None = None
    review_finding_dispositions: dict[str, FindingDisposition] = field(default_factory=dict)
    authority_adherence: AuthorityAdherenceStatus | None = None
    latency_seconds: float | None = None
    provider_wait_seconds: float | None = None
    usage: Usage = field(default_factory=Usage)
    direct_cost: Cost = field(default_factory=lambda: Cost(None, "USD", CostState.UNKNOWN))
    total_cost_to_accepted: Cost | None = None
    time_to_accepted_seconds: float | None = None
    task_difficulty: str | None = None
    evidence_refs: tuple[str, ...] = ()
    originating_ids: dict[str, str] = field(default_factory=dict)
    attribution: str = "unknown"
    event_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must be non-empty")
        if not self.schema_version.strip():
            raise ValueError("schema_version must be non-empty")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if not self.project_id.strip():
            raise ValueError("project_id must be non-empty")
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if not self.execution_role.strip():
            raise ValueError("execution_role must be non-empty")
        if self.latency_seconds is not None and self.latency_seconds < 0:
            raise ValueError("latency_seconds must be non-negative")
        if self.provider_wait_seconds is not None and self.provider_wait_seconds < 0:
            raise ValueError("provider_wait_seconds must be non-negative")
        if self.time_to_accepted_seconds is not None and self.time_to_accepted_seconds < 0:
            raise ValueError("time_to_accepted_seconds must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "schema_version": self.schema_version,
            "timestamp": isoformat(self.timestamp),
            "event_type": self.event_type.value,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "execution_role": self.execution_role,
            "task_class": self.task_class,
            "risk": self.risk,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "route_id": self.route_id,
            "failure_domain": self.failure_domain,
            "context_strategy": self.context_strategy,
            "language_framework": self.language_framework,
            "outcome_category": self.outcome_category.value,
            "acceptance_status": self.acceptance_status.value,
            "first_pass": self.first_pass,
            "plan_valid": self.plan_valid,
            "validation_result_summary": self.validation_result_summary,
            "repair_metadata": self.repair_metadata.to_dict() if self.repair_metadata else None,
            "review_finding_dispositions": {
                k: v.value for k, v in sorted(self.review_finding_dispositions.items())
            },
            "authority_adherence": self.authority_adherence.value
            if self.authority_adherence
            else None,
            "latency_seconds": self.latency_seconds,
            "provider_wait_seconds": self.provider_wait_seconds,
            "usage": self.usage.to_dict(),
            "direct_cost": self.direct_cost.to_dict(),
            "total_cost_to_accepted": self.total_cost_to_accepted.to_dict()
            if self.total_cost_to_accepted
            else None,
            "time_to_accepted_seconds": self.time_to_accepted_seconds,
            "task_difficulty": self.task_difficulty,
            "evidence_refs": list(self.evidence_refs),
            "originating_ids": dict(sorted(self.originating_ids.items())),
            "attribution": self.attribution,
            "event_fingerprint": self.event_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerformanceEvent:
        timestamp = datetime.datetime.fromisoformat(data["timestamp"])
        ensure_aware(timestamp)
        repair_raw = data.get("repair_metadata")
        repair_metadata = RepairMetadata.from_dict(repair_raw) if repair_raw else None
        total_raw = data.get("total_cost_to_accepted")
        total_cost = Cost.from_dict(total_raw) if total_raw else None
        usage_raw = data.get("usage")
        usage = Usage.from_dict(usage_raw) if usage_raw else Usage()
        direct_raw = data.get("direct_cost")
        direct_cost = (
            Cost.from_dict(direct_raw) if direct_raw else Cost(None, "USD", CostState.UNKNOWN)
        )
        auth_raw = data.get("authority_adherence")
        authority_adherence = AuthorityAdherenceStatus(auth_raw) if auth_raw else None
        review_dispositions = {
            k: FindingDisposition(v)
            for k, v in (data.get("review_finding_dispositions") or {}).items()
        }
        return cls(
            event_id=str(data["event_id"]),
            schema_version=str(data["schema_version"]),
            timestamp=timestamp,
            event_type=PerformanceEventType(data["event_type"]),
            project_id=str(data["project_id"]),
            task_id=str(data["task_id"]),
            run_id=str(data["run_id"]),
            execution_role=str(data["execution_role"]),
            task_class=str(data["task_class"]),
            risk=str(data["risk"]),
            provider_id=data.get("provider_id"),
            model_id=data.get("model_id"),
            route_id=data.get("route_id"),
            failure_domain=data.get("failure_domain"),
            context_strategy=data.get("context_strategy"),
            language_framework=data.get("language_framework"),
            outcome_category=OutcomeCategory(
                data.get("outcome_category", OutcomeCategory.UNKNOWN.value)
            ),
            acceptance_status=AcceptanceStatus(
                data.get("acceptance_status", AcceptanceStatus.PENDING.value)
            ),
            first_pass=data.get("first_pass"),
            plan_valid=data.get("plan_valid"),
            validation_result_summary=dict(data.get("validation_result_summary", {})),
            repair_metadata=repair_metadata,
            review_finding_dispositions=review_dispositions,
            authority_adherence=authority_adherence,
            latency_seconds=_optional_float(data.get("latency_seconds")),
            provider_wait_seconds=_optional_float(data.get("provider_wait_seconds")),
            usage=usage,
            direct_cost=direct_cost,
            total_cost_to_accepted=total_cost,
            time_to_accepted_seconds=_optional_float(data.get("time_to_accepted_seconds")),
            task_difficulty=data.get("task_difficulty"),
            evidence_refs=tuple(data.get("evidence_refs", [])),
            originating_ids=dict(data.get("originating_ids", {})),
            attribution=str(data.get("attribution", "unknown")),
            event_fingerprint=str(data.get("event_fingerprint", "")),
        )

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a redacted copy safe for persistence."""
        return cast(dict[str, Any], redact(self.to_dict()))


def event_identity(
    *,
    event_type: PerformanceEventType,
    task_id: str,
    run_id: str,
    timestamp: datetime.datetime,
    model_id: str | None,
    provider_id: str | None,
    route_id: str | None,
    outcome_category: OutcomeCategory,
    sequence: int = 0,
) -> str:
    """Deterministic identity for a performance event.

    Same logical event must produce the same event_id across restarts so replay
    is idempotent. Include a sequence number to disambiguate multiple events of
    the same type for the same task at the same timestamp.
    """
    payload = {
        "event_type": event_type.value,
        "task_id": task_id,
        "run_id": run_id,
        "timestamp": isoformat(timestamp),
        "model_id": model_id,
        "provider_id": provider_id,
        "route_id": route_id,
        "outcome_category": outcome_category.value,
        "sequence": sequence,
    }
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
