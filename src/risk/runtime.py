"""Runtime risk escalation engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.policy.risk import RiskLevel

from .assessment import OperationType
from .authority import AuthoritySensitivePolicy
from .path_utils import normalize_repo_path
from .security import SecuritySensitivePolicy


class RiskRuntimeEventType(StrEnum):
    """Canonical runtime escalation event types."""

    TEST_FAILURE = "test_failure"
    MODEL_DISAGREEMENT = "model_disagreement"
    UNEXPECTED_FILE_TOUCH = "unexpected_file_touch"
    MERGE_CONFLICT = "merge_conflict"
    REPAIR_LOOP = "repair_loop"
    AUTHORITY_VIOLATION = "authority_violation"
    INTEGRATION_ANOMALY = "integration_anomaly"


class RiskEscalationError(ValueError):
    """Raised when a runtime escalation request is invalid."""


@dataclass(frozen=True)
class RiskRuntimeEvent:
    """One runtime event that may escalate task risk.

    ``affected_paths`` carries structured repository-relative paths for events
    such as ``UNEXPECTED_FILE_TOUCH``. Decisions must use these structured
    paths, not free-text ``evidence``.
    """

    event_type: RiskRuntimeEventType
    material: bool
    evidence: str
    count: int = 1
    threshold: int = 1
    affected_paths: tuple[str, ...] = ()
    operation: OperationType | str | None = None

    def __post_init__(self) -> None:
        if not self.evidence.strip():
            raise ValueError("risk event evidence must be non-empty")
        if self.count < 0:
            raise ValueError("risk event count must be non-negative")
        if self.threshold < 1:
            raise ValueError("risk event threshold must be positive")
        object.__setattr__(
            self,
            "affected_paths",
            tuple(normalize_repo_path(p) for p in self.affected_paths),
        )


@dataclass(frozen=True)
class RiskEscalationRecord:
    """Structured record of one risk escalation."""

    previous_risk: RiskLevel
    new_risk: RiskLevel
    event_type: RiskRuntimeEventType
    reason: str
    evidence: str
    threshold: int
    sequence: int


class RuntimeRiskEscalator:
    """Apply monotonic, deterministic runtime risk escalation."""

    def __init__(
        self,
        test_failure_threshold: int = 3,
        repair_loop_threshold: int = 3,
        authority_policy: AuthoritySensitivePolicy | None = None,
        security_policy: SecuritySensitivePolicy | None = None,
        unexpected_file_touch_floor: RiskLevel = RiskLevel.R3_HIGH,
    ) -> None:
        self._test_failure_threshold = test_failure_threshold
        self._repair_loop_threshold = repair_loop_threshold
        self._authority_policy = authority_policy
        self._security_policy = security_policy
        self._unexpected_file_touch_floor = unexpected_file_touch_floor

    @classmethod
    def default(
        cls,
        authority_policy: AuthoritySensitivePolicy | None = None,
        security_policy: SecuritySensitivePolicy | None = None,
    ) -> RuntimeRiskEscalator:
        return cls(
            authority_policy=authority_policy,
            security_policy=security_policy,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeRiskEscalator:
        return cls(
            test_failure_threshold=int(data.get("test_failure_threshold", 3)),
            repair_loop_threshold=int(data.get("repair_loop_threshold", 3)),
        )

    def escalate(
        self,
        current: RiskLevel,
        event: RiskRuntimeEvent,
        sequence: int = 1,
    ) -> tuple[RiskLevel, RiskEscalationRecord | None]:
        """Return monotonic new risk and an escalation record.

        Runtime escalation never lowers risk. Non-material events or events
        below threshold are recorded with no level change.

        For ``TEST_FAILURE`` and ``REPAIR_LOOP`` the effective threshold is the
        engine's configured threshold, not any caller-supplied event threshold.
        """
        effective_threshold = self._effective_threshold(event)

        if not event.material:
            return current, RiskEscalationRecord(
                previous_risk=current,
                new_risk=current,
                event_type=event.event_type,
                reason=f"{event.event_type.value} not material",
                evidence=event.evidence,
                threshold=effective_threshold,
                sequence=sequence,
            )

        if event.count < effective_threshold:
            reason = (
                f"{event.event_type.value} below threshold ({event.count} < {effective_threshold})"
            )
            return current, RiskEscalationRecord(
                previous_risk=current,
                new_risk=current,
                event_type=event.event_type,
                reason=reason,
                evidence=event.evidence,
                threshold=effective_threshold,
                sequence=sequence,
            )

        new_risk = self._target_risk(event)
        if new_risk <= current:
            new_risk = current

        reason = (
            f"{event.event_type.value} at or above threshold "
            f"({event.count} >= {effective_threshold})"
        )
        record = RiskEscalationRecord(
            previous_risk=current,
            new_risk=new_risk,
            event_type=event.event_type,
            reason=reason,
            evidence=event.evidence,
            threshold=effective_threshold,
            sequence=sequence,
        )
        return new_risk, record

    def _effective_threshold(self, event: RiskRuntimeEvent) -> int:
        """Return the authoritative threshold for an event type.

        The engine's configured thresholds for ``TEST_FAILURE`` and
        ``REPAIR_LOOP`` override any caller-supplied event threshold, preventing
        bypass of safety policy.
        """
        if event.event_type == RiskRuntimeEventType.TEST_FAILURE:
            return self._test_failure_threshold
        if event.event_type == RiskRuntimeEventType.REPAIR_LOOP:
            return self._repair_loop_threshold
        return event.threshold

    def _target_risk(self, event: RiskRuntimeEvent) -> RiskLevel:
        if event.event_type == RiskRuntimeEventType.AUTHORITY_VIOLATION:
            return RiskLevel.R4_CRITICAL_AUTHORITY
        if event.event_type == RiskRuntimeEventType.UNEXPECTED_FILE_TOUCH:
            return self._unexpected_file_touch_risk(event)
        if event.event_type == RiskRuntimeEventType.MERGE_CONFLICT:
            return RiskLevel.R3_HIGH
        if event.event_type == RiskRuntimeEventType.INTEGRATION_ANOMALY:
            return RiskLevel.R3_HIGH
        if event.event_type == RiskRuntimeEventType.MODEL_DISAGREEMENT:
            return RiskLevel.R3_HIGH
        if event.event_type == RiskRuntimeEventType.TEST_FAILURE:
            return (
                RiskLevel.R3_HIGH
                if event.count >= self._test_failure_threshold
                else RiskLevel.R2_NORMAL
            )
        if event.event_type == RiskRuntimeEventType.REPAIR_LOOP:
            return (
                RiskLevel.R3_HIGH
                if event.count >= self._repair_loop_threshold
                else RiskLevel.R2_NORMAL
            )
        return RiskLevel.R3_HIGH

    def _unexpected_file_touch_risk(self, event: RiskRuntimeEvent) -> RiskLevel:
        """Derive risk from structured affected paths, not evidence text."""
        if not event.affected_paths:
            return self._unexpected_file_touch_floor

        operation = event.operation or OperationType.MODIFY
        if self._authority_policy is not None:
            authority_factor = self._authority_policy.assess(event.affected_paths, operation)
            if authority_factor is not None:
                return authority_factor.risk_level

        if self._security_policy is not None:
            security_factor = self._security_policy.assess(event.affected_paths)
            if security_factor is not None:
                return max(security_factor.risk_level, self._unexpected_file_touch_floor)

        return self._unexpected_file_touch_floor

    def apply_all(
        self,
        current: RiskLevel,
        events: tuple[RiskRuntimeEvent, ...],
        start_sequence: int = 1,
    ) -> tuple[RiskLevel, tuple[RiskEscalationRecord, ...]]:
        """Apply events in order and return final monotonic risk.

        Because escalation is monotonic and uses max(), the final risk is
        independent of event ordering.
        """
        records: list[RiskEscalationRecord] = []
        level = current
        for i, event in enumerate(events):
            level, record = self.escalate(level, event, sequence=start_sequence + i)
            if record is not None:
                records.append(record)
        return level, tuple(records)
