"""Runtime risk escalation engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.policy.risk import RiskLevel


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
    """One runtime event that may escalate task risk."""

    event_type: RiskRuntimeEventType
    material: bool
    evidence: str
    count: int = 1
    threshold: int = 1

    def __post_init__(self) -> None:
        if not self.evidence.strip():
            raise ValueError("risk event evidence must be non-empty")
        if self.count < 0:
            raise ValueError("risk event count must be non-negative")
        if self.threshold < 1:
            raise ValueError("risk event threshold must be positive")


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
    ) -> None:
        self._test_failure_threshold = test_failure_threshold
        self._repair_loop_threshold = repair_loop_threshold

    @classmethod
    def default(cls) -> RuntimeRiskEscalator:
        return cls()

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
        """
        if not event.material:
            return current, RiskEscalationRecord(
                previous_risk=current,
                new_risk=current,
                event_type=event.event_type,
                reason=f"{event.event_type.value} not material",
                evidence=event.evidence,
                threshold=event.threshold,
                sequence=sequence,
            )

        if event.count < event.threshold:
            reason = f"{event.event_type.value} below threshold ({event.count} < {event.threshold})"
            return current, RiskEscalationRecord(
                previous_risk=current,
                new_risk=current,
                event_type=event.event_type,
                reason=reason,
                evidence=event.evidence,
                threshold=event.threshold,
                sequence=sequence,
            )

        new_risk = self._target_risk(event)
        if new_risk <= current:
            new_risk = current

        reason = (
            f"{event.event_type.value} at or above threshold ({event.count} >= {event.threshold})"
        )
        record = RiskEscalationRecord(
            previous_risk=current,
            new_risk=new_risk,
            event_type=event.event_type,
            reason=reason,
            evidence=event.evidence,
            threshold=event.threshold,
            sequence=sequence,
        )
        return new_risk, record

    def _target_risk(self, event: RiskRuntimeEvent) -> RiskLevel:
        if event.event_type == RiskRuntimeEventType.AUTHORITY_VIOLATION:
            return RiskLevel.R4_CRITICAL_AUTHORITY
        if event.event_type == RiskRuntimeEventType.UNEXPECTED_FILE_TOUCH:
            # Authority/security unexpected files handled by authority/security
            # classification already; runtime touch itself is high but not
            # automatically critical unless the path is authority-sensitive.
            return RiskLevel.R3_HIGH
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
