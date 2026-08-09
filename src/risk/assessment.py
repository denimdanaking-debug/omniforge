"""Canonical risk assessment request and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.policy.risk import RiskLevel
from src.routing.roles import ExecutionRole


class RiskFactorCode(StrEnum):
    """Deterministic risk factor codes."""

    TASK_METADATA = "task_metadata"
    FILE_SCOPE = "file_scope"
    LINE_SCOPE = "line_scope"
    AUTHORITY_SENSITIVE = "authority_sensitive"
    SECURITY_SENSITIVE = "security_sensitive"
    ARCHITECTURAL_CHANGE = "architectural_change"
    INTEGRATION_SENSITIVE = "integration_sensitive"
    DEPENDENCY_CHANGE = "dependency_change"
    REPEATED_FAILURES = "repeated_failures"
    MODEL_DISAGREEMENT = "model_disagreement"
    UNEXPECTED_FILES = "unexpected_files"
    MERGE_CONFLICT = "merge_conflict"
    REPAIR_LOOP = "repair_loop"
    AUTHORITY_VIOLATION = "authority_violation"
    INTEGRATION_ANOMALY = "integration_anomaly"
    PROJECT_OVERRIDE = "project_override"


class OperationType(StrEnum):
    """Canonical operation types for repository changes."""

    READ = "read"
    REFERENCE = "reference"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


def coerce_risk_level(value: Any) -> RiskLevel:
    """Parse a RiskLevel from its name, integer value, or an existing member."""
    if isinstance(value, RiskLevel):
        return value
    if isinstance(value, int):
        try:
            return RiskLevel(value)
        except ValueError as exc:
            raise ValueError(f"invalid risk level: {value!r}") from exc
    if isinstance(value, str):
        try:
            return RiskLevel[value]
        except KeyError as exc:
            raise ValueError(f"invalid risk level: {value!r}") from exc
    raise ValueError(f"invalid risk level: {value!r}")


@dataclass(frozen=True)
class RiskFactor:
    """One deterministic risk factor with evidence and provenance."""

    code: RiskFactorCode
    evidence: str
    risk_level: RiskLevel
    provenance: str


@dataclass(frozen=True)
class RiskAssessmentRequest:
    """Normalized inputs for deterministic risk classification."""

    project_id: str
    task_id: str
    role: ExecutionRole
    task_class: str
    operation: OperationType | str = OperationType.MODIFY
    changed_files: tuple[str, ...] = ()
    changed_lines_estimate: int = 0
    dependency_changes: tuple[str, ...] = ()
    generated_files: tuple[str, ...] = ()
    explicit_paths: tuple[str, ...] = ()
    baseline_risk: RiskLevel | None = None
    runtime_events: tuple[Any, ...] = ()
    project_policy: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.project_id.strip():
            raise ValueError("project_id must be non-empty")
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not self.task_class.strip():
            raise ValueError("task_class must be non-empty")
        if self.changed_lines_estimate < 0:
            raise ValueError("changed_lines_estimate must be non-negative")


@dataclass(frozen=True)
class RiskAssessmentResult:
    """Immutable result of a risk assessment."""

    baseline_risk: RiskLevel
    final_risk: RiskLevel
    factors: tuple[RiskFactor, ...]
    policy_effects: dict[str, Any]
    fingerprint: str
    explanation: str

    def escalate(
        self, target: RiskLevel, reason: str, code: RiskFactorCode
    ) -> RiskAssessmentResult:
        """Return a new result with monotonic risk escalation."""
        from src.policy.risk import RiskAssessment

        base = RiskAssessment(self.final_risk, tuple(f.evidence for f in self.factors))
        escalated = base.escalate(target, reason)
        new_factor = RiskFactor(
            code=code,
            evidence=reason,
            risk_level=target,
            provenance="runtime_escalation",
        )
        return RiskAssessmentResult(
            baseline_risk=self.baseline_risk,
            final_risk=escalated.level,
            factors=self.factors + (new_factor,),
            policy_effects=self.policy_effects,
            fingerprint=self.fingerprint,
            explanation=self.explanation,
        )
