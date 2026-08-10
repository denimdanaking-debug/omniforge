"""Bounded failure evidence structures for deterministic repair and escalation.

Evidence packets carry enough structured information for a model or reviewer to
repair a failure without exposing full prompts, secrets, or unbounded logs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from src.security.redaction import redact


@dataclass(frozen=True)
class ImplementationFailureEvidence:
    """Canonical deterministic implementation failure evidence packet."""

    command: str
    exit_status: int | None
    failing_check_names: tuple[str, ...]
    error_excerpts: tuple[str, ...]
    affected_files: tuple[str, ...]
    validation_artifact_refs: tuple[str, ...]
    prior_implementation_fingerprint: str | None
    attempt_number: int = 1
    repair_count: int = 0

    def __post_init__(self) -> None:
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be positive")
        if self.repair_count < 0:
            raise ValueError("repair_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "exit_status": self.exit_status,
            "failing_check_names": sorted(self.failing_check_names),
            "error_excerpts": list(self.error_excerpts),
            "affected_files": sorted(self.affected_files),
            "validation_artifact_refs": sorted(self.validation_artifact_refs),
            "prior_implementation_fingerprint": self.prior_implementation_fingerprint,
            "attempt_number": self.attempt_number,
            "repair_count": self.repair_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImplementationFailureEvidence:
        return cls(
            command=str(data.get("command", "")),
            exit_status=data.get("exit_status"),
            failing_check_names=tuple(data.get("failing_check_names", [])),
            error_excerpts=tuple(data.get("error_excerpts", [])),
            affected_files=tuple(data.get("affected_files", [])),
            validation_artifact_refs=tuple(data.get("validation_artifact_refs", [])),
            prior_implementation_fingerprint=data.get("prior_implementation_fingerprint"),
            attempt_number=int(data.get("attempt_number", 1)),
            repair_count=int(data.get("repair_count", 0)),
        )


def implementation_failure_signature(evidence: ImplementationFailureEvidence) -> str:
    """Deterministic signature of the material failure.

    Same failing test names, validation category, or affected invariant map to
    the same signature. Different textual noise but same underlying failure maps
    to the same signature.
    """
    data = {
        "command": evidence.command,
        "failing_check_names": sorted(evidence.failing_check_names),
        "affected_files": sorted(evidence.affected_files),
        "validation_category": _validation_category(evidence.command),
    }
    canonical = json.dumps(redact(data), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validation_category(command: str) -> str:
    lowered = command.lower()
    if "compile" in lowered or "build" in lowered:
        return "build"
    if "mypy" in lowered or "type" in lowered:
        return "type_check"
    if "lint" in lowered:
        return "lint"
    if "test" in lowered:
        return "test"
    if "invariant" in lowered or "architecture" in lowered:
        return "invariant"
    return "validation"


@dataclass(frozen=True)
class PlanningRejectionEvidence:
    """Preserved planning rejection evidence for downstream planners/reviewers."""

    rejected_plan_fingerprint: str
    validation_findings: tuple[str, ...]
    rejection_reason: str
    planner_provider_id: str | None
    planner_model_id: str | None
    planner_route_id: str | None
    authority_snapshot_refs: tuple[str, ...]
    attempt_number: int = 1

    def __post_init__(self) -> None:
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rejected_plan_fingerprint": self.rejected_plan_fingerprint,
            "validation_findings": list(self.validation_findings),
            "rejection_reason": self.rejection_reason,
            "planner_provider_id": self.planner_provider_id,
            "planner_model_id": self.planner_model_id,
            "planner_route_id": self.planner_route_id,
            "authority_snapshot_refs": sorted(self.authority_snapshot_refs),
            "attempt_number": self.attempt_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanningRejectionEvidence:
        return cls(
            rejected_plan_fingerprint=str(data.get("rejected_plan_fingerprint", "")),
            validation_findings=tuple(data.get("validation_findings", [])),
            rejection_reason=str(data.get("rejection_reason", "")),
            planner_provider_id=data.get("planner_provider_id"),
            planner_model_id=data.get("planner_model_id"),
            planner_route_id=data.get("planner_route_id"),
            authority_snapshot_refs=tuple(data.get("authority_snapshot_refs", [])),
            attempt_number=int(data.get("attempt_number", 1)),
        )


def planning_failure_signature(evidence: PlanningRejectionEvidence) -> str:
    """Deterministic signature of a planning rejection.

    Based on validation findings and authority violations, not volatile IDs or
    timestamps.
    """
    data = {
        "validation_findings": sorted(evidence.validation_findings),
        "authority_violations": sorted(
            ref for ref in evidence.authority_snapshot_refs if "violation" in ref.lower()
        ),
        "rejection_reason": evidence.rejection_reason,
    }
    canonical = json.dumps(redact(data), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
