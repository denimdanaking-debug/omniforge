"""Deterministic input fingerprint for risk assessments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from src.routing.roles import ExecutionRole

from .architecture import ArchitectureThresholds
from .assessment import RiskAssessmentRequest
from .authority import AuthoritySensitivePolicy
from .path_utils import normalize_repo_path
from .project_policy import ProjectRiskPolicy
from .runtime import RiskRuntimeEvent, RuntimeRiskEscalator
from .security import SecuritySensitivePolicy


def _canonical_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable canonical form."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value
    if isinstance(value, tuple | list):
        return [_canonical_value(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted(str(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _canonical_value(v) for k, v in sorted(value.items())}
    if isinstance(value, ExecutionRole):
        return value.value
    return str(value)


@dataclass(frozen=True)
class RiskDecisionInputs:
    """Canonical effective decision inputs for a risk assessment fingerprint."""

    request: RiskAssessmentRequest
    project_policy: ProjectRiskPolicy
    authority_policy: AuthoritySensitivePolicy
    security_policy: SecuritySensitivePolicy
    architecture_thresholds: ArchitectureThresholds
    runtime_escalator: RuntimeRiskEscalator


def _effective_event_threshold(event: RiskRuntimeEvent, escalator: RuntimeRiskEscalator) -> int:
    """Return the threshold that actually drives the escalation decision."""
    if event.event_type.value == "test_failure":
        return escalator._test_failure_threshold
    if event.event_type.value == "repair_loop":
        return escalator._repair_loop_threshold
    return event.threshold


def _normalize_runtime_event(event: Any, escalator: RuntimeRiskEscalator) -> dict[str, Any]:
    """Return a deterministic, evidence-free fingerprint payload for an event.

    The fingerprint covers every decision-driving field: event type, materiality,
    count, effective threshold, structured affected paths, and operation. Free-text
    evidence is excluded because it is for display/audit only.
    """
    if isinstance(event, RiskRuntimeEvent):
        return {
            "event_type": event.event_type.value,
            "material": event.material,
            "count": event.count,
            "threshold": _effective_event_threshold(event, escalator),
            "affected_paths": sorted(normalize_repo_path(p) for p in event.affected_paths),
            "operation": str(event.operation) if event.operation is not None else None,
        }
    if isinstance(event, dict):
        return {
            "event_type": str(event.get("event_type", "")),
            "material": bool(event.get("material", True)),
            "count": int(event.get("count", 1)),
            "threshold": int(event.get("threshold", 1)),
            "affected_paths": sorted(
                normalize_repo_path(p) for p in event.get("affected_paths", ())
            ),
            "operation": str(event.get("operation"))
            if event.get("operation") is not None
            else None,
        }
    return {"repr": str(event)}


def risk_assessment_fingerprint(inputs: RiskDecisionInputs) -> str:
    """Return a deterministic SHA-256 fingerprint of the effective decision inputs.

    The fingerprint describes the exact normalized logical state that produced the
    risk decision, including effective project/authority/security/architecture/runtime
    policies. Free-text evidence is excluded.
    """
    request = inputs.request
    payload: dict[str, Any] = {
        "project_id": request.project_id,
        "task_id": request.task_id,
        "role": request.role.value,
        "task_class": request.task_class,
        "operation": str(request.operation),
        "changed_files": sorted(normalize_repo_path(p) for p in request.changed_files),
        "changed_lines_estimate": request.changed_lines_estimate,
        "dependency_changes": sorted(request.dependency_changes),
        "generated_files": sorted(normalize_repo_path(p) for p in request.generated_files),
        "explicit_paths": sorted(normalize_repo_path(p) for p in request.explicit_paths),
        "baseline_risk": request.baseline_risk.value if request.baseline_risk else None,
        "runtime_events": sorted(
            (_normalize_runtime_event(e, inputs.runtime_escalator) for e in request.runtime_events),
            key=lambda d: json.dumps(d, sort_keys=True),
        ),
        "project_policy": _canonical_value(inputs.project_policy.to_dict()),
        "authority_policy": _canonical_value(inputs.authority_policy.to_dict()),
        "security_policy": _canonical_value(inputs.security_policy.to_dict()),
        "architecture_thresholds": _canonical_value(inputs.architecture_thresholds.to_dict()),
        "runtime_thresholds": {
            "test_failure_threshold": inputs.runtime_escalator._test_failure_threshold,
            "repair_loop_threshold": inputs.runtime_escalator._repair_loop_threshold,
        },
    }
    canonical = _canonical_value(payload)
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
