"""Deterministic input fingerprint for risk assessments."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.routing.roles import ExecutionRole

from .assessment import RiskAssessmentRequest
from .runtime import RiskRuntimeEvent


def _normalize_path(path: str) -> str:
    """Return a deterministic, repository-root-relative normalized path."""
    from pathlib import PurePosixPath

    p = PurePosixPath(path)
    # Remove leading ./ and ../ attempts that stay within repo root.
    parts: list[str] = []
    for part in p.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != "." and part != "/":
            parts.append(part)
    return "/".join(parts)


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


def _normalize_runtime_event(event: Any) -> dict[str, Any]:
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
            "threshold": event.threshold,
            "affected_paths": sorted(_normalize_path(p) for p in event.affected_paths),
            "operation": str(event.operation) if event.operation is not None else None,
        }
    if isinstance(event, dict):
        return {
            "event_type": str(event.get("event_type", "")),
            "material": bool(event.get("material", True)),
            "count": int(event.get("count", 1)),
            "threshold": int(event.get("threshold", 1)),
            "affected_paths": sorted(_normalize_path(p) for p in event.get("affected_paths", ())),
            "operation": str(event.get("operation"))
            if event.get("operation") is not None
            else None,
        }
    return {"repr": str(event)}


def risk_assessment_fingerprint(request: RiskAssessmentRequest) -> str:
    """Return a deterministic SHA-256 fingerprint of the assessment input.

    The fingerprint describes the full logical state that produced the risk
    decision, including normalized runtime events because they affect the final
    risk level. Free-text evidence is excluded.
    """
    payload: dict[str, Any] = {
        "project_id": request.project_id,
        "task_id": request.task_id,
        "role": request.role.value,
        "task_class": request.task_class,
        "operation": str(request.operation),
        "changed_files": sorted(_normalize_path(p) for p in request.changed_files),
        "changed_lines_estimate": request.changed_lines_estimate,
        "dependency_changes": sorted(request.dependency_changes),
        "generated_files": sorted(_normalize_path(p) for p in request.generated_files),
        "explicit_paths": sorted(_normalize_path(p) for p in request.explicit_paths),
        "baseline_risk": request.baseline_risk.value if request.baseline_risk else None,
        "runtime_events": sorted(
            (_normalize_runtime_event(e) for e in request.runtime_events),
            key=lambda d: json.dumps(d, sort_keys=True),
        ),
        "project_policy": _canonical_value(request.project_policy),
    }
    canonical = _canonical_value(payload)
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
