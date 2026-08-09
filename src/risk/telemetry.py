"""Risk-engine telemetry events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class RiskTelemetryEvent:
    """Canonical telemetry event names emitted by the risk engine."""

    RISK_CLASSIFIED = "RISK_CLASSIFIED"
    RISK_ESCALATED = "RISK_ESCALATED"
    AUTHORITY_SENSITIVE_DETECTED = "AUTHORITY_SENSITIVE_DETECTED"
    SECURITY_SENSITIVE_DETECTED = "SECURITY_SENSITIVE_DETECTED"
    ARCHITECTURE_SENSITIVE_DETECTED = "ARCHITECTURE_SENSITIVE_DETECTED"
    REVIEW_REQUIREMENT_CHANGED = "REVIEW_REQUIREMENT_CHANGED"
    EXPERIMENTATION_DISALLOWED = "EXPERIMENTATION_DISALLOWED"
    CONTEXT_DEPTH_RAISED = "CONTEXT_DEPTH_RAISED"
    PROJECT_RISK_OVERRIDE_APPLIED = "PROJECT_RISK_OVERRIDE_APPLIED"


@dataclass(frozen=True)
class RiskTelemetryRecord:
    """One structured risk telemetry record."""

    event_name: str
    task_id: str
    project_id: str
    payload: dict[str, Any]
