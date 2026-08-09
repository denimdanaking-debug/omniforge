"""Authority-violation recovery integration with Phase 9 risk escalation.

Authority violations are severe task-integrity failures. This module provides
the canonical seam for escalating risk and blocking unsafe integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.policy.risk import RiskLevel
from src.recovery.failure_classification import AuthorityViolationData
from src.risk.runtime import RiskRuntimeEvent, RiskRuntimeEventType, RuntimeRiskEscalator


@dataclass(frozen=True)
class AuthorityRecoveryResult:
    """Result of applying authority-violation recovery logic."""

    escalated_risk: RiskLevel
    block_integration: bool
    evidence: dict[str, Any]
    task_local_penalties: dict[str, Any]


def apply_authority_violation_recovery(
    current_risk: RiskLevel,
    data: AuthorityViolationData,
    *,
    escalator: RuntimeRiskEscalator | None = None,
    model_id: str | None = None,
    provider_id: str | None = None,
    route_id: str | None = None,
) -> AuthorityRecoveryResult:
    """Escalate risk and produce task-local penalties for an authority violation.

    This does not implement Phase 11 permanent reputation; it only records
    task-local avoidance metadata.
    """
    escalator = escalator or RuntimeRiskEscalator.default()
    event = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.AUTHORITY_VIOLATION,
        material=True,
        evidence=_authority_violation_evidence_text(data),
        affected_paths=data.touched_authority_paths,
    )
    new_risk, _record = escalator.escalate(current_risk, event)

    penalties: dict[str, Any] = {
        "avoid_model_for_authority_repair": model_id,
        "avoid_provider_for_authority_repair": provider_id,
        "require_independent_reviewer": True,
        "require_review_count": max(2, _review_count_for_risk(new_risk)),
    }

    return AuthorityRecoveryResult(
        escalated_risk=new_risk,
        block_integration=True,
        evidence={
            "touched_authority_paths": sorted(data.touched_authority_paths),
            "attempted_state_advancement": data.attempted_state_advancement,
            "ignored_immutable_authority": data.ignored_immutable_authority,
            "summary_substituted_for_raw": data.summary_substituted_for_raw,
            "integration_state_mismatch": data.integration_state_mismatch,
        },
        task_local_penalties=penalties,
    )


def _authority_violation_evidence_text(data: AuthorityViolationData) -> str:
    parts: list[str] = []
    if data.touched_authority_paths:
        parts.append(f"touched authority paths: {', '.join(sorted(data.touched_authority_paths))}")
    if data.attempted_state_advancement:
        parts.append("attempted authority advancement")
    if data.ignored_immutable_authority:
        parts.append("ignored immutable authority")
    if data.summary_substituted_for_raw:
        parts.append("summary substituted for required raw authority")
    if data.integration_state_mismatch:
        parts.append("integration state mismatch")
    return "; ".join(parts) if parts else "authority violation"


def _review_count_for_risk(risk: RiskLevel) -> int:
    if risk.value >= RiskLevel.R4_CRITICAL_AUTHORITY.value:
        return 2
    if risk.value >= RiskLevel.R3_HIGH.value:
        return 1
    return 0
