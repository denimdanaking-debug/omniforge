"""Execution roles and role-aware routing request primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from src.policy.risk import RiskLevel


class ExecutionRole(StrEnum):
    PLANNING = "planning"
    ARCHITECTURE = "architecture"
    CODING = "coding"
    DEBUGGING = "debugging"
    REPAIR = "repair"
    REVIEW = "review"
    HIGH_RISK_REVIEW = "high_risk_review"
    ARBITRATION = "arbitration"
    CONTEXT_ANALYSIS = "context_analysis"
    INTEGRATION_ANALYSIS = "integration_analysis"


@dataclass(frozen=True)
class RoutingRequest:
    """Provider-neutral dispatch request; role and risk are mandatory by construction."""

    task_id: str
    role: ExecutionRole
    risk: RiskLevel
    project_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if self.project_id is not None and not self.project_id.strip():
            raise ValueError("project_id must be non-empty when provided")


@dataclass(frozen=True)
class RolePerformance:
    attempts: int = 0
    accepted: int = 0
    first_pass_accepted: int = 0

    def __post_init__(self) -> None:
        if min(self.attempts, self.accepted, self.first_pass_accepted) < 0:
            raise ValueError("role performance counters cannot be negative")
        if self.accepted > self.attempts:
            raise ValueError("accepted cannot exceed attempts")
        if self.first_pass_accepted > self.accepted:
            raise ValueError("first_pass_accepted cannot exceed accepted")


class RolePerformanceRegistry:
    """Keeps model evidence isolated by execution role."""

    def __init__(self) -> None:
        self._profiles: dict[tuple[str, ExecutionRole], RolePerformance] = {}

    def set(self, model_id: str, role: ExecutionRole, performance: RolePerformance) -> None:
        if not model_id.strip():
            raise ValueError("model_id must be non-empty")
        self._profiles[(model_id, role)] = performance

    def get(self, model_id: str, role: ExecutionRole) -> RolePerformance:
        return self._profiles.get((model_id, role), RolePerformance())

    def roles_for_model(self, model_id: str) -> dict[ExecutionRole, RolePerformance]:
        return {
            role: self._profiles[(candidate, role)]
            for candidate, role in sorted(
                self._profiles,
                key=lambda item: (item[0], item[1].value),
            )
            if candidate == model_id
        }
