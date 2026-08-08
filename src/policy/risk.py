"""Risk taxonomy, monotonic escalation, and policy effects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from src.routing.model_identity import ModelLifecycle


class RiskLevel(IntEnum):
    R0_TRIVIAL = 0
    R1_LOW = 1
    R2_NORMAL = 2
    R3_HIGH = 3
    R4_CRITICAL_AUTHORITY = 4


class ContextDepth(StrEnum):
    TARGETED = "targeted"
    NORMAL = "normal"
    BROAD = "broad"
    AUTHORITY_PRIMARY = "authority_primary"


@dataclass(frozen=True)
class RiskPolicyEffects:
    minimum_model_lifecycle: ModelLifecycle
    required_reviewers: int
    exploration_allowed: bool
    context_depth: ContextDepth


@dataclass(frozen=True)
class RiskAssessment:
    level: RiskLevel
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not reason.strip() for reason in self.reasons):
            raise ValueError("risk reasons must be non-empty strings")

    def escalate(self, target: RiskLevel, reason: str) -> "RiskAssessment":
        """Increase risk only; a runtime event cannot silently lower risk."""

        if not reason.strip():
            raise ValueError("escalation reason must be non-empty")
        if target < self.level:
            raise ValueError(
                f"risk cannot be downgraded during execution: {self.level.name}->{target.name}"
            )
        if target == self.level:
            return RiskAssessment(self.level, self.reasons + (reason,))
        return RiskAssessment(target, self.reasons + (reason,))


def policy_for(level: RiskLevel) -> RiskPolicyEffects:
    policies = {
        RiskLevel.R0_TRIVIAL: RiskPolicyEffects(
            minimum_model_lifecycle=ModelLifecycle.LOW_RISK,
            required_reviewers=1,
            exploration_allowed=True,
            context_depth=ContextDepth.TARGETED,
        ),
        RiskLevel.R1_LOW: RiskPolicyEffects(
            minimum_model_lifecycle=ModelLifecycle.LOW_RISK,
            required_reviewers=1,
            exploration_allowed=True,
            context_depth=ContextDepth.TARGETED,
        ),
        RiskLevel.R2_NORMAL: RiskPolicyEffects(
            minimum_model_lifecycle=ModelLifecycle.NORMAL,
            required_reviewers=1,
            exploration_allowed=False,
            context_depth=ContextDepth.NORMAL,
        ),
        RiskLevel.R3_HIGH: RiskPolicyEffects(
            minimum_model_lifecycle=ModelLifecycle.HIGH_RISK,
            required_reviewers=2,
            exploration_allowed=False,
            context_depth=ContextDepth.BROAD,
        ),
        RiskLevel.R4_CRITICAL_AUTHORITY: RiskPolicyEffects(
            minimum_model_lifecycle=ModelLifecycle.HIGH_RISK,
            required_reviewers=2,
            exploration_allowed=False,
            context_depth=ContextDepth.AUTHORITY_PRIMARY,
        ),
    }
    return policies[level]


def lifecycle_eligible(lifecycle: ModelLifecycle, level: RiskLevel) -> bool:
    """Return whether lifecycle status satisfies the risk gate.

    DISABLED and SHADOW never execute authoritative work. HIGH_RISK can execute
    all production levels; NORMAL cannot execute R3/R4; LOW_RISK only R0/R1.
    """

    if lifecycle in {ModelLifecycle.DISABLED, ModelLifecycle.SHADOW}:
        return False
    rank = {
        ModelLifecycle.LOW_RISK: 1,
        ModelLifecycle.NORMAL: 2,
        ModelLifecycle.HIGH_RISK: 3,
    }
    required = policy_for(level).minimum_model_lifecycle
    return rank[lifecycle] >= rank[required]
