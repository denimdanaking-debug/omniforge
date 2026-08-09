"""Risk-driven experimentation eligibility policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.policy.risk import RiskLevel

from .assessment import coerce_risk_level


@dataclass(frozen=True)
class ExperimentationEligibility:
    """Result of experimentation eligibility check."""

    allowed: bool
    reason: str


class ExperimentationEligibilityPolicy:
    """Determine whether exploration may be permitted for a given risk."""

    def __init__(self, max_risk: RiskLevel | None = None) -> None:
        self._max_risk = max_risk

    @classmethod
    def default(cls) -> ExperimentationEligibilityPolicy:
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentationEligibilityPolicy:
        max_risk = data.get("max_risk")
        return cls(max_risk=coerce_risk_level(max_risk) if max_risk is not None else None)

    def check(
        self,
        risk: RiskLevel,
        global_exploration_enabled: bool,
        project_exploration_allowed: bool | None,
    ) -> ExperimentationEligibility:
        """Return whether experimentation is permitted.

        Requires global ON, project not prohibiting, and risk within floor.
        """
        if not global_exploration_enabled:
            return ExperimentationEligibility(
                allowed=False,
                reason="global exploration_enabled is false",
            )
        if project_exploration_allowed is False:
            return ExperimentationEligibility(
                allowed=False,
                reason="project prohibits exploration",
            )

        effective_max = self._max_risk
        if effective_max is None:
            # Default: R0/R1 may be explored; R2+ denied.
            effective_max = RiskLevel.R1_LOW

        if risk > effective_max:
            return ExperimentationEligibility(
                allowed=False,
                reason=f"risk {risk.name} exceeds experimentation ceiling {effective_max.name}",
            )

        return ExperimentationEligibility(
            allowed=True,
            reason=f"risk {risk.name} within experimentation ceiling {effective_max.name}",
        )
