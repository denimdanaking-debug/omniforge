"""Risk-driven context depth and authority requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.policy.risk import RiskLevel


@dataclass(frozen=True)
class RiskContextRequirements:
    """Context strategy requirements derived from risk."""

    strategy_preference: str
    authority_required: bool
    require_raw_authority: bool
    include_test_evidence: bool
    include_historical_findings: bool
    budget_multiplier: float
    rationale: str


class RiskContextPolicy:
    """Map risk level to deterministic context depth requirements."""

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        self._overrides = overrides or {}

    @classmethod
    def default(cls) -> RiskContextPolicy:
        return cls()

    def requirements_for(self, risk: RiskLevel) -> RiskContextRequirements:
        override = self._overrides.get(risk.name)
        if override is not None:
            return RiskContextRequirements(
                strategy_preference=str(override["strategy_preference"]),
                authority_required=bool(override.get("authority_required", False)),
                require_raw_authority=bool(override.get("require_raw_authority", False)),
                include_test_evidence=bool(override.get("include_test_evidence", False)),
                include_historical_findings=bool(
                    override.get("include_historical_findings", False)
                ),
                budget_multiplier=float(override.get("budget_multiplier", 1.0)),
                rationale=str(override["rationale"]),
            )

        if risk == RiskLevel.R0_TRIVIAL:
            return RiskContextRequirements(
                strategy_preference="targeted",
                authority_required=False,
                require_raw_authority=False,
                include_test_evidence=False,
                include_historical_findings=False,
                budget_multiplier=1.0,
                rationale="r0 minimal context",
            )
        if risk == RiskLevel.R1_LOW:
            return RiskContextRequirements(
                strategy_preference="targeted",
                authority_required=False,
                require_raw_authority=False,
                include_test_evidence=False,
                include_historical_findings=False,
                budget_multiplier=1.0,
                rationale="r1 targeted context",
            )
        if risk == RiskLevel.R2_NORMAL:
            return RiskContextRequirements(
                strategy_preference="hybrid",
                authority_required=True,
                require_raw_authority=False,
                include_test_evidence=True,
                include_historical_findings=False,
                budget_multiplier=1.0,
                rationale="r2 standard hybrid context",
            )
        if risk == RiskLevel.R3_HIGH:
            return RiskContextRequirements(
                strategy_preference="hybrid",
                authority_required=True,
                require_raw_authority=True,
                include_test_evidence=True,
                include_historical_findings=True,
                budget_multiplier=1.5,
                rationale="r3 deeper raw evidence",
            )
        # R4_CRITICAL_AUTHORITY
        return RiskContextRequirements(
            strategy_preference="large_context",
            authority_required=True,
            require_raw_authority=True,
            include_test_evidence=True,
            include_historical_findings=True,
            budget_multiplier=2.0,
            rationale="r4 authority-protected large context",
        )
