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

    # Context depth ordering from shallowest to deepest.
    _DEPTH_RANK = {
        "targeted": 0,
        "hybrid": 1,
        "large_context": 2,
    }

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        self._overrides = overrides or {}

    @classmethod
    def default(cls) -> RiskContextPolicy:
        return cls()

    @classmethod
    def _depth_rank(cls, depth: str) -> int:
        return cls._DEPTH_RANK.get(depth, -1)

    def requirements_for(self, risk: RiskLevel) -> RiskContextRequirements:
        override = self._overrides.get(risk.name)
        base = self._base_requirements(risk)
        if override is not None:
            return self._merge_requirements(base, override)
        return base

    def _merge_requirements(
        self, base: RiskContextRequirements, override: dict[str, Any]
    ) -> RiskContextRequirements:
        """Merge an override monotonically with the base requirement.

        Overrides may deepen context or strengthen authority requirements, but
        they can never remove raw authority at R4 or weaken other safety flags.
        """
        override_strategy = str(override["strategy_preference"])
        base_rank = self._depth_rank(base.strategy_preference)
        override_rank = self._depth_rank(override_strategy)
        strategy_preference = (
            override_strategy if override_rank > base_rank else base.strategy_preference
        )

        authority_required = base.authority_required or bool(
            override.get("authority_required", False)
        )
        require_raw_authority = base.require_raw_authority or bool(
            override.get("require_raw_authority", False)
        )
        include_test_evidence = base.include_test_evidence or bool(
            override.get("include_test_evidence", False)
        )
        include_historical_findings = base.include_historical_findings or bool(
            override.get("include_historical_findings", False)
        )
        budget_multiplier = max(
            base.budget_multiplier, float(override.get("budget_multiplier", 1.0))
        )
        rationale = str(override.get("rationale", base.rationale))

        return RiskContextRequirements(
            strategy_preference=strategy_preference,
            authority_required=authority_required,
            require_raw_authority=require_raw_authority,
            include_test_evidence=include_test_evidence,
            include_historical_findings=include_historical_findings,
            budget_multiplier=budget_multiplier,
            rationale=rationale,
        )

    def _base_requirements(self, risk: RiskLevel) -> RiskContextRequirements:
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
