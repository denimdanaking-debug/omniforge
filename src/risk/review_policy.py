"""Risk-driven review requirement policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.policy.risk import RiskLevel


@dataclass(frozen=True)
class RiskReviewRequirement:
    """Structured review requirement derived from risk."""

    reviewer_count: int
    minimum_independence: str | None
    require_high_risk_reviewer: bool
    prohibit_coder_identity: bool
    distinct_failure_domains_required: bool
    rationale_codes: tuple[str, ...]


class RiskReviewPolicy:
    """Produce deterministic review requirements from risk level."""

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        self._overrides = overrides or {}

    @classmethod
    def default(cls) -> RiskReviewPolicy:
        return cls()

    def requirement_for(
        self, risk: RiskLevel, project_minimum: int | None = None
    ) -> RiskReviewRequirement:
        """Return review requirement for a risk level."""
        base = self._base_requirement(risk)
        reviewer_count = max(base.reviewer_count, project_minimum or 0)
        return RiskReviewRequirement(
            reviewer_count=reviewer_count,
            minimum_independence=base.minimum_independence,
            require_high_risk_reviewer=base.require_high_risk_reviewer,
            prohibit_coder_identity=base.prohibit_coder_identity,
            distinct_failure_domains_required=base.distinct_failure_domains_required,
            rationale_codes=base.rationale_codes,
        )

    def _base_requirement(self, risk: RiskLevel) -> RiskReviewRequirement:
        override = self._overrides.get(risk.name)
        if override is not None:
            return RiskReviewRequirement(
                reviewer_count=int(override["reviewer_count"]),
                minimum_independence=override.get("minimum_independence"),
                require_high_risk_reviewer=bool(override.get("require_high_risk_reviewer", False)),
                prohibit_coder_identity=bool(override.get("prohibit_coder_identity", False)),
                distinct_failure_domains_required=bool(
                    override.get("distinct_failure_domains_required", False)
                ),
                rationale_codes=tuple(override.get("rationale_codes", [])),
            )

        if risk == RiskLevel.R0_TRIVIAL:
            return RiskReviewRequirement(
                reviewer_count=0,
                minimum_independence=None,
                require_high_risk_reviewer=False,
                prohibit_coder_identity=False,
                distinct_failure_domains_required=False,
                rationale_codes=("r0_trivial_review_optional",),
            )
        if risk == RiskLevel.R1_LOW:
            return RiskReviewRequirement(
                reviewer_count=1,
                minimum_independence="same_provider",
                require_high_risk_reviewer=False,
                prohibit_coder_identity=True,
                distinct_failure_domains_required=False,
                rationale_codes=("r1_independent_review",),
            )
        if risk == RiskLevel.R2_NORMAL:
            return RiskReviewRequirement(
                reviewer_count=1,
                minimum_independence="same_model",
                require_high_risk_reviewer=False,
                prohibit_coder_identity=True,
                distinct_failure_domains_required=False,
                rationale_codes=("r2_independent_review",),
            )
        if risk == RiskLevel.R3_HIGH:
            return RiskReviewRequirement(
                reviewer_count=2,
                minimum_independence="independent",
                require_high_risk_reviewer=True,
                prohibit_coder_identity=True,
                distinct_failure_domains_required=True,
                rationale_codes=("r3_two_independent_reviewers",),
            )
        # R4_CRITICAL_AUTHORITY
        return RiskReviewRequirement(
            reviewer_count=2,
            minimum_independence="independent",
            require_high_risk_reviewer=True,
            prohibit_coder_identity=True,
            distinct_failure_domains_required=True,
            rationale_codes=("r4_authority_review",),
        )
