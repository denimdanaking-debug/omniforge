"""Project-specific risk policy and overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.policy.risk import RiskLevel

from .assessment import RiskFactor, RiskFactorCode, coerce_risk_level
from .authority import AuthoritySensitivePolicy
from .context_policy import RiskContextPolicy, RiskContextRequirements
from .experimentation import ExperimentationEligibilityPolicy
from .review_policy import RiskReviewPolicy, RiskReviewRequirement
from .security import SecuritySensitivePolicy

# Canonical context depth ordering (shallow -> deep).
_CONTEXT_DEPTH_ORDER = ("targeted", "hybrid", "large_context")


def _context_depth_rank(depth: str) -> int:
    try:
        return _CONTEXT_DEPTH_ORDER.index(depth)
    except ValueError as exc:
        raise ValueError(f"unknown context depth: {depth!r}") from exc


class ProjectRiskPolicyError(ValueError):
    """Raised when a project risk policy is invalid or attempts to weaken safety."""


@dataclass(frozen=True)
class ProjectRiskPolicy:
    """Project-specific risk overrides that may tighten but not weaken safety."""

    minimum_risk: RiskLevel | None = None
    authority_policy: AuthoritySensitivePolicy = field(
        default_factory=AuthoritySensitivePolicy.default
    )
    security_policy: SecuritySensitivePolicy = field(
        default_factory=SecuritySensitivePolicy.default
    )
    architecture_thresholds: dict[str, Any] = field(default_factory=dict)
    review_minimum: int | None = None
    exploration_max_risk: RiskLevel | None = None
    context_depth_minimum: str | None = None
    path_risk_floors: dict[str, int] = field(default_factory=dict)

    @classmethod
    def default(cls) -> ProjectRiskPolicy:
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectRiskPolicy:
        minimum_risk = data.get("minimum_risk")
        exploration_max_risk = data.get("exploration_max_risk")
        review_minimum = data.get("review_minimum")
        context_depth_minimum = data.get("context_depth_minimum")

        if review_minimum is not None and (
            not isinstance(review_minimum, int) or review_minimum < 0
        ):
            raise ProjectRiskPolicyError("review_minimum must be a non-negative integer")

        if context_depth_minimum is not None:
            if not isinstance(context_depth_minimum, str):
                raise ProjectRiskPolicyError("context_depth_minimum must be a string")
            _context_depth_rank(context_depth_minimum)

        path_floors = {str(k): int(v) for k, v in data.get("path_risk_floors", {}).items()}
        for floor_value in path_floors.values():
            coerce_risk_level(floor_value)

        return cls(
            minimum_risk=coerce_risk_level(minimum_risk) if minimum_risk is not None else None,
            authority_policy=AuthoritySensitivePolicy.from_dict(data.get("authority_policy", {})),
            security_policy=SecuritySensitivePolicy.from_dict(data.get("security_policy", {})),
            architecture_thresholds=dict(data.get("architecture_thresholds", {})),
            review_minimum=review_minimum,
            exploration_max_risk=coerce_risk_level(exploration_max_risk)
            if exploration_max_risk is not None
            else None,
            context_depth_minimum=context_depth_minimum,
            path_risk_floors=path_floors,
        )

    def apply_floor(
        self,
        paths: tuple[str, ...],
        current_risk: RiskLevel,
    ) -> tuple[RiskLevel, RiskFactor | None]:
        """Apply configured per-path risk floors. Returns (risk, factor_or_none)."""
        for raw_path, floor_value in sorted(self.path_risk_floors.items()):
            floor = RiskLevel(floor_value)
            if floor <= current_risk:
                continue
            for path in paths:
                normalized = path.replace("\\", "/")
                if normalized == raw_path or normalized.startswith(raw_path.rstrip("/") + "/"):
                    evidence = f"project policy imposes risk floor {floor.name} for path {raw_path}"
                    return floor, RiskFactor(
                        code=RiskFactorCode.PROJECT_OVERRIDE,
                        evidence=evidence,
                        risk_level=floor,
                        provenance="project_risk_policy",
                    )
        return current_risk, None

    def apply_minimum(self, current_risk: RiskLevel) -> tuple[RiskLevel, RiskFactor | None]:
        """Apply configured project minimum risk."""
        if self.minimum_risk is not None and self.minimum_risk > current_risk:
            return self.minimum_risk, RiskFactor(
                code=RiskFactorCode.PROJECT_OVERRIDE,
                evidence=f"project minimum risk is {self.minimum_risk.name}",
                risk_level=self.minimum_risk,
                provenance="project_risk_policy",
            )
        return current_risk, None

    def review_requirement(self, risk: RiskLevel) -> RiskReviewRequirement:
        """Return the effective review requirement for a risk level.

        Combines the canonical base requirement with the project review minimum.
        Project minimum can only strengthen the base; it cannot weaken R3/R4
        independence or high-risk-reviewer requirements.
        """
        base = RiskReviewPolicy.default().requirement_for(risk, project_minimum=self.review_minimum)
        if self.review_minimum is not None and self.review_minimum > base.reviewer_count:
            reviewer_count = self.review_minimum
        else:
            reviewer_count = base.reviewer_count

        return RiskReviewRequirement(
            reviewer_count=reviewer_count,
            minimum_independence=base.minimum_independence,
            require_high_risk_reviewer=base.require_high_risk_reviewer,
            prohibit_coder_identity=base.prohibit_coder_identity,
            distinct_failure_domains_required=base.distinct_failure_domains_required,
            rationale_codes=base.rationale_codes
            + (
                ("project_review_minimum",)
                if self.review_minimum is not None and self.review_minimum > reviewer_count
                else ()
            ),
        )

    def experimentation_policy(self) -> ExperimentationEligibilityPolicy:
        """Return the effective experimentation policy.

        Project exploration_max_risk may only tighten the core ceiling.
        """
        core_max = ExperimentationEligibilityPolicy.default()._max_risk or RiskLevel.R1_LOW
        if self.exploration_max_risk is not None and self.exploration_max_risk < core_max:
            return ExperimentationEligibilityPolicy(max_risk=self.exploration_max_risk)
        return ExperimentationEligibilityPolicy.default()

    def context_requirements(self, risk: RiskLevel) -> RiskContextRequirements:
        """Return the effective context requirements for a risk level.

        Project context_depth_minimum may only deepen context; R4 raw authority
        requirement is immutable.
        """
        base = RiskContextPolicy.default().requirements_for(risk)
        if self.context_depth_minimum is None:
            return base

        project_rank = _context_depth_rank(self.context_depth_minimum)
        base_rank = _context_depth_rank(base.strategy_preference)
        if project_rank <= base_rank:
            return base

        return RiskContextRequirements(
            strategy_preference=self.context_depth_minimum,
            authority_required=True,
            require_raw_authority=base.require_raw_authority
            or risk == RiskLevel.R4_CRITICAL_AUTHORITY,
            include_test_evidence=True,
            include_historical_findings=True,
            budget_multiplier=max(base.budget_multiplier, 1.5),
            rationale=f"{base.rationale}; project context minimum {self.context_depth_minimum}",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy to a normalized dict."""
        result: dict[str, Any] = {
            "architecture_thresholds": self.architecture_thresholds,
            "path_risk_floors": self.path_risk_floors,
        }
        if self.minimum_risk is not None:
            result["minimum_risk"] = self.minimum_risk.value
        if self.review_minimum is not None:
            result["review_minimum"] = self.review_minimum
        if self.exploration_max_risk is not None:
            result["exploration_max_risk"] = self.exploration_max_risk.value
        if self.context_depth_minimum is not None:
            result["context_depth_minimum"] = self.context_depth_minimum
        result["authority_policy"] = {
            "protected_paths": sorted(self.authority_policy.protected_paths),
            "protected_path_prefixes": list(self.authority_policy.protected_path_prefixes),
            "modify_risk_floor": self.authority_policy.modify_risk_floor.value,
            "delete_risk_floor": self.authority_policy.delete_risk_floor.value,
            "rename_risk_floor": self.authority_policy.rename_risk_floor.value,
            "read_risk_floor": self.authority_policy.read_risk_floor.value,
        }
        result["security_policy"] = {
            "sensitive_paths": sorted(self.security_policy.sensitive_paths),
            "sensitive_path_prefixes": list(self.security_policy.sensitive_path_prefixes),
            "default_risk_floor": self.security_policy.default_risk_floor.value,
        }
        return result
