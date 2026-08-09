"""Project-specific risk policy and overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.policy.risk import RiskLevel

from .assessment import RiskFactor, RiskFactorCode, coerce_risk_level
from .authority import AuthoritySensitivePolicy
from .security import SecuritySensitivePolicy


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
        path_floors = {str(k): int(v) for k, v in data.get("path_risk_floors", {}).items()}
        for floor_value in path_floors.values():
            coerce_risk_level(floor_value)
        return cls(
            minimum_risk=coerce_risk_level(minimum_risk) if minimum_risk is not None else None,
            authority_policy=AuthoritySensitivePolicy.from_dict(data.get("authority_policy", {})),
            security_policy=SecuritySensitivePolicy.from_dict(data.get("security_policy", {})),
            architecture_thresholds=dict(data.get("architecture_thresholds", {})),
            review_minimum=data.get("review_minimum"),
            exploration_max_risk=coerce_risk_level(exploration_max_risk)
            if exploration_max_risk is not None
            else None,
            context_depth_minimum=data.get("context_depth_minimum"),
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
