"""Broad architectural change detection for risk classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from src.policy.risk import RiskLevel

from .assessment import RiskFactor, RiskFactorCode, coerce_risk_level

# Paths that represent central abstractions or public interfaces. A small change
# here can have broad impact.
_CENTRAL_ABSTRACTION_PREFIXES: tuple[str, ...] = (
    "src/providers/adapter.py",
    "src/providers/request.py",
    "src/providers/response.py",
    "src/routing/",
    "src/policy/risk.py",
    "src/context/schema.py",
    "src/context/strategy.py",
    "src/persistence/configuration.py",
    "src/persistence/runtime_state.py",
    "src/orchestration/",
)

_PUBLIC_INTERFACE_PATTERNS: tuple[str, ...] = (
    "src/providers/adapter",
    "src/routing/capabilities.py",
    "src/routing/inference_route.py",
    "src/routing/model_identity.py",
    "src/routing/policy.py",
    "src/routing/roles.py",
    "src/context/schema.py",
    "src/context/budget.py",
)

_PERSISTENCE_SCHEMA_PATTERNS: tuple[str, ...] = (
    "src/persistence/configuration.py",
    "src/persistence/runtime_state.py",
    "src/context/schema.py",
)


@dataclass(frozen=True)
class ArchitectureImpact:
    """Structured architectural impact signals."""

    subsystem_count: int
    public_interface_changed: bool
    persistence_schema_changed: bool
    cross_package_dependency_change: bool
    central_abstraction_changed: bool
    file_count: int
    changed_lines: int
    generated_only: bool


@dataclass(frozen=True)
class ArchitectureThresholds:
    """Configurable thresholds for architectural risk escalation."""

    subsystem_risk_floor: int = 3
    file_count_risk_floor: int = 6
    line_count_risk_floor: int = 500
    central_abstraction_risk_level: RiskLevel = RiskLevel.R3_HIGH
    public_interface_risk_level: RiskLevel = RiskLevel.R3_HIGH
    persistence_schema_risk_level: RiskLevel = RiskLevel.R3_HIGH
    broad_change_risk_level: RiskLevel = RiskLevel.R3_HIGH

    @classmethod
    def default(cls) -> ArchitectureThresholds:
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchitectureThresholds:
        return cls(
            subsystem_risk_floor=int(data.get("subsystem_risk_floor", 3)),
            file_count_risk_floor=int(data.get("file_count_risk_floor", 6)),
            line_count_risk_floor=int(data.get("line_count_risk_floor", 500)),
            central_abstraction_risk_level=coerce_risk_level(
                data.get("central_abstraction_risk_level", RiskLevel.R3_HIGH)
            ),
            public_interface_risk_level=coerce_risk_level(
                data.get("public_interface_risk_level", RiskLevel.R3_HIGH)
            ),
            persistence_schema_risk_level=coerce_risk_level(
                data.get("persistence_schema_risk_level", RiskLevel.R3_HIGH)
            ),
            broad_change_risk_level=coerce_risk_level(
                data.get("broad_change_risk_level", RiskLevel.R3_HIGH)
            ),
        )


def _normalize_repo_path(path: str) -> str:
    """Return a deterministic repo-relative path string."""
    p = PurePosixPath(path)
    parts: list[str] = []
    for part in p.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != "." and part != "/":
            parts.append(part)
    return "/".join(parts)


def _subsystem_for(path: str) -> str:
    """Return the top-level subsystem name for a path."""
    normalized = _normalize_repo_path(path)
    parts = normalized.split("/")
    if len(parts) >= 2 and parts[0] == "src":
        return parts[1]
    if parts:
        return parts[0]
    return ""


class ArchitectureImpactDetector:
    """Detect broad architectural impact from deterministic repository signals."""

    def __init__(self, thresholds: ArchitectureThresholds | None = None) -> None:
        self._thresholds = thresholds or ArchitectureThresholds.default()

    @classmethod
    def default(cls) -> ArchitectureImpactDetector:
        return cls()

    def detect(
        self,
        paths: tuple[str, ...],
        changed_lines: int,
        generated_files: tuple[str, ...],
    ) -> RiskFactor | None:
        """Return a risk factor if the change appears broadly architectural."""
        non_generated = [
            _normalize_repo_path(p) for p in paths if _normalize_repo_path(p) not in generated_files
        ]
        generated_only = len(non_generated) == 0 and len(paths) > 0

        subsystems = sorted({_subsystem_for(p) for p in non_generated if _subsystem_for(p)})
        subsystem_count = len(subsystems)

        central_abstraction_changed = any(
            any(
                _normalize_repo_path(p).startswith(prefix)
                for prefix in _CENTRAL_ABSTRACTION_PREFIXES
            )
            for p in non_generated
        )
        public_interface_changed = any(
            _normalize_repo_path(p).startswith(pattern)
            for p in non_generated
            for pattern in _PUBLIC_INTERFACE_PATTERNS
        )
        persistence_schema_changed = any(
            _normalize_repo_path(p).startswith(pattern)
            for p in non_generated
            for pattern in _PERSISTENCE_SCHEMA_PATTERNS
        )
        cross_package_dependency_change = subsystem_count >= self._thresholds.subsystem_risk_floor

        if generated_only:
            return None

        reasons: list[str] = []
        max_level = RiskLevel.R0_TRIVIAL

        if central_abstraction_changed:
            reasons.append("central abstraction changed")
            max_level = max(max_level, self._thresholds.central_abstraction_risk_level)
        if public_interface_changed:
            reasons.append("public interface changed")
            max_level = max(max_level, self._thresholds.public_interface_risk_level)
        if persistence_schema_changed:
            reasons.append("persistence/schema changed")
            max_level = max(max_level, self._thresholds.persistence_schema_risk_level)
        if cross_package_dependency_change:
            reasons.append(f"cross-package change spans {subsystem_count} subsystems")
            max_level = max(max_level, self._thresholds.broad_change_risk_level)
        if len(non_generated) >= self._thresholds.file_count_risk_floor:
            reasons.append(f"changed {len(non_generated)} non-generated files")
            max_level = max(max_level, self._thresholds.broad_change_risk_level)
        if changed_lines >= self._thresholds.line_count_risk_floor and not generated_only:
            reasons.append(f"changed approximately {changed_lines} lines")
            max_level = max(max_level, self._thresholds.broad_change_risk_level)

        if not reasons:
            return None

        return RiskFactor(
            code=RiskFactorCode.ARCHITECTURAL_CHANGE,
            evidence=f"architectural impact: {', '.join(reasons)}",
            risk_level=max_level,
            provenance="architecture_impact_detector",
        )
