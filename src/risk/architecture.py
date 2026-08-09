"""Broad architectural change detection for risk classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.policy.risk import RiskLevel

from .assessment import RiskFactor, RiskFactorCode, coerce_risk_level
from .path_utils import normalize_repo_path

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


# Core architectural safety defaults. Project configuration may tighten these
# (lower numeric thresholds trigger sooner; higher risk levels escalate further)
# but may not weaken them.
_CORE_SUBSYSTEM_FLOOR = 3
_CORE_FILE_COUNT_FLOOR = 6
_CORE_LINE_COUNT_FLOOR = 500
_CORE_ARCHITECTURE_RISK_LEVEL = RiskLevel.R3_HIGH


@dataclass(frozen=True)
class ArchitectureThresholds:
    """Configurable thresholds for architectural risk escalation."""

    subsystem_risk_floor: int = _CORE_SUBSYSTEM_FLOOR
    file_count_risk_floor: int = _CORE_FILE_COUNT_FLOOR
    line_count_risk_floor: int = _CORE_LINE_COUNT_FLOOR
    central_abstraction_risk_level: RiskLevel = _CORE_ARCHITECTURE_RISK_LEVEL
    public_interface_risk_level: RiskLevel = _CORE_ARCHITECTURE_RISK_LEVEL
    persistence_schema_risk_level: RiskLevel = _CORE_ARCHITECTURE_RISK_LEVEL
    broad_change_risk_level: RiskLevel = _CORE_ARCHITECTURE_RISK_LEVEL

    @classmethod
    def default(cls) -> ArchitectureThresholds:
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchitectureThresholds:
        """Build thresholds that tighten the core defaults.

        Lower numeric thresholds are stricter because they trigger architectural
        classification sooner. Higher risk levels are stricter because they
        escalate further. Project values below core safety floors are rejected.
        """
        allowed = {
            "subsystem_risk_floor",
            "file_count_risk_floor",
            "line_count_risk_floor",
            "central_abstraction_risk_level",
            "public_interface_risk_level",
            "persistence_schema_risk_level",
            "broad_change_risk_level",
        }
        unknown = set(data.keys()) - allowed
        if unknown:
            raise ArchitectureThresholdsError(
                f"unknown architecture_thresholds fields: {sorted(unknown)}"
            )

        def _positive_int(value: Any, name: str) -> int:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ArchitectureThresholdsError(
                    f"{name} must be a positive integer, got {value!r}"
                )
            if value <= 0:
                raise ArchitectureThresholdsError(f"{name} must be positive, got {value}")
            return value

        subsystem_floor = _positive_int(
            data.get("subsystem_risk_floor", _CORE_SUBSYSTEM_FLOOR),
            "subsystem_risk_floor",
        )
        file_count_floor = _positive_int(
            data.get("file_count_risk_floor", _CORE_FILE_COUNT_FLOOR),
            "file_count_risk_floor",
        )
        line_count_floor = _positive_int(
            data.get("line_count_risk_floor", _CORE_LINE_COUNT_FLOOR),
            "line_count_risk_floor",
        )

        central_level = coerce_risk_level(
            data.get("central_abstraction_risk_level", _CORE_ARCHITECTURE_RISK_LEVEL)
        )
        public_level = coerce_risk_level(
            data.get("public_interface_risk_level", _CORE_ARCHITECTURE_RISK_LEVEL)
        )
        persistence_level = coerce_risk_level(
            data.get("persistence_schema_risk_level", _CORE_ARCHITECTURE_RISK_LEVEL)
        )
        broad_level = coerce_risk_level(
            data.get("broad_change_risk_level", _CORE_ARCHITECTURE_RISK_LEVEL)
        )

        for name, level in (
            ("central_abstraction_risk_level", central_level),
            ("public_interface_risk_level", public_level),
            ("persistence_schema_risk_level", persistence_level),
            ("broad_change_risk_level", broad_level),
        ):
            if level < _CORE_ARCHITECTURE_RISK_LEVEL:
                raise ArchitectureThresholdsError(
                    f"{name} cannot be lower than {_CORE_ARCHITECTURE_RISK_LEVEL.name}"
                )

        return cls(
            subsystem_risk_floor=min(_CORE_SUBSYSTEM_FLOOR, subsystem_floor),
            file_count_risk_floor=min(_CORE_FILE_COUNT_FLOOR, file_count_floor),
            line_count_risk_floor=min(_CORE_LINE_COUNT_FLOOR, line_count_floor),
            central_abstraction_risk_level=central_level,
            public_interface_risk_level=public_level,
            persistence_schema_risk_level=persistence_level,
            broad_change_risk_level=broad_level,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize thresholds to a normalized dict."""
        return {
            "subsystem_risk_floor": self.subsystem_risk_floor,
            "file_count_risk_floor": self.file_count_risk_floor,
            "line_count_risk_floor": self.line_count_risk_floor,
            "central_abstraction_risk_level": self.central_abstraction_risk_level.value,
            "public_interface_risk_level": self.public_interface_risk_level.value,
            "persistence_schema_risk_level": self.persistence_schema_risk_level.value,
            "broad_change_risk_level": self.broad_change_risk_level.value,
        }


class ArchitectureThresholdsError(ValueError):
    """Raised when architecture threshold configuration violates safety rules."""


def _subsystem_for(path: str) -> str:
    """Return the top-level subsystem name for a path."""
    normalized = normalize_repo_path(path)
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
        normalized_paths = [normalize_repo_path(p) for p in paths]
        normalized_generated = {normalize_repo_path(p) for p in generated_files}
        non_generated = [p for p in normalized_paths if p not in normalized_generated]
        generated_only = len(non_generated) == 0 and len(paths) > 0

        subsystems = sorted({_subsystem_for(p) for p in non_generated if _subsystem_for(p)})
        subsystem_count = len(subsystems)

        central_abstraction_changed = any(
            any(p.startswith(prefix) for prefix in _CENTRAL_ABSTRACTION_PREFIXES)
            for p in non_generated
        )
        public_interface_changed = any(
            p.startswith(pattern) for p in non_generated for pattern in _PUBLIC_INTERFACE_PATTERNS
        )
        persistence_schema_changed = any(
            p.startswith(pattern) for p in non_generated for pattern in _PERSISTENCE_SCHEMA_PATTERNS
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
