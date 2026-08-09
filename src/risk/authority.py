"""Authority-sensitive change detection for risk classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from src.policy.risk import RiskLevel

from .assessment import OperationType, RiskFactor, RiskFactorCode, coerce_risk_level

# Immutable core authority surfaces. These protect the authoritative roadmap
# and project state regardless of project configuration.
_CORE_AUTHORITY_PATHS: frozenset[str] = frozenset(
    {
        "docs/OMNIFORGE_FULL_ROADMAP_v1.0.md",
        "docs/PROJECT_STATE.json",
        "docs/ROADMAP_AUTHORITY.json",
    }
)

_CORE_AUTHORITY_PREFIXES: tuple[str, ...] = (
    "docs/authority/",
    "docs/roadmap/",
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


@dataclass(frozen=True)
class AuthoritySensitivePolicy:
    """Policy defining authority-sensitive paths and their risk floor."""

    protected_paths: frozenset[str] = frozenset()
    protected_path_prefixes: tuple[str, ...] = ()
    modify_risk_floor: RiskLevel = RiskLevel.R4_CRITICAL_AUTHORITY
    delete_risk_floor: RiskLevel = RiskLevel.R4_CRITICAL_AUTHORITY
    rename_risk_floor: RiskLevel = RiskLevel.R4_CRITICAL_AUTHORITY
    read_risk_floor: RiskLevel = RiskLevel.R1_LOW

    @classmethod
    def default(cls) -> AuthoritySensitivePolicy:
        return cls(
            protected_paths=_CORE_AUTHORITY_PATHS,
            protected_path_prefixes=_CORE_AUTHORITY_PREFIXES,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthoritySensitivePolicy:
        return cls(
            protected_paths=frozenset(data.get("protected_paths", _CORE_AUTHORITY_PATHS)),
            protected_path_prefixes=tuple(
                data.get("protected_path_prefixes", _CORE_AUTHORITY_PREFIXES)
            ),
            modify_risk_floor=coerce_risk_level(
                data.get("modify_risk_floor", RiskLevel.R4_CRITICAL_AUTHORITY)
            ),
            delete_risk_floor=coerce_risk_level(
                data.get("delete_risk_floor", RiskLevel.R4_CRITICAL_AUTHORITY)
            ),
            rename_risk_floor=coerce_risk_level(
                data.get("rename_risk_floor", RiskLevel.R4_CRITICAL_AUTHORITY)
            ),
            read_risk_floor=coerce_risk_level(data.get("read_risk_floor", RiskLevel.R1_LOW)),
        )

    def _is_protected(self, path: str) -> bool:
        normalized = _normalize_repo_path(path)
        if normalized in self.protected_paths:
            return True
        return any(normalized.startswith(prefix) for prefix in self.protected_path_prefixes)

    def assess(self, paths: tuple[str, ...], operation: OperationType | str) -> RiskFactor | None:
        """Return a risk factor if any protected authority surface is touched."""
        if isinstance(operation, str):
            try:
                op = OperationType(operation)
            except ValueError:
                op = OperationType.MODIFY
        else:
            op = operation
        touched = [_normalize_repo_path(p) for p in paths if self._is_protected(p)]
        if not touched:
            return None

        floor = {
            OperationType.READ: self.read_risk_floor,
            OperationType.REFERENCE: self.read_risk_floor,
            OperationType.MODIFY: self.modify_risk_floor,
            OperationType.DELETE: self.delete_risk_floor,
            OperationType.RENAME: self.rename_risk_floor,
        }.get(op, self.modify_risk_floor)

        return RiskFactor(
            code=RiskFactorCode.AUTHORITY_SENSITIVE,
            evidence=f"{op.value} on authority-sensitive path(s): {', '.join(sorted(touched))}",
            risk_level=floor,
            provenance="authority_sensitive_policy",
        )
