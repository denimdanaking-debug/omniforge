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

# Core safety floors for authority-mutating operations. These may only be
# raised by project policy; they can never be lowered.
_MIN_MODIFY_FLOOR = RiskLevel.R4_CRITICAL_AUTHORITY
_MIN_DELETE_FLOOR = RiskLevel.R4_CRITICAL_AUTHORITY
_MIN_RENAME_FLOOR = RiskLevel.R4_CRITICAL_AUTHORITY
_MIN_READ_FLOOR = RiskLevel.R0_TRIVIAL


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


class AuthorityPolicyError(ValueError):
    """Raised when a project authority policy attempts to weaken core safety."""


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
        """Build a policy that merges project additions with immutable core surfaces.

        Project config may add protected paths/prefixes and raise floors. It may
        not remove core authority surfaces or lower core floors.
        """
        project_paths = frozenset(data.get("protected_paths", set()))
        project_prefixes = tuple(data.get("protected_path_prefixes", ()))

        modify_floor = coerce_risk_level(
            data.get("modify_risk_floor", RiskLevel.R4_CRITICAL_AUTHORITY)
        )
        delete_floor = coerce_risk_level(
            data.get("delete_risk_floor", RiskLevel.R4_CRITICAL_AUTHORITY)
        )
        rename_floor = coerce_risk_level(
            data.get("rename_risk_floor", RiskLevel.R4_CRITICAL_AUTHORITY)
        )
        read_floor = coerce_risk_level(data.get("read_risk_floor", RiskLevel.R1_LOW))

        if modify_floor < _MIN_MODIFY_FLOOR:
            raise AuthorityPolicyError(
                f"authority modify_risk_floor cannot be lower than {_MIN_MODIFY_FLOOR.name}"
            )
        if delete_floor < _MIN_DELETE_FLOOR:
            raise AuthorityPolicyError(
                f"authority delete_risk_floor cannot be lower than {_MIN_DELETE_FLOOR.name}"
            )
        if rename_floor < _MIN_RENAME_FLOOR:
            raise AuthorityPolicyError(
                f"authority rename_risk_floor cannot be lower than {_MIN_RENAME_FLOOR.name}"
            )
        if read_floor < _MIN_READ_FLOOR:
            raise AuthorityPolicyError(
                f"authority read_risk_floor cannot be lower than {_MIN_READ_FLOOR.name}"
            )

        return cls(
            protected_paths=_CORE_AUTHORITY_PATHS | project_paths,
            protected_path_prefixes=_CORE_AUTHORITY_PREFIXES + project_prefixes,
            modify_risk_floor=modify_floor,
            delete_risk_floor=delete_floor,
            rename_risk_floor=rename_floor,
            read_risk_floor=read_floor,
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
