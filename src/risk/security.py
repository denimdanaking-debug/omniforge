"""Security-sensitive change detection for risk classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from src.policy.risk import RiskLevel

from .assessment import RiskFactor, RiskFactorCode, coerce_risk_level

# Canonical security-sensitive surfaces. These are explicit path-based rules,
# not text-pattern heuristics, to avoid false positives on ordinary code words.
_DEFAULT_SECURITY_PATHS: frozenset[str] = frozenset(
    {
        "src/security/redaction.py",
        "src/security/secrets.py",
        "src/providers/identity.py",
        "src/persistence/configuration.py",
    }
)

_DEFAULT_SECURITY_PREFIXES: tuple[str, ...] = (
    "src/security/",
    "src/providers/credentials/",
    "src/auth/",
    "src/permissions/",
    "src/crypto/",
    "src/signing/",
    "src/key_management/",
    "src/sandbox/",
    ".github/workflows/",
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
class SecuritySensitivePolicy:
    """Policy defining security-sensitive paths and their risk floor."""

    sensitive_paths: frozenset[str] = frozenset()
    sensitive_path_prefixes: tuple[str, ...] = ()
    default_risk_floor: RiskLevel = RiskLevel.R3_HIGH

    @classmethod
    def default(cls) -> SecuritySensitivePolicy:
        return cls(
            sensitive_paths=_DEFAULT_SECURITY_PATHS,
            sensitive_path_prefixes=_DEFAULT_SECURITY_PREFIXES,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecuritySensitivePolicy:
        return cls(
            sensitive_paths=frozenset(data.get("sensitive_paths", _DEFAULT_SECURITY_PATHS)),
            sensitive_path_prefixes=tuple(
                data.get("sensitive_path_prefixes", _DEFAULT_SECURITY_PREFIXES)
            ),
            default_risk_floor=coerce_risk_level(data.get("default_risk_floor", RiskLevel.R3_HIGH)),
        )

    def _is_sensitive(self, path: str) -> bool:
        normalized = _normalize_repo_path(path)
        if normalized in self.sensitive_paths:
            return True
        return any(normalized.startswith(prefix) for prefix in self.sensitive_path_prefixes)

    def assess(self, paths: tuple[str, ...]) -> RiskFactor | None:
        """Return a risk factor if any security-sensitive surface is touched."""
        touched = [_normalize_repo_path(p) for p in paths if self._is_sensitive(p)]
        if not touched:
            return None
        return RiskFactor(
            code=RiskFactorCode.SECURITY_SENSITIVE,
            evidence=f"change touches security-sensitive path(s): {', '.join(sorted(touched))}",
            risk_level=self.default_risk_floor,
            provenance="security_sensitive_policy",
        )
