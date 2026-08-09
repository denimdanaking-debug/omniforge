"""Security-sensitive change detection for risk classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.policy.risk import RiskLevel

from .assessment import RiskFactor, RiskFactorCode, coerce_risk_level
from .path_utils import normalize_repo_path

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

# Core safety floor for security-sensitive changes. Project policy may raise it
# but never lower it.
_MIN_SECURITY_FLOOR = RiskLevel.R3_HIGH


class SecurityPolicyError(ValueError):
    """Raised when a project security policy attempts to weaken core safety."""


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
        """Build a policy that merges project additions with immutable core surfaces.

        Project config may add sensitive paths/prefixes and raise the floor. It
        may not remove core security surfaces or lower the core floor.
        """
        project_paths = frozenset(data.get("sensitive_paths", set()))
        project_prefixes = tuple(data.get("sensitive_path_prefixes", ()))
        floor = coerce_risk_level(data.get("default_risk_floor", RiskLevel.R3_HIGH))

        if floor < _MIN_SECURITY_FLOOR:
            raise SecurityPolicyError(
                f"security default_risk_floor cannot be lower than {_MIN_SECURITY_FLOOR.name}"
            )

        return cls(
            sensitive_paths=_DEFAULT_SECURITY_PATHS | project_paths,
            sensitive_path_prefixes=_DEFAULT_SECURITY_PREFIXES + project_prefixes,
            default_risk_floor=floor,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy to a normalized dict."""
        return {
            "sensitive_paths": sorted(self.sensitive_paths),
            "sensitive_path_prefixes": list(self.sensitive_path_prefixes),
            "default_risk_floor": self.default_risk_floor.value,
        }

    def _is_sensitive(self, path: str) -> bool:
        normalized = normalize_repo_path(path)
        if normalized in self.sensitive_paths:
            return True
        return any(normalized.startswith(prefix) for prefix in self.sensitive_path_prefixes)

    def assess(self, paths: tuple[str, ...]) -> RiskFactor | None:
        """Return a risk factor if any security-sensitive surface is touched."""
        touched = [normalize_repo_path(p) for p in paths if self._is_sensitive(p)]
        if not touched:
            return None
        return RiskFactor(
            code=RiskFactorCode.SECURITY_SENSITIVE,
            evidence=f"change touches security-sensitive path(s): {', '.join(sorted(touched))}",
            risk_level=self.default_risk_floor,
            provenance="security_sensitive_policy",
        )
