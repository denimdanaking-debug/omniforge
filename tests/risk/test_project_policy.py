from __future__ import annotations

import pytest

from src.policy.risk import RiskLevel
from src.risk import ProjectRiskPolicy, RiskFactorCode


def test_default_policy_no_floor() -> None:
    policy = ProjectRiskPolicy.default()
    risk, factor = policy.apply_minimum(RiskLevel.R1_LOW)
    assert risk == RiskLevel.R1_LOW
    assert factor is None


def test_minimum_risk_raises_lower_risk() -> None:
    policy = ProjectRiskPolicy(minimum_risk=RiskLevel.R3_HIGH)
    risk, factor = policy.apply_minimum(RiskLevel.R1_LOW)
    assert risk == RiskLevel.R3_HIGH
    assert factor is not None
    assert factor.code == RiskFactorCode.PROJECT_OVERRIDE


def test_minimum_risk_does_not_raise_when_already_high() -> None:
    policy = ProjectRiskPolicy(minimum_risk=RiskLevel.R3_HIGH)
    risk, factor = policy.apply_minimum(RiskLevel.R4_CRITICAL_AUTHORITY)
    assert risk == RiskLevel.R4_CRITICAL_AUTHORITY
    assert factor is None


def test_path_floor_raises_risk() -> None:
    policy = ProjectRiskPolicy(path_risk_floors={"src/security": 3})
    risk, factor = policy.apply_floor(("src/security/secrets.py",), RiskLevel.R1_LOW)
    assert risk == RiskLevel.R3_HIGH
    assert factor is not None
    assert factor.code == RiskFactorCode.PROJECT_OVERRIDE


def test_path_floor_does_not_lower() -> None:
    policy = ProjectRiskPolicy(path_risk_floors={"src/security": 1})
    risk, factor = policy.apply_floor(("src/security/secrets.py",), RiskLevel.R3_HIGH)
    assert risk == RiskLevel.R3_HIGH
    assert factor is None


def test_path_floor_matches_exact_path() -> None:
    policy = ProjectRiskPolicy(path_risk_floors={"src/security/secrets.py": 3})
    risk, factor = policy.apply_floor(("src/security/secrets.py",), RiskLevel.R1_LOW)
    assert risk == RiskLevel.R3_HIGH
    assert factor is not None


def test_project_override_cannot_lower_authority_r4_floor() -> None:
    # A project minimum of R1 must not lower an already-R4 authority result.
    policy = ProjectRiskPolicy(minimum_risk=RiskLevel.R1_LOW)
    risk, factor = policy.apply_minimum(RiskLevel.R4_CRITICAL_AUTHORITY)
    assert risk == RiskLevel.R4_CRITICAL_AUTHORITY
    assert factor is None


def test_invalid_risk_enum_rejected() -> None:
    with pytest.raises(ValueError):
        ProjectRiskPolicy.from_dict({"minimum_risk": "R5_DOES_NOT_EXIST"})


def test_negative_path_floor_rejected_by_type() -> None:
    with pytest.raises(ValueError):
        ProjectRiskPolicy.from_dict({"path_risk_floors": {"src/x": -1}})


def test_from_dict_preserves_security_policy() -> None:
    data = {
        "minimum_risk": "R3_HIGH",
        "authority_policy": {
            "protected_paths": ["custom/authority.md"],
        },
        "security_policy": {
            "sensitive_paths": ["custom/secrets.py"],
            "default_risk_floor": "R4_CRITICAL_AUTHORITY",
        },
        "path_risk_floors": {"src/security": 3},
    }
    policy = ProjectRiskPolicy.from_dict(data)
    assert policy.minimum_risk == RiskLevel.R3_HIGH
    assert "custom/authority.md" in policy.authority_policy.protected_paths
    assert policy.security_policy.default_risk_floor == RiskLevel.R4_CRITICAL_AUTHORITY
    assert policy.path_risk_floors == {"src/security": 3}
