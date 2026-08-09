from __future__ import annotations

import pytest

from src.policy.risk import RiskLevel
from src.risk import AuthoritySensitivePolicy, RiskFactorCode
from src.risk.assessment import OperationType


def test_default_policy_protects_core_authority_files() -> None:
    policy = AuthoritySensitivePolicy.default()
    factor = policy.assess(("docs/PROJECT_STATE.json",), OperationType.MODIFY)
    assert factor is not None
    assert factor.code == RiskFactorCode.AUTHORITY_SENSITIVE
    assert factor.risk_level == RiskLevel.R4_CRITICAL_AUTHORITY


def test_default_policy_protects_roadmap() -> None:
    policy = AuthoritySensitivePolicy.default()
    factor = policy.assess(("docs/OMNIFORGE_FULL_ROADMAP_v1.0.md",), OperationType.MODIFY)
    assert factor is not None
    assert factor.risk_level == RiskLevel.R4_CRITICAL_AUTHORITY


def test_default_policy_protects_roadmap_authority() -> None:
    policy = AuthoritySensitivePolicy.default()
    factor = policy.assess(("docs/ROADMAP_AUTHORITY.json",), OperationType.MODIFY)
    assert factor is not None
    assert factor.risk_level == RiskLevel.R4_CRITICAL_AUTHORITY


def test_authority_prefixes_escalate() -> None:
    policy = AuthoritySensitivePolicy.default()
    factor = policy.assess(("docs/authority/policy.md",), OperationType.MODIFY)
    assert factor is not None
    assert factor.risk_level == RiskLevel.R4_CRITICAL_AUTHORITY


def test_path_normalization_prevents_bypass() -> None:
    policy = AuthoritySensitivePolicy.default()
    for path in (
        "./docs/PROJECT_STATE.json",
        "docs/../docs/PROJECT_STATE.json",
        "docs/./PROJECT_STATE.json",
        "docs/PROJECT_STATE.json",
    ):
        factor = policy.assess((path,), OperationType.MODIFY)
        assert factor is not None, path
        assert factor.risk_level == RiskLevel.R4_CRITICAL_AUTHORITY, path


def test_read_authority_is_lower_risk() -> None:
    policy = AuthoritySensitivePolicy.default()
    factor = policy.assess(("docs/PROJECT_STATE.json",), OperationType.READ)
    assert factor is not None
    assert factor.risk_level == RiskLevel.R1_LOW


def test_reference_authority_is_lower_risk() -> None:
    policy = AuthoritySensitivePolicy.default()
    factor = policy.assess(("docs/PROJECT_STATE.json",), OperationType.REFERENCE)
    assert factor is not None
    assert factor.risk_level == RiskLevel.R1_LOW


def test_delete_authority_is_critical() -> None:
    policy = AuthoritySensitivePolicy.default()
    factor = policy.assess(("docs/PROJECT_STATE.json",), OperationType.DELETE)
    assert factor is not None
    assert factor.risk_level == RiskLevel.R4_CRITICAL_AUTHORITY


def test_rename_authority_is_critical() -> None:
    policy = AuthoritySensitivePolicy.default()
    factor = policy.assess(("docs/PROJECT_STATE.json",), OperationType.RENAME)
    assert factor is not None
    assert factor.risk_level == RiskLevel.R4_CRITICAL_AUTHORITY


def test_non_authority_file_does_not_trigger() -> None:
    policy = AuthoritySensitivePolicy.default()
    factor = policy.assess(("src/providers/openai/adapter.py",), OperationType.MODIFY)
    assert factor is None


def test_authority_adjacent_name_is_not_auto_r4() -> None:
    policy = AuthoritySensitivePolicy.default()
    factor = policy.assess(("docs/authority_notes.md",), OperationType.MODIFY)
    assert factor is None


def test_custom_authority_paths() -> None:
    policy = AuthoritySensitivePolicy(
        protected_paths=frozenset({"custom/authority.toml"}),
        protected_path_prefixes=(),
    )
    factor = policy.assess(("custom/authority.toml",), OperationType.MODIFY)
    assert factor is not None
    assert factor.risk_level == RiskLevel.R4_CRITICAL_AUTHORITY


def test_custom_read_floor() -> None:
    policy = AuthoritySensitivePolicy(
        protected_paths=frozenset({"custom/authority.toml"}),
        protected_path_prefixes=(),
        read_risk_floor=RiskLevel.R2_NORMAL,
    )
    factor = policy.assess(("custom/authority.toml",), OperationType.READ)
    assert factor is not None
    assert factor.risk_level == RiskLevel.R2_NORMAL


def test_invalid_operation_defaults_to_modify_floor() -> None:
    policy = AuthoritySensitivePolicy.default()
    factor = policy.assess(("docs/PROJECT_STATE.json",), "unknown_op")
    assert factor is not None
    assert factor.risk_level == RiskLevel.R4_CRITICAL_AUTHORITY


def test_project_authority_paths_merge_with_core() -> None:
    policy = AuthoritySensitivePolicy.from_dict(
        {"protected_paths": ["custom/project_authority.json"]}
    )
    assert "docs/PROJECT_STATE.json" in policy.protected_paths
    assert "custom/project_authority.json" in policy.protected_paths
    factor = policy.assess(("docs/PROJECT_STATE.json",), OperationType.MODIFY)
    assert factor is not None
    assert factor.risk_level == RiskLevel.R4_CRITICAL_AUTHORITY


def test_empty_project_paths_keep_core_authority_protected() -> None:
    policy = AuthoritySensitivePolicy.from_dict({})
    assert "docs/PROJECT_STATE.json" in policy.protected_paths
    factor = policy.assess(("docs/PROJECT_STATE.json",), OperationType.MODIFY)
    assert factor is not None
    assert factor.risk_level == RiskLevel.R4_CRITICAL_AUTHORITY


def test_project_cannot_lower_authority_modify_floor() -> None:
    from src.risk.authority import AuthorityPolicyError

    with pytest.raises(AuthorityPolicyError):
        AuthoritySensitivePolicy.from_dict({"modify_risk_floor": "R2_NORMAL"})


def test_project_cannot_lower_authority_delete_floor() -> None:
    from src.risk.authority import AuthorityPolicyError

    with pytest.raises(AuthorityPolicyError):
        AuthoritySensitivePolicy.from_dict({"delete_risk_floor": "R1_LOW"})


def test_project_cannot_lower_authority_rename_floor() -> None:
    from src.risk.authority import AuthorityPolicyError

    with pytest.raises(AuthorityPolicyError):
        AuthoritySensitivePolicy.from_dict({"rename_risk_floor": "R0_TRIVIAL"})


def test_project_can_raise_authority_floor() -> None:
    policy = AuthoritySensitivePolicy.from_dict({"read_risk_floor": "R3_HIGH"})
    factor = policy.assess(("docs/PROJECT_STATE.json",), OperationType.READ)
    assert factor is not None
    assert factor.risk_level == RiskLevel.R3_HIGH
