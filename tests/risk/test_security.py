from __future__ import annotations

import pytest

from src.policy.risk import RiskLevel
from src.risk import RiskFactorCode, SecuritySensitivePolicy


def test_credential_resolution_path_is_security_sensitive() -> None:
    policy = SecuritySensitivePolicy.default()
    factor = policy.assess(("src/security/secrets.py",))
    assert factor is not None
    assert factor.code == RiskFactorCode.SECURITY_SENSITIVE
    assert factor.risk_level == RiskLevel.R3_HIGH


def test_auth_prefix_is_security_sensitive() -> None:
    policy = SecuritySensitivePolicy.default()
    factor = policy.assess(("src/auth/password_policy.py",))
    assert factor is not None
    assert factor.risk_level == RiskLevel.R3_HIGH


def test_permissions_prefix_is_security_sensitive() -> None:
    policy = SecuritySensitivePolicy.default()
    factor = policy.assess(("src/permissions/grants.py",))
    assert factor is not None
    assert factor.risk_level == RiskLevel.R3_HIGH


def test_crypto_prefix_is_security_sensitive() -> None:
    policy = SecuritySensitivePolicy.default()
    factor = policy.assess(("src/crypto/signer.py",))
    assert factor is not None
    assert factor.risk_level == RiskLevel.R3_HIGH


def test_ci_workflow_prefix_is_security_sensitive() -> None:
    policy = SecuritySensitivePolicy.default()
    factor = policy.assess((".github/workflows/deploy.yml",))
    assert factor is not None
    assert factor.risk_level == RiskLevel.R3_HIGH


def test_ordinary_text_password_does_not_trigger() -> None:
    policy = SecuritySensitivePolicy.default()
    factor = policy.assess(("src/providers/openai/password_policy_comment.py",))
    assert factor is None


def test_fake_secret_fixture_does_not_trigger() -> None:
    policy = SecuritySensitivePolicy.default()
    factor = policy.assess(("tests/fixtures/fake_secret.json",))
    assert factor is None


def test_provider_credentials_prefix_is_security_sensitive() -> None:
    policy = SecuritySensitivePolicy.default()
    factor = policy.assess(("src/providers/credentials/store.py",))
    assert factor is not None
    assert factor.risk_level == RiskLevel.R3_HIGH


def test_configuration_path_is_security_sensitive() -> None:
    policy = SecuritySensitivePolicy.default()
    factor = policy.assess(("src/persistence/configuration.py",))
    assert factor is not None
    assert factor.risk_level == RiskLevel.R3_HIGH


def test_non_security_file_does_not_trigger() -> None:
    policy = SecuritySensitivePolicy.default()
    factor = policy.assess(("src/providers/openai/adapter.py",))
    assert factor is None


def test_custom_security_floor() -> None:
    policy = SecuritySensitivePolicy(
        sensitive_paths=frozenset({"custom/secrets.ini"}),
        sensitive_path_prefixes=(),
        default_risk_floor=RiskLevel.R4_CRITICAL_AUTHORITY,
    )
    factor = policy.assess(("custom/secrets.ini",))
    assert factor is not None
    assert factor.risk_level == RiskLevel.R4_CRITICAL_AUTHORITY


def test_custom_prefix() -> None:
    policy = SecuritySensitivePolicy(
        sensitive_paths=frozenset(),
        sensitive_path_prefixes=("custom/vault/",),
        default_risk_floor=RiskLevel.R3_HIGH,
    )
    factor = policy.assess(("custom/vault/key.py",))
    assert factor is not None
    assert factor.risk_level == RiskLevel.R3_HIGH


def test_path_normalization_for_prefix() -> None:
    policy = SecuritySensitivePolicy.default()
    factor = policy.assess(("./src/security/../security/secrets.py",))
    assert factor is not None
    assert factor.risk_level == RiskLevel.R3_HIGH


def test_project_security_paths_merge_with_core() -> None:
    policy = SecuritySensitivePolicy.from_dict({"sensitive_paths": ["custom/vault.py"]})
    assert "src/security/secrets.py" in policy.sensitive_paths
    assert "custom/vault.py" in policy.sensitive_paths
    factor = policy.assess(("src/security/secrets.py",))
    assert factor is not None
    assert factor.risk_level == RiskLevel.R3_HIGH


def test_project_security_prefixes_merge_with_core() -> None:
    policy = SecuritySensitivePolicy.from_dict({"sensitive_path_prefixes": ["custom/vault/"]})
    assert "src/security/" in policy.sensitive_path_prefixes
    factor = policy.assess(("src/security/secrets.py",))
    assert factor is not None
    assert factor.risk_level == RiskLevel.R3_HIGH


def test_project_cannot_lower_security_floor() -> None:
    from src.risk.security import SecurityPolicyError

    with pytest.raises(SecurityPolicyError):
        SecuritySensitivePolicy.from_dict({"default_risk_floor": "R1_LOW"})


def test_project_can_raise_security_floor() -> None:
    policy = SecuritySensitivePolicy.from_dict({"default_risk_floor": "R4_CRITICAL_AUTHORITY"})
    factor = policy.assess(("src/security/secrets.py",))
    assert factor is not None
    assert factor.risk_level == RiskLevel.R4_CRITICAL_AUTHORITY
