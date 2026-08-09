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


def test_review_requirement_applies_project_minimum() -> None:
    policy = ProjectRiskPolicy(review_minimum=3)
    req = policy.review_requirement(RiskLevel.R2_NORMAL)
    assert req.reviewer_count == 3


def test_review_requirement_preserves_r3_independence() -> None:
    policy = ProjectRiskPolicy(review_minimum=1)
    req = policy.review_requirement(RiskLevel.R3_HIGH)
    assert req.reviewer_count >= 2
    assert req.minimum_independence == "independent"
    assert req.require_high_risk_reviewer
    assert req.prohibit_coder_identity
    assert req.distinct_failure_domains_required


def test_experimentation_policy_tightens_ceiling() -> None:
    policy = ProjectRiskPolicy(exploration_max_risk=RiskLevel.R0_TRIVIAL)
    exp_policy = policy.experimentation_policy()
    assert (
        exp_policy.check(
            RiskLevel.R1_LOW,
            global_exploration_enabled=True,
            project_exploration_allowed=True,
        ).allowed
        is False
    )


def test_experimentation_policy_cannot_loosen_core_ceiling() -> None:
    policy = ProjectRiskPolicy(exploration_max_risk=RiskLevel.R3_HIGH)
    exp_policy = policy.experimentation_policy()
    assert (
        exp_policy.check(
            RiskLevel.R2_NORMAL,
            global_exploration_enabled=True,
            project_exploration_allowed=True,
        ).allowed
        is False
    )


def test_context_requirements_deepen_with_project_minimum() -> None:
    policy = ProjectRiskPolicy(context_depth_minimum="large_context")
    req = policy.context_requirements(RiskLevel.R1_LOW)
    assert req.strategy_preference == "large_context"


def test_context_requirements_preserve_r4_raw_authority() -> None:
    policy = ProjectRiskPolicy(context_depth_minimum="targeted")
    req = policy.context_requirements(RiskLevel.R4_CRITICAL_AUTHORITY)
    assert req.strategy_preference == "large_context"
    assert req.require_raw_authority


def test_to_dict_round_trip() -> None:
    policy = ProjectRiskPolicy(
        minimum_risk=RiskLevel.R2_NORMAL,
        review_minimum=2,
        exploration_max_risk=RiskLevel.R1_LOW,
        context_depth_minimum="hybrid",
    )
    data = policy.to_dict()
    assert data["minimum_risk"] == 2
    assert data["review_minimum"] == 2
    assert data["exploration_max_risk"] == 1
    assert data["context_depth_minimum"] == "hybrid"


def test_path_floor_normalizes_dotted_input() -> None:
    policy = ProjectRiskPolicy(path_risk_floors={"src/security": 3})
    risk, factor = policy.apply_floor(("./src/security/secrets.py",), RiskLevel.R1_LOW)
    assert risk == RiskLevel.R3_HIGH
    assert factor is not None


def test_path_floor_normalizes_dotdot_input() -> None:
    policy = ProjectRiskPolicy(path_risk_floors={"src/security": 3})
    risk, factor = policy.apply_floor(("src/foo/../security/secrets.py",), RiskLevel.R1_LOW)
    assert risk == RiskLevel.R3_HIGH
    assert factor is not None


def test_path_floor_normalizes_backslash_input() -> None:
    policy = ProjectRiskPolicy(path_risk_floors={"src/security": 3})
    risk, factor = policy.apply_floor(("src\\security\\secrets.py",), RiskLevel.R1_LOW)
    assert risk == RiskLevel.R3_HIGH
    assert factor is not None


def test_path_floor_normalizes_configured_key() -> None:
    policy = ProjectRiskPolicy.from_dict({"path_risk_floors": {"./src/security/": 3}})
    assert "src/security" in policy.path_risk_floors
    risk, factor = policy.apply_floor(("src/security/secrets.py",), RiskLevel.R1_LOW)
    assert risk == RiskLevel.R3_HIGH
    assert factor is not None


def test_path_floor_rejects_root_escape_input() -> None:
    policy = ProjectRiskPolicy(path_risk_floors={"src/security": 3})
    with pytest.raises(ValueError):
        policy.apply_floor(("../../outside/file.py",), RiskLevel.R1_LOW)


def test_path_floor_rejects_root_escape_key() -> None:
    with pytest.raises(ValueError):
        ProjectRiskPolicy.from_dict({"path_risk_floors": {"../../outside": 3}})


def test_same_logical_floor_configuration_is_deterministic() -> None:
    p1 = ProjectRiskPolicy.from_dict({"path_risk_floors": {"./src/security/": 3}})
    p2 = ProjectRiskPolicy.from_dict({"path_risk_floors": {"src/security": 3}})
    assert p1.path_risk_floors == p2.path_risk_floors
