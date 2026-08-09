from __future__ import annotations

from src.policy.risk import RiskLevel
from src.risk import ExperimentationEligibilityPolicy


def test_global_exploration_disabled_denies_all() -> None:
    policy = ExperimentationEligibilityPolicy.default()
    for risk in RiskLevel:
        result = policy.check(
            risk, global_exploration_enabled=False, project_exploration_allowed=True
        )
        assert not result.allowed
        assert "global" in result.reason


def test_project_prohibition_denies() -> None:
    policy = ExperimentationEligibilityPolicy.default()
    result = policy.check(
        RiskLevel.R0_TRIVIAL,
        global_exploration_enabled=True,
        project_exploration_allowed=False,
    )
    assert not result.allowed
    assert "project prohibits" in result.reason


def test_r0_may_explore_when_enabled() -> None:
    policy = ExperimentationEligibilityPolicy.default()
    result = policy.check(
        RiskLevel.R0_TRIVIAL,
        global_exploration_enabled=True,
        project_exploration_allowed=True,
    )
    assert result.allowed


def test_r1_may_explore_when_enabled() -> None:
    policy = ExperimentationEligibilityPolicy.default()
    result = policy.check(
        RiskLevel.R1_LOW,
        global_exploration_enabled=True,
        project_exploration_allowed=True,
    )
    assert result.allowed


def test_r2_denied_by_default() -> None:
    policy = ExperimentationEligibilityPolicy.default()
    result = policy.check(
        RiskLevel.R2_NORMAL,
        global_exploration_enabled=True,
        project_exploration_allowed=True,
    )
    assert not result.allowed


def test_r3_denied() -> None:
    policy = ExperimentationEligibilityPolicy.default()
    result = policy.check(
        RiskLevel.R3_HIGH,
        global_exploration_enabled=True,
        project_exploration_allowed=True,
    )
    assert not result.allowed


def test_r4_denied() -> None:
    policy = ExperimentationEligibilityPolicy.default()
    result = policy.check(
        RiskLevel.R4_CRITICAL_AUTHORITY,
        global_exploration_enabled=True,
        project_exploration_allowed=True,
    )
    assert not result.allowed


def test_custom_max_risk_allows_r2() -> None:
    policy = ExperimentationEligibilityPolicy(max_risk=RiskLevel.R2_NORMAL)
    result = policy.check(
        RiskLevel.R2_NORMAL,
        global_exploration_enabled=True,
        project_exploration_allowed=True,
    )
    assert result.allowed


def test_custom_max_risk_denies_higher() -> None:
    policy = ExperimentationEligibilityPolicy(max_risk=RiskLevel.R2_NORMAL)
    result = policy.check(
        RiskLevel.R3_HIGH,
        global_exploration_enabled=True,
        project_exploration_allowed=True,
    )
    assert not result.allowed


def test_project_none_is_neutral() -> None:
    policy = ExperimentationEligibilityPolicy.default()
    result = policy.check(
        RiskLevel.R1_LOW,
        global_exploration_enabled=True,
        project_exploration_allowed=None,
    )
    assert result.allowed


def test_from_dict_custom_max() -> None:
    policy = ExperimentationEligibilityPolicy.from_dict({"max_risk": "R2_NORMAL"})
    result = policy.check(
        RiskLevel.R2_NORMAL,
        global_exploration_enabled=True,
        project_exploration_allowed=True,
    )
    assert result.allowed
