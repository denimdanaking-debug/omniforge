from __future__ import annotations

from src.policy.risk import RiskLevel
from src.risk import RiskContextPolicy, RiskContextRequirements


def test_r0_minimal_context() -> None:
    policy = RiskContextPolicy.default()
    req = policy.requirements_for(RiskLevel.R0_TRIVIAL)
    assert req.strategy_preference == "targeted"
    assert not req.authority_required
    assert not req.require_raw_authority
    assert req.budget_multiplier == 1.0


def test_r1_targeted_context() -> None:
    policy = RiskContextPolicy.default()
    req = policy.requirements_for(RiskLevel.R1_LOW)
    assert req.strategy_preference == "targeted"
    assert not req.authority_required


def test_r2_standard_hybrid_context() -> None:
    policy = RiskContextPolicy.default()
    req = policy.requirements_for(RiskLevel.R2_NORMAL)
    assert req.strategy_preference == "hybrid"
    assert req.authority_required
    assert req.include_test_evidence
    assert not req.require_raw_authority
    assert req.budget_multiplier == 1.0


def test_r3_deeper_raw_evidence() -> None:
    policy = RiskContextPolicy.default()
    req = policy.requirements_for(RiskLevel.R3_HIGH)
    assert req.strategy_preference == "hybrid"
    assert req.authority_required
    assert req.require_raw_authority
    assert req.include_test_evidence
    assert req.include_historical_findings
    assert req.budget_multiplier == 1.5


def test_r4_authority_protected_large_context() -> None:
    policy = RiskContextPolicy.default()
    req = policy.requirements_for(RiskLevel.R4_CRITICAL_AUTHORITY)
    assert req.strategy_preference == "large_context"
    assert req.authority_required
    assert req.require_raw_authority
    assert req.include_test_evidence
    assert req.include_historical_findings
    assert req.budget_multiplier == 2.0


def test_r4_never_summary_only_authority() -> None:
    policy = RiskContextPolicy.default()
    req = policy.requirements_for(RiskLevel.R4_CRITICAL_AUTHORITY)
    assert req.require_raw_authority


def test_override() -> None:
    policy = RiskContextPolicy(
        overrides={
            RiskLevel.R2_NORMAL.name: {
                "strategy_preference": "large_context",
                "authority_required": True,
                "require_raw_authority": True,
                "include_test_evidence": True,
                "include_historical_findings": True,
                "budget_multiplier": 2.5,
                "rationale": "custom",
            }
        }
    )
    req = policy.requirements_for(RiskLevel.R2_NORMAL)
    assert req.strategy_preference == "large_context"
    assert req.require_raw_authority
    assert req.budget_multiplier == 2.5
    assert req.rationale == "custom"


def test_requirements_is_frozen() -> None:
    req = RiskContextRequirements(
        strategy_preference="targeted",
        authority_required=False,
        require_raw_authority=False,
        include_test_evidence=False,
        include_historical_findings=False,
        budget_multiplier=1.0,
        rationale="x",
    )
    assert req.strategy_preference == "targeted"


def test_override_cannot_remove_r4_raw_authority() -> None:
    policy = RiskContextPolicy(
        overrides={
            RiskLevel.R4_CRITICAL_AUTHORITY.name: {
                "strategy_preference": "targeted",
                "authority_required": False,
                "require_raw_authority": False,
                "include_test_evidence": False,
                "include_historical_findings": False,
                "budget_multiplier": 0.5,
                "rationale": "weakened",
            }
        }
    )
    req = policy.requirements_for(RiskLevel.R4_CRITICAL_AUTHORITY)
    assert req.strategy_preference == "large_context"
    assert req.authority_required
    assert req.require_raw_authority
    assert req.include_test_evidence
    assert req.include_historical_findings
    assert req.budget_multiplier >= 2.0
