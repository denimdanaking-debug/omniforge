from __future__ import annotations

from src.policy.risk import RiskLevel
from src.risk import RiskReviewPolicy, RiskReviewRequirement


def test_r0_review_optional() -> None:
    policy = RiskReviewPolicy.default()
    req = policy.requirement_for(RiskLevel.R0_TRIVIAL)
    assert req.reviewer_count == 0
    assert not req.require_high_risk_reviewer


def test_r1_requires_one_reviewer() -> None:
    policy = RiskReviewPolicy.default()
    req = policy.requirement_for(RiskLevel.R1_LOW)
    assert req.reviewer_count == 1
    assert req.prohibit_coder_identity
    assert req.minimum_independence == "same_provider"


def test_r2_requires_one_reviewer() -> None:
    policy = RiskReviewPolicy.default()
    req = policy.requirement_for(RiskLevel.R2_NORMAL)
    assert req.reviewer_count == 1
    assert req.prohibit_coder_identity
    assert req.minimum_independence == "same_model"


def test_r3_requires_two_independent_reviewers() -> None:
    policy = RiskReviewPolicy.default()
    req = policy.requirement_for(RiskLevel.R3_HIGH)
    assert req.reviewer_count == 2
    assert req.require_high_risk_reviewer
    assert req.prohibit_coder_identity
    assert req.minimum_independence == "independent"
    assert req.distinct_failure_domains_required


def test_r4_requires_two_strict_independent_reviewers() -> None:
    policy = RiskReviewPolicy.default()
    req = policy.requirement_for(RiskLevel.R4_CRITICAL_AUTHORITY)
    assert req.reviewer_count == 2
    assert req.require_high_risk_reviewer
    assert req.prohibit_coder_identity
    assert req.minimum_independence == "independent"
    assert req.distinct_failure_domains_required
    assert "r4_authority_review" in req.rationale_codes


def test_project_minimum_review_count_strengthens() -> None:
    policy = RiskReviewPolicy.default()
    req = policy.requirement_for(RiskLevel.R1_LOW, project_minimum=2)
    assert req.reviewer_count == 2


def test_project_minimum_does_not_weaken() -> None:
    policy = RiskReviewPolicy.default()
    req = policy.requirement_for(RiskLevel.R3_HIGH, project_minimum=1)
    assert req.reviewer_count == 2


def test_r3_cannot_be_satisfied_by_coder() -> None:
    policy = RiskReviewPolicy.default()
    req = policy.requirement_for(RiskLevel.R3_HIGH)
    assert req.prohibit_coder_identity


def test_r3_requires_distinct_failure_domains() -> None:
    policy = RiskReviewPolicy.default()
    req = policy.requirement_for(RiskLevel.R3_HIGH)
    assert req.distinct_failure_domains_required


def test_override_r2_to_two_reviewers() -> None:
    policy = RiskReviewPolicy(
        overrides={
            RiskLevel.R2_NORMAL.name: {
                "reviewer_count": 2,
                "minimum_independence": "independent",
                "require_high_risk_reviewer": True,
                "prohibit_coder_identity": True,
                "distinct_failure_domains_required": True,
                "rationale_codes": ["custom_r2"],
            }
        }
    )
    req = policy.requirement_for(RiskLevel.R2_NORMAL)
    assert req.reviewer_count == 2
    assert req.minimum_independence == "independent"
    assert req.require_high_risk_reviewer
    assert req.distinct_failure_domains_required
    assert "custom_r2" in req.rationale_codes
    assert "r2_independent_review" in req.rationale_codes


def test_requirement_is_frozen() -> None:
    req = RiskReviewRequirement(
        reviewer_count=2,
        minimum_independence="independent",
        require_high_risk_reviewer=True,
        prohibit_coder_identity=True,
        distinct_failure_domains_required=True,
        rationale_codes=("x",),
    )
    assert req.reviewer_count == 2


def test_override_cannot_weaken_review_count() -> None:
    policy = RiskReviewPolicy(
        overrides={
            RiskLevel.R3_HIGH.name: {
                "reviewer_count": 0,
                "minimum_independence": "same_provider",
                "require_high_risk_reviewer": False,
                "prohibit_coder_identity": False,
                "distinct_failure_domains_required": False,
                "rationale_codes": [],
            }
        }
    )
    req = policy.requirement_for(RiskLevel.R3_HIGH)
    assert req.reviewer_count == 2
    assert req.minimum_independence == "independent"
    assert req.require_high_risk_reviewer
    assert req.prohibit_coder_identity
    assert req.distinct_failure_domains_required
