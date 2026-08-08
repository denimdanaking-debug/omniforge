from __future__ import annotations

import unittest

from src.policy.risk import (
    ContextDepth,
    RiskAssessment,
    RiskLevel,
    lifecycle_eligible,
    policy_for,
)
from src.routing.model_identity import ModelLifecycle


class RiskPolicyTests(unittest.TestCase):
    def test_authoritative_five_level_taxonomy_is_ordered(self) -> None:
        self.assertEqual(
            [0, 1, 2, 3, 4],
            [
                RiskLevel.R0_TRIVIAL,
                RiskLevel.R1_LOW,
                RiskLevel.R2_NORMAL,
                RiskLevel.R3_HIGH,
                RiskLevel.R4_CRITICAL_AUTHORITY,
            ],
        )

    def test_runtime_risk_can_escalate(self) -> None:
        assessment = RiskAssessment(RiskLevel.R1_LOW, ("initial low-risk task",))
        escalated = assessment.escalate(RiskLevel.R3_HIGH, "authority file touched")
        self.assertEqual(RiskLevel.R3_HIGH, escalated.level)
        self.assertIn("authority file touched", escalated.reasons)

    def test_runtime_risk_cannot_downgrade(self) -> None:
        assessment = RiskAssessment(RiskLevel.R3_HIGH)
        with self.assertRaises(ValueError):
            assessment.escalate(RiskLevel.R1_LOW, "looks easier now")

    def test_high_risk_changes_review_experiment_and_context_policy(self) -> None:
        normal = policy_for(RiskLevel.R2_NORMAL)
        high = policy_for(RiskLevel.R3_HIGH)
        critical = policy_for(RiskLevel.R4_CRITICAL_AUTHORITY)

        self.assertEqual(1, normal.required_reviewers)
        self.assertEqual(2, high.required_reviewers)
        self.assertFalse(high.exploration_allowed)
        self.assertEqual(ContextDepth.BROAD, high.context_depth)
        self.assertEqual(ContextDepth.AUTHORITY_PRIMARY, critical.context_depth)

    def test_risk_controls_model_lifecycle_eligibility(self) -> None:
        self.assertTrue(lifecycle_eligible(ModelLifecycle.LOW_RISK, RiskLevel.R1_LOW))
        self.assertFalse(lifecycle_eligible(ModelLifecycle.LOW_RISK, RiskLevel.R2_NORMAL))
        self.assertTrue(lifecycle_eligible(ModelLifecycle.NORMAL, RiskLevel.R2_NORMAL))
        self.assertFalse(lifecycle_eligible(ModelLifecycle.NORMAL, RiskLevel.R3_HIGH))
        self.assertTrue(
            lifecycle_eligible(ModelLifecycle.HIGH_RISK, RiskLevel.R4_CRITICAL_AUTHORITY)
        )

    def test_shadow_and_disabled_never_execute_authoritative_work(self) -> None:
        for lifecycle in (ModelLifecycle.SHADOW, ModelLifecycle.DISABLED):
            for level in RiskLevel:
                self.assertFalse(lifecycle_eligible(lifecycle, level))

    def test_low_risk_allows_controlled_exploration(self) -> None:
        self.assertTrue(policy_for(RiskLevel.R0_TRIVIAL).exploration_allowed)
        self.assertTrue(policy_for(RiskLevel.R1_LOW).exploration_allowed)
        self.assertFalse(policy_for(RiskLevel.R2_NORMAL).exploration_allowed)


if __name__ == "__main__":
    unittest.main()
