from __future__ import annotations

import unittest

from src.policy.risk import RiskLevel
from src.routing.roles import (
    ExecutionRole,
    RolePerformance,
    RolePerformanceRegistry,
    RoutingRequest,
)


class ExecutionRoleTests(unittest.TestCase):
    def test_all_authoritative_roles_are_formalized(self) -> None:
        self.assertEqual(
            {
                "planning",
                "architecture",
                "coding",
                "debugging",
                "repair",
                "review",
                "high_risk_review",
                "arbitration",
                "context_analysis",
                "integration_analysis",
            },
            {role.value for role in ExecutionRole},
        )

    def test_routing_request_requires_explicit_role_and_risk(self) -> None:
        request = RoutingRequest(
            task_id="task-1", role=ExecutionRole.CODING, risk=RiskLevel.R2_NORMAL
        )
        self.assertEqual(ExecutionRole.CODING, request.role)
        self.assertEqual(RiskLevel.R2_NORMAL, request.risk)
        with self.assertRaises(TypeError):
            RoutingRequest(task_id="task-2", risk=RiskLevel.R1_LOW)  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            RoutingRequest(task_id="task-3", role=ExecutionRole.CODING)  # type: ignore[call-arg]

    def test_model_performance_isolated_by_role(self) -> None:
        registry = RolePerformanceRegistry()
        registry.set(
            "model-a",
            ExecutionRole.CODING,
            RolePerformance(attempts=100, accepted=95, first_pass_accepted=90),
        )
        registry.set(
            "model-a",
            ExecutionRole.REVIEW,
            RolePerformance(attempts=100, accepted=60, first_pass_accepted=50),
        )
        self.assertEqual(95, registry.get("model-a", ExecutionRole.CODING).accepted)
        self.assertEqual(60, registry.get("model-a", ExecutionRole.REVIEW).accepted)

    def test_missing_role_profile_does_not_borrow_another_role(self) -> None:
        registry = RolePerformanceRegistry()
        registry.set("model-a", ExecutionRole.CODING, RolePerformance(attempts=10, accepted=9))
        self.assertEqual(RolePerformance(), registry.get("model-a", ExecutionRole.PLANNING))

    def test_role_performance_rejects_impossible_counts(self) -> None:
        with self.assertRaises(ValueError):
            RolePerformance(attempts=1, accepted=2)
        with self.assertRaises(ValueError):
            RolePerformance(attempts=2, accepted=1, first_pass_accepted=2)

    def test_roles_for_model_are_deterministic(self) -> None:
        registry = RolePerformanceRegistry()
        registry.set("m", ExecutionRole.REVIEW, RolePerformance(attempts=1))
        registry.set("m", ExecutionRole.CODING, RolePerformance(attempts=2))
        self.assertEqual(
            [ExecutionRole.CODING, ExecutionRole.REVIEW],
            list(registry.roles_for_model("m")),
        )


if __name__ == "__main__":
    unittest.main()
