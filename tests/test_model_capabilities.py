from __future__ import annotations

import unittest

from src.routing.capabilities import (
    CapabilityError,
    CapabilityRequirement,
    CostMetadata,
    DeploymentMode,
    ModelCapabilities,
    RateMetadata,
    filter_capable_models,
    match_capabilities,
)


class ModelCapabilityTests(unittest.TestCase):
    def capable(self) -> ModelCapabilities:
        return ModelCapabilities(
            context_tokens=1_000_000,
            structured_output=True,
            tool_use=True,
            streaming=True,
            reasoning=True,
            code_generation=True,
            multimodal=True,
            deployment_mode=DeploymentMode.CLOUD,
            cost=CostMetadata(input_per_million=1.0, output_per_million=3.0),
            rate=RateMetadata(max_concurrency=8, requests_per_minute=120),
            supported_roles=frozenset({"planning", "coding", "review"}),
        )

    def test_full_requirements_match(self) -> None:
        requirement = CapabilityRequirement(
            min_context_tokens=500_000,
            structured_output=True,
            tool_use=True,
            reasoning=True,
            code_generation=True,
            multimodal=True,
            allowed_deployment_modes=frozenset({DeploymentMode.CLOUD}),
            required_roles=frozenset({"coding"}),
        )
        result = match_capabilities(self.capable(), requirement)
        self.assertTrue(result.eligible)
        self.assertEqual((), result.missing)

    def test_unsupported_features_fail_hard_eligibility(self) -> None:
        limited = ModelCapabilities(
            context_tokens=128_000,
            structured_output=False,
            tool_use=False,
            code_generation=True,
            supported_roles=frozenset({"coding"}),
        )
        requirement = CapabilityRequirement(
            min_context_tokens=256_000,
            structured_output=True,
            tool_use=True,
            reasoning=True,
        )
        result = match_capabilities(limited, requirement)
        self.assertFalse(result.eligible)
        self.assertEqual(
            ("context_tokens", "structured_output", "tool_use", "reasoning"),
            result.missing,
        )

    def test_required_role_is_a_hard_capability(self) -> None:
        result = match_capabilities(
            self.capable(), CapabilityRequirement(required_roles=frozenset({"arbitration"}))
        )
        self.assertFalse(result.eligible)
        self.assertEqual(("role:arbitration",), result.missing)

    def test_router_filter_is_deterministic(self) -> None:
        candidates = {
            "model-b": ModelCapabilities(context_tokens=32_000, code_generation=True),
            "model-a": self.capable(),
            "model-c": ModelCapabilities(context_tokens=500_000, code_generation=True),
        }
        eligible = filter_capable_models(
            candidates,
            CapabilityRequirement(min_context_tokens=100_000, code_generation=True),
        )
        self.assertEqual(("model-a", "model-c"), eligible)

    def test_cost_and_rate_metadata_reject_invalid_values(self) -> None:
        with self.assertRaises(CapabilityError):
            CostMetadata(input_per_million=-0.1)
        with self.assertRaises(CapabilityError):
            RateMetadata(max_concurrency=0)

    def test_context_size_must_be_positive(self) -> None:
        with self.assertRaises(CapabilityError):
            ModelCapabilities(context_tokens=0)


if __name__ == "__main__":
    unittest.main()
