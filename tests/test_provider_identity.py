from __future__ import annotations

import unittest

from src.providers.identity import (
    ProviderHealth,
    ProviderIdentity,
    ProviderIdentityError,
    ProviderOperationalState,
    ProviderQuotaState,
    ProviderRegistry,
)


class ProviderIdentityTests(unittest.TestCase):
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            provider_id="openai",
            display_name="OpenAI",
            failure_domain="api.openai.com",
        )

    def test_provider_id_is_stable_and_validated(self) -> None:
        identity = self.identity()
        self.assertEqual("openai", identity.provider_id)
        with self.assertRaises(ProviderIdentityError):
            ProviderIdentity("Open AI", "OpenAI", "api.openai.com")

    def test_conflicting_identity_cannot_rebind_same_provider_id(self) -> None:
        registry = ProviderRegistry()
        registry.register(self.identity())
        with self.assertRaises(ProviderIdentityError):
            registry.register(ProviderIdentity("openai", "Different", "different.example"))

    def test_health_and_quota_are_independent_of_identity(self) -> None:
        registry = ProviderRegistry()
        identity = self.identity()
        registry.register(identity)
        registry.set_operational_state(
            "openai",
            ProviderOperationalState(
                health=ProviderHealth.QUOTA_EXHAUSTED,
                quota=ProviderQuotaState(remaining_fraction=0.0, reset_at="2026-08-09T00:00:00Z"),
            ),
        )
        self.assertEqual(identity, registry.get("openai").identity)
        self.assertEqual(
            ProviderHealth.QUOTA_EXHAUSTED,
            registry.operational_state("openai").health,
        )

    def test_provider_can_expose_multiple_models_and_routes(self) -> None:
        registry = ProviderRegistry()
        registry.register(self.identity())
        registry.attach_model("openai", "gpt-codex-a")
        registry.attach_model("openai", "gpt-codex-b")
        registry.attach_route("openai", "openai-direct")
        registry.attach_route("openai", "openrouter-openai")

        registration = registry.get("openai")
        self.assertEqual(frozenset({"gpt-codex-a", "gpt-codex-b"}), registration.model_ids)
        self.assertEqual(frozenset({"openai-direct", "openrouter-openai"}), registration.route_ids)

    def test_quota_fraction_is_bounded(self) -> None:
        with self.assertRaises(ProviderIdentityError):
            ProviderQuotaState(remaining_fraction=1.1)

    def test_registry_order_is_deterministic(self) -> None:
        registry = ProviderRegistry()
        registry.register(ProviderIdentity("qwen", "Qwen", "dashscope"))
        registry.register(self.identity())
        self.assertEqual(
            ["openai", "qwen"],
            [entry.identity.provider_id for entry in registry.registrations()],
        )


if __name__ == "__main__":
    unittest.main()
