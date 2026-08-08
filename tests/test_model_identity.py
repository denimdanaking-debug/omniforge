from __future__ import annotations

import unittest

from src.routing.model_identity import (
    ModelIdentity,
    ModelIdentityError,
    ModelLifecycle,
    ModelRegistry,
    ModelReputation,
)


class ModelIdentityTests(unittest.TestCase):
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            model_id="kimi:k3-code",
            family="kimi-k3",
            version="k3",
            revision="2026-08",
            capability_metadata={"context_tokens": 1000000, "tool_use": True},
            lifecycle=ModelLifecycle.HIGH_RISK,
        )

    def test_identity_carries_family_version_revision_capabilities_and_lifecycle(self) -> None:
        identity = self.identity()
        self.assertEqual("kimi-k3", identity.family)
        self.assertEqual("k3", identity.version)
        self.assertEqual("2026-08", identity.revision)
        self.assertEqual(1000000, identity.capability_metadata["context_tokens"])
        self.assertEqual(ModelLifecycle.HIGH_RISK, identity.lifecycle)

    def test_invalid_model_id_is_rejected(self) -> None:
        with self.assertRaises(ModelIdentityError):
            ModelIdentity(model_id="Kimi K3", family="kimi")

    def test_same_model_id_cannot_be_rebound_to_different_identity(self) -> None:
        registry = ModelRegistry()
        registry.register(self.identity())
        with self.assertRaises(ModelIdentityError):
            registry.register(
                ModelIdentity(
                    model_id="kimi:k3-code",
                    family="different-family",
                    lifecycle=ModelLifecycle.NORMAL,
                )
            )

    def test_reputation_changes_do_not_change_identity(self) -> None:
        registry = ModelRegistry()
        identity = self.identity()
        registry.register(identity)
        registry.set_reputation(
            identity.model_id,
            ModelReputation(attempts=10, accepted=8, authority_violations=0, score_hint=0.8),
        )
        registration = registry.get(identity.model_id)
        self.assertEqual(identity, registration.identity)
        self.assertEqual(10, registration.reputation.attempts)

    def test_lifecycle_can_change_without_resetting_reputation(self) -> None:
        registry = ModelRegistry()
        identity = self.identity()
        registry.register(identity)
        reputation = ModelReputation(attempts=20, accepted=18)
        registry.set_reputation(identity.model_id, reputation)
        registry.set_lifecycle(identity.model_id, ModelLifecycle.NORMAL)
        registration = registry.get(identity.model_id)
        self.assertEqual(ModelLifecycle.NORMAL, registration.identity.lifecycle)
        self.assertEqual(reputation, registration.reputation)

    def test_reputation_validation_rejects_impossible_counts(self) -> None:
        with self.assertRaises(ModelIdentityError):
            ModelReputation(attempts=1, accepted=2)

    def test_registry_is_deterministic(self) -> None:
        registry = ModelRegistry()
        registry.register(self.identity())
        registry.register(
            ModelIdentity(
                model_id="openai:codex",
                family="codex",
                lifecycle=ModelLifecycle.HIGH_RISK,
            )
        )
        self.assertEqual(
            ["kimi:k3-code", "openai:codex"],
            [entry.identity.model_id for entry in registry.registrations()],
        )


if __name__ == "__main__":
    unittest.main()
