from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pytest

from src.persistence import configuration, runtime_state
from src.security.redaction import contains_secret


class ConfigurationVersioningTests(unittest.TestCase):
    def setUp(self) -> None:
        configuration._MIGRATIONS.clear()
        # Re-register the production Phase 5, Phase 8, and Phase 9 migrations after clearing.
        configuration.register_migration("1.0.0", "1.1.0")(configuration._migrate_1_0_0_to_1_1_0)
        configuration.register_migration("1.1.0", "1.2.0")(configuration._migrate_1_1_0_to_1_2_0)
        configuration.register_migration("1.2.0", "1.3.0")(configuration._migrate_1_2_0_to_1_3_0)

    def valid_config(self) -> dict:
        return {
            "schema_version": "1.3.0",
            "routing_mode": "legacy",
            "exploration_enabled": False,
            "providers": {"openai": {"enabled": True, "models": {}, "routes": {}}},
            "pins": {},
            "project_policies": {},
            "router_config": {},
            "risk_policy": {},
        }

    def test_current_version_is_accepted(self) -> None:
        result = configuration.validate_config(self.valid_config())
        self.assertEqual("1.3.0", result["schema_version"])

    def test_legacy_config_migrates_to_current_version(self) -> None:
        legacy = {
            "schema_version": "1.0.0",
            "routing_mode": "legacy",
            "exploration_enabled": False,
            "providers": {"openai": {"enabled": True}},
        }
        result = configuration.validate_config(legacy)
        self.assertEqual("1.3.0", result["schema_version"])
        self.assertIn("models", result["providers"]["openai"])
        self.assertIn("routes", result["providers"]["openai"])
        self.assertIn("pins", result)
        self.assertIn("project_policies", result)
        self.assertIn("router_config", result)
        self.assertIn("risk_policy", result)

    def test_missing_version_fails_closed(self) -> None:
        config = self.valid_config()
        del config["schema_version"]
        with self.assertRaises(configuration.InvalidConfiguration):
            configuration.validate_config(config)

    def test_unknown_future_version_fails_closed(self) -> None:
        config = self.valid_config()
        config["schema_version"] = "99.0.0"
        with self.assertRaises(configuration.UnsupportedConfigVersion):
            configuration.validate_config(config)

    def test_registered_migration_hook_is_applied_without_mutating_input(self) -> None:
        @configuration.register_migration("0.9.0", "1.0.0")
        def migrate(old: dict) -> dict:
            old.setdefault("providers", {})
            old.setdefault("routing_mode", "legacy")
            return old

        original = {"schema_version": "0.9.0"}
        result = configuration.validate_config(original)
        self.assertEqual({"schema_version": "0.9.0"}, original)
        self.assertEqual("1.3.0", result["schema_version"])
        self.assertEqual("legacy", result["routing_mode"])

    def test_migration_cycle_fails_closed(self) -> None:
        @configuration.register_migration("0.8.0", "0.9.0")
        def first(old: dict) -> dict:
            return old

        @configuration.register_migration("0.9.0", "0.8.0")
        def second(old: dict) -> dict:
            return old

        with self.assertRaises(configuration.UnsupportedConfigVersion):
            configuration.migrate_config({"schema_version": "0.8.0"})

    def test_load_config_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{not-json", encoding="utf-8")
            with self.assertRaises(configuration.InvalidConfiguration):
                configuration.load_config(path)

    def test_load_config_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(self.valid_config()), encoding="utf-8")
            result = configuration.load_config(path)
            self.assertEqual("legacy", result["routing_mode"])

    def test_model_enable_disable_validated(self) -> None:
        config = self.valid_config()
        config["providers"]["openai"]["models"] = {
            "gpt-4o": {"enabled": True},
            "gpt-4o-mini": {"enabled": False},
        }
        result = configuration.validate_config(config)
        self.assertTrue(result["providers"]["openai"]["models"]["gpt-4o"]["enabled"])
        self.assertFalse(result["providers"]["openai"]["models"]["gpt-4o-mini"]["enabled"])

    def test_route_enable_disable_validated(self) -> None:
        config = self.valid_config()
        config["providers"]["openai"]["routes"] = {
            "openai-direct": {"enabled": True},
            "openrouter-openai": {"enabled": False},
        }
        result = configuration.validate_config(config)
        self.assertTrue(result["providers"]["openai"]["routes"]["openai-direct"]["enabled"])
        self.assertFalse(result["providers"]["openai"]["routes"]["openrouter-openai"]["enabled"])

    def test_invalid_model_enabled_rejected(self) -> None:
        config = self.valid_config()
        config["providers"]["openai"]["models"] = {"gpt-4o": {"enabled": "yes"}}
        with self.assertRaises(configuration.InvalidConfiguration):
            configuration.validate_config(config)

    def test_invalid_route_enabled_rejected(self) -> None:
        config = self.valid_config()
        config["providers"]["openai"]["routes"] = {"openai-direct": {"enabled": "yes"}}
        with self.assertRaises(configuration.InvalidConfiguration):
            configuration.validate_config(config)

    def test_raw_api_key_rejected(self) -> None:
        config = self.valid_config()
        config["providers"]["openai"]["api_key"] = "sk-secret"
        with self.assertRaises(configuration.RawSecretInConfigurationError):
            configuration.validate_config(config)

    def test_raw_secret_key_rejected(self) -> None:
        config = self.valid_config()
        config["providers"]["openai"]["secret_key"] = "secret"
        with self.assertRaises(configuration.RawSecretInConfigurationError):
            configuration.validate_config(config)

    def test_credential_ref_allowed(self) -> None:
        config = self.valid_config()
        config["providers"]["openai"]["credential_ref"] = {
            "source": "environment",
            "name": "OPENAI_API_KEY",
        }
        result = configuration.validate_config(config)
        self.assertEqual("environment", result["providers"]["openai"]["credential_ref"]["source"])

    def test_credential_ref_requires_name(self) -> None:
        config = self.valid_config()
        config["providers"]["openai"]["credential_ref"] = {"source": "environment"}
        with self.assertRaises(configuration.SecretValidationError):
            configuration.validate_config(config)

    def test_pins_validated(self) -> None:
        config = self.valid_config()
        config["pins"] = {
            "debug": {"provider_id": "openai", "model_id": "gpt-4o"},
        }
        result = configuration.validate_config(config)
        self.assertEqual("openai", result["pins"]["debug"]["provider_id"])

    def test_pin_rejects_unknown_field(self) -> None:
        config = self.valid_config()
        config["pins"] = {"debug": {"provider_id": "openai", "extra": "x"}}
        with self.assertRaises(configuration.InvalidConfiguration):
            configuration.validate_config(config)

    def test_project_policies_validated(self) -> None:
        config = self.valid_config()
        config["project_policies"] = {
            "project-x": {
                "prohibited_provider_ids": ["xai"],
                "allowed_deployment_modes": ["cloud"],
                "minimum_review_independence": "independent",
                "allow_exploration": False,
                "routing_mode_override": "legacy",
            }
        }
        result = configuration.validate_config(config)
        prohibited = result["project_policies"]["project-x"]["prohibited_provider_ids"]
        self.assertEqual(["xai"], prohibited)

    def test_project_policy_rejects_invalid_routing_mode_override(self) -> None:
        config = self.valid_config()
        config["project_policies"] = {"project-x": {"routing_mode_override": "magic"}}
        with self.assertRaises(configuration.InvalidConfiguration):
            configuration.validate_config(config)

    def test_invalid_routing_mode_rejected(self) -> None:
        config = self.valid_config()
        config["routing_mode"] = "magic"
        with self.assertRaises(configuration.InvalidConfiguration):
            configuration.validate_config(config)

    def test_dynamic_routing_mode_accepted(self) -> None:
        config = self.valid_config()
        config["routing_mode"] = "dynamic"
        result = configuration.validate_config(config)
        self.assertEqual("dynamic", result["routing_mode"])

    def test_exploration_enabled_round_trips(self) -> None:
        config = self.valid_config()
        config["exploration_enabled"] = True
        result = configuration.validate_config(config)
        self.assertTrue(result["exploration_enabled"])

    def test_extract_administrative_state_includes_disablements(self) -> None:
        config = self.valid_config()
        config["providers"]["openai"]["enabled"] = False
        config["providers"]["openai"]["models"] = {"gpt-4o": {"enabled": False}}
        config["providers"]["openai"]["routes"] = {"openai-direct": {"enabled": False}}
        state = configuration.extract_administrative_state(config)
        self.assertFalse(state["provider_status"]["openai"]["enabled"])
        self.assertFalse(state["model_status"]["gpt-4o"]["enabled"])
        self.assertFalse(state["route_status"]["openai-direct"]["enabled"])
        self.assertFalse(state["exploration_enabled"])

    def test_extract_administrative_state_has_no_credential_values(self) -> None:
        config = self.valid_config()
        config["providers"]["openai"]["credential_ref"] = {
            "source": "environment",
            "name": "OPENAI_API_KEY",
        }
        state = configuration.extract_administrative_state(config)
        text = json.dumps(state, sort_keys=True)
        self.assertNotIn("sk-", text)
        # Administrative state contains enablement/policy only, not credential refs.
        self.assertNotIn("OPENAI_API_KEY", text)

    def test_project_policy_round_trips_through_runtime_state(self) -> None:
        config = self.valid_config()
        config["project_policies"] = {
            "project-a": {
                "prohibited_provider_ids": ["xai"],
                "prohibited_model_ids": ["some-model"],
                "prohibited_route_ids": ["some-route"],
                "allowed_deployment_modes": ["cloud"],
                "minimum_review_independence": "independent",
                "allow_exploration": False,
                "routing_mode_override": "legacy",
            }
        }
        validated = configuration.validate_config(config)
        policy = validated["project_policies"]["project-a"]
        self.assertIsInstance(policy, dict)
        self.assertEqual(["xai"], policy["prohibited_provider_ids"])

        state = configuration.extract_administrative_state(config)
        self.assertEqual("legacy", state["routing_mode"])
        self.assertEqual(["xai"], state["project_policies"]["project-a"]["prohibited_provider_ids"])

        state["run_id"] = "run-test"
        state["workflow_state"] = "STOPPED"
        state["checkpoint"] = {}
        validated_state = runtime_state.validate_runtime_state(state)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.json"
            runtime_state.save_runtime_state(path, validated_state)
            loaded = runtime_state.load_runtime_state(path)

        self.assertEqual(
            ["xai"],
            loaded["project_policies"]["project-a"]["prohibited_provider_ids"],
        )
        self.assertEqual(
            "independent",
            loaded["project_policies"]["project-a"]["minimum_review_independence"],
        )

    def test_secret_reference_with_extra_fields_is_rejected(self) -> None:
        config = self.valid_config()
        config["providers"]["openai"]["credential_ref"] = {
            "source": "environment",
            "name": "OPENAI_API_KEY",
            "extra": "forbidden",
        }
        with self.assertRaises(configuration.SecretValidationError):
            configuration.validate_config(config)

    def test_secret_reference_with_sensitive_extra_field_is_rejected(self) -> None:
        config = self.valid_config()
        config["providers"]["openai"]["credential_ref"] = {
            "source": "environment",
            "name": "OPENAI_API_KEY",
            "password": "OMNIFORGE_TEST_SECRET_SENTINEL_config_1",
        }
        with self.assertRaises(configuration.InvalidConfiguration):
            configuration.validate_config(config)

    def test_secret_reference_metadata_cannot_smuggle_secrets(self) -> None:
        config = self.valid_config()
        config["providers"]["openai"]["credential_ref"] = {
            "source": "environment",
            "name": "OPENAI_API_KEY",
            "metadata": {"password": "OMNIFORGE_TEST_SECRET_SENTINEL_config_2"},
        }
        with self.assertRaises(configuration.InvalidConfiguration):
            configuration.validate_config(config)


SENTINEL = "OMNIFORGE_TEST_SECRET_SENTINEL_config_3"


def test_runtime_state_save_contains_no_credential_sentinel() -> None:
    config = {
        "schema_version": "1.3.0",
        "routing_mode": "legacy",
        "exploration_enabled": False,
        "providers": {
            "openai": {
                "enabled": True,
                "credential_ref": {"source": "environment", "name": "OPENAI_API_KEY"},
                "models": {},
                "routes": {},
            }
        },
        "pins": {},
        "project_policies": {
            "project-a": {
                "prohibited_provider_ids": ["xai"],
                "allowed_deployment_modes": ["cloud"],
                "minimum_review_independence": "independent",
                "allow_exploration": False,
                "routing_mode_override": "legacy",
            }
        },
        "router_config": {},
        "risk_policy": {},
    }
    state = configuration.extract_administrative_state(config)
    state["run_id"] = "run-test"
    state["workflow_state"] = "STOPPED"
    state["checkpoint"] = {"diagnostic": f"error password={SENTINEL}", "password": SENTINEL}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runtime.json"
        runtime_state.save_runtime_state(path, state)
        text = path.read_text(encoding="utf-8")
        assert not contains_secret(text, SENTINEL)
        assert "<redacted>" in text


def test_sensitive_key_variants_are_rejected_in_config() -> None:
    for key in ["api-key", "apikey", "API_KEY", "secret-key", "access-token"]:
        config = {
            "schema_version": "1.3.0",
            "routing_mode": "legacy",
            "exploration_enabled": False,
            "providers": {"openai": {"enabled": True, key: SENTINEL}},
            "pins": {},
            "project_policies": {},
            "router_config": {},
            "risk_policy": {},
        }
        with pytest.raises(configuration.RawSecretInConfigurationError):
            configuration.validate_config(config)


if __name__ == "__main__":
    unittest.main()
