from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.persistence import configuration


class ConfigurationVersioningTests(unittest.TestCase):
    def setUp(self) -> None:
        configuration._MIGRATIONS.clear()

    def valid_config(self) -> dict:
        return {
            "schema_version": "1.0.0",
            "routing_mode": "legacy",
            "exploration_enabled": False,
            "providers": {"openai": {"enabled": True}},
        }

    def test_current_version_is_accepted(self) -> None:
        result = configuration.validate_config(self.valid_config())
        self.assertEqual("1.0.0", result["schema_version"])

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
        self.assertEqual("1.0.0", result["schema_version"])
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


if __name__ == "__main__":
    unittest.main()
