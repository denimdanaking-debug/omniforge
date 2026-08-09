from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.persistence import runtime_state


class RuntimeStateVersioningTests(unittest.TestCase):
    def setUp(self) -> None:
        runtime_state._RUNTIME_MIGRATIONS.clear()
        runtime_state.register_runtime_migration("1.0.0", "1.1.0")(
            runtime_state._migrate_runtime_1_0_0_to_1_1_0
        )
        runtime_state.register_runtime_migration("1.1.0", "1.2.0")(
            runtime_state._migrate_runtime_1_1_0_to_1_2_0
        )

    def valid_state(self) -> dict:
        return {
            "schema_version": "1.2.0",
            "run_id": "run-123",
            "workflow_state": "EXECUTING",
            "checkpoint": {"phase": "0", "step": "0.5", "task": "persist"},
            "updated_at": None,
            "next_wakeup_at": None,
            "last_error": None,
            "provider_status": {},
            "model_status": {},
            "route_status": {},
            "routing_mode": "legacy",
            "exploration_enabled": False,
            "pins": {},
            "project_policies": {},
            "provider_recovery_state": {},
            "route_recovery_state": {},
            "failure_domain_index": {},
            "recovery_scheduler": {},
            "waiting_tasks": {},
        }

    def test_state_survives_save_and_restart_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.json"
            runtime_state.save_runtime_state(path, self.valid_state())
            reloaded = runtime_state.load_runtime_state(path)
            self.assertEqual(self.valid_state(), reloaded)

    def test_unknown_version_fails_closed_with_diagnostic(self) -> None:
        state = self.valid_state()
        state["schema_version"] = "99.0.0"
        with self.assertRaises(runtime_state.UnsupportedRuntimeStateVersion) as caught:
            runtime_state.validate_runtime_state(state)
        self.assertEqual("UNSUPPORTED_SCHEMA_VERSION", caught.exception.diagnostic.code)
        self.assertTrue(caught.exception.diagnostic.recoverable)

    def test_corrupt_json_fails_closed_and_preserves_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaises(runtime_state.CorruptRuntimeState) as caught:
                runtime_state.load_runtime_state(path)
            self.assertEqual("CORRUPT_JSON", caught.exception.diagnostic.code)
            self.assertEqual(str(path), caught.exception.diagnostic.path)
            self.assertEqual("{broken", path.read_text(encoding="utf-8"))

    def test_invalid_workflow_state_is_rejected(self) -> None:
        state = self.valid_state()
        state["workflow_state"] = "MAGIC"
        with self.assertRaises(runtime_state.CorruptRuntimeState) as caught:
            runtime_state.validate_runtime_state(state)
        self.assertEqual("INVALID_WORKFLOW_STATE", caught.exception.diagnostic.code)

    def test_legacy_state_migrates_to_current_version(self) -> None:
        legacy = {
            "schema_version": "1.0.0",
            "run_id": "run-123",
            "workflow_state": "EXECUTING",
            "checkpoint": {},
        }
        result = runtime_state.validate_runtime_state(legacy)
        self.assertEqual("1.2.0", result["schema_version"])
        self.assertEqual("legacy", result["routing_mode"])
        self.assertFalse(result["exploration_enabled"])
        self.assertIn("provider_status", result)
        self.assertIn("model_status", result)
        self.assertIn("route_status", result)
        self.assertIn("pins", result)
        self.assertIn("project_policies", result)
        self.assertIn("provider_recovery_state", result)
        self.assertIn("route_recovery_state", result)
        self.assertIn("failure_domain_index", result)
        self.assertIn("recovery_scheduler", result)
        self.assertIn("waiting_tasks", result)

    def test_explicit_migration_is_applied(self) -> None:
        @runtime_state.register_runtime_migration("0.9.0", "1.0.0")
        def migrate(old: dict) -> dict:
            old.setdefault("checkpoint", {})
            old.setdefault("workflow_state", "STOPPED")
            return old

        result = runtime_state.validate_runtime_state(
            {"schema_version": "0.9.0", "run_id": "legacy"}
        )
        self.assertEqual("1.2.0", result["schema_version"])
        self.assertEqual("STOPPED", result["workflow_state"])

    def test_migration_cycle_is_rejected(self) -> None:
        @runtime_state.register_runtime_migration("0.8.0", "0.9.0")
        def first(old: dict) -> dict:
            return old

        @runtime_state.register_runtime_migration("0.9.0", "0.8.0")
        def second(old: dict) -> dict:
            return old

        with self.assertRaises(runtime_state.UnsupportedRuntimeStateVersion) as caught:
            runtime_state.migrate_runtime_state({"schema_version": "0.8.0"})
        self.assertEqual("MIGRATION_CYCLE", caught.exception.diagnostic.code)

    def test_invalid_state_is_never_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.json"
            state = self.valid_state()
            state["checkpoint"] = "invalid"
            with self.assertRaises(runtime_state.CorruptRuntimeState):
                runtime_state.save_runtime_state(path, state)
            self.assertFalse(path.exists())

    def test_provider_status_validated(self) -> None:
        state = self.valid_state()
        state["provider_status"] = {"openai": {"enabled": True}, "anthropic": {"enabled": "yes"}}
        with self.assertRaises(runtime_state.CorruptRuntimeState) as caught:
            runtime_state.validate_runtime_state(state)
        self.assertEqual("INVALID_STATUS_ENABLED", caught.exception.diagnostic.code)

    def test_model_and_route_status_validated(self) -> None:
        state = self.valid_state()
        state["model_status"] = {"gpt-4o": {"enabled": True}}
        state["route_status"] = {"openai-direct": {"enabled": False}}
        result = runtime_state.validate_runtime_state(state)
        self.assertTrue(result["model_status"]["gpt-4o"]["enabled"])
        self.assertFalse(result["route_status"]["openai-direct"]["enabled"])

    def test_invalid_routing_mode_rejected(self) -> None:
        state = self.valid_state()
        state["routing_mode"] = "magic"
        with self.assertRaises(runtime_state.CorruptRuntimeState) as caught:
            runtime_state.validate_runtime_state(state)
        self.assertEqual("INVALID_ROUTING_MODE", caught.exception.diagnostic.code)

    def test_dynamic_routing_mode_round_trips(self) -> None:
        state = self.valid_state()
        state["routing_mode"] = "dynamic"
        result = runtime_state.validate_runtime_state(state)
        self.assertEqual("dynamic", result["routing_mode"])

    def test_exploration_flag_round_trips(self) -> None:
        state = self.valid_state()
        state["exploration_enabled"] = True
        result = runtime_state.validate_runtime_state(state)
        self.assertTrue(result["exploration_enabled"])

    def test_invalid_exploration_rejected(self) -> None:
        state = self.valid_state()
        state["exploration_enabled"] = "yes"
        with self.assertRaises(runtime_state.CorruptRuntimeState) as caught:
            runtime_state.validate_runtime_state(state)
        self.assertEqual("INVALID_EXPLORATION", caught.exception.diagnostic.code)

    def test_pins_validated(self) -> None:
        state = self.valid_state()
        state["pins"] = {"debug": {"provider_id": "openai", "model_id": "gpt-4o"}}
        result = runtime_state.validate_runtime_state(state)
        self.assertEqual("openai", result["pins"]["debug"]["provider_id"])

    def test_pin_rejects_unknown_field(self) -> None:
        state = self.valid_state()
        state["pins"] = {"debug": {"provider_id": "openai", "extra": "x"}}
        with self.assertRaises(runtime_state.CorruptRuntimeState) as caught:
            runtime_state.validate_runtime_state(state)
        self.assertEqual("UNKNOWN_PIN_FIELD", caught.exception.diagnostic.code)

    def test_recovery_state_rejects_unknown_health(self) -> None:
        state = self.valid_state()
        state["route_recovery_state"] = {
            "openai-direct": {"health": "banana", "consecutive_failures": 0}
        }
        with self.assertRaises(runtime_state.CorruptRuntimeState) as caught:
            runtime_state.validate_runtime_state(state)
        self.assertEqual("INVALID_RECOVERY_HEALTH", caught.exception.diagnostic.code)

    def test_recovery_state_accepts_all_canonical_healths(self) -> None:
        from src.providers.identity import ProviderHealth

        for health in ProviderHealth:
            state = self.valid_state()
            state["route_recovery_state"] = {
                "route-1": {"health": health.value, "consecutive_failures": 0}
            }
            result = runtime_state.validate_runtime_state(state)
            self.assertEqual(health.value, result["route_recovery_state"]["route-1"]["health"])

    def test_recovery_state_rejects_naive_timestamp(self) -> None:
        state = self.valid_state()
        state["provider_recovery_state"] = {
            "openai": {
                "health": "healthy",
                "consecutive_failures": 0,
                "next_recheck_at": "2026-01-01T12:00:00",
            }
        }
        with self.assertRaises(runtime_state.CorruptRuntimeState) as caught:
            runtime_state.validate_runtime_state(state)
        self.assertEqual("INVALID_RECOVERY_TIMESTAMP", caught.exception.diagnostic.code)

    def test_recovery_state_rejects_disabled_with_next_recheck(self) -> None:
        state = self.valid_state()
        state["route_recovery_state"] = {
            "route-1": {
                "health": "disabled",
                "consecutive_failures": 0,
                "next_recheck_at": "2026-01-01T12:00:00+00:00",
            }
        }
        with self.assertRaises(runtime_state.CorruptRuntimeState) as caught:
            runtime_state.validate_runtime_state(state)
        self.assertEqual("INCONSISTENT_RECOVERY_STATE", caught.exception.diagnostic.code)

    def test_scheduler_rejects_naive_timestamp(self) -> None:
        state = self.valid_state()
        state["recovery_scheduler"] = {"route-1": "2026-01-01T12:00:00"}
        with self.assertRaises(runtime_state.CorruptRuntimeState) as caught:
            runtime_state.validate_runtime_state(state)
        self.assertEqual("INVALID_RECOVERY_TIMESTAMP", caught.exception.diagnostic.code)

    def test_scheduler_rejects_malformed_timestamp(self) -> None:
        state = self.valid_state()
        state["recovery_scheduler"] = {"route-1": "tomorrow sometime"}
        with self.assertRaises(runtime_state.CorruptRuntimeState) as caught:
            runtime_state.validate_runtime_state(state)
        self.assertEqual("INVALID_RECOVERY_TIMESTAMP", caught.exception.diagnostic.code)

    def test_waiting_task_rejects_invalid_role(self) -> None:
        state = self.valid_state()
        state["waiting_tasks"] = {
            "task-1": {
                "task_id": "task-1",
                "role": "wizard",
                "reason": "outage",
                "next_recheck_at": "2026-01-01T12:00:00+00:00",
            }
        }
        with self.assertRaises(runtime_state.CorruptRuntimeState) as caught:
            runtime_state.validate_runtime_state(state)
        self.assertEqual("INVALID_WAITING_TASK_ROLE", caught.exception.diagnostic.code)

    def test_waiting_task_rejects_mismatched_id(self) -> None:
        state = self.valid_state()
        state["waiting_tasks"] = {
            "task-1": {
                "task_id": "task-2",
                "role": "coding",
                "reason": "outage",
                "next_recheck_at": "2026-01-01T12:00:00+00:00",
            }
        }
        with self.assertRaises(runtime_state.CorruptRuntimeState) as caught:
            runtime_state.validate_runtime_state(state)
        self.assertEqual("MISMATCHED_WAITING_TASK_ID", caught.exception.diagnostic.code)

    def test_waiting_task_rejects_naive_timestamp(self) -> None:
        state = self.valid_state()
        state["waiting_tasks"] = {
            "task-1": {
                "task_id": "task-1",
                "role": "coding",
                "reason": "outage",
                "next_recheck_at": "2026-01-01T12:00:00",
            }
        }
        with self.assertRaises(runtime_state.CorruptRuntimeState) as caught:
            runtime_state.validate_runtime_state(state)
        self.assertEqual("INVALID_RECOVERY_TIMESTAMP", caught.exception.diagnostic.code)


if __name__ == "__main__":
    unittest.main()
