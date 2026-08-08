from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.persistence import runtime_state


class RuntimeStateVersioningTests(unittest.TestCase):
    def setUp(self) -> None:
        runtime_state._RUNTIME_MIGRATIONS.clear()

    def valid_state(self) -> dict:
        return {
            "schema_version": "1.0.0",
            "run_id": "run-123",
            "workflow_state": "EXECUTING",
            "checkpoint": {"phase": "0", "step": "0.5", "task": "persist"},
            "updated_at": None,
            "next_wakeup_at": None,
            "last_error": None,
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

    def test_explicit_migration_is_applied(self) -> None:
        @runtime_state.register_runtime_migration("0.9.0", "1.0.0")
        def migrate(old: dict) -> dict:
            old.setdefault("checkpoint", {})
            old.setdefault("workflow_state", "STOPPED")
            return old

        result = runtime_state.validate_runtime_state(
            {"schema_version": "0.9.0", "run_id": "legacy"}
        )
        self.assertEqual("1.0.0", result["schema_version"])
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


if __name__ == "__main__":
    unittest.main()
