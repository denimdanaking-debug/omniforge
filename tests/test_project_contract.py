from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.orchestration.project_contract import (
    AdvancementEvidence,
    AdvancementRejected,
    AuthorityChangedError,
    ProjectContractError,
    assert_authority_unchanged,
    authorize_advancement,
    load_project_contract,
    snapshot_authority,
)


class ProjectContractTests(unittest.TestCase):
    def contract(self):
        return load_project_contract(
            {
                "schema_version": "1.0.0",
                "project_id": "demo",
                "roadmap_file": "docs/ROADMAP.md",
                "project_state_file": "docs/PROJECT_STATE.json",
                "source_branch": "main",
                "integration_branch": "integration",
                "validation_profile": ["python -m unittest"],
                "policy": {"require_review": True},
            }
        )

    def write_authority(self, root: Path, current_step: str = "0.1") -> None:
        docs = root / "docs"
        docs.mkdir(parents=True)
        (docs / "ROADMAP.md").write_text(
            "# Roadmap\n\n## 0.1 First step\n\n## 0.2 Second step\n",
            encoding="utf-8",
        )
        (docs / "PROJECT_STATE.json").write_text(
            json.dumps(
                {
                    "current_phase": "0",
                    "current_step": current_step,
                    "completed_verified_steps": 0,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_contract_requires_explicit_supported_schema(self) -> None:
        with self.assertRaises(ProjectContractError):
            load_project_contract(
                {
                    "schema_version": "99.0.0",
                    "project_id": "demo",
                    "roadmap_file": "r",
                    "project_state_file": "s",
                    "source_branch": "main",
                    "integration_branch": "integration",
                    "validation_profile": ["test"],
                }
            )

    def test_snapshot_determines_current_roadmap_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_authority(root, "0.2")
            snapshot = snapshot_authority(root, self.contract())
            self.assertEqual("0.2", snapshot.current_step)
            self.assertEqual(2, snapshot.roadmap_step_count)

    def test_snapshot_rejects_state_position_missing_from_roadmap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_authority(root, "9.9")
            with self.assertRaises(ProjectContractError):
                snapshot_authority(root, self.contract())

    def test_authority_snapshot_detects_mid_cycle_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_authority(root)
            contract = self.contract()
            snapshot = snapshot_authority(root, contract)
            (root / "docs" / "ROADMAP.md").write_text(
                "# Roadmap\n\n## 0.1 Changed step\n\n## 0.2 Second step\n",
                encoding="utf-8",
            )
            with self.assertRaises(AuthorityChangedError):
                assert_authority_unchanged(root, contract, snapshot)

    def test_full_evidence_authorizes_advancement(self) -> None:
        authorize_advancement(
            AdvancementEvidence(
                implemented=True,
                deterministic_validation_passed=True,
                independent_review_satisfied=True,
                safely_integrated=True,
                planner_declared_complete=True,
            )
        )

    def test_planner_completion_alone_never_advances_authority(self) -> None:
        with self.assertRaises(AdvancementRejected) as caught:
            authorize_advancement(
                AdvancementEvidence(
                    implemented=False,
                    deterministic_validation_passed=False,
                    independent_review_satisfied=False,
                    safely_integrated=False,
                    planner_declared_complete=True,
                )
            )
        message = str(caught.exception)
        self.assertIn("implementation", message)
        self.assertIn("deterministic_validation", message)
        self.assertIn("independent_review", message)
        self.assertIn("safe_integration", message)

    def test_missing_integration_rejects_advancement_even_if_planner_says_done(self) -> None:
        with self.assertRaises(AdvancementRejected):
            authorize_advancement(
                AdvancementEvidence(
                    implemented=True,
                    deterministic_validation_passed=True,
                    independent_review_satisfied=True,
                    safely_integrated=False,
                    planner_declared_complete=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
