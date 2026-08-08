"""Authoritative project contract and immutable execution-cycle snapshots."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_CONTRACT_SCHEMA_VERSION = "1.0.0"


class ProjectContractError(RuntimeError):
    pass


class AuthorityChangedError(ProjectContractError):
    pass


class AdvancementRejected(ProjectContractError):
    pass


@dataclass(frozen=True)
class ProjectContract:
    project_id: str
    roadmap_file: str
    project_state_file: str
    source_branch: str
    integration_branch: str
    validation_profile: tuple[str, ...]
    policy: Mapping[str, Any]


@dataclass(frozen=True)
class AuthoritySnapshot:
    project_id: str
    roadmap_sha256: str
    project_state_sha256: str
    roadmap_step_count: int
    current_phase: str
    current_step: str
    completed_verified_steps: int


@dataclass(frozen=True)
class AdvancementEvidence:
    implemented: bool
    deterministic_validation_passed: bool
    independent_review_satisfied: bool
    safely_integrated: bool
    planner_declared_complete: bool = False


def load_project_contract(raw: Mapping[str, Any]) -> ProjectContract:
    version = raw.get("schema_version")
    if version != PROJECT_CONTRACT_SCHEMA_VERSION:
        raise ProjectContractError(
            f"unsupported project contract schema_version {version!r}; expected {PROJECT_CONTRACT_SCHEMA_VERSION!r}"
        )

    required_strings = (
        "project_id",
        "roadmap_file",
        "project_state_file",
        "source_branch",
        "integration_branch",
    )
    values: dict[str, str] = {}
    for field in required_strings:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ProjectContractError(f"project contract requires non-empty string {field}")
        values[field] = value

    profile = raw.get("validation_profile")
    if not isinstance(profile, list) or not profile or not all(
        isinstance(command, str) and command.strip() for command in profile
    ):
        raise ProjectContractError("validation_profile must contain at least one non-empty command")

    policy = raw.get("policy", {})
    if not isinstance(policy, dict):
        raise ProjectContractError("policy must be an object")

    return ProjectContract(
        project_id=values["project_id"],
        roadmap_file=values["roadmap_file"],
        project_state_file=values["project_state_file"],
        source_branch=values["source_branch"],
        integration_branch=values["integration_branch"],
        validation_profile=tuple(profile),
        policy=dict(policy),
    )


def _read_required(root: Path, relative_path: str) -> bytes:
    path = root / relative_path
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ProjectContractError(f"unable to read authority file {relative_path}: {exc}") from exc


def snapshot_authority(root: str | Path, contract: ProjectContract) -> AuthoritySnapshot:
    """Capture immutable fingerprints and deterministic roadmap position for one cycle."""

    root_path = Path(root)
    roadmap_bytes = _read_required(root_path, contract.roadmap_file)
    state_bytes = _read_required(root_path, contract.project_state_file)

    try:
        roadmap_text = roadmap_bytes.decode("utf-8")
        state = json.loads(state_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectContractError(f"authority files are not valid UTF-8/JSON: {exc}") from exc

    if not isinstance(state, dict):
        raise ProjectContractError("project state root must be an object")

    current_phase = state.get("current_phase")
    current_step = state.get("current_step")
    completed = state.get("completed_verified_steps")
    if not isinstance(current_phase, str) or not isinstance(current_step, str):
        raise ProjectContractError("project state requires string current_phase/current_step")
    if not isinstance(completed, int) or completed < 0:
        raise ProjectContractError("project state requires non-negative completed_verified_steps")

    marker = f"## {current_step} "
    if marker not in roadmap_text:
        raise ProjectContractError(
            f"project state current_step {current_step!r} does not exist in authoritative roadmap"
        )

    step_count = len(re.findall(r"^##\s+\d+\.\d+\s+", roadmap_text, flags=re.MULTILINE))

    return AuthoritySnapshot(
        project_id=contract.project_id,
        roadmap_sha256=hashlib.sha256(roadmap_bytes).hexdigest(),
        project_state_sha256=hashlib.sha256(state_bytes).hexdigest(),
        roadmap_step_count=step_count,
        current_phase=current_phase,
        current_step=current_step,
        completed_verified_steps=completed,
    )


def assert_authority_unchanged(
    root: str | Path, contract: ProjectContract, snapshot: AuthoritySnapshot
) -> None:
    current = snapshot_authority(root, contract)
    if current != snapshot:
        raise AuthorityChangedError(
            "authoritative roadmap/project state changed during the active execution cycle"
        )


def authorize_advancement(evidence: AdvancementEvidence) -> None:
    """Reject advancement unless implementation, validation, review, and integration all passed.

    planner_declared_complete is deliberately informational and can never substitute for evidence.
    """

    missing: list[str] = []
    if not evidence.implemented:
        missing.append("implementation")
    if not evidence.deterministic_validation_passed:
        missing.append("deterministic_validation")
    if not evidence.independent_review_satisfied:
        missing.append("independent_review")
    if not evidence.safely_integrated:
        missing.append("safe_integration")

    if missing:
        raise AdvancementRejected(
            "authority advancement rejected; missing evidence: " + ", ".join(missing)
        )
