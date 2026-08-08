"""Authority validation: ensure protected authority files are intact.

This test enforces the invariant that the authoritative roadmap and project
state files are present, valid, and unmodified relative to the recorded
authority metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
_ROADMAP_PATH = _PROJECT_ROOT / "docs" / "OMNIFORGE_FULL_ROADMAP_v1.0.md"
_PROJECT_STATE_PATH = _PROJECT_ROOT / "docs" / "PROJECT_STATE.json"
_AUTHORITY_PATH = _PROJECT_ROOT / "docs" / "ROADMAP_AUTHORITY.json"


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 after normalizing line endings to LF (repository standard)."""
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


@pytest.mark.authority
def test_roadmap_authority_matches_recorded_hash() -> None:
    """The authoritative roadmap must match the SHA-256 recorded in authority metadata."""
    assert _ROADMAP_PATH.exists(), "Authoritative roadmap is missing"
    assert _AUTHORITY_PATH.exists(), "Roadmap authority metadata is missing"

    authority = json.loads(_AUTHORITY_PATH.read_text(encoding="utf-8"))
    recorded_hash = authority.get("sha256")
    actual_hash = _sha256_file(_ROADMAP_PATH)

    assert recorded_hash == actual_hash, (
        f"Roadmap SHA-256 mismatch: recorded={recorded_hash}, actual={actual_hash}. "
        "The authoritative roadmap must not be modified outside the authority process."
    )


@pytest.mark.authority
def test_project_state_is_valid_json() -> None:
    """The project state file must be present and parseable JSON."""
    assert _PROJECT_STATE_PATH.exists(), "Project state is missing"
    state = json.loads(_PROJECT_STATE_PATH.read_text(encoding="utf-8"))
    assert "schema_version" in state
    assert "roadmap_file" in state


@pytest.mark.authority
def test_authority_metadata_is_valid_json() -> None:
    """The roadmap authority metadata must be present and parseable JSON."""
    assert _AUTHORITY_PATH.exists(), "Roadmap authority metadata is missing"
    authority = json.loads(_AUTHORITY_PATH.read_text(encoding="utf-8"))
    assert authority.get("roadmap_path") == "docs/OMNIFORGE_FULL_ROADMAP_v1.0.md"
    assert authority.get("roadmap_version") == "1.0"
