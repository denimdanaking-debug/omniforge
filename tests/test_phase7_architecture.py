"""Architectural enforcement tests for Phase 7 context construction engine."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.context.budget import ContextBudget
from src.context.hybrid import HybridContextStrategy
from src.context.schema import AuthorityPresence, ContextPacket
from src.context.strategy import ContextBuildRequest
from src.policy.risk import RiskLevel
from src.routing.capabilities import ModelCapabilities
from src.routing.roles import ExecutionRole
from src.security.redaction import contains_secret, redact

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "OMNIFORGE_TEST_SECRET_SENTINEL_PHASE7_ARCH_999"


def _source_files(glob: str) -> list[Path]:
    return sorted(ROOT.glob(glob))


@pytest.mark.architecture
def test_context_modules_do_not_branch_on_provider() -> None:
    """Context construction must remain provider-agnostic."""
    violations: list[str] = []
    providers = {
        "anthropic",
        "openai",
        "google",
        "gemini",
        "kimi",
        "qwen",
        "mistral",
        "xai",
        "minimax",
    }
    for path in _source_files("src/context/**/*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for comparator in ast.walk(node):
                    if (
                        isinstance(comparator, ast.Constant)
                        and isinstance(comparator.value, str)
                        and comparator.value.lower() in providers
                    ):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.lower()
                if lowered in providers and "provider" in lowered:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, "context modules must not branch on provider: " + "; ".join(
        violations[:10]
    )


@pytest.mark.architecture
def test_summary_does_not_replace_required_authority() -> None:
    """Hybrid strategy keeps authority raw and never replaces it with a summary."""
    strategy = HybridContextStrategy()
    request = ContextBuildRequest(
        task_id="task-1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R3_HIGH,
        model_capabilities=ModelCapabilities(context_tokens=4096),
        authority_refs=("roadmap.md",),
        changed_files=("src/a.py",),
        constraints=("must pass tests",),
        explicit_paths=("src/b.py",),
        referenced_symbols=("func_a",),
        budget=ContextBudget(primary_budget=10_000),
    )
    result = strategy.build(request)
    assert result.packet.authority_presence == AuthorityPresence.RAW_INCLUDED
    assert result.packet.authority == ("roadmap.md",)
    assert result.packet.summary_count > 0
    for item in result.packet.authority:
        assert "summary" not in item.lower()


@pytest.mark.architecture
def test_context_packet_has_deterministic_content_hash() -> None:
    packet = ContextPacket(
        packet_id="p",
        authority=("a",),
        provenance_index={
            "p1": __import__("src.context.schema", fromlist=["ProvenanceRef"]).ProvenanceRef(
                "git", path="a.py"
            )
        },
    )
    assert packet.content_hash() == packet.content_hash()


@pytest.mark.architecture
def test_no_live_calls_in_context_tests() -> None:
    """Normal context tests must not perform live network/LLM calls."""
    for path in _source_files("tests/context/**/*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in {
                    "requests",
                    "httpx",
                    "openai",
                    "anthropic",
                }:
                    raise AssertionError(f"live call in {path.relative_to(ROOT)}:{node.lineno}")


@pytest.mark.architecture
def test_redaction_sentinel_not_in_context_payload() -> None:
    packet = ContextPacket(
        packet_id="p",
        authority=(f"error bearer {SENTINEL}",),
    )
    data = packet.to_dict()
    safe_data = redact(data)
    serialized = json.dumps(safe_data, sort_keys=True)
    assert not contains_secret(serialized, SENTINEL)


@pytest.mark.architecture
def test_authority_files_unchanged() -> None:
    """Phase 7 must not mutate the pinned authority files."""
    import subprocess

    result = subprocess.run(
        [
            "git",
            "diff",
            "origin/main",
            "--",
            "docs/OMNIFORGE_FULL_ROADMAP_v1.0.md",
            "docs/PROJECT_STATE.json",
            "docs/ROADMAP_AUTHORITY.json",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.stdout == "", "authority files must not be modified"


@pytest.mark.architecture
def test_no_phase8_scoring_code() -> None:
    """Phase 7 implementation must not include Phase 8 router scoring."""
    forbidden = ["expected_success", "role_fit", "risk_fit", "empirical_reliability"]
    violations: list[str] = []
    for path in _source_files("src/context/**/*.py"):
        source = path.read_text(encoding="utf-8").lower()
        for phrase in forbidden:
            if phrase in source:
                violations.append(f"{path.relative_to(ROOT)}: {phrase}")
    assert not violations, "Phase 8 scoring code must not appear: " + "; ".join(violations[:10])
