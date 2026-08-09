"""Regression tests for the six context-module repairs.

These tests are provider-neutral and deterministic.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.context.arbitration import ArbitrationEvidencePacket
from src.context.budget import ContextBudget, ContextBudgetError
from src.context.hierarchical import HierarchicalContextStrategy
from src.context.hybrid import HybridContextStrategy
from src.context.large_context import LargeContextStrategy
from src.context.provenance import ProvenanceIndex
from src.context.schema import (
    AcceptanceCriterion,
    ArbitrationContext,
    AuthorityContextItem,
    AuthorityPresence,
    ContextPacket,
    ContextSummary,
    DiffInfo,
    DisputedFinding,
    ProvenanceRef,
    ReviewerPosition,
    TaskMetadata,
    TestEvidence,
)
from src.context.strategy import ContextBuildRequest
from src.context.targeted import TargetedContextStrategy, _provenance_coverage
from src.context.validation import ContextPacketValidator
from src.policy.risk import RiskLevel
from src.routing.capabilities import ModelCapabilities
from src.routing.roles import ExecutionRole
from src.security.secrets import SecretValue

_FAKE_COMMIT = "a" * 40
_FAKE_HASH = "b" * 64


def _targeted_request(
    budget_chars: int = 10_000,
    authority: tuple[str, ...] = ("roadmap.md",),
    authority_entries: tuple[dict[str, Any], ...] = (),
    changed_files: tuple[str, ...] = ("src/a.py",),
    symbols: tuple[str, ...] = ("func_a",),
    constraints: tuple[str, ...] = (),
    max_items: int | None = None,
) -> ContextBuildRequest:
    return ContextBuildRequest(
        task_id="task-1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        model_capabilities=ModelCapabilities(context_tokens=4096),
        authority_refs=authority if not authority_entries else (),
        authority_entries=authority_entries,
        changed_files=changed_files,
        referenced_symbols=symbols,
        constraints=constraints,
        budget=ContextBudget(primary_budget=budget_chars, max_items=max_items),
    )


def _hybrid_request(
    budget_chars: int = 10_000,
    authority: tuple[str, ...] = ("roadmap.md",),
    paths: tuple[str, ...] = ("src/b.py",),
    symbols: tuple[str, ...] = ("func_a",),
) -> ContextBuildRequest:
    return ContextBuildRequest(
        task_id="task-1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        model_capabilities=ModelCapabilities(context_tokens=4096),
        authority_refs=authority,
        changed_files=("src/a.py",),
        constraints=("must pass tests",),
        explicit_paths=paths,
        referenced_symbols=symbols,
        budget=ContextBudget(primary_budget=budget_chars),
    )


def _hierarchical_request(
    budget_chars: int = 10_000,
    authority: tuple[str, ...] = ("roadmap.md",),
    paths: tuple[str, ...] = ("src/b.py",),
    symbols: tuple[str, ...] = ("func_a",),
) -> ContextBuildRequest:
    return ContextBuildRequest(
        task_id="task-1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        model_capabilities=ModelCapabilities(context_tokens=4096),
        authority_refs=authority,
        changed_files=("src/a.py",),
        explicit_paths=paths,
        referenced_symbols=symbols,
        budget=ContextBudget(primary_budget=budget_chars),
    )


def _large_request(
    budget_chars: int = 10_000,
    authority: tuple[str, ...] = ("roadmap.md",),
) -> ContextBuildRequest:
    return ContextBuildRequest(
        task_id="task-1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        model_capabilities=ModelCapabilities(context_tokens=4096),
        authority_refs=authority,
        changed_files=("src/a.py",),
        budget=ContextBudget(primary_budget=budget_chars),
    )


# REPAIR 1: first-class summaries


def test_summary_count_mismatch_raises_value_error() -> None:
    with pytest.raises(ValueError, match="summary_count"):
        ContextPacket(summary_count=1, summaries=())


def test_summary_stored_in_summaries_not_payload() -> None:
    result = HybridContextStrategy().build(_hybrid_request())
    assert len(result.packet.summaries) == 1
    assert result.packet.payload.get("summary") is None


def test_summary_source_provenance_ids_reference_actual_sources() -> None:
    result = HierarchicalContextStrategy().build(_hierarchical_request())
    summary = result.packet.summaries[0]
    assert "file-0" in summary.source_provenance_ids
    assert "symbol-0" in summary.source_provenance_ids
    assert all(not sid.startswith("summary-source-") for sid in summary.source_provenance_ids)


def test_validator_catches_dangling_summary_source_provenance() -> None:
    packet = ContextPacket(
        summaries=(
            ContextSummary(
                text="summary",
                source_provenance_ids=("missing-source",),
                source_hashes=(_FAKE_HASH,),
                level="section",
            ),
        ),
        summary_count=1,
        provenance_index={
            "summary-1": ProvenanceRef("summary", path="summary", content_hash=_FAKE_HASH),
        },
    )
    issues = ContextPacketValidator().validate(packet)
    assert any(issue.code == "SUMMARY_DANGLING_SOURCE_PROVENANCE" for issue in issues)


# REPAIR 2: authority budget protection


def test_targeted_authority_presence_never_not_required() -> None:
    result = TargetedContextStrategy().build(_targeted_request())
    assert result.packet.authority_presence != AuthorityPresence.NOT_REQUIRED


def test_hybrid_authority_presence_never_not_required() -> None:
    result = HybridContextStrategy().build(_hybrid_request())
    assert result.packet.authority_presence != AuthorityPresence.NOT_REQUIRED


def test_hierarchical_authority_presence_never_not_required() -> None:
    result = HierarchicalContextStrategy().build(_hierarchical_request())
    assert result.packet.authority_presence != AuthorityPresence.NOT_REQUIRED


def test_large_context_authority_presence_never_not_required() -> None:
    result = LargeContextStrategy().build(_large_request())
    assert result.packet.authority_presence != AuthorityPresence.NOT_REQUIRED


def test_tiny_budget_legacy_authority_raises_context_budget_error() -> None:
    with pytest.raises(ContextBudgetError):
        TargetedContextStrategy().build(_targeted_request(budget_chars=1))


def test_tiny_budget_rich_authority_entries_produce_raw_referenced() -> None:
    result = TargetedContextStrategy().build(
        _targeted_request(
            budget_chars=1,
            authority=(),
            authority_entries=(
                {
                    "authority_id": "auth-1",
                    "provenance_id": "auth-prov-1",
                    "full_source_ref": "roadmap.md",
                    "revision": _FAKE_COMMIT,
                    "content_hash": _FAKE_HASH,
                    "content": "roadmap.md",
                    "raw_included": True,
                },
            ),
        )
    )
    assert result.packet.authority_presence == AuthorityPresence.RAW_REFERENCED
    assert result.packet.authority[0].content is None
    assert result.packet.authority[0].raw_included is False


def test_max_items_zero_legacy_authority_raises_context_budget_error() -> None:
    with pytest.raises(ContextBudgetError):
        TargetedContextStrategy().build(_targeted_request(max_items=0))


def test_validator_catches_raw_referenced_missing_immutable_provenance() -> None:
    packet = ContextPacket(
        authority=(
            AuthorityContextItem(
                authority_id="auth-1",
                provenance_id="auth-prov-1",
                full_source_ref="roadmap.md",
                revision="",
                content_hash="",
                content=None,
                raw_included=False,
            ),
        ),
        authority_presence=AuthorityPresence.RAW_REFERENCED,
        provenance_index={
            "auth-prov-1": ProvenanceRef("authority", path="roadmap.md"),
        },
    )
    issues = ContextPacketValidator().validate(packet)
    assert any(issue.code == "RAW_REFERENCED_MISSING_IMMUTABLE_PROVENANCE" for issue in issues)


# REPAIR 3: arbitration separation


def _arbitration_packet() -> ArbitrationEvidencePacket:
    return ArbitrationEvidencePacket(
        dispute_id="dispute-1",
        disputed_findings=(
            DisputedFinding(
                finding_id="f1",
                exact_text="Line 42 should use X",
                positions=(
                    ReviewerPosition("alice", "BLOCKING", "breaks invariant", ("e1",)),
                    ReviewerPosition("bob", "NOT_BLOCKING", "edge case only", ("e2",)),
                ),
                evidence_refs=("e1", "e2"),
                authority_refs=("roadmap.md",),
                unresolved_question="Does this violate the authority rule?",
            ),
        ),
        authority_refs=("roadmap.md",),
        diff_excerpts=(DiffInfo("d1", "src/a.py"),),
        test_evidence=(TestEvidence("e1", "test_a", "failed"),),
        provenance_index={
            "d1": ProvenanceRef("git", path="src/a.py"),
            "e1": ProvenanceRef("test", path="test_a"),
            "e2": ProvenanceRef("test", path="test_b"),
            "roadmap.md": ProvenanceRef(
                "authority",
                path="roadmap.md",
                revision=_FAKE_COMMIT,
                content_hash=_FAKE_HASH,
                authority_level="roadmap",
            ),
        },
    )


def test_arbitration_round_trip_preserves_reviewer_positions() -> None:
    packet = _arbitration_packet().to_context_packet()
    data = packet.to_dict()
    restored = ContextPacket.from_dict(data)
    assert restored.arbitration is not None
    assert len(restored.arbitration.reviewer_positions) == 2
    assert {p.position for p in restored.arbitration.reviewer_positions} == {
        "BLOCKING",
        "NOT_BLOCKING",
    }


def test_arbitration_authority_does_not_contain_finding_text() -> None:
    packet = _arbitration_packet().to_context_packet()
    assert packet.arbitration is not None
    finding_text = packet.arbitration.disputed_findings[0].exact_text
    position_text = packet.arbitration.disputed_findings[0].positions[0].reasoning
    for item in packet.authority:
        assert finding_text not in (item.content or "")
        assert position_text not in (item.content or "")
        assert item.raw_included is False


def test_validator_catches_arbitration_dangling_evidence_ref() -> None:
    packet = ContextPacket(
        arbitration=ArbitrationContext(
            dispute_id="dispute-1",
            disputed_findings=(),
            reviewer_positions=(),
            evidence_refs=("missing-evidence",),
            authority_refs=(),
        ),
        provenance_index={},
    )
    issues = ContextPacketValidator().validate(packet)
    assert any(issue.code == "ARBITRATION_DANGLING_EVIDENCE_REF" for issue in issues)


# REPAIR 4: secret-safe serialization


def test_safe_dict_redacts_secret_value_api_key() -> None:
    secret = SecretValue("sk-1234567890abcdef")
    packet = ContextPacket(
        packet_id="secret-pkt",
        payload={"api_key": secret, "other": "visible"},
    )
    safe = packet.to_safe_dict()
    assert "sk-1234567890abcdef" not in json.dumps(safe, default=str)
    assert safe["payload"]["api_key"] == "<redacted>"
    assert safe["payload"]["other"] == "visible"


def test_safe_dict_preserves_password_variable_name() -> None:
    packet = ContextPacket(
        packet_id="source-pkt",
        current_diff=(
            DiffInfo(
                diff_id="d1",
                path="src/auth.py",
                content="password = user_input\n",
            ),
        ),
    )
    safe = packet.to_safe_dict()
    assert "password = user_input" in json.dumps(safe, default=str)


# REPAIR 5: provenance coverage


def test_coverage_includes_authority_criteria_summaries_metadata() -> None:
    index = ProvenanceIndex()
    index.register("auth-1", ProvenanceRef("authority", path="roadmap.md"))
    index.register("criterion-1", ProvenanceRef("criterion", path="constraint:0"))
    index.register("summary-1", ProvenanceRef("summary", path="summary", content_hash=_FAKE_HASH))
    index.register("task-1", ProvenanceRef("task_metadata", path="task-1"))

    packet = ContextPacket(
        packet_id="coverage-pkt",
        authority=(
            AuthorityContextItem(
                authority_id="auth-1",
                provenance_id="auth-1",
                full_source_ref="roadmap.md",
                revision=_FAKE_COMMIT,
                content_hash=_FAKE_HASH,
                content="roadmap.md",
                raw_included=True,
            ),
        ),
        acceptance_criteria=(
            AcceptanceCriterion(
                criterion_id="criterion-1",
                text="must pass",
                authority_refs=("auth-1",),
            ),
        ),
        summaries=(
            ContextSummary(
                text="summary",
                source_provenance_ids=("auth-1",),
                source_hashes=(_FAKE_HASH,),
                level="section",
            ),
        ),
        summary_count=1,
        task_metadata=TaskMetadata("task-1"),
        provenance_index=dict(index._refs),
        authority_presence=AuthorityPresence.RAW_INCLUDED,
    )
    assert _provenance_coverage(packet, index) == 1.0


def test_validator_catches_criterion_dangling_authority_ref() -> None:
    packet = ContextPacket(
        acceptance_criteria=(
            AcceptanceCriterion(
                criterion_id="c1",
                text="must pass",
                authority_refs=("missing-authority",),
            ),
        ),
        provenance_index={},
    )
    issues = ContextPacketValidator().validate(packet)
    assert any(issue.code == "CRITERION_DANGLING_AUTHORITY_REF" for issue in issues)


# REPAIR 6: immutable authority provenance


def test_raw_authority_with_revision_and_hash_passes_validation() -> None:
    packet = ContextPacket(
        packet_id="immutable-pkt",
        authority=(
            AuthorityContextItem(
                authority_id="auth-1",
                provenance_id="auth-prov-1",
                full_source_ref="roadmap.md",
                revision=_FAKE_COMMIT,
                content_hash=_FAKE_HASH,
                content="roadmap.md",
                raw_included=True,
            ),
        ),
        authority_presence=AuthorityPresence.RAW_INCLUDED,
        provenance_index={
            "auth-prov-1": ProvenanceRef(
                "authority",
                path="roadmap.md",
                revision=_FAKE_COMMIT,
                content_hash=_FAKE_HASH,
                authority_level="roadmap",
            ),
        },
    )
    issues = ContextPacketValidator().validate(packet)
    assert not issues
