"""Tests for the canonical context packet schema."""

from __future__ import annotations

import pytest

from src.context.schema import (
    CONTEXT_PACKET_SCHEMA_VERSION,
    AcceptanceCriterion,
    AuthorityPresence,
    ContextPacket,
    DiffInfo,
    Exclusion,
    HistoricalFinding,
    ProvenanceRef,
    RelevantFile,
    TaskMetadata,
    TestEvidence,
)


def test_schema_version_constant() -> None:
    assert CONTEXT_PACKET_SCHEMA_VERSION == "1.0.0"


def test_context_packet_requires_supported_schema_version() -> None:
    with pytest.raises(ValueError):
        ContextPacket(schema_version="0.0.0")


def test_context_packet_round_trip() -> None:
    packet = ContextPacket(
        packet_id="pkt-1",
        authority=("roadmap.md",),
        acceptance_criteria=(AcceptanceCriterion("c1", "must pass tests"),),
        relevant_files=(RelevantFile("f1", "src/a.py"),),
        current_diff=(DiffInfo("d1", "src/a.py"),),
        test_evidence=(TestEvidence("e1", "test_a", "passed"),),
        historical_findings=(HistoricalFinding("h1", "prior issue"),),
        task_metadata=TaskMetadata("t1", role="coding"),
        exclusions=(Exclusion("x1", "too large"),),
        provenance_index={"p1": ProvenanceRef("authority", path="roadmap.md")},
        authority_presence=AuthorityPresence.RAW_INCLUDED,
        raw_item_count=5,
        summary_count=0,
        estimated_input_chars=123,
        budget={"usable_chars": 1000},
    )
    data = packet.to_dict()
    restored = ContextPacket.from_dict(data)
    assert restored.packet_id == packet.packet_id
    assert restored.authority == packet.authority
    assert restored.estimated_input_chars == packet.estimated_input_chars
    assert restored.authority_presence == packet.authority_presence
    assert restored.content_hash() == packet.content_hash()


def test_content_hash_is_deterministic() -> None:
    packet1 = ContextPacket(packet_id="p", authority=("a", "b"))
    packet2 = ContextPacket(packet_id="p", authority=("b", "a"))
    # Authority order matters for hash, so sort externally for canonical equality.
    packet3 = ContextPacket(packet_id="p", authority=("a", "b"))
    assert packet1.content_hash() == packet3.content_hash()
    assert packet1.content_hash() != packet2.content_hash()


def test_backward_compatible_minimal_packet() -> None:
    packet = ContextPacket(kind="authority", source="roadmap.md")
    assert packet.kind == "authority"
    assert packet.source == "roadmap.md"
    assert packet.schema_version == CONTEXT_PACKET_SCHEMA_VERSION


def test_required_categories_present() -> None:
    packet = ContextPacket(
        packet_id="pkt",
        authority=("roadmap",),
        acceptance_criteria=(AcceptanceCriterion("c1", "text"),),
        relevant_files=(RelevantFile("f1", "a.py"),),
        current_diff=(DiffInfo("d1", "a.py"),),
        test_evidence=(TestEvidence("e1", "t", "pass"),),
        historical_findings=(HistoricalFinding("h1", "summary"),),
        task_metadata=TaskMetadata("t1"),
        exclusions=(Exclusion("x1", "reason"),),
        provenance_index={"p1": ProvenanceRef("git", path="a.py")},
    )
    assert packet.authority
    assert packet.acceptance_criteria
    assert packet.relevant_files
    assert packet.current_diff
    assert packet.test_evidence
    assert packet.historical_findings
    assert packet.task_metadata is not None
    assert packet.exclusions
    assert packet.provenance_index
