"""Tests for context packet validation."""

from __future__ import annotations

from src.context.schema import (
    AuthorityPresence,
    ContextPacket,
    ProvenanceRef,
    RelevantFile,
)
from src.context.validation import ContextPacketValidator, ValidationIssue


def test_authority_protection() -> None:
    packet = ContextPacket(
        authority_presence=AuthorityPresence.RAW_INCLUDED,
        authority=(),
    )
    issues = ContextPacketValidator().validate(packet)
    assert any(issue.code == "MISSING_AUTHORITY" for issue in issues)


def test_summary_claiming_authority_rejected() -> None:
    packet = ContextPacket(
        summary_count=1,
        authority=("summary of roadmap",),
        provenance_index={"s1": ProvenanceRef("summary", path="summary")},
    )
    issues = ContextPacketValidator().validate(packet)
    assert any(issue.code == "SUMMARY_CLAIMS_AUTHORITY" for issue in issues)


def test_dangling_provenance_detected() -> None:
    packet = ContextPacket(
        relevant_files=(RelevantFile("f1", "a.py", provenance_id="missing"),),
    )
    issues = ContextPacketValidator().validate(packet)
    assert any(issue.code == "DANGLING_PROVENANCE" for issue in issues)


def test_valid_packet_has_no_issues() -> None:
    packet = ContextPacket(
        packet_id="p1",
        authority=("roadmap.md",),
        authority_presence=AuthorityPresence.RAW_INCLUDED,
        provenance_index={
            "p1": ProvenanceRef("authority", path="roadmap.md", authority_level="primary")
        },
    )
    issues = ContextPacketValidator().validate(packet)
    assert not issues


def test_validation_issue_is_frozen() -> None:
    issue = ValidationIssue("error", "CODE", "message")
    assert issue.severity == "error"
