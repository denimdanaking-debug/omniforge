"""Tests for arbitration evidence packets."""

from __future__ import annotations

from src.context.arbitration import (
    ArbitrationEvidencePacket,
    DisputedFinding,
    ReviewerPosition,
)
from src.context.schema import DiffInfo, ProvenanceRef, TestEvidence
from src.context.validation import ContextPacketValidator


def _packet() -> ArbitrationEvidencePacket:
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
        },
    )


def test_reviewer_positions_preserved_separately() -> None:
    packet = _packet()
    finding = packet.disputed_findings[0]
    assert len(finding.positions) == 2
    assert {p.position for p in finding.positions} == {"BLOCKING", "NOT_BLOCKING"}


def test_primary_evidence_included() -> None:
    packet = _packet()
    context = packet.to_context_packet()
    assert context.current_diff
    assert context.test_evidence


def test_validation_rejects_missing_evidence() -> None:
    bad_finding = DisputedFinding(
        finding_id="f2",
        exact_text="no evidence",
        positions=(ReviewerPosition("alice", "BLOCKING", "no evidence", ()),),
        evidence_refs=(),
        authority_refs=(),
        unresolved_question="?",
    )
    bad_packet = ArbitrationEvidencePacket(
        dispute_id="dispute-2",
        disputed_findings=(bad_finding,),
        authority_refs=(),
        diff_excerpts=(),
        test_evidence=(),
        provenance_index={},
    )
    context = bad_packet.to_context_packet()
    issues = ContextPacketValidator().validate(context)
    assert any(issue.code == "ARBITRATION_FINDING_WITHOUT_EVIDENCE" for issue in issues)
