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


def _valid_authority_provenance() -> ProvenanceRef:
    return ProvenanceRef(
        "authority",
        path="roadmap.md",
        revision="a" * 40,
        content_hash="b" * 64,
        authority_level="roadmap",
    )


def test_missing_finding_evidence_ref_is_rejected() -> None:
    """A. Finding evidence ref missing from provenance_index must stay missing."""
    packet = ArbitrationEvidencePacket(
        dispute_id="dispute-missing-finding-evidence",
        disputed_findings=(
            DisputedFinding(
                finding_id="f1",
                exact_text="Line 42 should use X",
                positions=(ReviewerPosition("alice", "BLOCKING", "breaks", ("missing-evidence",)),),
                evidence_refs=("missing-evidence",),
                authority_refs=(),
                unresolved_question="?",
            ),
        ),
        authority_refs=(),
        diff_excerpts=(),
        test_evidence=(),
        provenance_index={},
    )
    context = packet.to_context_packet()
    issues = ContextPacketValidator().validate(context)
    assert any(issue.code == "ARBITRATION_DANGLING_EVIDENCE_REF" for issue in issues)


def test_missing_reviewer_evidence_ref_is_rejected() -> None:
    """B. ReviewerPosition evidence ref missing from provenance_index must stay missing."""
    packet = ArbitrationEvidencePacket(
        dispute_id="dispute-missing-reviewer-evidence",
        disputed_findings=(
            DisputedFinding(
                finding_id="f1",
                exact_text="Line 42 should use X",
                positions=(ReviewerPosition("alice", "BLOCKING", "breaks", ("missing-evidence",)),),
                evidence_refs=("existing-evidence",),
                authority_refs=(),
                unresolved_question="?",
            ),
        ),
        authority_refs=(),
        diff_excerpts=(),
        test_evidence=(),
        provenance_index={"existing-evidence": ProvenanceRef("test", path="test_a")},
    )
    context = packet.to_context_packet()
    issues = ContextPacketValidator().validate(context)
    assert any(issue.code == "ARBITRATION_DANGLING_REVIEWER_EVIDENCE_REF" for issue in issues)


def test_missing_finding_authority_ref_is_rejected() -> None:
    """C. DisputedFinding authority ref missing from provenance_index must stay missing."""
    packet = ArbitrationEvidencePacket(
        dispute_id="dispute-missing-finding-authority",
        disputed_findings=(
            DisputedFinding(
                finding_id="f1",
                exact_text="Line 42 should use X",
                positions=(ReviewerPosition("alice", "BLOCKING", "breaks", ()),),
                evidence_refs=(),
                authority_refs=("missing-authority",),
                unresolved_question="?",
            ),
        ),
        authority_refs=(),
        diff_excerpts=(),
        test_evidence=(),
        provenance_index={},
    )
    context = packet.to_context_packet()
    issues = ContextPacketValidator().validate(context)
    assert any(issue.code == "ARBITRATION_FINDING_DANGLING_AUTHORITY_REF" for issue in issues)


def test_missing_arbitration_authority_ref_is_rejected() -> None:
    """D. ArbitrationContext authority ref missing from provenance_index must stay missing."""
    packet = ArbitrationEvidencePacket(
        dispute_id="dispute-missing-arb-authority",
        disputed_findings=(),
        authority_refs=("missing-authority",),
        diff_excerpts=(),
        test_evidence=(),
        provenance_index={},
    )
    context = packet.to_context_packet()
    issues = ContextPacketValidator().validate(context)
    assert any(
        issue.code in {"ARBITRATION_DANGLING_AUTHORITY_REF", "AUTHORITY_DANGLING_PROVENANCE"}
        for issue in issues
    )


def test_non_authority_ref_rejected_for_arbitration_authority() -> None:
    """E. An authority ref that resolves to non-authority provenance is rejected."""
    packet = ArbitrationEvidencePacket(
        dispute_id="dispute-non-authority",
        disputed_findings=(),
        authority_refs=("git-record",),
        diff_excerpts=(),
        test_evidence=(),
        provenance_index={
            "git-record": ProvenanceRef("git", path="src/a.py"),
        },
    )
    context = packet.to_context_packet()
    issues = ContextPacketValidator().validate(context)
    assert any(
        issue.code
        in {
            "ARBITRATION_AUTHORITY_REF_NOT_AUTHORITY",
            "AUTHORITY_REF_NOT_AUTHORITY_SOURCE",
        }
        for issue in issues
    )


def test_manufactured_provenance_not_added_for_missing_refs() -> None:
    """Conversion must not synthesize provenance entries for missing refs."""
    packet = ArbitrationEvidencePacket(
        dispute_id="dispute-no-manufacture",
        disputed_findings=(
            DisputedFinding(
                finding_id="f1",
                exact_text="Line 42 should use X",
                positions=(ReviewerPosition("alice", "BLOCKING", "breaks", ("missing",)),),
                evidence_refs=("missing",),
                authority_refs=("also-missing",),
                unresolved_question="?",
            ),
        ),
        authority_refs=("also-missing",),
        diff_excerpts=(),
        test_evidence=(),
        provenance_index={},
    )
    context = packet.to_context_packet()
    assert "missing" not in context.provenance_index
    assert "also-missing" not in context.provenance_index
