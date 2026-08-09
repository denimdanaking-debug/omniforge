"""Arbitration evidence packet preserving independent reviewer positions."""

from __future__ import annotations

from dataclasses import dataclass

from src.context.schema import (
    AuthorityPresence,
    ContextPacket,
    DiffInfo,
    ProvenanceRef,
    TestEvidence,
)


@dataclass(frozen=True)
class ReviewerPosition:
    """One reviewer's stance on a disputed finding."""

    reviewer_id: str
    position: str
    reasoning: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class DisputedFinding:
    """A finding under arbitration with independent reviewer positions."""

    finding_id: str
    exact_text: str
    positions: tuple[ReviewerPosition, ...]
    evidence_refs: tuple[str, ...]
    authority_refs: tuple[str, ...]
    unresolved_question: str


@dataclass(frozen=True)
class ArbitrationEvidencePacket:
    """Primary evidence package sent to an arbitrator."""

    dispute_id: str
    disputed_findings: tuple[DisputedFinding, ...]
    authority_refs: tuple[str, ...]
    diff_excerpts: tuple[DiffInfo, ...]
    test_evidence: tuple[TestEvidence, ...]
    provenance_index: dict[str, ProvenanceRef]

    def to_context_packet(self) -> ContextPacket:
        """Convert arbitration evidence into a canonical context packet."""
        findings_text = "\n\n".join(
            f"Finding {f.finding_id}: {f.exact_text}\nQuestion: {f.unresolved_question}"
            for f in self.disputed_findings
        )
        positions_text = "\n\n".join(
            f"Reviewer {p.reviewer_id} ({p.position}): {p.reasoning}"
            for f in self.disputed_findings
            for p in f.positions
        )
        authority = tuple(self.authority_refs) + (findings_text, positions_text)
        disputed_findings_data = [
            {
                "finding_id": f.finding_id,
                "exact_text": f.exact_text,
                "evidence_refs": list(f.evidence_refs),
            }
            for f in self.disputed_findings
        ]
        return ContextPacket(
            packet_id=self.dispute_id,
            authority=authority,
            current_diff=self.diff_excerpts,
            test_evidence=self.test_evidence,
            provenance_index=dict(self.provenance_index),
            authority_presence=AuthorityPresence.RAW_INCLUDED,
            payload={"disputed_findings": disputed_findings_data},
        )
