"""Arbitration evidence packet preserving independent reviewer positions."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ArbitrationEvidencePacket", "DisputedFinding", "ReviewerPosition"]

from src.context.schema import (
    ArbitrationContext,
    AuthorityContextItem,
    AuthorityPresence,
    ContextPacket,
    DiffInfo,
    DisputedFinding,
    ProvenanceRef,
    ReviewerPosition,
    TestEvidence,
)


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
        provenance_index = dict(self.provenance_index)

        authority: list[AuthorityContextItem] = []
        for i, ref in enumerate(self.authority_refs):
            existing = provenance_index.get(ref)
            if existing is not None:
                authority.append(
                    AuthorityContextItem(
                        authority_id=f"arb-authority-{i}",
                        provenance_id=ref,
                        full_source_ref=existing.path or ref,
                        revision=existing.revision or "",
                        content_hash=existing.content_hash or "",
                        content=None,
                        raw_included=False,
                    )
                )
            else:
                authority.append(
                    AuthorityContextItem(
                        authority_id=f"arb-authority-{i}",
                        provenance_id=ref,
                        full_source_ref=ref,
                        revision="",
                        content_hash="",
                        content=ref,
                        raw_included=True,
                    )
                )

        for item in authority:
            if item.provenance_id not in provenance_index:
                provenance_index[item.provenance_id] = ProvenanceRef(
                    source_type="authority",
                    path=item.full_source_ref,
                    authority_level="roadmap",
                    revision=item.revision or None,
                    content_hash=item.content_hash or None,
                )

        all_evidence_refs: list[str] = []
        all_positions: list[ReviewerPosition] = []
        for finding in self.disputed_findings:
            all_evidence_refs.extend(finding.evidence_refs)
            all_positions.extend(finding.positions)
            for ref in finding.evidence_refs:
                if ref not in provenance_index:
                    provenance_index[ref] = ProvenanceRef(
                        source_type="arbitration_evidence", path=ref
                    )

        arbitration = ArbitrationContext(
            dispute_id=self.dispute_id,
            disputed_findings=self.disputed_findings,
            reviewer_positions=tuple(all_positions),
            evidence_refs=tuple(sorted(set(all_evidence_refs))),
            authority_refs=tuple(self.authority_refs),
        )

        return ContextPacket(
            packet_id=self.dispute_id,
            authority=tuple(authority),
            current_diff=self.diff_excerpts,
            test_evidence=self.test_evidence,
            arbitration=arbitration,
            provenance_index=provenance_index,
            authority_presence=AuthorityPresence.RAW_INCLUDED
            if authority
            else AuthorityPresence.NOT_REQUIRED,
        )
