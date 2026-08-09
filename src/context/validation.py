"""Validation rules for context packets."""

from __future__ import annotations

from dataclasses import dataclass

from src.context.schema import AuthorityPresence, ContextPacket


@dataclass(frozen=True)
class ValidationIssue:
    """One validation issue."""

    severity: str
    code: str
    message: str


class ContextPacketValidator:
    """Validate a context packet against authority and provenance rules."""

    def validate(self, packet: ContextPacket) -> list[ValidationIssue]:
        """Return a list of validation issues for the packet."""
        issues: list[ValidationIssue] = []
        issues.extend(self._check_authority_presence(packet))
        issues.extend(self._check_dangling_provenance(packet))
        issues.extend(self._check_summary_authority(packet))
        issues.extend(self._check_summary_provenance(packet))
        issues.extend(self._check_arbitration_evidence(packet))
        return issues

    def _check_authority_presence(self, packet: ContextPacket) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if (
            packet.authority_presence
            in {
                AuthorityPresence.RAW_INCLUDED,
                AuthorityPresence.RAW_REFERENCED,
            }
            and not packet.authority
        ):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="MISSING_AUTHORITY",
                    message="Authority-required packet has no raw authority entries",
                )
            )
        return issues

    def _check_dangling_provenance(self, packet: ContextPacket) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        provenanced_ids: set[str] = set()
        for file in packet.relevant_files:
            if file.provenance_id:
                provenanced_ids.add(file.provenance_id)
        for diff in packet.current_diff:
            if diff.provenance_id:
                provenanced_ids.add(diff.provenance_id)
        for evidence in packet.test_evidence:
            if evidence.provenance_id:
                provenanced_ids.add(evidence.provenance_id)
        for finding in packet.historical_findings:
            if finding.provenance_id:
                provenanced_ids.add(finding.provenance_id)

        dangling = sorted(
            item_id for item_id in provenanced_ids if item_id not in packet.provenance_index
        )
        for item_id in dangling:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="DANGLING_PROVENANCE",
                    message=f"Item {item_id!r} references unknown provenance",
                )
            )
        return issues

    def _check_summary_authority(self, packet: ContextPacket) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if packet.summary_count > 0 and packet.authority_presence != AuthorityPresence.RAW_INCLUDED:
            for item in packet.authority:
                if "summary" in str(item).lower():
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="SUMMARY_CLAIMS_AUTHORITY",
                            message="Summary text appears in authority section",
                        )
                    )
                    break
        return issues

    def _check_summary_provenance(self, packet: ContextPacket) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if packet.summary_count > 0:
            summary_keys = [
                key for key, ref in packet.provenance_index.items() if ref.source_type == "summary"
            ]
            if not summary_keys:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="SUMMARY_WITHOUT_PROVENANCE",
                        message="Packet contains a summary without summary provenance",
                    )
                )
        return issues

    def _check_arbitration_evidence(self, packet: ContextPacket) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        # Arbitration packets include dispute findings in authority as unstructured text;
        # if the payload carries structured findings, ensure each has evidence.
        payload = packet.payload or {}
        findings = payload.get("disputed_findings")
        if isinstance(findings, list):
            for finding in findings:
                if not isinstance(finding, dict):
                    continue
                evidence_refs = finding.get("evidence_refs") or []
                if not evidence_refs:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="ARBITRATION_FINDING_WITHOUT_EVIDENCE",
                            message=f"Finding {finding.get('finding_id')!r} has no evidence refs",
                        )
                    )
        return issues
