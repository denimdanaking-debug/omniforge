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
        issues.extend(self._check_authority_immutability(packet))
        issues.extend(self._check_dangling_provenance(packet))
        issues.extend(self._check_summary_authority(packet))
        issues.extend(self._check_summary_validity(packet))
        issues.extend(self._check_summary_provenance(packet))
        issues.extend(self._check_acceptance_criterion_authority(packet))
        issues.extend(self._check_arbitration_evidence(packet))
        return issues

    def _authority_requested(self, packet: ContextPacket) -> bool:
        """Return True if the packet is expected to carry authority."""
        return bool(packet.authority) or bool(
            packet.arbitration and packet.arbitration.authority_refs
        )

    def _check_authority_presence(self, packet: ContextPacket) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        authority_requested = self._authority_requested(packet)
        if authority_requested and packet.authority_presence == AuthorityPresence.NOT_REQUIRED:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="AUTHORITY_REQUIRED_BUT_NOT_REQUIRED",
                    message="Authority is required but authority_presence is NOT_REQUIRED",
                )
            )
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

    def _check_authority_immutability(self, packet: ContextPacket) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for item in packet.authority:
            if not item.revision or not item.content_hash:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="AUTHORITY_ITEM_MISSING_IMMUTABLE_IDENTITY",
                        message=f"Authority item {item.authority_id!r} is missing revision or "
                        "content_hash",
                    )
                )
            if item.raw_included and item.content is None:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="RAW_INCLUDED_MISSING_CONTENT",
                        message=f"Authority item {item.authority_id!r} is RAW_INCLUDED but "
                        "has no content",
                    )
                )
            provenance = packet.provenance_index.get(item.provenance_id)
            if provenance is None:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="AUTHORITY_DANGLING_PROVENANCE",
                        message=f"Authority item {item.authority_id!r} references unknown "
                        f"provenance {item.provenance_id!r}",
                    )
                )
                continue
            if provenance.source_type != "authority":
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="AUTHORITY_REF_NOT_AUTHORITY_SOURCE",
                        message=f"Authority item {item.authority_id!r} resolves to "
                        f"source_type={provenance.source_type!r}, not 'authority'",
                    )
                )
            if not provenance.authority_level:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="AUTHORITY_PROVENANCE_MISSING_LEVEL",
                        message=f"Authority provenance {item.provenance_id!r} is missing "
                        "authority_level",
                    )
                )
            if not provenance.revision or not provenance.content_hash:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="AUTHORITY_PROVENANCE_MISSING_REVISION_OR_HASH",
                        message=f"Authority provenance {item.provenance_id!r} is missing "
                        "revision or content_hash",
                    )
                )
            if provenance.revision != item.revision:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="AUTHORITY_REVISION_MISMATCH",
                        message=f"Authority item {item.authority_id!r} revision "
                        f"{item.revision!r} does not match provenance revision "
                        f"{provenance.revision!r}",
                    )
                )
            if provenance.content_hash != item.content_hash:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="AUTHORITY_HASH_MISMATCH",
                        message=f"Authority item {item.authority_id!r} content_hash "
                        f"{item.content_hash!r} does not match provenance content_hash "
                        f"{provenance.content_hash!r}",
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
        for item in packet.authority:
            if item.provenance_id:
                provenanced_ids.add(item.provenance_id)
        for summary in packet.summaries:
            provenanced_ids.update(summary.source_provenance_ids)
        if packet.task_metadata is not None:
            provenanced_ids.add(packet.task_metadata.task_id)

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
        if packet.summaries and packet.authority_presence != AuthorityPresence.RAW_INCLUDED:
            for item in packet.authority:
                if item.content and "summary" in item.content.lower():
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="SUMMARY_CLAIMS_AUTHORITY",
                            message="Summary text appears in authority section",
                        )
                    )
                    break
        return issues

    def _check_summary_validity(self, packet: ContextPacket) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for summary in packet.summaries:
            if not summary.lossy:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="SUMMARY_NOT_LOSSY",
                        message="ContextSummary must be lossy",
                    )
                )
            if not summary.source_provenance_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="SUMMARY_MISSING_SOURCE_PROVENANCE",
                        message="ContextSummary has no source_provenance_ids",
                    )
                )
            if not summary.source_hashes:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="SUMMARY_MISSING_SOURCE_HASHES",
                        message="ContextSummary has no source_hashes",
                    )
                )
            if len(summary.source_provenance_ids) != len(summary.source_hashes):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="SUMMARY_PROVENANCE_HASH_MISMATCH",
                        message="ContextSummary source_provenance_ids and source_hashes length "
                        "mismatch",
                    )
                )
        return issues

    def _check_summary_provenance(self, packet: ContextPacket) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for summary in packet.summaries:
            for source_id in summary.source_provenance_ids:
                if source_id not in packet.provenance_index:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="SUMMARY_DANGLING_SOURCE_PROVENANCE",
                            message=f"Summary references unknown provenance {source_id!r}",
                        )
                    )
        if packet.summaries and not any(
            ref.source_type == "summary" for ref in packet.provenance_index.values()
        ):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="SUMMARY_WITHOUT_PROVENANCE",
                    message="Packet contains a summary without summary provenance",
                )
            )
        return issues

    def _check_acceptance_criterion_authority(self, packet: ContextPacket) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for criterion in packet.acceptance_criteria:
            for ref in criterion.authority_refs:
                if ref not in packet.provenance_index:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="CRITERION_DANGLING_AUTHORITY_REF",
                            message=f"Criterion {criterion.criterion_id!r} references unknown "
                            f"authority provenance {ref!r}",
                        )
                    )
        return issues

    def _check_arbitration_evidence(self, packet: ContextPacket) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        arbitration = packet.arbitration
        if arbitration is None:
            return issues

        for ref in arbitration.evidence_refs:
            if ref not in packet.provenance_index:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="ARBITRATION_DANGLING_EVIDENCE_REF",
                        message=f"Arbitration evidence ref {ref!r} has no provenance",
                    )
                )

        for position in arbitration.reviewer_positions:
            for ref in position.evidence_refs:
                if ref not in packet.provenance_index:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="ARBITRATION_DANGLING_REVIEWER_EVIDENCE_REF",
                            message=f"Reviewer {position.reviewer_id!r} evidence ref {ref!r} "
                            "has no provenance",
                        )
                    )

        for item in packet.authority:
            provenance = packet.provenance_index.get(item.provenance_id)
            if provenance is not None and provenance.source_type == "arbitration_evidence":
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="ARBITRATION_EVIDENCE_TREATED_AS_AUTHORITY",
                        message=f"Arbitration evidence {item.provenance_id!r} appears in authority",
                    )
                )

        for finding in arbitration.disputed_findings:
            if not finding.evidence_refs:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="ARBITRATION_FINDING_WITHOUT_EVIDENCE",
                        message=f"Finding {finding.finding_id!r} has no evidence refs",
                    )
                )
            for ref in finding.authority_refs:
                if ref not in packet.provenance_index:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="ARBITRATION_FINDING_DANGLING_AUTHORITY_REF",
                            message=f"Finding {finding.finding_id!r} authority ref {ref!r} "
                            "has no provenance",
                        )
                    )
                else:
                    provenance = packet.provenance_index[ref]
                    if provenance.source_type != "authority":
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                code="ARBITRATION_FINDING_AUTHORITY_REF_NOT_AUTHORITY",
                                message=f"Finding {finding.finding_id!r} authority ref {ref!r} "
                                f"is source_type={provenance.source_type!r}, not 'authority'",
                            )
                        )

        for ref in arbitration.authority_refs:
            if ref not in packet.provenance_index:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="ARBITRATION_DANGLING_AUTHORITY_REF",
                        message=f"Arbitration authority ref {ref!r} has no provenance",
                    )
                )
            else:
                provenance = packet.provenance_index[ref]
                if provenance.source_type != "authority":
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="ARBITRATION_AUTHORITY_REF_NOT_AUTHORITY",
                            message=f"Arbitration authority ref {ref!r} is "
                            f"source_type={provenance.source_type!r}, not 'authority'",
                        )
                    )

        return issues
