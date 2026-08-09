"""Targeted context strategy: deterministic priority under tight budget."""

from __future__ import annotations

from typing import Any

from src.context._utils import deterministic_sort, normalize_path
from src.context.budget import compute_usable_budget
from src.context.provenance import ProvenanceIndex
from src.context.schema import (
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
from src.context.strategy import ContextBuildRequest, ContextStrategy, ContextStrategyResult
from src.context.telemetry import ContextStrategyTelemetry


class TargetedContextStrategy(ContextStrategy):
    """Deterministic priority strategy for normal-to-tight context budgets."""

    @property
    def name(self) -> str:
        return "targeted"

    def build(self, request: ContextBuildRequest) -> ContextStrategyResult:
        packet_id = f"targeted-{request.task_id}"
        budget_result = compute_usable_budget(
            request.model_capabilities.context_tokens if request.model_capabilities else None,
            request.budget,
        )
        usable_chars = min(budget_result.usable, request.budget.primary_budget)
        max_items = request.budget.max_items

        provenance_index = ProvenanceIndex()
        items: list[tuple[str, Any, int, str]] = []  # (priority, item, chars, source_id)

        # Priority 1: authority
        for idx, ref in enumerate(deterministic_sort(request.authority_refs, key=str)):
            source_id = f"authority-{idx}"
            text = str(ref)
            provenance_index.register(
                source_id,
                ProvenanceRef(source_type="authority", path=text, authority_level="roadmap"),
            )
            items.append(("authority", text, len(text), source_id))

        # Priority 2: acceptance criteria (synthesized from constraints)
        for idx, constraint in enumerate(request.constraints):
            criterion_id = f"criterion-{idx}"
            criterion = AcceptanceCriterion(
                criterion_id=criterion_id,
                text=constraint,
                authority_refs=("authority-0",) if request.authority_refs else (),
            )
            provenance_index.register(
                criterion_id,
                ProvenanceRef(source_type="criterion", path=f"constraint:{idx}"),
            )
            items.append(("criterion", criterion, len(criterion.text), criterion_id))

        # Priority 3: changed files
        for idx, path in enumerate(deterministic_sort(request.changed_files)):
            file_id = f"changed-{idx}"
            norm = normalize_path(path)
            file_item = RelevantFile(
                file_id=file_id,
                path=norm,
                reason="changed",
                provenance_id=file_id,
            )
            provenance_index.register(
                file_id,
                ProvenanceRef(source_type="git", path=norm),
            )
            items.append(("changed_file", file_item, len(norm) + 20, file_id))

        # Priority 4: referenced symbols
        for idx, symbol in enumerate(deterministic_sort(request.referenced_symbols)):
            symbol_id = f"symbol-{idx}"
            finding = HistoricalFinding(
                finding_id=symbol_id,
                summary=symbol,
                provenance_id=symbol_id,
                related_symbols=(symbol,),
            )
            provenance_index.register(
                symbol_id,
                ProvenanceRef(source_type="symbol", path=symbol),
            )
            items.append(("referenced_symbol", finding, len(symbol) + 20, symbol_id))

        # Priority 5: relevant tests (test failures)
        for idx, failure in enumerate(request.test_failures):
            test_id = f"test-{idx}"
            name = str(failure) if not isinstance(failure, dict) else failure.get("name", test_id)
            log = str(failure) if not isinstance(failure, dict) else failure.get("log", "")
            evidence = TestEvidence(
                evidence_id=test_id,
                test_name=name,
                outcome="failed",
                log_excerpt=log,
                provenance_id=test_id,
            )
            provenance_index.register(
                test_id,
                ProvenanceRef(source_type="test", path=name),
            )
            items.append(("test_evidence", evidence, len(name) + len(log) + 20, test_id))

        # Priority 6: prior findings
        for idx, finding in enumerate(request.prior_findings):
            finding_id = f"finding-{idx}"
            summary = str(finding) if not isinstance(finding, dict) else finding.get("summary", "")
            hf = HistoricalFinding(
                finding_id=finding_id,
                summary=summary,
                provenance_id=finding_id,
            )
            provenance_index.register(
                finding_id,
                ProvenanceRef(source_type="finding", path=finding_id),
            )
            items.append(("prior_finding", hf, len(summary) + 20, finding_id))

        # Priority 7: nearby context from explicit paths
        for idx, path in enumerate(deterministic_sort(request.explicit_paths)):
            path_id = f"nearby-{idx}"
            norm = normalize_path(path)
            file_item = RelevantFile(
                file_id=path_id,
                path=norm,
                reason="nearby_context",
                provenance_id=path_id,
            )
            provenance_index.register(
                path_id,
                ProvenanceRef(source_type="filesystem", path=norm),
            )
            items.append(("nearby_context", file_item, len(norm) + 20, path_id))

        # Build packet within budget, preserving priority order.
        authority: list[str] = []
        acceptance_criteria: list[AcceptanceCriterion] = []
        relevant_files: list[RelevantFile] = []
        current_diff: list[DiffInfo] = []
        test_evidence: list[TestEvidence] = []
        historical_findings: list[HistoricalFinding] = []
        exclusions: list[Exclusion] = []

        priority_order = [
            "authority",
            "criterion",
            "changed_file",
            "referenced_symbol",
            "test_evidence",
            "prior_finding",
            "nearby_context",
        ]

        consumed = 0
        item_count = 0
        truncation_events: list[str] = []

        for priority in priority_order:
            for _priority, item, chars, source_id in items:
                if _priority != priority:
                    continue
                if max_items is not None and item_count >= max_items:
                    exclusions.append(
                        Exclusion(
                            exclusion_id=f"excluded-{source_id}",
                            reason="max_items exceeded",
                            source_item_id=source_id,
                            estimated_chars=chars,
                        )
                    )
                    truncation_events.append(f"max_items_exceeded:{source_id}")
                    continue
                if consumed + chars > usable_chars:
                    exclusions.append(
                        Exclusion(
                            exclusion_id=f"excluded-{source_id}",
                            reason="budget exhausted",
                            source_item_id=source_id,
                            estimated_chars=chars,
                        )
                    )
                    truncation_events.append(f"budget_exhausted:{source_id}")
                    continue

                consumed += chars
                item_count += 1

                if priority == "authority":
                    authority.append(str(item))
                elif priority == "criterion":
                    acceptance_criteria.append(item)
                elif priority in {"changed_file", "nearby_context"}:
                    relevant_files.append(item)
                elif priority == "referenced_symbol":
                    historical_findings.append(item)
                elif priority == "test_evidence":
                    test_evidence.append(item)
                elif priority == "prior_finding":
                    historical_findings.append(item)

        authority_presence = (
            AuthorityPresence.RAW_INCLUDED
            if authority
            else AuthorityPresence.NOT_REQUIRED
            if not request.authority_refs
            else AuthorityPresence.RAW_REFERENCED
        )

        task_metadata = TaskMetadata(
            task_id=request.task_id,
            role=request.role.value,
            risk=request.risk.name,
            requested_objective=request.requested_objective,
            constraints=request.constraints,
        )

        raw_item_count = (
            len(authority)
            + len(acceptance_criteria)
            + len(relevant_files)
            + len(current_diff)
            + len(test_evidence)
            + len(historical_findings)
        )

        packet = ContextPacket(
            packet_id=packet_id,
            authority=tuple(authority),
            acceptance_criteria=tuple(acceptance_criteria),
            relevant_files=tuple(relevant_files),
            current_diff=tuple(current_diff),
            test_evidence=tuple(test_evidence),
            historical_findings=tuple(historical_findings),
            task_metadata=task_metadata,
            exclusions=tuple(exclusions),
            provenance_index=dict(provenance_index._refs),
            authority_presence=authority_presence,
            raw_item_count=raw_item_count,
            summary_count=0,
            estimated_input_chars=consumed,
            budget={
                "usable_chars": usable_chars,
                "reserved": budget_result.reserved,
                "total": budget_result.total_budget,
            },
        )

        telemetry = ContextStrategyTelemetry(
            strategy=self.name,
            packet_id=packet_id,
            source_item_count=len(items),
            raw_item_count=raw_item_count,
            summary_count=0,
            estimated_input_chars=consumed,
            context_capacity=request.model_capabilities.context_tokens
            if request.model_capabilities
            else None,
            budget_consumed={"chars": consumed},
            excluded_count=len(exclusions),
            authority_presence=authority_presence,
            provenance_coverage=_provenance_coverage(packet, provenance_index),
            truncation_events=tuple(truncation_events),
        )

        return ContextStrategyResult(strategy_name=self.name, packet=packet, telemetry=telemetry)


def _provenance_coverage(packet: ContextPacket, index: ProvenanceIndex) -> float:
    """Return the fraction of packet items with registered provenance."""
    provenanced = 0
    total = 0
    for file in packet.relevant_files:
        total += 1
        if file.provenance_id is not None and file.provenance_id in index:
            provenanced += 1
    for diff in packet.current_diff:
        total += 1
        if diff.provenance_id is not None and diff.provenance_id in index:
            provenanced += 1
    for evidence in packet.test_evidence:
        total += 1
        if evidence.provenance_id is not None and evidence.provenance_id in index:
            provenanced += 1
    for finding in packet.historical_findings:
        total += 1
        if finding.provenance_id is not None and finding.provenance_id in index:
            provenanced += 1
    if total == 0:
        return 1.0
    return provenanced / total
