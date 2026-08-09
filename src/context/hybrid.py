"""Hybrid context strategy: raw authority + summaries for supporting context."""

from __future__ import annotations

from typing import Any

from src.context._utils import deterministic_sort, hash_text, normalize_path
from src.context.budget import compute_usable_budget
from src.context.hierarchical import DeterministicTestSummarizer, Summarizer
from src.context.provenance import ProvenanceIndex
from src.context.schema import (
    AcceptanceCriterion,
    AuthorityPresence,
    ContextPacket,
    ContextSummary,
    DiffInfo,
    Exclusion,
    HistoricalFinding,
    ProvenanceRef,
    RelevantFile,
    TaskMetadata,
    TestEvidence,
)
from src.context.strategy import ContextBuildRequest, ContextStrategy, ContextStrategyResult
from src.context.targeted import _provenance_coverage
from src.context.telemetry import ContextStrategyTelemetry


class HybridContextStrategy(ContextStrategy):
    """Combines raw authority/acceptance criteria/diff/evidence with summaries."""

    def __init__(self, summarizer: Summarizer | None = None) -> None:
        self._summarizer = summarizer or DeterministicTestSummarizer()

    @property
    def name(self) -> str:
        return "hybrid"

    def build(self, request: ContextBuildRequest) -> ContextStrategyResult:
        packet_id = f"hybrid-{request.task_id}"
        budget_result = compute_usable_budget(
            request.model_capabilities.context_tokens if request.model_capabilities else None,
            request.budget,
        )
        usable_chars = budget_result.usable
        max_items = request.budget.max_items

        provenance_index = ProvenanceIndex()
        raw_items: list[tuple[Any, int, str, str]] = []
        summary_candidates: list[Any] = []

        # Authority is always raw and never summarized.
        for idx, ref in enumerate(deterministic_sort(request.authority_refs, key=str)):
            source_id = f"authority-{idx}"
            text = str(ref)
            provenance_index.register(
                source_id,
                ProvenanceRef(source_type="authority", path=text, authority_level="roadmap"),
            )
            raw_items.append((text, len(text), source_id, "authority"))

        # Acceptance criteria are raw.
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
            raw_items.append((criterion, len(criterion.text), criterion_id, "criterion"))

        # Current diff is raw.
        for idx, path in enumerate(deterministic_sort(request.changed_files)):
            diff_id = f"diff-{idx}"
            norm = normalize_path(path)
            diff = DiffInfo(
                diff_id=diff_id,
                path=norm,
                before_ref="HEAD~1",
                after_ref="HEAD",
                content=f"diff --git a/{norm} b/{norm}\n",
                provenance_id=diff_id,
            )
            provenance_index.register(diff_id, ProvenanceRef(source_type="git", path=norm))
            raw_items.append((diff, len(diff.content) + len(norm) + 20, diff_id, "diff"))

        # Test evidence is raw.
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
            provenance_index.register(test_id, ProvenanceRef(source_type="test", path=name))
            raw_items.append((evidence, len(name) + len(log) + 20, test_id, "test"))

        # Supporting material is summarized when possible.
        for idx, path in enumerate(deterministic_sort(request.explicit_paths)):
            file_id = f"file-{idx}"
            norm = normalize_path(path)
            file_item = RelevantFile(
                file_id=file_id,
                path=norm,
                reason="supporting",
                provenance_id=file_id,
            )
            provenance_index.register(file_id, ProvenanceRef(source_type="filesystem", path=norm))
            summary_candidates.append(file_item)

        for idx, symbol in enumerate(deterministic_sort(request.referenced_symbols)):
            symbol_id = f"symbol-{idx}"
            finding = HistoricalFinding(
                finding_id=symbol_id,
                summary=symbol,
                provenance_id=symbol_id,
                related_symbols=(symbol,),
            )
            provenance_index.register(symbol_id, ProvenanceRef(source_type="symbol", path=symbol))
            summary_candidates.append(finding)

        for idx, finding in enumerate(request.prior_findings):
            finding_id = f"finding-{idx}"
            finding_summary = (
                str(finding) if not isinstance(finding, dict) else finding.get("summary", "")
            )
            hf = HistoricalFinding(
                finding_id=finding_id,
                summary=finding_summary,
                provenance_id=finding_id,
            )
            provenance_index.register(
                finding_id, ProvenanceRef(source_type="finding", path=finding_id)
            )
            summary_candidates.append(hf)

        summary: ContextSummary | None = None
        if summary_candidates:
            summary_result = self._summarizer.summarize(summary_candidates)
            summary_id = f"summary-{request.task_id}"
            provenance_index.register(
                summary_id,
                ProvenanceRef(
                    source_type="summary",
                    path=f"summary:{request.task_id}",
                    content_hash=hash_text(summary_result.text),
                ),
            )
            summary = ContextSummary(
                text=summary_result.text,
                source_provenance_ids=summary_result.source_provenance_ids,
                source_hashes=summary_result.source_hashes,
                level=summary_result.level,
                lossy=True,
                generated_by=summary_result.generated_by,
                source_revision="HEAD",
            )

        # Add raw items by priority until budget pressure.
        authority: list[str] = []
        acceptance_criteria: list[AcceptanceCriterion] = []
        current_diff: list[DiffInfo] = []
        test_evidence: list[TestEvidence] = []
        exclusions: list[Exclusion] = []
        consumed = 0
        item_count = 0
        truncation_events: list[str] = []

        priority_order = ["authority", "criterion", "diff", "test"]
        for priority in priority_order:
            for item, chars, source_id, category in raw_items:
                if category != priority:
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
                if category == "authority":
                    authority.append(str(item))
                elif category == "criterion":
                    acceptance_criteria.append(item)
                elif category == "diff":
                    current_diff.append(item)
                elif category == "test":
                    test_evidence.append(item)

        # Add summary last; drop lower-priority raw items if needed already handled above.
        summary_count = 0
        if summary is not None:
            summary_chars = len(summary.text) + 20
            if (
                max_items is None or item_count + 1 <= max_items
            ) and consumed + summary_chars <= usable_chars:
                consumed += summary_chars
                item_count += 1
                summary_count = 1
            else:
                exclusions.append(
                    Exclusion(
                        exclusion_id="excluded-summary",
                        reason="budget exhausted",
                        source_item_id="summary",
                        estimated_chars=summary_chars,
                    )
                )
                truncation_events.append("budget_exceeded:summary")

        authority_presence = (
            AuthorityPresence.RAW_INCLUDED if authority else AuthorityPresence.NOT_REQUIRED
        )

        task_metadata = TaskMetadata(
            task_id=request.task_id,
            role=request.role.value,
            risk=request.risk.name,
            requested_objective=request.requested_objective,
            constraints=request.constraints,
        )

        raw_item_count = (
            len(authority) + len(acceptance_criteria) + len(current_diff) + len(test_evidence)
        )

        packet = ContextPacket(
            packet_id=packet_id,
            authority=tuple(authority),
            acceptance_criteria=tuple(acceptance_criteria),
            current_diff=tuple(current_diff),
            test_evidence=tuple(test_evidence),
            task_metadata=task_metadata,
            exclusions=tuple(exclusions),
            provenance_index=dict(provenance_index._refs),
            authority_presence=authority_presence,
            raw_item_count=raw_item_count,
            summary_count=summary_count,
            estimated_input_chars=consumed,
            budget={
                "usable_chars": usable_chars,
                "reserved": budget_result.reserved,
                "total": budget_result.total_budget,
            },
            payload={"summary": summary.to_dict() if summary else None},
        )

        telemetry = ContextStrategyTelemetry(
            strategy=self.name,
            packet_id=packet_id,
            source_item_count=len(raw_items) + len(summary_candidates),
            raw_item_count=raw_item_count,
            summary_count=summary_count,
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
