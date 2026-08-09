"""Hybrid context strategy: raw authority + summaries for supporting context."""

from __future__ import annotations

from typing import Any

from src.context._utils import deterministic_sort, hash_text, normalize_path
from src.context.budget import ContextBudgetError, compute_usable_budget
from src.context.hierarchical import DeterministicTestSummarizer, Summarizer
from src.context.provenance import ProvenanceIndex
from src.context.schema import (
    AcceptanceCriterion,
    AuthorityContextItem,
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
from src.context.strategy import (
    ContextBuildRequest,
    ContextStrategy,
    ContextStrategyResult,
    _authority_items_from_request,
)
from src.context.targeted import _provenance_coverage
from src.context.telemetry import ContextStrategyTelemetry


def _allocate_authority_items(
    authority_items: tuple[AuthorityContextItem, ...],
    usable_chars: int,
    max_items: int | None,
    consumed: int,
    item_count: int,
) -> tuple[list[AuthorityContextItem], int, int, int]:
    """Allocate authority items first, falling back to RAW_REFERENCED when needed."""
    authority: list[AuthorityContextItem] = []
    raw_count = 0
    for item in authority_items:
        if item.raw_included:
            chars = len(item.content) if item.content else 0
            fits = (max_items is None or item_count + 1 <= max_items) and (
                consumed + chars <= usable_chars
            )
            if fits:
                authority.append(item)
                consumed += chars
                item_count += 1
                raw_count += 1
            else:
                if item.revision and item.content_hash:
                    authority.append(
                        AuthorityContextItem(
                            authority_id=item.authority_id,
                            provenance_id=item.provenance_id,
                            full_source_ref=item.full_source_ref,
                            revision=item.revision,
                            content_hash=item.content_hash,
                            content=None,
                            raw_included=False,
                        )
                    )
                else:
                    raise ContextBudgetError(
                        f"Authority item {item.authority_id!r} cannot fit in budget and lacks "
                        "revision/content_hash for reference"
                    )
        else:
            if not item.revision or not item.content_hash:
                raise ValueError(
                    f"Authority item {item.authority_id!r} marked raw_included=False but missing "
                    "revision or content_hash"
                )
            authority.append(
                AuthorityContextItem(
                    authority_id=item.authority_id,
                    provenance_id=item.provenance_id,
                    full_source_ref=item.full_source_ref,
                    revision=item.revision,
                    content_hash=item.content_hash,
                    content=None,
                    raw_included=False,
                )
            )
    return authority, consumed, item_count, raw_count


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

        # Authority is always raw or referenced first.
        authority_items = _authority_items_from_request(request)
        for item in authority_items:
            provenance_index.register(
                item.provenance_id,
                ProvenanceRef(
                    source_type="authority",
                    path=item.full_source_ref,
                    authority_level="roadmap",
                    revision=item.revision or None,
                    content_hash=item.content_hash or None,
                ),
            )

        consumed = 0
        item_count = 0
        authority, consumed, item_count, raw_authority_count = _allocate_authority_items(
            authority_items, usable_chars, max_items, consumed, item_count
        )

        first_authority_provenance_id = (
            authority_items[0].provenance_id if authority_items else None
        )

        raw_items: list[tuple[Any, int, str, str]] = []
        summary_candidates: list[Any] = []

        # Acceptance criteria are raw.
        for idx, constraint in enumerate(request.constraints):
            criterion_id = f"criterion-{idx}"
            criterion = AcceptanceCriterion(
                criterion_id=criterion_id,
                text=constraint,
                authority_refs=(first_authority_provenance_id,)
                if first_authority_provenance_id
                else (),
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
        acceptance_criteria: list[AcceptanceCriterion] = []
        current_diff: list[DiffInfo] = []
        test_evidence: list[TestEvidence] = []
        exclusions: list[Exclusion] = []
        truncation_events: list[str] = []

        priority_order = ["criterion", "diff", "test"]
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
                if category == "criterion":
                    acceptance_criteria.append(item)
                elif category == "diff":
                    current_diff.append(item)
                elif category == "test":
                    test_evidence.append(item)

        # Add summary last; drop lower-priority raw items if needed already handled above.
        summaries: list[ContextSummary] = []
        if summary is not None:
            summary_chars = len(summary.text) + 20
            if (
                max_items is None or item_count + 1 <= max_items
            ) and consumed + summary_chars <= usable_chars:
                consumed += summary_chars
                item_count += 1
                summaries.append(summary)
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

        if authority_items:
            authority_presence = (
                AuthorityPresence.RAW_INCLUDED
                if raw_authority_count > 0
                else AuthorityPresence.RAW_REFERENCED
            )
        else:
            authority_presence = AuthorityPresence.NOT_REQUIRED

        task_metadata = TaskMetadata(
            task_id=request.task_id,
            role=request.role.value,
            risk=request.risk.name,
            requested_objective=request.requested_objective,
            constraints=request.constraints,
        )
        provenance_index.register(
            task_metadata.task_id,
            ProvenanceRef(source_type="task_metadata", path=task_metadata.task_id),
        )

        raw_item_count = (
            raw_authority_count + len(acceptance_criteria) + len(current_diff) + len(test_evidence)
        )

        packet = ContextPacket(
            packet_id=packet_id,
            authority=tuple(authority),
            acceptance_criteria=tuple(acceptance_criteria),
            current_diff=tuple(current_diff),
            test_evidence=tuple(test_evidence),
            summaries=tuple(summaries),
            task_metadata=task_metadata,
            exclusions=tuple(exclusions),
            provenance_index=dict(provenance_index._refs),
            authority_presence=authority_presence,
            raw_item_count=raw_item_count,
            summary_count=len(summaries),
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
            source_item_count=len(raw_items) + len(summary_candidates) + len(authority_items),
            raw_item_count=raw_item_count,
            summary_count=len(summaries),
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
