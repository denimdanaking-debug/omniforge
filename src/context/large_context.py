"""Large-context strategy: use available model capacity while reserving margins."""

from __future__ import annotations

from typing import Any

from src.context._utils import deterministic_sort, normalize_path
from src.context.budget import ContextBudgetError, compute_usable_budget
from src.context.provenance import ProvenanceIndex
from src.context.schema import (
    AuthorityContextItem,
    AuthorityPresence,
    ContextPacket,
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


class LargeContextStrategy(ContextStrategy):
    """Strategy that includes more raw source when model capacity is known."""

    @property
    def name(self) -> str:
        return "large_context"

    def build(self, request: ContextBuildRequest) -> ContextStrategyResult:
        packet_id = f"large-{request.task_id}"
        budget_result = compute_usable_budget(
            request.model_capabilities.context_tokens if request.model_capabilities else None,
            request.budget,
        )
        usable_chars = budget_result.usable
        max_items = request.budget.max_items

        provenance_index = ProvenanceIndex()

        # Authority first.
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

        items: list[tuple[Any, int, str, str]] = []  # (item, chars, source_id, category)

        # Include raw content for all changed files (large-context assumption).
        for idx, path in enumerate(deterministic_sort(request.changed_files)):
            file_id = f"changed-{idx}"
            norm = normalize_path(path)
            file_item = RelevantFile(
                file_id=file_id,
                path=norm,
                reason="changed_raw",
                provenance_id=file_id,
            )
            provenance_index.register(file_id, ProvenanceRef(source_type="git", path=norm))
            # Simulate larger raw inclusion with file content placeholder.
            content = f"// full content of {norm}\n"
            chars = len(content) + len(norm) + 20
            items.append((file_item, chars, file_id, "file"))

        # Referenced symbols.
        for idx, symbol in enumerate(deterministic_sort(request.referenced_symbols)):
            symbol_id = f"symbol-{idx}"
            finding = HistoricalFinding(
                finding_id=symbol_id,
                summary=symbol,
                provenance_id=symbol_id,
                related_symbols=(symbol,),
            )
            provenance_index.register(symbol_id, ProvenanceRef(source_type="symbol", path=symbol))
            items.append((finding, len(symbol) + 20, symbol_id, "finding"))

        # Test failures.
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
            items.append((evidence, len(name) + len(log) + 20, test_id, "test"))

        # Prior findings.
        for idx, finding in enumerate(request.prior_findings):
            finding_id = f"finding-{idx}"
            summary = str(finding) if not isinstance(finding, dict) else finding.get("summary", "")
            hf = HistoricalFinding(
                finding_id=finding_id,
                summary=summary,
                provenance_id=finding_id,
            )
            provenance_index.register(
                finding_id, ProvenanceRef(source_type="finding", path=finding_id)
            )
            items.append((hf, len(summary) + 20, finding_id, "finding"))

        # Explicit paths.
        for idx, path in enumerate(deterministic_sort(request.explicit_paths)):
            path_id = f"nearby-{idx}"
            norm = normalize_path(path)
            file_item = RelevantFile(
                file_id=path_id,
                path=norm,
                reason="nearby_context",
                provenance_id=path_id,
            )
            provenance_index.register(path_id, ProvenanceRef(source_type="filesystem", path=norm))
            items.append((file_item, len(norm) + 20, path_id, "file"))

        relevant_files: list[RelevantFile] = []
        test_evidence: list[TestEvidence] = []
        historical_findings: list[HistoricalFinding] = []
        exclusions: list[Exclusion] = []
        truncation_events: list[str] = []

        for item, chars, source_id, category in items:
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
            if category == "file":
                relevant_files.append(item)
            elif category == "test":
                test_evidence.append(item)
            elif category == "finding":
                historical_findings.append(item)

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
            raw_authority_count
            + len(relevant_files)
            + len(test_evidence)
            + len(historical_findings)
        )

        packet = ContextPacket(
            packet_id=packet_id,
            authority=tuple(authority),
            relevant_files=tuple(relevant_files),
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
            source_item_count=len(items) + len(authority_items),
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
