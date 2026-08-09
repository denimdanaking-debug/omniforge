"""Context-overflow recovery integration with Phase 7 context construction.

Ensures required authority is never silently dropped when rebuilding context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.context.strategy import ContextBuildRequest
from src.recovery.failure_classification import ContextOverflowMetadata


@dataclass(frozen=True)
class ContextRebuildResult:
    """Outcome of a context rebuild attempt."""

    success: bool
    strategy_name: str
    authority_presence: str
    estimated_input_chars: int
    excluded_material: tuple[str, ...]
    authority_items_present: int
    authority_items_raw: int
    rebuild_attempt: int


def build_context_overflow_metadata(
    request: ContextBuildRequest,
    result: ContextRebuildResult | None,
    model_context_tokens: int | None,
) -> ContextOverflowMetadata:
    """Build overflow metadata from a context build request and result."""
    estimated = 0
    if result is not None:
        estimated = result.estimated_input_chars
    elif request.budget and hasattr(request.budget, "usable"):
        estimated = request.budget.usable
    return ContextOverflowMetadata(
        estimated_input_chars=estimated,
        model_context_tokens=model_context_tokens,
        authority_required=bool(request.authority_refs or request.authority_entries),
        authority_items_present=len(request.authority_refs) + len(request.authority_entries),
        authority_items_raw=result.authority_items_raw if result else 0,
        rebuild_attempts=result.rebuild_attempt if result else 0,
    )


def context_rebuild_attempt_exceeds_budget(
    metadata: ContextOverflowMetadata,
) -> bool:
    """Return True if the estimated input exceeds model context capacity."""
    if metadata.estimated_input_chars is None or metadata.model_context_tokens is None:
        return False
    return metadata.estimated_input_chars > metadata.model_context_tokens


def context_recovery_evidence(
    metadata: ContextOverflowMetadata,
    rebuild_result: ContextRebuildResult | None,
) -> dict[str, Any]:
    """Build bounded evidence packet for context recovery decisions."""
    return {
        "estimated_input_chars": metadata.estimated_input_chars,
        "model_context_tokens": metadata.model_context_tokens,
        "authority_required": metadata.authority_required,
        "authority_items_present": metadata.authority_items_present,
        "authority_items_raw": metadata.authority_items_raw,
        "rebuild_attempts": metadata.rebuild_attempts,
        "rebuild_strategy": rebuild_result.strategy_name if rebuild_result else None,
        "excluded_material": sorted(rebuild_result.excluded_material) if rebuild_result else [],
    }
