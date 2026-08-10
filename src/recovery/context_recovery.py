"""Context-overflow recovery integration with Phase 7 context construction.

Ensures required authority is never silently dropped when rebuilding context.
All capacity comparisons use a single canonical unit (tokens).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.context.budget import BudgetType, estimate_tokens
from src.context.strategy import ContextBuildRequest
from src.recovery.failure_classification import ContextOverflowMetadata


@dataclass(frozen=True)
class ContextRebuildResult:
    """Outcome of a context rebuild attempt."""

    success: bool
    strategy_name: str
    authority_presence: str
    estimated_input_chars: int
    estimated_input_tokens: int
    required_context_tokens: int
    excluded_material: tuple[str, ...]
    authority_items_present: int
    authority_items_raw: int
    rebuild_attempt: int


def _required_tokens_from_request(request: ContextBuildRequest) -> int:
    """Return the canonical required-token estimate for a build request."""
    budget = request.budget
    if budget.budget_type is BudgetType.TOKENS_ESTIMATE:
        return max(0, budget.primary_budget)
    return estimate_tokens(max(0, budget.primary_budget))


def build_context_overflow_metadata(
    request: ContextBuildRequest,
    result: ContextRebuildResult | None,
    model_context_tokens: int | None,
) -> ContextOverflowMetadata:
    """Build overflow metadata from a context build request and result."""
    required_tokens = _required_tokens_from_request(request)
    estimated_chars = 0
    estimated_tokens = 0
    if result is not None:
        estimated_chars = result.estimated_input_chars
        estimated_tokens = result.estimated_input_tokens
        required_tokens = result.required_context_tokens
    elif request.budget is not None:
        estimated_chars = max(0, request.budget.primary_budget)
        estimated_tokens = estimate_tokens(estimated_chars)
    return ContextOverflowMetadata(
        estimated_input_chars=estimated_chars,
        estimated_input_tokens=estimated_tokens,
        model_context_tokens=model_context_tokens,
        required_context_tokens=required_tokens,
        authority_required=bool(request.authority_refs or request.authority_entries),
        authority_items_present=len(request.authority_refs) + len(request.authority_entries),
        authority_items_raw=result.authority_items_raw if result else 0,
        rebuild_attempts=result.rebuild_attempt if result else 0,
    )


def context_rebuild_attempt_exceeds_budget(
    metadata: ContextOverflowMetadata,
) -> bool:
    """Return True if the required context exceeds model token capacity."""
    required = metadata.required_context_tokens
    if required is None and metadata.estimated_input_tokens is not None:
        required = metadata.estimated_input_tokens
    if required is None and metadata.estimated_input_chars is not None:
        required = estimate_tokens(metadata.estimated_input_chars)
    capacity = metadata.model_context_tokens
    if required is None or capacity is None:
        return False
    return required > capacity


def context_recovery_evidence(
    metadata: ContextOverflowMetadata,
    rebuild_result: ContextRebuildResult | None,
) -> dict[str, Any]:
    """Build bounded evidence packet for context recovery decisions."""
    return {
        "estimated_input_chars": metadata.estimated_input_chars,
        "estimated_input_tokens": metadata.estimated_input_tokens,
        "required_context_tokens": metadata.required_context_tokens,
        "model_context_tokens": metadata.model_context_tokens,
        "authority_required": metadata.authority_required,
        "authority_items_present": metadata.authority_items_present,
        "authority_items_raw": metadata.authority_items_raw,
        "rebuild_attempts": metadata.rebuild_attempts,
        "rebuild_strategy": rebuild_result.strategy_name if rebuild_result else None,
        "excluded_material": sorted(rebuild_result.excluded_material) if rebuild_result else [],
    }
