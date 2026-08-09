"""Context budget estimation and capacity-aware reservation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class BudgetType(Enum):
    """Unit for a context budget."""

    CHARACTERS = auto()
    TOKENS_ESTIMATE = auto()
    BYTES = auto()
    ITEM_COUNT = auto()


UNKNOWN_CAPACITY_CHARACTERS = 8_000
DEFAULT_TOKEN_TO_CHAR_RATIO = 4


@dataclass(frozen=True)
class ContextBudget:
    """Budget allocation for a context packet."""

    primary_budget: int
    reserve_for_system: int = 0
    reserve_for_response: int = 0
    safety_margin_fraction: float = 0.0
    max_items: int | None = None
    budget_type: BudgetType = BudgetType.CHARACTERS

    def __post_init__(self) -> None:
        if self.primary_budget < 0:
            raise ValueError("primary_budget must be non-negative")
        if self.reserve_for_system < 0:
            raise ValueError("reserve_for_system must be non-negative")
        if self.reserve_for_response < 0:
            raise ValueError("reserve_for_response must be non-negative")
        if not 0.0 <= self.safety_margin_fraction <= 1.0:
            raise ValueError("safety_margin_fraction must be between 0.0 and 1.0")
        if self.max_items is not None and self.max_items < 0:
            raise ValueError("max_items must be non-negative when provided")


@dataclass(frozen=True)
class BudgetResult:
    """Result of a budget computation."""

    usable: int
    total_budget: int
    consumed: int
    reserved: int
    truncated: bool


def estimate_tokens(chars: int) -> int:
    """Return a conservative token estimate from a character count.

    This is a coarse heuristic (approximately four characters per token) and
    should not be treated as exact tokenizer output.
    """
    if chars <= 0:
        return 0
    return max(1, chars // DEFAULT_TOKEN_TO_CHAR_RATIO)


def compute_usable_budget(
    model_context_tokens: int | None,
    budget: ContextBudget,
) -> BudgetResult:
    """Compute usable budget given model capacity and reservations.

    If ``model_context_tokens`` is unknown, a conservative character budget is
    returned.
    """
    if budget.budget_type == BudgetType.TOKENS_ESTIMATE and model_context_tokens is not None:
        total = model_context_tokens
        reserved = budget.reserve_for_system + budget.reserve_for_response
        safety = int(total * budget.safety_margin_fraction)
        usable = total - reserved - safety
        usable = max(0, usable)
        return BudgetResult(
            usable=usable,
            total_budget=total,
            consumed=0,
            reserved=reserved + safety,
            truncated=False,
        )

    if model_context_tokens is None:
        capacity = UNKNOWN_CAPACITY_CHARACTERS
    else:
        capacity = model_context_tokens * DEFAULT_TOKEN_TO_CHAR_RATIO

    total = min(capacity, budget.primary_budget) if budget.primary_budget > 0 else capacity
    reserved = budget.reserve_for_system + budget.reserve_for_response
    safety = int(total * budget.safety_margin_fraction)
    usable = total - reserved - safety
    usable = max(0, usable)

    return BudgetResult(
        usable=usable,
        total_budget=total,
        consumed=0,
        reserved=reserved + safety,
        truncated=False,
    )
