"""Tests for context budget estimation."""

from __future__ import annotations

import pytest

from src.context.budget import (
    BudgetType,
    ContextBudget,
    compute_usable_budget,
    estimate_tokens,
)


def test_estimate_tokens_conservative() -> None:
    assert estimate_tokens(0) == 0
    assert estimate_tokens(4) == 1
    assert estimate_tokens(100) == 25


def test_compute_usable_budget_with_known_tokens() -> None:
    budget = ContextBudget(
        primary_budget=10_000,
        reserve_for_system=1000,
        reserve_for_response=1000,
        safety_margin_fraction=0.1,
        budget_type=BudgetType.TOKENS_ESTIMATE,
    )
    result = compute_usable_budget(100_000, budget)
    assert result.total_budget == 100_000
    assert result.reserved == 2_000 + 10_000  # reserves + 10% safety
    assert result.usable == 88_000


def test_compute_usable_budget_unknown_capacity_is_conservative() -> None:
    budget = ContextBudget(primary_budget=10_000, budget_type=BudgetType.CHARACTERS)
    result = compute_usable_budget(None, budget)
    assert result.total_budget == 8_000
    assert result.usable <= 8_000


def test_context_budget_rejects_invalid_safety_margin() -> None:
    with pytest.raises(ValueError):
        ContextBudget(primary_budget=100, safety_margin_fraction=1.5)


def test_context_budget_rejects_negative_values() -> None:
    with pytest.raises(ValueError):
        ContextBudget(primary_budget=-1)
