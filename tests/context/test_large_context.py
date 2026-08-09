"""Tests for the large-context strategy."""

from __future__ import annotations

from src.context.budget import ContextBudget
from src.context.large_context import LargeContextStrategy
from src.context.strategy import ContextBuildRequest
from src.policy.risk import RiskLevel
from src.routing.capabilities import ModelCapabilities
from src.routing.roles import ExecutionRole


def test_capacity_reserve_reduces_usable_budget() -> None:
    strategy = LargeContextStrategy()
    request = ContextBuildRequest(
        task_id="task-1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        model_capabilities=ModelCapabilities(context_tokens=100_000),
        authority_refs=("roadmap.md",),
        changed_files=("src/a.py",),
        budget=ContextBudget(
            primary_budget=500_000,
            reserve_for_system=10_000,
            reserve_for_response=5_000,
            safety_margin_fraction=0.1,
        ),
    )
    result = strategy.build(request)
    assert result.strategy_name == "large_context"
    assert result.telemetry.context_capacity == 100_000
    # 100k tokens * 4 chars/token = 400k; minus 15k reserves and 40k safety = 345k usable.
    assert result.packet.budget["usable_chars"] == 345_000


def test_unknown_capacity_falls_back_conservatively() -> None:
    strategy = LargeContextStrategy()
    request = ContextBuildRequest(
        task_id="task-1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        model_capabilities=None,
        authority_refs=("roadmap.md",),
        budget=ContextBudget(primary_budget=10_000),
    )
    result = strategy.build(request)
    assert result.telemetry.context_capacity is None
    assert result.packet.budget["usable_chars"] <= 8_000
