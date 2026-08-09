"""Tests for context strategy telemetry."""

from __future__ import annotations

import pytest

from src.context.budget import ContextBudget
from src.context.schema import AuthorityPresence
from src.context.strategy import ContextBuildRequest
from src.context.targeted import TargetedContextStrategy
from src.context.telemetry import ContextStrategyTelemetry
from src.policy.risk import RiskLevel
from src.routing.capabilities import ModelCapabilities
from src.routing.roles import ExecutionRole


def test_telemetry_populated_by_strategy() -> None:
    strategy = TargetedContextStrategy()
    request = ContextBuildRequest(
        task_id="task-1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        model_capabilities=ModelCapabilities(context_tokens=4096),
        authority_refs=("roadmap.md",),
        changed_files=("src/a.py",),
        budget=ContextBudget(primary_budget=10_000),
    )
    result = strategy.build(request)
    telemetry = result.telemetry
    assert telemetry.strategy == "targeted"
    assert telemetry.packet_id == result.packet.packet_id
    assert telemetry.source_item_count > 0
    assert telemetry.provenance_coverage >= 0.0
    assert telemetry.authority_presence == AuthorityPresence.RAW_INCLUDED


def test_telemetry_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        ContextStrategyTelemetry(
            strategy="x",
            packet_id="p",
            source_item_count=-1,
            raw_item_count=0,
            summary_count=0,
            estimated_input_chars=0,
            context_capacity=None,
            budget_consumed={},
            excluded_count=0,
            authority_presence=AuthorityPresence.NOT_REQUIRED,
            provenance_coverage=1.0,
            truncation_events=(),
        )
