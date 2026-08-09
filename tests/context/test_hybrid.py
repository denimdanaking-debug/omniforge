"""Tests for the hybrid context strategy."""

from __future__ import annotations

from src.context.budget import ContextBudget
from src.context.hybrid import HybridContextStrategy
from src.context.schema import AuthorityPresence
from src.context.strategy import ContextBuildRequest
from src.policy.risk import RiskLevel
from src.routing.capabilities import ModelCapabilities
from src.routing.roles import ExecutionRole


def _request(
    budget_chars: int = 10_000,
    authority: tuple[str, ...] = ("roadmap.md",),
    changed_files: tuple[str, ...] = ("src/a.py",),
    constraints: tuple[str, ...] = ("must pass tests",),
    paths: tuple[str, ...] = ("src/b.py",),
    symbols: tuple[str, ...] = ("func_a",),
    findings: tuple[object, ...] = (),
) -> ContextBuildRequest:
    return ContextBuildRequest(
        task_id="task-1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        model_capabilities=ModelCapabilities(context_tokens=4096),
        authority_refs=authority,
        changed_files=changed_files,
        constraints=constraints,
        explicit_paths=paths,
        referenced_symbols=symbols,
        prior_findings=findings,
        budget=ContextBudget(primary_budget=budget_chars),
    )


def test_raw_authority_preserved() -> None:
    strategy = HybridContextStrategy()
    result = strategy.build(_request())
    assert result.packet.authority == ("roadmap.md",)
    assert result.packet.authority_presence == AuthorityPresence.RAW_INCLUDED


def test_lower_priority_summarized() -> None:
    strategy = HybridContextStrategy()
    result = strategy.build(_request())
    assert result.packet.summary_count == 1
    assert result.packet.payload.get("summary") is not None


def test_exclusions_recorded_under_budget_pressure() -> None:
    strategy = HybridContextStrategy()
    result = strategy.build(_request(budget_chars=25))
    assert result.packet.exclusions
    assert result.telemetry.excluded_count > 0
