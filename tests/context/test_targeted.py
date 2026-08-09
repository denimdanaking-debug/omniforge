"""Tests for the targeted context strategy."""

from __future__ import annotations

from typing import Any

from src.context.budget import ContextBudget
from src.context.schema import AuthorityPresence
from src.context.strategy import ContextBuildRequest
from src.context.targeted import TargetedContextStrategy
from src.policy.risk import RiskLevel
from src.routing.capabilities import ModelCapabilities
from src.routing.roles import ExecutionRole


def _request(
    budget_chars: int = 10_000,
    context_tokens: int = 4096,
    authority: tuple[str, ...] = ("roadmap.md",),
    authority_entries: tuple[dict[str, Any], ...] = (),
    changed_files: tuple[str, ...] = ("src/a.py",),
    symbols: tuple[str, ...] = ("func_a",),
    tests: tuple[object, ...] = (),
    findings: tuple[object, ...] = (),
    paths: tuple[str, ...] = (),
) -> ContextBuildRequest:
    return ContextBuildRequest(
        task_id="task-1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        model_capabilities=ModelCapabilities(context_tokens=context_tokens),
        authority_refs=authority if not authority_entries else (),
        authority_entries=authority_entries,
        changed_files=changed_files,
        referenced_symbols=symbols,
        test_failures=tests,
        prior_findings=findings,
        explicit_paths=paths,
        budget=ContextBudget(primary_budget=budget_chars),
    )


def test_prioritization_order() -> None:
    strategy = TargetedContextStrategy()
    result = strategy.build(_request())
    assert result.strategy_name == "targeted"
    assert result.packet.authority
    assert result.packet.relevant_files
    assert result.packet.historical_findings
    assert result.packet.authority_presence == AuthorityPresence.RAW_INCLUDED


def test_budget_protects_authority() -> None:
    strategy = TargetedContextStrategy()
    result = strategy.build(
        _request(
            budget_chars=50,
            authority=("very important authority text",),
            changed_files=("src/one.py", "src/two.py"),
            symbols=("sym",),
        )
    )
    # Authority should be included; lower-priority items dropped.
    assert result.packet.authority
    assert len(result.packet.exclusions) > 0
    # All exclusions should be lower-priority items, not authority.
    for exclusion in result.packet.exclusions:
        assert exclusion.source_item_id is not None
        assert not exclusion.source_item_id.startswith("authority")


def test_exclusions_recorded() -> None:
    strategy = TargetedContextStrategy()
    # Use RAW_REFERENCED authority so the tiny budget does not raise ContextBudgetError.
    result = strategy.build(
        _request(
            budget_chars=5,
            authority=(),
            authority_entries=(
                {
                    "authority_id": "auth-1",
                    "provenance_id": "auth-prov-1",
                    "full_source_ref": "roadmap.md",
                    "revision": "a" * 40,
                    "content_hash": "b" * 64,
                    "content": "roadmap.md",
                    "raw_included": True,
                },
            ),
        )
    )
    assert result.packet.exclusions
    assert result.telemetry.excluded_count == len(result.packet.exclusions)
