"""Tests for the hierarchical summary strategy."""

from __future__ import annotations

from dataclasses import replace

from src.context.budget import ContextBudget
from src.context.hierarchical import (
    DeterministicTestSummarizer,
    HierarchicalContextStrategy,
)
from src.context.schema import AuthorityPresence
from src.context.strategy import ContextBuildRequest
from src.context.validation import ContextPacketValidator
from src.policy.risk import RiskLevel
from src.routing.capabilities import ModelCapabilities
from src.routing.roles import ExecutionRole


def _request(
    budget_chars: int = 10_000,
    context_tokens: int | None = 4096,
    authority: tuple[str, ...] = ("roadmap.md",),
    changed_files: tuple[str, ...] = ("src/a.py",),
    paths: tuple[str, ...] = ("src/b.py",),
    symbols: tuple[str, ...] = ("func_a",),
) -> ContextBuildRequest:
    return ContextBuildRequest(
        task_id="task-1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        model_capabilities=ModelCapabilities(context_tokens=context_tokens)
        if context_tokens
        else None,
        authority_refs=authority,
        changed_files=changed_files,
        explicit_paths=paths,
        referenced_symbols=symbols,
        budget=ContextBudget(primary_budget=budget_chars),
    )


def test_deterministic_fake_summarizer() -> None:
    summarizer = DeterministicTestSummarizer()
    result = summarizer.summarize(["a", "b", "c"])
    assert "3 item(s)" in result.text
    assert len(result.source_hashes) == 3
    assert result.level == "section"


def test_summary_provenance_preserved() -> None:
    strategy = HierarchicalContextStrategy()
    result = strategy.build(_request())
    assert result.packet.summary_count == 1
    assert len(result.packet.summaries) == 1
    summary = result.packet.summaries[0]
    assert summary.text
    assert summary.source_provenance_ids
    # Source provenance IDs should reference actual source IDs, not summary-source-0.
    assert all(not sid.startswith("summary-source-") for sid in summary.source_provenance_ids)
    assert any(ref.source_type == "summary" for ref in result.packet.provenance_index.values())


def test_authority_kept_raw() -> None:
    strategy = HierarchicalContextStrategy()
    result = strategy.build(_request())
    assert len(result.packet.authority) == 1
    assert result.packet.authority[0].content == "roadmap.md"
    assert result.packet.authority[0].raw_included is True
    assert result.packet.authority_presence == AuthorityPresence.RAW_INCLUDED


def test_validator_rejects_summary_without_provenance() -> None:
    strategy = HierarchicalContextStrategy()
    result = strategy.build(_request())
    # Manually strip summary provenance to simulate violation.
    bad_packet = replace(result.packet, summary_count=1, provenance_index={})
    issues = ContextPacketValidator().validate(bad_packet)
    assert any(issue.code == "SUMMARY_WITHOUT_PROVENANCE" for issue in issues)
