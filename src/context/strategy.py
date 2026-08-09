"""Abstract context strategy and normalized build request."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.context.budget import ContextBudget
from src.context.schema import AuthorityContextItem, ContextPacket
from src.context.telemetry import ContextStrategyTelemetry
from src.policy.risk import RiskLevel
from src.routing.capabilities import ModelCapabilities
from src.routing.roles import ExecutionRole


def _authority_items_from_request(
    request: ContextBuildRequest,
) -> tuple[AuthorityContextItem, ...]:
    """Normalize authority_refs and authority_entries into typed items.

    Legacy string-only refs are upgraded to AuthorityContextItem with empty
    revision/content_hash, which forces RAW_INCLUDED and fails the build if
    they cannot fit. Rich entries may be used for RAW_REFERENCED fallback.
    """
    entries: list[AuthorityContextItem] = []
    seen_ids: set[str] = set()

    for i, entry in enumerate(request.authority_entries):
        if isinstance(entry, AuthorityContextItem):
            item = entry
        elif isinstance(entry, dict):
            item = AuthorityContextItem(
                authority_id=entry.get("authority_id", f"authority-entry-{i}"),
                provenance_id=entry.get("provenance_id", f"authority-entry-{i}"),
                full_source_ref=entry.get("full_source_ref", ""),
                revision=entry.get("revision", ""),
                content_hash=entry.get("content_hash", ""),
                content=entry.get("content"),
                raw_included=bool(entry.get("raw_included", True)),
            )
        else:
            text = str(entry)
            item = AuthorityContextItem(
                authority_id=f"authority-{i}",
                provenance_id=f"authority-{i}",
                full_source_ref=text,
                revision="",
                content_hash="",
                content=text,
                raw_included=True,
            )
        if item.authority_id in seen_ids:
            raise ValueError(f"duplicate authority_id {item.authority_id!r}")
        seen_ids.add(item.authority_id)
        entries.append(item)

    # Legacy authority_refs are appended after explicit entries so existing tests
    # keep working while richer entries take precedence.
    offset = len(entries)
    for i, ref in enumerate(request.authority_refs):
        text = str(ref)
        item = AuthorityContextItem(
            authority_id=f"authority-{offset + i}",
            provenance_id=f"authority-{offset + i}",
            full_source_ref=text,
            revision="",
            content_hash="",
            content=text,
            raw_included=True,
        )
        if item.authority_id in seen_ids:
            raise ValueError(f"duplicate authority_id {item.authority_id!r}")
        seen_ids.add(item.authority_id)
        entries.append(item)

    return tuple(entries)


@dataclass(frozen=True)
class ContextBuildRequest:
    """Inputs for building a context packet."""

    task_id: str
    role: ExecutionRole
    risk: RiskLevel
    model_capabilities: ModelCapabilities | None = None
    authority_refs: tuple[Any, ...] = ()
    authority_entries: tuple[AuthorityContextItem | dict[str, Any] | str, ...] = ()
    changed_files: tuple[str, ...] = ()
    referenced_symbols: tuple[str, ...] = ()
    test_failures: tuple[Any, ...] = ()
    prior_findings: tuple[Any, ...] = ()
    explicit_paths: tuple[str, ...] = ()
    budget: ContextBudget = field(default_factory=lambda: ContextBudget(primary_budget=0))
    exclusions: tuple[Any, ...] = ()
    repository_snapshot: dict[str, Any] = field(default_factory=dict)
    requested_objective: str = ""
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id or not self.task_id.strip():
            raise ValueError("task_id must be non-empty")


@dataclass(frozen=True)
class ContextStrategyResult:
    """Output of a context strategy."""

    strategy_name: str
    packet: ContextPacket
    telemetry: ContextStrategyTelemetry


class ContextStrategy(ABC):
    """Abstract base class for deterministic context construction strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def build(self, request: ContextBuildRequest) -> ContextStrategyResult:
        """Build a context packet from the supplied request."""
