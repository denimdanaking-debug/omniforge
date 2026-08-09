"""Abstract context strategy and normalized build request."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.context.budget import ContextBudget
from src.context.schema import ContextPacket
from src.context.telemetry import ContextStrategyTelemetry
from src.policy.risk import RiskLevel
from src.routing.capabilities import ModelCapabilities
from src.routing.roles import ExecutionRole


@dataclass(frozen=True)
class ContextBuildRequest:
    """Inputs for building a context packet."""

    task_id: str
    role: ExecutionRole
    risk: RiskLevel
    model_capabilities: ModelCapabilities | None = None
    authority_refs: tuple[Any, ...] = ()
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
