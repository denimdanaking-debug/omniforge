"""Outcome records for context-quality learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextOutcomeRecord:
    """Learning record linking a context packet to execution outcome."""

    packet_id: str
    strategy: str
    model_id: str
    role: str
    risk: str
    task_class: str
    context_size: int
    accepted: bool
    validation_result: dict[str, Any]
    review_result: str | None
    repair_required: bool
    failure_category: str | None

    def __post_init__(self) -> None:
        if self.context_size < 0:
            raise ValueError("context_size must be non-negative")
