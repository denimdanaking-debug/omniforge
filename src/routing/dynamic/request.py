"""Dynamic routing request primitive."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.policy.risk import RiskLevel
from src.routing.capabilities import CapabilityRequirement
from src.routing.policy import RoutingPin
from src.routing.roles import ExecutionRole


@dataclass(frozen=True)
class DynamicRoutingRequest:
    """Provider-neutral dynamic routing request with full execution context."""

    task_id: str
    project_id: str
    role: ExecutionRole
    risk: RiskLevel
    task_class: str
    capability_requirement: CapabilityRequirement | None = None
    required_context_tokens: int | None = None
    pin: RoutingPin | None = None
    reviewer_identities: tuple[str, ...] = ()
    coder_identities: tuple[str, ...] = ()
    expected_input_tokens: int | None = None
    expected_output_tokens: int | None = None
    timestamp: datetime | None = None
    state_snapshot_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not self.project_id.strip():
            raise ValueError("project_id must be non-empty")
        if not self.task_class.strip():
            raise ValueError("task_class must be non-empty")
        if self.required_context_tokens is not None and self.required_context_tokens < 0:
            raise ValueError("required_context_tokens must be non-negative")
        if self.expected_input_tokens is not None and self.expected_input_tokens < 0:
            raise ValueError("expected_input_tokens must be non-negative")
        if self.expected_output_tokens is not None and self.expected_output_tokens < 0:
            raise ValueError("expected_output_tokens must be non-negative")
