"""Normalized, provider-neutral request contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from src.context.schema import ContextPacket
from src.policy.risk import RiskLevel
from src.providers.identity import ProviderIdentity
from src.routing.inference_route import InferenceRouteIdentity
from src.routing.model_identity import ModelIdentity
from src.routing.roles import ExecutionRole

__all__ = [
    "CapabilityRequirement",
    "ContextPacket",
    "Message",
    "MessageRole",
    "ProviderRequest",
    "ReasoningMode",
    "StructuredOutputRequirement",
    "TaskLineage",
    "ToolChoiceMode",
    "ToolDefinition",
    "ToolParameter",
]


class MessageRole(Enum):
    """Normalized message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolChoiceMode(Enum):
    """Normalized tool-choice modes."""

    AUTO = auto()
    REQUIRED = auto()
    NONE = auto()
    FORBIDDEN = auto()


class ReasoningMode(Enum):
    """Normalized reasoning control modes.

    Provider adapters translate these into provider-specific controls; unsupported
    modes must fail eligibility rather than being silently ignored.
    """

    DEFAULT = auto()
    EFFORT_LOW = auto()
    EFFORT_MEDIUM = auto()
    EFFORT_HIGH = auto()
    DISABLED = auto()


@dataclass(frozen=True)
class Message:
    """A single normalized message."""

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolParameter:
    """JSON-schema-like parameter descriptor for a tool."""

    name: str
    schema: dict[str, Any]
    required: bool = True
    description: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    """Normalized tool definition."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)


@dataclass(frozen=True)
class StructuredOutputRequirement:
    """Structured-output / JSON schema requirement."""

    schema: dict[str, Any] | None = None
    name: str | None = None
    enforce_schema: bool = True


@dataclass(frozen=True)
class CapabilityRequirement:
    """Distinguishes required, preferred, and unsupported features."""

    feature: str
    required: bool = False
    preferred: bool = False


@dataclass(frozen=True)
class TaskLineage:
    """Project/run/task lineage."""

    project_id: str
    run_id: str | None = None
    task_id: str | None = None
    parent_task_id: str | None = None


@dataclass(frozen=True)
class ProviderRequest:
    """Provider-neutral request representation.

    Provider-neutral does not mean pretending all providers support every feature.
    Required capabilities that are unsupported must fail eligibility/validation
    before dispatch rather than being silently ignored.
    """

    request_id: str
    execution_role: ExecutionRole
    risk_level: RiskLevel
    system_instructions: list[str] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    context_packets: list[ContextPacket] = field(default_factory=list)
    tools: list[ToolDefinition] = field(default_factory=list)
    tool_choice: ToolChoiceMode = ToolChoiceMode.AUTO
    structured_output: StructuredOutputRequirement | None = None
    temperature: float | None = None
    reasoning: ReasoningMode = ReasoningMode.DEFAULT
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = False
    target_model: ModelIdentity | None = None
    target_route: InferenceRouteIdentity | None = None
    target_provider: ProviderIdentity | None = None
    capability_requirements: list[CapabilityRequirement] = field(default_factory=list)
    lineage: TaskLineage | None = None
    correlation_id: str | None = None
    cancellation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id or not self.request_id.strip():
            raise ValueError("ProviderRequest.request_id is required")

    def required_capabilities(self) -> list[str]:
        """Return features marked as required."""
        return [req.feature for req in self.capability_requirements if req.required]

    def preferred_capabilities(self) -> list[str]:
        """Return features marked as preferred."""
        return [req.feature for req in self.capability_requirements if req.preferred]

    def requires_streaming(self) -> bool:
        """Return True if streaming is a required capability."""
        return any(
            req.feature == "streaming" and req.required for req in self.capability_requirements
        )

    def requires_tools(self) -> bool:
        """Return True if tool use is a required capability."""
        return bool(self.tools) or any(
            req.feature == "tool_use" and req.required for req in self.capability_requirements
        )

    def requires_structured_output(self) -> bool:
        """Return True if structured output is a required capability."""
        enforced = self.structured_output is not None and self.structured_output.enforce_schema
        required = any(
            req.feature == "structured_output" and req.required
            for req in self.capability_requirements
        )
        return enforced or required
