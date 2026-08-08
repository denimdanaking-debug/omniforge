"""Normalized, provider-neutral response contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from src.providers.identity import ProviderIdentity
from src.routing.inference_route import InferenceRouteIdentity
from src.routing.model_identity import ModelIdentity


class FinishReason(Enum):
    """Normalized finish reasons."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class StreamingState(Enum):
    """Streaming completion state."""

    NOT_STREAMING = auto()
    STARTED = auto()
    IN_PROGRESS = auto()
    COMPLETE = auto()
    FAILED = auto()


@dataclass(frozen=True)
class Usage:
    """Token usage with explicit unknown values."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None

    def has_any_known(self) -> bool:
        """Return True if at least one usage metric is known."""
        return any(
            value is not None
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.cached_tokens,
                self.reasoning_tokens,
                self.total_tokens,
            )
        )


@dataclass(frozen=True)
class ToolCallArgument:
    """A single tool-call argument."""

    name: str
    value: Any


@dataclass(frozen=True)
class ToolCall:
    """Normalized tool call."""

    id: str
    tool_name: str
    arguments: list[ToolCallArgument] = field(default_factory=list)
    raw_arguments: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProviderResponse:
    """Provider-neutral response representation.

    Do not invent measurements providers do not expose. Unknown/unavailable
    metrics remain explicitly ``None`` rather than fabricated as zero.
    """

    request_id: str
    provider_id: ProviderIdentity
    model_id: ModelIdentity
    route_id: InferenceRouteIdentity | None = None
    text: str | None = None
    structured_result: dict[str, Any] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: FinishReason = FinishReason.UNKNOWN
    streaming_state: StreamingState = StreamingState.NOT_STREAMING
    usage: Usage = field(default_factory=Usage)
    latency_seconds: float | None = None
    provider_request_id: str | None = None
    warnings: list[str] = field(default_factory=list)
    error_reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id or not self.request_id.strip():
            raise ValueError("ProviderResponse.request_id is required")

    def has_tool_calls(self) -> bool:
        """Return True if the response includes at least one tool call."""
        return bool(self.tool_calls)

    def has_structured_result(self) -> bool:
        """Return True if the response includes a structured result."""
        return self.structured_result is not None
