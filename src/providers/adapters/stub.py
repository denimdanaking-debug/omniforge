"""Deterministic stub adapter for contract tests.

This adapter does not call any external API. It produces predictable responses
and can be configured to exercise error, health, quota, streaming, tool-call,
structured-output, and cancellation scenarios.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from providers.contracts.adapter import AdapterIdentity, ProviderAdapter
from providers.contracts.capabilities import (
    CapabilitySupport,
    ExecutionRole,
    ModelCapabilities,
)
from providers.contracts.errors import ProviderError, ProviderErrorCode
from providers.contracts.health import HealthStatus, ProviderHealth
from providers.contracts.identity import ModelId, ProviderId, RouteId
from providers.contracts.quota import ProviderQuota, QuotaSignal
from providers.contracts.request import ProviderRequest, ToolChoiceMode
from providers.contracts.response import (
    FinishReason,
    ProviderResponse,
    StreamingState,
    ToolCall,
    ToolCallArgument,
    Usage,
)


@dataclass
class StubAdapterConfig:
    """Configuration for the stub adapter's deterministic behavior."""

    provider_id: ProviderId = field(default_factory=lambda: ProviderId("stub"))
    model_id: ModelId = field(default_factory=lambda: ModelId("stub-model", "1.0"))
    route_id: RouteId = field(default_factory=lambda: RouteId("direct"))
    capabilities: ModelCapabilities | None = None
    health_status: HealthStatus = HealthStatus.HEALTHY
    quota_signal: QuotaSignal = QuotaSignal.AVAILABLE
    fail_with_error: ProviderError | None = None
    simulate_stream: bool = False
    structured_output_schema: dict[str, Any] | None = None
    echo_messages: bool = False


class StubProviderAdapter(ProviderAdapter):
    """Deterministic provider adapter used for contract tests."""

    def __init__(self, config: StubAdapterConfig | None = None) -> None:
        self.config = config or StubAdapterConfig()
        self._cancelled: set[str] = set()
        self._calls: list[ProviderRequest] = []

    @property
    def identity(self) -> AdapterIdentity:
        caps = self.config.capabilities
        if caps is None:
            caps = ModelCapabilities(
                context_size=128_000,
                supports_structured_output=CapabilitySupport.SUPPORTED,
                supports_tool_use=CapabilitySupport.SUPPORTED,
                supports_streaming=CapabilitySupport.SUPPORTED,
                supports_reasoning=CapabilitySupport.SUPPORTED,
                supports_multimodal=CapabilitySupport.UNSUPPORTED,
                supports_temperature=CapabilitySupport.SUPPORTED,
                supports_max_tokens=CapabilitySupport.SUPPORTED,
                supports_stop_sequences=CapabilitySupport.SUPPORTED,
                supported_roles=frozenset(ExecutionRole),
            )
        return AdapterIdentity(
            provider_id=self.config.provider_id,
            supported_models=(self.config.model_id,),
            supported_routes=(self.config.route_id,),
            capabilities=caps,
        )

    async def submit(self, request: ProviderRequest) -> ProviderResponse:
        self._calls.append(request)
        can_serve, error = self.can_serve(request)
        if not can_serve and error is not None:
            return self._error_response(request, error)
        if self.config.fail_with_error is not None:
            return self._error_response(request, self.config.fail_with_error)
        if request.cancellation_id in self._cancelled:
            return self._error_response(
                request,
                ProviderError(
                    code=ProviderErrorCode.CANCELLED,
                    message="Request was cancelled",
                    provider_id=self.identity.provider_id,
                ),
            )
        return self._success_response(request)

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderResponse]:
        self._calls.append(request)
        can_serve, error = self.can_serve(request)
        if not can_serve and error is not None:
            yield self._error_response(request, error)
            return
        if self.config.fail_with_error is not None:
            yield self._error_response(request, self.config.fail_with_error)
            return
        if request.cancellation_id in self._cancelled:
            yield self._error_response(
                request,
                ProviderError(
                    code=ProviderErrorCode.CANCELLED,
                    message="Request was cancelled",
                    provider_id=self.identity.provider_id,
                ),
            )
            return

        text = self._build_text(request)
        words = text.split()
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            yield ProviderResponse(
                request_id=request.request_id,
                provider_id=self.identity.provider_id,
                model_id=self.config.model_id,
                route_id=self.config.route_id,
                text=word + (" " if not is_last else ""),
                streaming_state=StreamingState.COMPLETE if is_last else StreamingState.IN_PROGRESS,
                usage=Usage(),
                finish_reason=FinishReason.STOP if is_last else FinishReason.UNKNOWN,
            )

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            status=self.config.health_status,
            provider_id=self.identity.provider_id,
            route_id=self.config.route_id,
            reason="stub-configured",
        )

    async def quota(self) -> ProviderQuota:
        return ProviderQuota(
            provider_id=self.identity.provider_id,
            route_id=self.config.route_id,
            provider_signal=self.config.quota_signal,
        )

    async def cancel(self, cancellation_id: str) -> bool:
        if not cancellation_id:
            return False
        self._cancelled.add(cancellation_id)
        return True

    def translate_error(self, raw_error: Any) -> ProviderError:
        return ProviderError(
            code=ProviderErrorCode.UNKNOWN,
            message=str(raw_error),
            provider_id=self.identity.provider_id,
            route_id=self.config.route_id,
        )

    def _success_response(self, request: ProviderRequest) -> ProviderResponse:
        text = self._build_text(request)
        tool_calls: list[ToolCall] = []
        structured: dict[str, Any] | None = None

        if request.structured_output is not None:
            structured = {"request_id": request.request_id}
            if request.structured_output.schema:
                structured["schema_keys"] = list(request.structured_output.schema.keys())
        elif request.tools and request.tool_choice is not ToolChoiceMode.NONE:
            tool_calls = [
                ToolCall(
                    id=f"tc-{request.request_id}-0",
                    tool_name=request.tools[0].name,
                    arguments=[ToolCallArgument("request_id", request.request_id)],
                )
            ]

        return ProviderResponse(
            request_id=request.request_id,
            provider_id=self.identity.provider_id,
            model_id=self.config.model_id,
            route_id=self.config.route_id,
            text=text if structured is None else None,
            structured_result=structured,
            tool_calls=tool_calls,
            finish_reason=FinishReason.STOP,
            streaming_state=StreamingState.NOT_STREAMING,
            usage=Usage(input_tokens=10, output_tokens=5),
            latency_seconds=0.042,
            provider_request_id=f"stub-req-{request.request_id}",
            model_version=self.config.model_id.version,
        )

    def _error_response(self, request: ProviderRequest, error: ProviderError) -> ProviderResponse:
        return ProviderResponse(
            request_id=request.request_id,
            provider_id=error.provider_id or self.identity.provider_id,
            model_id=self.config.model_id,
            route_id=error.route_id or self.config.route_id,
            text=None,
            finish_reason=FinishReason.UNKNOWN,
            usage=Usage(),
            error_reference=error.code.value,
            metadata={"error": error},
        )

    def _build_text(self, request: ProviderRequest) -> str:
        if self.config.echo_messages:
            parts = [f"{m.role.value}: {m.content}" for m in request.messages]
            return " | ".join(parts) or "empty"
        if request.execution_role == ExecutionRole.PLANNING:
            return "plan"
        if request.execution_role == ExecutionRole.CODING:
            return "code"
        if request.execution_role == ExecutionRole.REVIEW:
            return "review"
        return "ack"
