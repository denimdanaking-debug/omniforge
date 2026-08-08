"""Deterministic stub adapter for contract tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.errors import ProviderError, ProviderErrorCode
from src.providers.identity import (
    ProviderHealth,
    ProviderIdentity,
    ProviderOperationalState,
    ProviderQuotaState,
    QuotaSignal,
)
from src.providers.request import ProviderRequest, ToolChoiceMode
from src.providers.response import (
    FinishReason,
    ProviderResponse,
    StreamingState,
    ToolCall,
    ToolCallArgument,
    Usage,
)
from src.routing.model_identity import ModelIdentity


@dataclass
class StubAdapterConfig:
    """Configuration for the stub adapter's deterministic behavior."""

    provider: ProviderIdentity = field(
        default_factory=lambda: ProviderIdentity("stub", "Stub Provider", "stub.example")
    )
    model: ModelIdentity = field(
        default_factory=lambda: ModelIdentity(model_id="stub-model", family="stub")
    )
    capabilities: ProviderAdapterCapabilities = field(
        default_factory=lambda: ProviderAdapterCapabilities(
            streaming=True,
            tool_calls=True,
            structured_output=True,
            cancellation=True,
        )
    )
    health: ProviderHealth = ProviderHealth.HEALTHY
    quota_signal: QuotaSignal = QuotaSignal.AVAILABLE
    fail_with_error: ProviderError | None = None
    echo_messages: bool = False


class StubProviderAdapter(ProviderAdapter):
    """Deterministic provider adapter used for contract tests."""

    def __init__(self, config: StubAdapterConfig | None = None) -> None:
        self.config = config or StubAdapterConfig()
        self._cancelled: set[str] = set()
        self._calls: list[ProviderRequest] = []

    @property
    def identity(self) -> ProviderIdentity:
        return self.config.provider

    @property
    def capabilities(self) -> ProviderAdapterCapabilities:
        return self.config.capabilities

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
                    provider_id=self.identity,
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
                    provider_id=self.identity,
                ),
            )
            return

        text = self._build_text(request)
        words = text.split()
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            yield ProviderResponse(
                request_id=request.request_id,
                provider_id=self.identity,
                model_id=self.config.model,
                text=word + (" " if not is_last else ""),
                streaming_state=StreamingState.COMPLETE if is_last else StreamingState.IN_PROGRESS,
                usage=Usage(),
                finish_reason=FinishReason.STOP if is_last else FinishReason.UNKNOWN,
            )

    async def cancel(self, request_id: str) -> None:
        if request_id:
            self._cancelled.add(request_id)

    async def health(self) -> ProviderOperationalState:
        return ProviderOperationalState(
            health=self.config.health,
            reason="stub-configured",
        )

    async def quota(self) -> ProviderQuotaState:
        return ProviderQuotaState(
            provider_signal=self.config.quota_signal,
        )

    def translate_error(self, raw_error: Any) -> ProviderError:
        return ProviderError(
            code=ProviderErrorCode.UNKNOWN,
            message=str(raw_error),
            provider_id=self.identity,
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
            provider_id=self.identity,
            model_id=self.config.model,
            text=text if structured is None else None,
            structured_result=structured,
            tool_calls=tool_calls,
            finish_reason=FinishReason.STOP,
            streaming_state=StreamingState.NOT_STREAMING,
            usage=Usage(input_tokens=10, output_tokens=5),
            latency_seconds=0.042,
            provider_request_id=f"stub-req-{request.request_id}",
        )

    def _error_response(self, request: ProviderRequest, error: ProviderError) -> ProviderResponse:
        return ProviderResponse(
            request_id=request.request_id,
            provider_id=error.provider_id or self.identity,
            model_id=self.config.model,
            route_id=error.route_id,
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
        return request.execution_role.value
