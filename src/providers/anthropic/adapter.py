"""Anthropic provider adapter for OmniForge.

This adapter translates normalized OmniForge request/response models into the
Anthropic Messages API. The ``anthropic`` SDK is imported lazily inside methods
so that OmniForge remains importable when the SDK is not installed.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from src.providers._common import redact_secrets, translate_exception
from src.providers._models import ModelDescriptor
from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.errors import ProviderError, ProviderErrorCode
from src.providers.identity import (
    ProviderHealth,
    ProviderIdentity,
    ProviderOperationalState,
    ProviderQuotaState,
)
from src.providers.request import (
    Message,
    MessageRole,
    ProviderRequest,
    ToolChoiceMode,
    ToolDefinition,
)
from src.providers.response import (
    FinishReason,
    ProviderResponse,
    StreamingState,
    ToolCall,
    ToolCallArgument,
    Usage,
)
from src.routing.model_identity import ModelIdentity, ModelLifecycle


def _default_model() -> ModelIdentity:
    """Return the default Claude model identity."""
    return ModelDescriptor(
        model_id="claude-sonnet-4-20250514",
        family="claude",
        lifecycle=ModelLifecycle.HIGH_RISK,
        context_tokens=200_000,
        structured_output=True,
        tool_use=True,
        streaming=True,
        reasoning=True,
    ).to_identity()


class AnthropicAdapter(ProviderAdapter):
    """Provider adapter for the Anthropic Messages API."""

    def __init__(
        self,
        *,
        model_id: ModelIdentity | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        capabilities: ProviderAdapterCapabilities | None = None,
    ) -> None:
        self._model_id = model_id or _default_model()
        self._api_key = api_key
        self._base_url = base_url
        self._capabilities = capabilities or ProviderAdapterCapabilities(
            streaming=True,
            tool_calls=True,
            structured_output=True,
            cancellation=True,
        )
        self._cancelled: set[str] = set()

    @property
    def identity(self) -> ProviderIdentity:
        """Stable Anthropic provider identity."""
        return ProviderIdentity("anthropic", "Anthropic", "anthropic.com")

    @property
    def capabilities(self) -> ProviderAdapterCapabilities:
        """Capabilities advertised by this adapter."""
        return self._capabilities

    async def submit(self, request: ProviderRequest) -> ProviderResponse:
        """Submit a request to Anthropic and return a normalized response."""
        if self._is_cancelled(request):
            return self._cancelled_response(request)

        can_serve, error = self.can_serve(request)
        if not can_serve and error is not None:
            return self._error_response(request, error)

        try:
            start = time.monotonic()
            client = self._client()
            params = self._build_params(request)
            response = await client.messages.create(**params)
            latency = time.monotonic() - start
            return self._normalize_response(request, response, latency=latency)
        except Exception as exc:
            normalized_error = self._translate_error(exc)
            return self._error_response(request, normalized_error)

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderResponse]:
        """Stream normalized response chunks from Anthropic."""
        if self._is_cancelled(request):
            yield self._cancelled_response(request)
            return

        can_serve, error = self.can_serve(request)
        if not can_serve and error is not None:
            yield self._error_response(request, error)
            return

        try:
            client = self._client()
            params = self._build_params(request)
            params["stream"] = True
            stream_obj = await client.messages.create(**params)

            start = time.monotonic()
            provider_request_id: str | None = None
            finish_reason = FinishReason.UNKNOWN
            usage = Usage()

            async for event in stream_obj:
                event_type = getattr(event, "type", None)
                if event_type == "message_start":
                    provider_request_id = getattr(event.message, "id", None)
                elif event_type == "content_block_delta":
                    delta = event.delta
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        yield ProviderResponse(
                            request_id=request.request_id,
                            provider_id=self.identity,
                            model_id=self._model_id,
                            text=getattr(delta, "text", None),
                            streaming_state=StreamingState.IN_PROGRESS,
                            usage=Usage(),
                            provider_request_id=provider_request_id,
                        )
                elif event_type == "message_delta":
                    finish_reason = self._parse_stop_reason(
                        getattr(event.delta, "stop_reason", None)
                    )
                    usage = self._parse_usage(getattr(event, "usage", None))

            latency = time.monotonic() - start
            yield ProviderResponse(
                request_id=request.request_id,
                provider_id=self.identity,
                model_id=self._model_id,
                text="",
                streaming_state=StreamingState.COMPLETE,
                usage=usage,
                finish_reason=finish_reason,
                latency_seconds=latency,
                provider_request_id=provider_request_id,
            )
        except Exception as exc:
            normalized_error = self._translate_error(exc)
            yield self._error_response(request, normalized_error)

    async def cancel(self, request_id: str) -> None:
        """Record a cancelled request identifier."""
        if request_id:
            self._cancelled.add(request_id)

    async def health(self) -> ProviderOperationalState:
        """Return the operational health of the Anthropic adapter."""
        return ProviderOperationalState(health=ProviderHealth.HEALTHY)

    async def quota(self) -> ProviderQuotaState:
        """Return quota state; Anthropic headers are not currently parsed."""
        return ProviderQuotaState()

    def translate_error(self, raw_error: Any) -> ProviderError:
        """Translate a raw Anthropic error into the normalized taxonomy."""
        return self._translate_error(raw_error)

    def _client(self) -> Any:
        """Build an async Anthropic client, importing the SDK lazily."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' SDK is required for this provider. "
                "Install it with: pip install 'omniforge[anthropic]'"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return AsyncAnthropic(**kwargs)

    def _build_params(self, request: ProviderRequest) -> dict[str, Any]:
        """Build Anthropic Messages API parameters from a normalized request."""
        system_text = "\n\n".join(request.system_instructions) or None
        messages = self._build_messages(request.messages)

        params: dict[str, Any] = {
            "model": self._resolve_model_name(request),
            "messages": messages,
            "max_tokens": request.max_output_tokens
            if request.max_output_tokens is not None
            else 4096,
        }
        if system_text is not None:
            params["system"] = system_text
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.stop_sequences:
            params["stop_sequences"] = request.stop_sequences
        if request.tools:
            params["tools"] = self._build_tools(request.tools)
            params["tool_choice"] = self._build_tool_choice(request.tool_choice)
        return params

    def _build_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Translate normalized messages into Anthropic message parameters."""
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                continue

            role = "assistant" if msg.role == MessageRole.ASSISTANT else "user"
            content: Any = msg.content
            if msg.role == MessageRole.TOOL:
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "",
                        "content": msg.content,
                    }
                ]

            entry: dict[str, Any] = {"role": role, "content": content}
            if msg.name:
                entry["name"] = msg.name
            result.append(entry)
        return result

    def _build_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Translate normalized tool definitions into Anthropic tool parameters."""
        result: list[dict[str, Any]] = []
        for tool in tools:
            schema: dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
            }
            for param in tool.parameters:
                schema["properties"][param.name] = param.schema
                if param.required:
                    schema["required"].append(param.name)
            result.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": schema,
                }
            )
        return result

    def _build_tool_choice(self, mode: ToolChoiceMode) -> dict[str, Any]:
        """Map a normalized tool-choice mode to an Anthropic tool_choice value."""
        mapping = {
            ToolChoiceMode.AUTO: {"type": "auto"},
            ToolChoiceMode.REQUIRED: {"type": "any"},
            ToolChoiceMode.NONE: {"type": "none"},
            ToolChoiceMode.FORBIDDEN: {"type": "none"},
        }
        return mapping.get(mode, {"type": "auto"})

    def _normalize_response(
        self, request: ProviderRequest, response: Any, latency: float | None = None
    ) -> ProviderResponse:
        """Translate an Anthropic Message into a normalized ProviderResponse."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif block_type == "tool_use":
                raw_input = getattr(block, "input", {}) or {}
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", "") or "",
                        tool_name=getattr(block, "name", "") or "",
                        arguments=[ToolCallArgument(k, v) for k, v in raw_input.items()],
                        raw_arguments=raw_input,
                    )
                )

        text = "".join(text_parts) if text_parts else None
        structured: dict[str, Any] | None = None
        if text is not None and request.structured_output is not None:
            try:
                structured = json.loads(text)
                text = None
            except Exception:
                structured = None

        return ProviderResponse(
            request_id=request.request_id,
            provider_id=self.identity,
            model_id=self._model_id,
            text=text,
            structured_result=structured,
            tool_calls=tool_calls,
            finish_reason=self._parse_stop_reason(getattr(response, "stop_reason", None)),
            streaming_state=StreamingState.NOT_STREAMING,
            usage=self._parse_usage(getattr(response, "usage", None)),
            latency_seconds=latency,
            provider_request_id=getattr(response, "id", None),
        )

    def _parse_usage(self, raw: Any) -> Usage:
        """Extract normalized usage from Anthropic usage metadata."""
        if raw is None:
            return Usage()
        return Usage(
            input_tokens=getattr(raw, "input_tokens", None),
            output_tokens=getattr(raw, "output_tokens", None),
            cached_tokens=getattr(raw, "cache_read_input_tokens", None),
        )

    def _parse_stop_reason(self, raw: str | None) -> FinishReason:
        """Map Anthropic stop_reason values to normalized FinishReason."""
        if not raw:
            return FinishReason.UNKNOWN
        lowered = str(raw).lower()
        if lowered in {"end_turn", "stop", "stop_sequence"}:
            return FinishReason.STOP
        if lowered == "max_tokens":
            return FinishReason.LENGTH
        if lowered == "tool_use":
            return FinishReason.TOOL_CALLS
        if lowered in {"content_filter", "refusal"}:
            return FinishReason.CONTENT_FILTER
        return FinishReason.UNKNOWN

    def _translate_error(self, exc: BaseException) -> ProviderError:
        """Map Anthropic SDK exceptions to normalized ProviderError values.

        Uses class-name dispatch and duck-typing rather than importing SDK
        exception classes, so translation remains robust when the ``anthropic``
        module is shadowed by tests or partially available.
        """
        provider_id = self.identity
        message = redact_secrets(str(exc))
        exc_name = type(exc).__name__
        status = getattr(exc, "status_code", None)

        if exc_name == "AuthenticationError":
            return ProviderError(
                code=ProviderErrorCode.AUTH_FAILURE,
                message=message,
                http_status=status,
                provider_id=provider_id,
                retryable=False,
            )

        if exc_name == "RateLimitError":
            retry_after = None
            response = getattr(exc, "response", None)
            if response is not None:
                retry_after = self._parse_retry_after(
                    getattr(response, "headers", {}).get("retry-after")
                )
            return ProviderError(
                code=ProviderErrorCode.RATE_LIMITED,
                message=message,
                http_status=status,
                provider_id=provider_id,
                retryable=True,
                retry_after_seconds=retry_after,
            )

        if exc_name == "APIConnectionError":
            return ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT,
                message=message,
                provider_id=provider_id,
                retryable=True,
            )

        if exc_name == "BadRequestError":
            code = (
                ProviderErrorCode.CONTEXT_OVERFLOW
                if status == 400 and self._looks_like_context_overflow(message)
                else ProviderErrorCode.UNKNOWN
            )
            return ProviderError(
                code=code,
                message=message,
                http_status=status,
                provider_id=provider_id,
                retryable=False,
            )

        if exc_name in {"APIStatusError", "InternalServerError"}:
            return self._translate_http_status(status, message)

        return translate_exception(exc, provider_id, None)

    def _translate_http_status(self, status: int | None, message: str) -> ProviderError:
        """Map an HTTP status code to a normalized provider error."""
        if status in {401, 403}:
            return ProviderError(
                code=ProviderErrorCode.AUTH_FAILURE,
                message=message,
                http_status=status,
                provider_id=self.identity,
                retryable=False,
            )
        if status == 429:
            return ProviderError(
                code=ProviderErrorCode.RATE_LIMITED,
                message=message,
                http_status=status,
                provider_id=self.identity,
                retryable=True,
            )
        if status in {500, 502, 503, 504}:
            return ProviderError(
                code=ProviderErrorCode.PROVIDER_UNAVAILABLE,
                message=message,
                http_status=status,
                provider_id=self.identity,
                retryable=True,
            )
        if status == 400 and self._looks_like_context_overflow(message):
            return ProviderError(
                code=ProviderErrorCode.CONTEXT_OVERFLOW,
                message=message,
                http_status=status,
                provider_id=self.identity,
                retryable=False,
            )
        return ProviderError(
            code=ProviderErrorCode.UNKNOWN,
            message=message,
            http_status=status,
            provider_id=self.identity,
            retryable=False,
        )

    def _looks_like_context_overflow(self, message: str) -> bool:
        """Return True when an error message suggests a context-length problem."""
        lowered = message.lower()
        return any(
            hint in lowered
            for hint in ("context", "too long", "context window", "max_tokens", "token limit")
        )

    def _parse_retry_after(self, value: Any) -> int | None:
        """Safely parse a retry-after header value."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _error_response(self, request: ProviderRequest, error: ProviderError) -> ProviderResponse:
        """Build a normalized error response."""
        return ProviderResponse(
            request_id=request.request_id,
            provider_id=self.identity,
            model_id=self._model_id,
            finish_reason=FinishReason.UNKNOWN,
            usage=Usage(),
            error_reference=error.code.value,
            metadata={"error": error},
        )

    def _cancelled_response(self, request: ProviderRequest) -> ProviderResponse:
        """Build a normalized cancelled response."""
        return ProviderResponse(
            request_id=request.request_id,
            provider_id=self.identity,
            model_id=self._model_id,
            finish_reason=FinishReason.UNKNOWN,
            usage=Usage(),
            error_reference=ProviderErrorCode.CANCELLED.value,
        )

    def _is_cancelled(self, request: ProviderRequest) -> bool:
        """Return True if the request has been cancelled."""
        return bool(request.cancellation_id and request.cancellation_id in self._cancelled) or bool(
            request.request_id in self._cancelled
        )

    def _resolve_model_name(self, request: ProviderRequest) -> str:
        """Resolve the model name, preferring an explicit target model."""
        if request.target_model is not None:
            return request.target_model.model_id
        return self._model_id.model_id
