"""Shared translation helpers for OpenAI-compatible provider endpoints.

This module is internal to the provider layer. It intentionally does NOT
import the optional ``openai`` SDK at module load time; adapters import it
lazily inside methods so that OmniForge remains importable without credentials
or SDKs installed.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from src.providers._common import (
    normalize_messages,
    normalize_system_messages,
    normalize_tools,
    parse_finish_reason,
    parse_tool_calls,
    parse_usage,
    translate_exception,
)
from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.errors import ProviderError, ProviderErrorCode
from src.providers.identity import (
    ProviderHealth,
    ProviderIdentity,
    ProviderOperationalState,
    ProviderQuotaState,
    QuotaSignal,
)
from src.providers.request import (
    MessageRole,
    ProviderRequest,
    ReasoningMode,
    StructuredOutputRequirement,
    ToolChoiceMode,
)
from src.providers.response import FinishReason, ProviderResponse, StreamingState, Usage
from src.routing.inference_route import InferenceRouteIdentity
from src.routing.model_identity import ModelIdentity


class OpenAICompatibleAdapter(ProviderAdapter):
    """Base adapter for providers exposing an OpenAI-compatible chat completions API.

    Subclasses supply provider-specific identity, model identity, and endpoint
    configuration. The translation logic is shared, but provider/model/route
    identities remain distinct for each subclass.
    """

    def __init__(
        self,
        *,
        provider_id: ProviderIdentity,
        model_id: ModelIdentity,
        route_id: InferenceRouteIdentity | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        capabilities: ProviderAdapterCapabilities | None = None,
    ) -> None:
        self._provider_id = provider_id
        self._model_id = model_id
        self._route_id = route_id
        self._api_key = api_key
        self._base_url = base_url
        self._capabilities = capabilities or ProviderAdapterCapabilities(
            streaming=True, tool_calls=True, structured_output=True, cancellation=True
        )
        self._cancelled: set[str] = set()

    @property
    def identity(self) -> ProviderIdentity:
        return self._provider_id

    @property
    def model_id(self) -> ModelIdentity:
        return self._model_id

    @property
    def route_id(self) -> InferenceRouteIdentity | None:
        return self._route_id

    @property
    def capabilities(self) -> ProviderAdapterCapabilities:
        return self._capabilities

    async def submit(self, request: ProviderRequest) -> ProviderResponse:
        if self._is_cancelled(request):
            return self._cancelled_response(request)
        can_serve, error = self.can_serve(request)
        if not can_serve and error is not None:
            return self._error_response(request, error)
        start = time.perf_counter()
        try:
            client = self._client()
            params = self._build_chat_params(request)
            response = await client.chat.completions.create(**params)
            latency = time.perf_counter() - start
            return self._normalize_response(request, response, latency_seconds=latency)
        except Exception as exc:
            error = translate_exception(exc, self._provider_id, self._route_id)
            return self._error_response(request, error)

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderResponse]:
        if self._is_cancelled(request):
            yield self._cancelled_response(request)
            return
        can_serve, error = self.can_serve(request)
        if not can_serve and error is not None:
            yield self._error_response(request, error)
            return
        start = time.perf_counter()
        try:
            client = self._client()
            params = self._build_chat_params(request)
            params["stream"] = True
            stream = await client.chat.completions.create(**params)
            final_finish = FinishReason.UNKNOWN
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                finish_raw = chunk.choices[0].finish_reason
                if finish_raw:
                    final_finish = parse_finish_reason(finish_raw)
                text = delta.content or ""
                if text:
                    yield ProviderResponse(
                        request_id=request.request_id,
                        provider_id=self._provider_id,
                        model_id=self._resolve_model_identity(request),
                        route_id=self._route_id,
                        text=text,
                        streaming_state=StreamingState.IN_PROGRESS,
                        usage=Usage(),
                        provider_request_id=chunk.id if hasattr(chunk, "id") else None,
                    )
            yield ProviderResponse(
                request_id=request.request_id,
                provider_id=self._provider_id,
                model_id=self._resolve_model_identity(request),
                route_id=self._route_id,
                text="",
                streaming_state=StreamingState.COMPLETE,
                usage=Usage(),
                finish_reason=final_finish,
                latency_seconds=time.perf_counter() - start,
            )
        except Exception as exc:
            error = translate_exception(exc, self._provider_id, self._route_id)
            yield self._error_response(request, error)

    async def cancel(self, request_id: str) -> None:
        if request_id:
            self._cancelled.add(request_id)

    async def health(self) -> ProviderOperationalState:
        # Phase 3: locally configured adapters are not the same as verified
        # external-provider health. Report DEGRADED with an explicit reason until
        # a real health observation is available (Phase 6 recovery engine).
        return ProviderOperationalState(
            health=ProviderHealth.DEGRADED,
            reason="Adapter initialized; no external health observation available",
        )

    async def quota(self) -> ProviderQuotaState:
        return ProviderQuotaState(provider_signal=QuotaSignal.UNKNOWN)

    def _client(self) -> Any:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' SDK is required for this provider. "
                "Install it with: pip install 'omniforge[openai]'"
            ) from exc
        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return AsyncOpenAI(**kwargs)

    def _build_chat_params(self, request: ProviderRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        messages.extend(normalize_system_messages(request.system_instructions))
        messages.extend(normalize_messages(request.messages))

        params: dict[str, Any] = {
            "model": self._resolve_model_name(request),
            "messages": messages,
        }
        if request.max_output_tokens is not None:
            params["max_tokens"] = request.max_output_tokens
        if request.max_total_tokens is not None and "max_tokens" not in params:
            params["max_tokens"] = request.max_total_tokens
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.stop_sequences:
            params["stop"] = request.stop_sequences
        if request.tools:
            params["tools"] = normalize_tools(request.tools)
            if request.tool_choice is ToolChoiceMode.REQUIRED:
                params["tool_choice"] = "required"
            elif request.tool_choice is ToolChoiceMode.NONE:
                params["tool_choice"] = "none"
            elif request.tool_choice is ToolChoiceMode.AUTO:
                params["tool_choice"] = "auto"
        if request.structured_output is not None and self._capabilities.structured_output:
            params["response_format"] = self._build_response_format(request.structured_output)
        if request.reasoning is not ReasoningMode.DEFAULT and self._capabilities.reasoning:
            params["reasoning_effort"] = _reasoning_effort(request.reasoning)
        return params

    def _build_response_format(self, requirement: StructuredOutputRequirement) -> dict[str, Any]:
        if requirement.schema:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": requirement.name or "response",
                    "schema": requirement.schema,
                    "strict": requirement.enforce_schema,
                },
            }
        return {"type": "json_object"}

    def _normalize_response(
        self,
        request: ProviderRequest,
        response: Any,
        *,
        latency_seconds: float | None = None,
    ) -> ProviderResponse:
        model_identity = self._resolve_model_identity(request)
        choice = response.choices[0]
        message = choice.message
        text = message.content or None
        tool_calls = parse_tool_calls(
            message.tool_calls if hasattr(message, "tool_calls") else None
        )
        structured: dict[str, Any] | None = None
        if text and request.structured_output is not None:
            try:
                structured = json.loads(text)
                text = None
            except Exception:
                structured = None
        return ProviderResponse(
            request_id=request.request_id,
            provider_id=self._provider_id,
            model_id=model_identity,
            route_id=self._route_id,
            text=text,
            structured_result=structured,
            tool_calls=tool_calls,
            finish_reason=parse_finish_reason(choice.finish_reason),
            usage=parse_usage(
                response.usage.to_dict() if hasattr(response, "usage") and response.usage else None
            ),
            latency_seconds=latency_seconds,
            provider_request_id=response.id if hasattr(response, "id") else None,
        )

    def _error_response(self, request: ProviderRequest, error: ProviderError) -> ProviderResponse:
        return ProviderResponse(
            request_id=request.request_id,
            provider_id=self._provider_id,
            model_id=self._resolve_model_identity(request),
            route_id=self._route_id,
            finish_reason=FinishReason.UNKNOWN,
            usage=Usage(),
            error_reference=error.code.value,
            metadata={"error": error},
        )

    def _cancelled_response(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            request_id=request.request_id,
            provider_id=self._provider_id,
            model_id=self._resolve_model_identity(request),
            route_id=self._route_id,
            finish_reason=FinishReason.UNKNOWN,
            usage=Usage(),
            error_reference=ProviderErrorCode.CANCELLED.value,
        )

    def _is_cancelled(self, request: ProviderRequest) -> bool:
        return bool(request.cancellation_id and request.cancellation_id in self._cancelled) or bool(
            request.request_id in self._cancelled
        )

    def _resolve_model_name(self, request: ProviderRequest) -> str:
        if request.target_model is not None:
            return request.target_model.model_id
        return self._model_id.model_id

    def _resolve_model_identity(self, request: ProviderRequest) -> ModelIdentity:
        """Return the model identity that actually handles the request.

        Prefer an explicitly selected target model; otherwise fall back to the
        adapter's configured default. This prevents reputation/cost evidence from
        being misattributed to the wrong model.
        """
        if request.target_model is not None:
            return request.target_model
        return self._model_id


def _reasoning_effort(mode: ReasoningMode) -> str:
    mapping = {
        ReasoningMode.EFFORT_LOW: "low",
        ReasoningMode.EFFORT_MEDIUM: "medium",
        ReasoningMode.EFFORT_HIGH: "high",
        ReasoningMode.DISABLED: "none",
    }
    return mapping.get(mode, "medium")


def split_messages_by_role(
    messages: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Separate system-string content from non-system messages for Anthropic-style APIs."""
    system_texts: list[str] = []
    non_system: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == MessageRole.SYSTEM.value:
            system_texts.append(message.get("content", ""))
        else:
            non_system.append(message)
    return system_texts, non_system
