"""Google Gemini provider adapter for OmniForge.

This module implements the canonical ``ProviderAdapter`` contract for the
official ``google-genai`` SDK. The SDK is imported lazily inside methods so
OmniForge remains importable without it installed.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from src.providers._common import (
    redact_secrets,
    translate_exception,
)
from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.errors import (
    ProviderError,
    ProviderErrorCode,
)
from src.providers.identity import (
    ProviderHealth,
    ProviderIdentity,
    ProviderOperationalState,
    ProviderQuotaState,
)
from src.providers.request import (
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
from src.routing.inference_route import InferenceRouteIdentity
from src.routing.model_identity import ModelIdentity

_DEFAULT_MODEL_ID = "gemini-2.5-pro"
_DEFAULT_FAMILY = "gemini"


def _default_model_identity(model_id: str | None = None) -> ModelIdentity:
    """Return the canonical model identity for a Gemini model ID."""
    return ModelIdentity(
        model_id=model_id or _DEFAULT_MODEL_ID,
        family=_DEFAULT_FAMILY,
    )


class GeminiAdapter(ProviderAdapter):
    """Provider adapter backed by the official Google GenAI SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str | None = None,
        model_identity: ModelIdentity | None = None,
        route_id: InferenceRouteIdentity | None = None,
        capabilities: ProviderAdapterCapabilities | None = None,
    ) -> None:
        """Initialize the Gemini adapter.

        Args:
            api_key: Optional API key for the Gemini API. When omitted, the SDK
                falls back to environment-based credentials.
            model_id: Optional model ID string. Ignored when ``model_identity``
                is supplied.
            model_identity: Optional canonical model identity. Takes precedence
                over ``model_id``.
            route_id: Optional inference route identity for diagnostics.
            capabilities: Optional capability advertisement. Defaults to a
                streaming/tool-call/structured-output capable adapter.
        """
        self._api_key = api_key
        self._model_id = model_identity or _default_model_identity(model_id)
        self._route_id = route_id
        self._capabilities = capabilities or ProviderAdapterCapabilities(
            streaming=True,
            tool_calls=True,
            structured_output=True,
            cancellation=True,
        )
        self._cancelled: set[str] = set()

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            provider_id="gemini",
            display_name="Google Gemini",
            failure_domain="googleapis.com",
        )

    @property
    def capabilities(self) -> ProviderAdapterCapabilities:
        return self._capabilities

    @property
    def model_id(self) -> ModelIdentity:
        return self._model_id

    async def submit(self, request: ProviderRequest) -> ProviderResponse:
        if self._is_cancelled(request):
            return self._cancelled_response(request)

        can_serve, error = self.can_serve(request)
        if not can_serve and error is not None:
            return self._error_response(request, error)

        started_at = time.perf_counter()
        try:
            client = self._client()
            model = self._resolve_model_name(request)
            contents = self._build_contents(request)
            config = self._build_config(request)
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            latency = time.perf_counter() - started_at
            return self._normalize_response(request, response, latency=latency)
        except Exception as exc:  # noqa: BLE001
            error = self.translate_error(exc)
            return self._error_response(request, error)

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderResponse]:
        if self._is_cancelled(request):
            yield self._cancelled_response(request)
            return

        can_serve, error = self.can_serve(request)
        if not can_serve and error is not None:
            yield self._error_response(request, error)
            return

        try:
            client = self._client()
            model = self._resolve_model_name(request)
            contents = self._build_contents(request)
            config = self._build_config(request)
            stream = await client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
            final_finish = FinishReason.UNKNOWN
            request_id: str | None = None
            async for chunk in stream:
                chunk_request_id = self._extract_request_id(chunk)
                if chunk_request_id:
                    request_id = chunk_request_id
                candidate = self._first_candidate(chunk)
                if candidate is None:
                    continue
                text = self._extract_text(candidate)
                chunk_tool_calls = self._extract_tool_calls(candidate)
                finish = self._map_finish_reason(candidate.finish_reason, chunk_tool_calls)
                if finish is not FinishReason.UNKNOWN:
                    final_finish = finish
                if text or chunk_tool_calls:
                    yield ProviderResponse(
                        request_id=request.request_id,
                        provider_id=self.identity,
                        model_id=self._resolve_model_identity(request),
                        route_id=self._route_id,
                        text=text,
                        tool_calls=chunk_tool_calls,
                        streaming_state=StreamingState.IN_PROGRESS,
                        usage=Usage(),
                        provider_request_id=request_id,
                    )
            yield ProviderResponse(
                request_id=request.request_id,
                provider_id=self.identity,
                model_id=self._resolve_model_identity(request),
                route_id=self._route_id,
                text="",
                streaming_state=StreamingState.COMPLETE,
                usage=Usage(),
                finish_reason=final_finish,
                provider_request_id=request_id,
            )
        except Exception as exc:  # noqa: BLE001
            error = self.translate_error(exc)
            yield self._error_response(request, error)

    async def cancel(self, request_id: str) -> None:
        if request_id:
            self._cancelled.add(request_id)

    async def health(self) -> ProviderOperationalState:
        return ProviderOperationalState(health=ProviderHealth.HEALTHY)

    async def quota(self) -> ProviderQuotaState:
        return ProviderQuotaState()

    def translate_error(self, raw_error: Any) -> ProviderError:
        """Translate a Gemini SDK error into the normalized taxonomy."""
        try:
            from google.genai.errors import APIError
        except Exception:  # noqa: BLE001
            return translate_exception(raw_error, self.identity, self._route_id)

        if isinstance(raw_error, APIError):
            code: int | None = getattr(raw_error, "code", None)
            message = redact_secrets(getattr(raw_error, "message", str(raw_error)))
            return self._map_api_error(code, message, raw_error)

        return translate_exception(raw_error, self.identity, self._route_id)

    def _client(self) -> Any:
        """Return a lazily-initialized ``google.genai.Client``."""
        from google import genai

        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return genai.Client(**kwargs)

    def _build_contents(self, request: ProviderRequest) -> list[Any]:
        """Convert normalized messages into Gemini ``Content`` values."""
        from google.genai import types

        contents: list[Any] = []
        for message in request.messages:
            if message.role is MessageRole.SYSTEM:
                continue
            role = "model" if message.role is MessageRole.ASSISTANT else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=message.content)],
                )
            )
        return contents

    def _build_config(self, request: ProviderRequest) -> Any:
        """Build a ``GenerateContentConfig`` from a normalized request."""
        from google.genai import types

        config = types.GenerateContentConfig()

        if request.system_instructions:
            config.system_instruction = "\n\n".join(request.system_instructions)

        if request.temperature is not None:
            config.temperature = request.temperature

        if request.max_output_tokens is not None:
            config.max_output_tokens = request.max_output_tokens
        elif request.max_total_tokens is not None:
            config.max_output_tokens = request.max_total_tokens

        if request.stop_sequences:
            config.stop_sequences = request.stop_sequences

        if request.tools:
            config.tools = [
                types.Tool(function_declarations=self._build_function_declarations(request.tools))
            ]
            config.tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=self._map_tool_choice(request.tool_choice)
                )
            )

        if request.structured_output is not None and self._capabilities.structured_output:
            config.response_mime_type = "application/json"
            if request.structured_output.schema:
                config.response_json_schema = request.structured_output.schema

        return config

    def _build_function_declarations(self, tools: list[ToolDefinition]) -> list[Any]:
        """Convert normalized tool definitions into Gemini function declarations."""
        from google.genai import types

        declarations: list[Any] = []
        for tool in tools:
            parameters: dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
            }
            for param in tool.parameters:
                parameters["properties"][param.name] = param.schema
                if param.required:
                    parameters["required"].append(param.name)
            declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=parameters,
                )
            )
        return declarations

    def _normalize_response(
        self,
        request: ProviderRequest,
        response: Any,
        latency: float | None = None,
    ) -> ProviderResponse:
        """Translate a Gemini response into a normalized ``ProviderResponse``."""
        candidate = self._first_candidate(response)
        text: str | None = None
        tool_calls: list[ToolCall] = []
        finish = FinishReason.UNKNOWN

        if candidate is not None:
            text = self._extract_text(candidate)
            tool_calls = self._extract_tool_calls(candidate)
            finish = self._map_finish_reason(candidate.finish_reason, tool_calls)
        elif getattr(response, "prompt_feedback", None) is not None:
            finish = FinishReason.CONTENT_FILTER

        structured: dict[str, Any] | None = None
        if text and request.structured_output is not None:
            try:
                structured = json.loads(text)
                text = None
            except Exception:  # noqa: BLE001
                structured = None

        return ProviderResponse(
            request_id=request.request_id,
            provider_id=self.identity,
            model_id=self._resolve_model_identity(request),
            route_id=self._route_id,
            text=text,
            structured_result=structured,
            tool_calls=tool_calls,
            finish_reason=finish,
            streaming_state=StreamingState.NOT_STREAMING,
            usage=self._parse_usage(getattr(response, "usage_metadata", None)),
            latency_seconds=latency,
            provider_request_id=self._extract_request_id(response),
        )

    def _error_response(self, request: ProviderRequest, error: ProviderError) -> ProviderResponse:
        return ProviderResponse(
            request_id=request.request_id,
            provider_id=error.provider_id or self.identity,
            model_id=self._resolve_model_identity(request),
            route_id=error.route_id or self._route_id,
            finish_reason=FinishReason.UNKNOWN,
            usage=Usage(),
            error_reference=error.code.value,
            metadata={"error": error},
        )

    def _cancelled_response(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            request_id=request.request_id,
            provider_id=self.identity,
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
        if request.target_model is not None:
            return request.target_model
        return self._model_id

    def _extract_text(self, candidate: Any) -> str | None:
        content = getattr(candidate, "content", None)
        if content is None:
            return None
        parts = getattr(content, "parts", None) or []
        texts = [part.text for part in parts if getattr(part, "text", None)]
        joined = "".join(texts)
        return joined or None

    def _extract_tool_calls(self, candidate: Any) -> list[ToolCall]:
        content = getattr(candidate, "content", None)
        if content is None:
            return []
        parts = getattr(content, "parts", None) or []
        calls: list[ToolCall] = []
        for part in parts:
            function_call = getattr(part, "function_call", None)
            if function_call is None:
                continue
            args = getattr(function_call, "args", None) or {}
            call_id = getattr(function_call, "id", None) or f"fc-{function_call.name}"
            calls.append(
                ToolCall(
                    id=call_id,
                    tool_name=function_call.name,
                    arguments=[ToolCallArgument(k, v) for k, v in args.items()],
                    raw_arguments=dict(args),
                )
            )
        return calls

    def _first_candidate(self, response: Any) -> Any | None:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None
        return candidates[0]

    def _extract_request_id(self, response: Any) -> str | None:
        return getattr(response, "response_id", None)

    def _parse_usage(self, usage_metadata: Any) -> Usage:
        if usage_metadata is None:
            return Usage()
        return Usage(
            input_tokens=getattr(usage_metadata, "prompt_token_count", None),
            output_tokens=getattr(usage_metadata, "candidates_token_count", None)
            or getattr(usage_metadata, "response_token_count", None),
            cached_tokens=getattr(usage_metadata, "cached_content_token_count", None),
            total_tokens=getattr(usage_metadata, "total_token_count", None),
        )

    def _map_finish_reason(self, raw: Any, has_tool_calls: list[ToolCall] | bool) -> FinishReason:
        if has_tool_calls:
            return FinishReason.TOOL_CALLS
        if raw is None:
            return FinishReason.UNKNOWN
        name = raw.name if hasattr(raw, "name") else str(raw)
        mapping = {
            "STOP": FinishReason.STOP,
            "MAX_TOKENS": FinishReason.LENGTH,
            "SAFETY": FinishReason.CONTENT_FILTER,
            "RECITATION": FinishReason.CONTENT_FILTER,
            "BLOCKLIST": FinishReason.CONTENT_FILTER,
            "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
            "SPII": FinishReason.CONTENT_FILTER,
            "IMAGE_SAFETY": FinishReason.CONTENT_FILTER,
            "IMAGE_PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
        }
        return mapping.get(name, FinishReason.UNKNOWN)

    def _map_tool_choice(self, mode: ToolChoiceMode) -> Any:
        from google.genai import types

        if mode is ToolChoiceMode.REQUIRED:
            return types.FunctionCallingConfigMode.ANY
        if mode is ToolChoiceMode.NONE:
            return types.FunctionCallingConfigMode.NONE
        return types.FunctionCallingConfigMode.AUTO

    def _map_api_error(
        self,
        code: int | None,
        message: str,
        raw_error: Any,
    ) -> ProviderError:
        if code in {401, 403}:
            return ProviderError(
                code=ProviderErrorCode.AUTH_FAILURE,
                message=message,
                http_status=code,
                provider_id=self.identity,
                route_id=self._route_id,
                provider_error_code=str(code),
                retryable=False,
            )
        if code == 429:
            return ProviderError(
                code=ProviderErrorCode.RATE_LIMITED,
                message=message,
                http_status=code,
                provider_id=self.identity,
                route_id=self._route_id,
                provider_error_code=str(code),
                retryable=True,
            )
        if code in {500, 502, 503, 504}:
            return ProviderError(
                code=ProviderErrorCode.PROVIDER_UNAVAILABLE,
                message=message,
                http_status=code,
                provider_id=self.identity,
                route_id=self._route_id,
                provider_error_code=str(code),
                retryable=True,
            )
        if code == 400 and ("context" in message.lower() or "too long" in message.lower()):
            return ProviderError(
                code=ProviderErrorCode.CONTEXT_OVERFLOW,
                message=message,
                http_status=code,
                provider_id=self.identity,
                route_id=self._route_id,
                provider_error_code=str(code),
                retryable=False,
            )
        return ProviderError(
            code=ProviderErrorCode.UNKNOWN,
            message=message,
            http_status=code,
            provider_id=self.identity,
            route_id=self._route_id,
            provider_error_code=str(code),
            retryable=False,
        )
