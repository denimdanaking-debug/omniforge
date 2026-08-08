"""Shared provider-adapter translation utilities.

These helpers are internal to the provider layer. They do not leak SDK types
and do not invent provider data.
"""

from __future__ import annotations

import re
from typing import Any

from src.providers.errors import ProviderError, ProviderErrorCode
from src.providers.identity import ProviderIdentity
from src.providers.request import Message, MessageRole, ToolDefinition
from src.providers.response import FinishReason, ToolCall, ToolCallArgument, Usage
from src.routing.inference_route import InferenceRouteIdentity


def redact_secrets(text: str) -> str:
    """Remove common credential patterns from diagnostic text."""
    patterns = [
        (r"sk-[a-zA-Z0-9_-]{10,}", "sk-***"),
        (r"Bearer\s+[a-zA-Z0-9_.-]{10,}", "Bearer ***"),
        (r"api[_-]?key['\"\s]*[:=]['\"\s]*[a-zA-Z0-9_.-]{10,}", "api_key=***"),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert normalized messages into a generic chat-message dict list."""
    result: list[dict[str, Any]] = []
    for message in messages:
        entry: dict[str, Any] = {"role": message.role.value, "content": message.content}
        if message.name:
            entry["name"] = message.name
        if message.tool_call_id:
            entry["tool_call_id"] = message.tool_call_id
        result.append(entry)
    return result


def normalize_system_messages(system_instructions: list[str]) -> list[dict[str, Any]]:
    """Convert system instructions into generic system-message dicts."""
    return [{"role": MessageRole.SYSTEM.value, "content": text} for text in system_instructions]


def normalize_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert normalized tool definitions into JSON-schema tool dicts."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        for param in tool.parameters:
            parameters["properties"][param.name] = param.schema
            if param.required:
                parameters["required"].append(param.name)
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                },
            }
        )
    return result


def parse_tool_calls(raw_calls: list[dict[str, Any]] | None) -> list[ToolCall]:
    """Parse provider tool-call objects into normalized ToolCall values."""
    if not raw_calls:
        return []
    calls: list[ToolCall] = []
    for raw in raw_calls:
        function = raw.get("function") or raw
        name = function.get("name") or raw.get("name") or "unknown"
        call_id = raw.get("id") or f"tc-{name}"
        arguments: dict[str, Any] = {}
        raw_args = function.get("arguments") or raw.get("arguments") or {}
        if isinstance(raw_args, str):
            try:
                import json

                arguments = json.loads(raw_args)
            except Exception:
                arguments = {"raw": raw_args}
        elif isinstance(raw_args, dict):
            arguments = raw_args
        parsed_args = [ToolCallArgument(k, v) for k, v in arguments.items()]
        calls.append(
            ToolCall(id=call_id, tool_name=name, arguments=parsed_args, raw_arguments=arguments)
        )
    return calls


def parse_usage(raw: dict[str, Any] | None) -> Usage:
    """Parse provider usage metadata into a normalized Usage value."""
    if not raw:
        return Usage()
    return Usage(
        input_tokens=_int_or_none(raw.get("prompt_tokens") or raw.get("input_tokens")),
        output_tokens=_int_or_none(raw.get("completion_tokens") or raw.get("output_tokens")),
        cached_tokens=_int_or_none(
            raw.get("cached_tokens") or raw.get("prompt_tokens_details", {}).get("cached_tokens")
        ),
        reasoning_tokens=_int_or_none(
            raw.get("reasoning_tokens") or raw.get("reasoning_effort_tokens")
        ),
        total_tokens=_int_or_none(raw.get("total_tokens")),
    )


def parse_finish_reason(raw: str | None) -> FinishReason:
    """Map a provider finish reason string to the normalized enum."""
    if not raw:
        return FinishReason.UNKNOWN
    lowered = raw.lower()
    if lowered in {"stop", "end_turn"}:
        return FinishReason.STOP
    if lowered in {"length", "max_tokens"}:
        return FinishReason.LENGTH
    if lowered in {"tool_calls", "function_call"}:
        return FinishReason.TOOL_CALLS
    if lowered in {"content_filter"}:
        return FinishReason.CONTENT_FILTER
    return FinishReason.UNKNOWN


def translate_http_error(
    status: int | None,
    body: str,
    provider_id: ProviderIdentity,
    route_id: InferenceRouteIdentity | None = None,
) -> ProviderError:
    """Translate a raw HTTP error into the normalized taxonomy."""
    safe_body = redact_secrets(body)
    if status == 401 or status == 403:
        return ProviderError(
            code=ProviderErrorCode.AUTH_FAILURE,
            message=f"Authentication failed: {safe_body}",
            http_status=status,
            provider_id=provider_id,
            route_id=route_id,
            retryable=False,
        )
    if status == 429:
        return ProviderError(
            code=ProviderErrorCode.RATE_LIMITED,
            message=f"Rate limited: {safe_body}",
            http_status=status,
            provider_id=provider_id,
            route_id=route_id,
            retryable=True,
        )
    if status in {500, 502, 503, 504}:
        return ProviderError(
            code=ProviderErrorCode.PROVIDER_UNAVAILABLE,
            message=f"Provider unavailable: {safe_body}",
            http_status=status,
            provider_id=provider_id,
            route_id=route_id,
            retryable=True,
        )
    if status == 400 and ("context" in safe_body.lower() or "too long" in safe_body.lower()):
        return ProviderError(
            code=ProviderErrorCode.CONTEXT_OVERFLOW,
            message=f"Context overflow: {safe_body}",
            http_status=status,
            provider_id=provider_id,
            route_id=route_id,
            retryable=False,
        )
    return ProviderError(
        code=ProviderErrorCode.UNKNOWN,
        message=f"Provider error ({status}): {safe_body}",
        http_status=status,
        provider_id=provider_id,
        route_id=route_id,
        retryable=False,
    )


def translate_exception(
    exc: BaseException,
    provider_id: ProviderIdentity,
    route_id: InferenceRouteIdentity | None = None,
) -> ProviderError:
    """Translate an arbitrary provider exception into the normalized taxonomy."""
    message = redact_secrets(str(exc))
    name = type(exc).__name__
    if "RateLimit" in name or "rate limit" in message.lower():
        return ProviderError(
            code=ProviderErrorCode.RATE_LIMITED,
            message=message,
            provider_id=provider_id,
            route_id=route_id,
            retryable=True,
        )
    if "Auth" in name or "auth" in message.lower() or "api key" in message.lower():
        return ProviderError(
            code=ProviderErrorCode.AUTH_FAILURE,
            message=message,
            provider_id=provider_id,
            route_id=route_id,
            retryable=False,
        )
    if "Timeout" in name or "Connect" in name or "Network" in name:
        return ProviderError(
            code=ProviderErrorCode.TRANSIENT_TRANSPORT,
            message=message,
            provider_id=provider_id,
            route_id=route_id,
            retryable=True,
        )
    return ProviderError(
        code=ProviderErrorCode.UNKNOWN,
        message=message,
        provider_id=provider_id,
        route_id=route_id,
        retryable=False,
    )


def _int_or_none(value: Any) -> int | None:
    """Coerce a value to int, returning None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
