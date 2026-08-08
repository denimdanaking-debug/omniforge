"""OpenAI provider adapter for OmniForge.

Implements the provider-neutral contract using the official ``openai`` SDK via the
shared OpenAI-compatible base class. OpenAI-specific behavior is limited to
identity, model descriptors, latency tracking, and robust handling of the OpenAI
SDK response object shape.
"""

from __future__ import annotations

import json
import time
from typing import Any

from src.providers._common import parse_finish_reason, parse_tool_calls, parse_usage
from src.providers._models import ModelDescriptor, full_eligibility_roles
from src.providers._openai_compat import OpenAICompatibleAdapter
from src.providers.adapter import ProviderAdapterCapabilities
from src.providers.identity import ProviderIdentity
from src.providers.request import ProviderRequest, ReasoningMode, ToolChoiceMode
from src.providers.response import ProviderResponse
from src.routing.capabilities import ModelCapabilities
from src.routing.model_identity import ModelLifecycle


class OpenAIAdapter(OpenAICompatibleAdapter):
    """OpenAI API adapter exposing chat completions with full OmniForge normalization."""

    DEFAULT_MODEL_ID: str = "codex-mini-latest"

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        capabilities: ProviderAdapterCapabilities | None = None,
    ) -> None:
        provider = ProviderIdentity("openai", "OpenAI", "openai.com")
        resolved_model = model_id or self.DEFAULT_MODEL_ID
        descriptor = ModelDescriptor(
            model_id=resolved_model,
            family="codex",
            lifecycle=ModelLifecycle.HIGH_RISK,
            structured_output=True,
            tool_use=True,
            streaming=True,
            reasoning=True,
            code_generation=True,
            supported_roles=full_eligibility_roles(),
        )
        super().__init__(
            provider_id=provider,
            model_id=descriptor.to_identity(),
            api_key=api_key,
            base_url=base_url,
            capabilities=capabilities
            or ProviderAdapterCapabilities(
                streaming=True,
                tool_calls=True,
                structured_output=True,
                cancellation=True,
                reasoning=True,
            ),
        )
        self._descriptor: ModelDescriptor = descriptor
        self._submit_start: float | None = None

    @property
    def model_capabilities(self) -> ModelCapabilities:
        """Return canonical capabilities for the configured model."""
        return self._descriptor.to_capabilities()

    async def submit(self, request: ProviderRequest) -> ProviderResponse:
        """Submit a request and return a normalized response with latency."""
        self._submit_start = time.monotonic()
        try:
            return await super().submit(request)
        finally:
            self._submit_start = None

    def _build_chat_params(self, request: ProviderRequest) -> dict[str, Any]:
        """Build OpenAI chat-completion parameters with OpenAI-specific mappings."""
        params = super()._build_chat_params(request)

        if request.reasoning is not ReasoningMode.DEFAULT:
            params["reasoning_effort"] = _map_reasoning_effort(request.reasoning)

        # The shared base class does not map FORBIDDEN; OpenAI's "none" is the
        # equivalent signal.
        if request.tool_choice is ToolChoiceMode.FORBIDDEN and request.tools:
            params["tool_choice"] = "none"

        return params

    def _normalize_response(
        self, request: ProviderRequest, response: Any, *, latency_seconds: float | None = None
    ) -> ProviderResponse:
        """Normalize an OpenAI chat-completion response, handling SDK objects or dicts."""
        latency: float | None = latency_seconds
        if self._submit_start is not None:
            latency = time.monotonic() - self._submit_start

        # Convert real OpenAI SDK objects to plain dicts when possible so the
        # shared parsers can operate on a known shape.
        maybe_dict = _response_to_dict(response)
        if isinstance(maybe_dict, dict):
            response = maybe_dict

        if isinstance(response, dict):
            choice: Any = (response.get("choices") or [{}])[0] or {}
            message: Any = choice.get("message") or {}
            text = message.get("content") or None
            raw_tool_calls = message.get("tool_calls")
            finish_raw = choice.get("finish_reason")
            usage_raw = response.get("usage")
            provider_request_id = response.get("id")
        else:
            choice = response.choices[0]
            message = choice.message
            text = message.content or None
            raw_tool_calls = message.tool_calls if hasattr(message, "tool_calls") else None
            finish_raw = choice.finish_reason
            usage_raw = (
                response.usage.to_dict() if hasattr(response, "usage") and response.usage else None
            )
            provider_request_id = response.id if hasattr(response, "id") else None

        tool_calls = parse_tool_calls(raw_tool_calls)
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
            model_id=self._resolve_model_identity(request),
            route_id=self._route_id,
            text=text,
            structured_result=structured,
            tool_calls=tool_calls,
            finish_reason=parse_finish_reason(finish_raw),
            usage=parse_usage(usage_raw),
            latency_seconds=latency,
            provider_request_id=provider_request_id,
        )


def _map_reasoning_effort(mode: ReasoningMode) -> str:
    mapping = {
        ReasoningMode.EFFORT_LOW: "low",
        ReasoningMode.EFFORT_MEDIUM: "medium",
        ReasoningMode.EFFORT_HIGH: "high",
        ReasoningMode.DISABLED: "none",
    }
    return mapping.get(mode, "medium")


def _response_to_dict(response: Any) -> Any:
    """Attempt to convert an OpenAI SDK response object to a dict."""
    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            result = model_dump()
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    return response
