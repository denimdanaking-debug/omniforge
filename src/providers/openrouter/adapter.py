"""OpenRouter gateway adapter for OmniForge.

OpenRouter is an inference gateway, not a model family. This adapter forwards
requests through OpenRouter's OpenAI-compatible API while preserving the
underlying provider and model identities in normalized responses. Gateway-specific
failures are attributed to the OpenRouter route so they do not corrupt the
underlying model's reputation evidence.
"""

from __future__ import annotations

from typing import Any

from src.providers._models import ModelDescriptor
from src.providers._openai_compat import OpenAICompatibleAdapter
from src.providers.identity import ProviderIdentity
from src.providers.request import ProviderRequest
from src.routing.capabilities import ModelCapabilities
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

_OPENROUTER_PROVIDER_IDENTITY = ProviderIdentity("openrouter", "OpenRouter", "openrouter.ai")


# Conservative default for an unknown underlying model routed through OpenRouter.
# OpenRouter is the transport route; it must not be treated as a source of model
# capability truth.
_DEFAULT_MODEL_CAPABILITIES = ModelCapabilities(
    context_tokens=4096,
    structured_output=False,
    tool_use=False,
    streaming=False,
    reasoning=False,
    code_generation=False,
    multimodal=False,
)


def _default_route_identity() -> InferenceRouteIdentity:
    """Return the canonical OpenRouter gateway route identity."""
    return InferenceRouteIdentity(
        route_id="openrouter-gateway",
        provider_id="openrouter",
        route_type=RouteType.GATEWAY,
        endpoint_key="openrouter://gateway",
        failure_domain="openrouter.ai",
    )


class OpenRouterAdapter(OpenAICompatibleAdapter):
    """Gateway adapter that routes requests through OpenRouter.

    The adapter's provider identity is the *underlying* provider (e.g. Anthropic,
    Qwen, DeepSeek). The OpenRouter gateway is represented by ``route_id``.
    This keeps model reputation evidence attached to the actual model while
    allowing route-specific failure attribution.

    The route identity must be a GATEWAY route. OpenRouter is categorically a
    gateway, not a direct/local/enterprise provider. The underlying model's
    capabilities must be supplied explicitly; the adapter will not infer them
    from the existence of the OpenRouter route.
    """

    def __init__(
        self,
        *,
        provider_identity: ProviderIdentity,
        model_identity: ModelIdentity,
        route_identity: InferenceRouteIdentity | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        capabilities: Any = None,
        model_descriptor: ModelDescriptor | None = None,
        model_capabilities: ModelCapabilities | None = None,
    ) -> None:
        resolved_route = route_identity or _default_route_identity()
        if resolved_route.route_type is not RouteType.GATEWAY:
            raise ValueError(
                f"OpenRouter route must be RouteType.GATEWAY, got {resolved_route.route_type.value}"
            )

        if model_capabilities is not None:
            self._model_capabilities = model_capabilities
        elif model_descriptor is not None:
            self._model_capabilities = model_descriptor.to_capabilities()
        else:
            self._model_capabilities = _DEFAULT_MODEL_CAPABILITIES

        # Keep a descriptor around for internal consistency; it reflects the
        # supplied underlying-model capabilities, not OpenRouter defaults.
        self._descriptor = ModelDescriptor(
            model_id=model_identity.model_id,
            family=model_identity.family,
            lifecycle=model_identity.lifecycle,
            context_tokens=self._model_capabilities.context_tokens,
            structured_output=self._model_capabilities.structured_output,
            tool_use=self._model_capabilities.tool_use,
            streaming=self._model_capabilities.streaming,
            reasoning=self._model_capabilities.reasoning,
            code_generation=self._model_capabilities.code_generation,
            multimodal=self._model_capabilities.multimodal,
            supported_roles=self._model_capabilities.supported_roles,
        )
        super().__init__(
            provider_id=provider_identity,
            model_id=model_identity,
            route_id=resolved_route,
            api_key=api_key,
            base_url=base_url or DEFAULT_BASE_URL,
            capabilities=capabilities,
        )

    @property
    def model_capabilities(self) -> ModelCapabilities:
        """Return canonical capabilities for the configured underlying model."""
        return self._model_capabilities

    def _build_chat_params(self, request: ProviderRequest) -> dict[str, Any]:
        params = super()._build_chat_params(request)
        # OpenRouter does not use OpenAI-style reasoning_effort; reasoning models
        # are selected by model ID.
        params.pop("reasoning_effort", None)
        return params
