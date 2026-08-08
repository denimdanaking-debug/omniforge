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
    ) -> None:
        self._descriptor = ModelDescriptor(
            model_id=model_identity.model_id,
            family=model_identity.family,
            lifecycle=model_identity.lifecycle,
            context_tokens=128_000,
            supported_roles=frozenset(),
        )
        super().__init__(
            provider_id=provider_identity,
            model_id=model_identity,
            route_id=route_identity or _default_route_identity(),
            api_key=api_key,
            base_url=base_url or DEFAULT_BASE_URL,
            capabilities=capabilities,
        )

    @property
    def model_capabilities(self) -> ModelCapabilities:
        """Return canonical capabilities for the configured model.

        OpenRouter exposes many models; capabilities should be supplied explicitly
        via ``model_descriptor`` or inherited from the model identity's metadata
        when available. By default we return a conservative cloud profile.
        """
        return self._descriptor.to_capabilities()

    def _build_chat_params(self, request: ProviderRequest) -> dict[str, Any]:
        params = super()._build_chat_params(request)
        # OpenRouter does not use OpenAI-style reasoning_effort; reasoning models
        # are selected by model ID.
        params.pop("reasoning_effort", None)
        return params
