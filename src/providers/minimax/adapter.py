"""OpenAI-compatible adapter for MiniMax.

MiniMax exposes an OpenAI-compatible chat completions API. This adapter reuses
OmniForge's shared compatible transport while preserving a distinct MiniMax
provider identity and configurable model identity.
"""

from __future__ import annotations

from typing import Any

from src.providers._models import ModelDescriptor, full_eligibility_roles
from src.providers._openai_compat import OpenAICompatibleAdapter
from src.providers.identity import ProviderIdentity
from src.providers.request import ProviderRequest
from src.routing.capabilities import ModelCapabilities
from src.routing.model_identity import ModelIdentity, ModelLifecycle

DEFAULT_BASE_URL = "https://api.minimaxi.chat/v1"

_PROVIDER_IDENTITY = ProviderIdentity("minimax", "MiniMax", "minimaxi.chat")

_DEFAULT_MODEL_IDENTITY = ModelDescriptor(
    model_id="minimax-text-01",
    family="minimax",
    lifecycle=ModelLifecycle.HIGH_RISK,
    context_tokens=256_000,
    supported_roles=full_eligibility_roles(),
).to_identity()


def _default_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        model_id="minimax-text-01",
        family="minimax",
        lifecycle=ModelLifecycle.HIGH_RISK,
        context_tokens=256_000,
        supported_roles=full_eligibility_roles(),
    )


class MiniMaxAdapter(OpenAICompatibleAdapter):
    """OpenAI-compatible adapter for MiniMax."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        model_identity: ModelIdentity | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        route_id: Any = None,
        capabilities: Any = None,
    ) -> None:
        resolved_identity = model_identity or _default_descriptor().to_identity()
        if model_id is not None and model_identity is None:
            resolved_identity = ModelIdentity(
                model_id=model_id,
                family=_DEFAULT_MODEL_IDENTITY.family,
                lifecycle=ModelLifecycle.HIGH_RISK,
            )

        self._descriptor = ModelDescriptor(
            model_id=resolved_identity.model_id,
            family=resolved_identity.family,
            lifecycle=resolved_identity.lifecycle,
            context_tokens=256_000,
            supported_roles=full_eligibility_roles(),
        )
        super().__init__(
            provider_id=_PROVIDER_IDENTITY,
            model_id=resolved_identity,
            route_id=route_id,
            api_key=api_key,
            base_url=base_url or DEFAULT_BASE_URL,
            capabilities=capabilities,
        )

    @property
    def model_capabilities(self) -> ModelCapabilities:
        """Return canonical capabilities for the configured model."""
        return self._descriptor.to_capabilities()

    def _build_chat_params(self, request: ProviderRequest) -> dict[str, Any]:
        params = super()._build_chat_params(request)
        # MiniMax does not support OpenAI-style reasoning_effort parameters.
        params.pop("reasoning_effort", None)
        return params
