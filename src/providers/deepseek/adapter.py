"""OpenAI-compatible adapter for DeepSeek."""

from __future__ import annotations

from typing import Any

from src.providers._models import ModelDescriptor, full_eligibility_roles
from src.providers._openai_compat import OpenAICompatibleAdapter
from src.providers.adapter import ProviderAdapterCapabilities
from src.providers.identity import ProviderIdentity
from src.routing.capabilities import ModelCapabilities
from src.routing.model_identity import ModelIdentity, ModelLifecycle

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

_PROVIDER_IDENTITY = ProviderIdentity("deepseek", "DeepSeek", "deepseek.com")

_CHAT_DESCRIPTOR = ModelDescriptor(
    model_id="deepseek-chat",
    family="deepseek",
    lifecycle=ModelLifecycle.NORMAL,
    context_tokens=64_000,
    supported_roles=full_eligibility_roles(),
)

_REASONER_DESCRIPTOR = ModelDescriptor(
    model_id="deepseek-reasoner",
    family="deepseek",
    lifecycle=ModelLifecycle.HIGH_RISK,
    reasoning=True,
    context_tokens=64_000,
    supported_roles=full_eligibility_roles(),
)

_DEFAULT_MODEL_IDENTITY = _CHAT_DESCRIPTOR.to_identity()


class DeepSeekAdapter(OpenAICompatibleAdapter):
    """OpenAI-compatible adapter for DeepSeek."""

    DEFAULT_MODEL: ModelIdentity = _DEFAULT_MODEL_IDENTITY
    SUPPORTED_MODELS: tuple[ModelIdentity, ...] = (
        _CHAT_DESCRIPTOR.to_identity(),
        _REASONER_DESCRIPTOR.to_identity(),
    )

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
        resolved_identity = model_identity or _DEFAULT_MODEL_IDENTITY
        if model_id is not None and model_identity is None:
            resolved_identity = ModelIdentity(
                model_id=model_id,
                family="deepseek",
                lifecycle=ModelLifecycle.HIGH_RISK
                if model_id == "deepseek-reasoner"
                else ModelLifecycle.NORMAL,
            )

        self._descriptor = _descriptor_for_identity(resolved_identity)
        super().__init__(
            provider_id=_PROVIDER_IDENTITY,
            model_id=resolved_identity,
            route_id=route_id,
            api_key=api_key,
            base_url=base_url or DEFAULT_BASE_URL,
            capabilities=capabilities
            or ProviderAdapterCapabilities(
                streaming=True,
                tool_calls=True,
                structured_output=True,
                cancellation=True,
                reasoning=resolved_identity.model_id == "deepseek-reasoner",
            ),
        )

    @property
    def model_capabilities(self) -> ModelCapabilities:
        """Return canonical capabilities for the configured model."""
        return self._descriptor.to_capabilities()


def _descriptor_for_identity(identity: ModelIdentity) -> ModelDescriptor:
    if identity.model_id == "deepseek-reasoner":
        return _REASONER_DESCRIPTOR
    return _CHAT_DESCRIPTOR
