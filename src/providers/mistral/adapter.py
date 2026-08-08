"""OpenAI-compatible adapter for Mistral and Devstral models.

Mistral exposes an OpenAI-compatible chat completions API. This adapter supports
general Mistral models as well as Devstral coding-focused models, with distinct
model identities and capability profiles that can be configured without source-code
edits.
"""

from __future__ import annotations

from typing import Any

from src.providers._models import ModelDescriptor, full_eligibility_roles
from src.providers._openai_compat import OpenAICompatibleAdapter
from src.providers.identity import ProviderIdentity
from src.providers.request import ProviderRequest
from src.routing.capabilities import ModelCapabilities
from src.routing.model_identity import ModelIdentity, ModelLifecycle

DEFAULT_BASE_URL = "https://api.mistral.ai/v1"

_PROVIDER_IDENTITY = ProviderIdentity("mistral", "Mistral AI", "mistral.ai")

_GENERAL_MODEL_ID = "mistral-large-latest"
_CODING_MODEL_ID = "codestral-latest"
_DEVSTRAL_MODEL_ID = "devstral-small-latest"

_GENERAL_DESCRIPTOR = ModelDescriptor(
    model_id=_GENERAL_MODEL_ID,
    family="mistral",
    lifecycle=ModelLifecycle.HIGH_RISK,
    context_tokens=128_000,
    supported_roles=full_eligibility_roles(),
)

_CODING_DESCRIPTOR = ModelDescriptor(
    model_id=_CODING_MODEL_ID,
    family="mistral",
    lifecycle=ModelLifecycle.HIGH_RISK,
    context_tokens=128_000,
    code_generation=True,
    supported_roles=full_eligibility_roles(),
)

_DEVSTRAL_DESCRIPTOR = ModelDescriptor(
    model_id=_DEVSTRAL_MODEL_ID,
    family="devstral",
    lifecycle=ModelLifecycle.HIGH_RISK,
    context_tokens=128_000,
    code_generation=True,
    supported_roles=full_eligibility_roles(),
)

_DEFAULT_MODEL_IDENTITY = _GENERAL_DESCRIPTOR.to_identity()


class MistralAdapter(OpenAICompatibleAdapter):
    """OpenAI-compatible adapter for Mistral and Devstral models."""

    DEFAULT_MODEL: ModelIdentity = _DEFAULT_MODEL_IDENTITY

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
        resolved_identity = model_identity or _descriptor_for_model_id(model_id).to_identity()
        if model_id is not None and model_identity is None:
            resolved_identity = ModelIdentity(
                model_id=model_id,
                family=_descriptor_for_model_id(model_id).family,
                lifecycle=ModelLifecycle.HIGH_RISK,
            )

        self._descriptor = _descriptor_for_identity(resolved_identity)
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
        # Mistral's OpenAI-compatible endpoint does not use reasoning_effort.
        params.pop("reasoning_effort", None)
        return params


def _descriptor_for_model_id(model_id: str | None) -> ModelDescriptor:
    if model_id is None:
        return _GENERAL_DESCRIPTOR
    lowered = model_id.lower()
    if "devstral" in lowered:
        return _DEVSTRAL_DESCRIPTOR
    if "codestral" in lowered:
        return _CODING_DESCRIPTOR
    return _GENERAL_DESCRIPTOR


def _descriptor_for_identity(identity: ModelIdentity) -> ModelDescriptor:
    return _descriptor_for_model_id(identity.model_id)
