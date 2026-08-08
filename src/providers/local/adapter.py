"""Local endpoint adapter for OmniForge.

This adapter connects to local OpenAI-compatible inference servers (Ollama,
vLLM, llama.cpp, LM Studio, SGLang, generic) using a ``LocalEndpointProfile``.
The runtime name is kept as a route label; the model identity remains distinct.
"""

from __future__ import annotations

from typing import Any

from src.providers._models import ModelDescriptor
from src.providers._openai_compat import OpenAICompatibleAdapter
from src.providers.adapter import ProviderAdapterCapabilities
from src.providers.identity import ProviderIdentity
from src.providers.local.profile import LocalEndpointProfile, LocalModelConfig
from src.providers.request import ProviderRequest
from src.routing.capabilities import DeploymentMode, ModelCapabilities
from src.routing.model_identity import ModelIdentity


class LocalEndpointAdapter(OpenAICompatibleAdapter):
    """Adapter for a local OpenAI-compatible inference endpoint.

    The provider identity is a local/self-hosted route identity. The underlying
    model identity is preserved separately so the same model served by Ollama and
    vLLM can coexist as distinct routes.
    """

    def __init__(
        self,
        *,
        model_config: LocalModelConfig,
        capabilities: ProviderAdapterCapabilities | None = None,
    ) -> None:
        profile = model_config.profile
        model_identity = ModelIdentity(
            model_id=model_config.model_id,
            family=model_config.family,
            capability_metadata={"deployment_mode": DeploymentMode.LOCAL.value},
        )
        provider_identity = ProviderIdentity(
            provider_id=f"local-{profile.runtime_kind.value}",
            display_name=f"Local ({profile.runtime_kind.value})",
            failure_domain=profile.failure_domain,
            metadata={"runtime_kind": profile.runtime_kind.value},
        )
        explicit_caps = model_config.explicit_capabilities
        descriptor = ModelDescriptor(
            model_id=model_identity.model_id,
            family=model_identity.family,
            context_tokens=explicit_caps.get("context_tokens", 4096),
            structured_output=explicit_caps.get("structured_output", False),
            tool_use=explicit_caps.get("tool_use", False),
            streaming=explicit_caps.get("streaming", False),
            reasoning=explicit_caps.get("reasoning", False),
            code_generation=explicit_caps.get("code_generation", False),
            multimodal=explicit_caps.get("multimodal", False),
            deployment_mode=DeploymentMode.LOCAL,
        )
        self._descriptor = descriptor
        self._profile = profile
        super().__init__(
            provider_id=provider_identity,
            model_id=model_identity,
            route_id=profile.route_identity,
            api_key=None,
            base_url=profile.base_url,
            capabilities=capabilities
            or ProviderAdapterCapabilities(
                streaming=descriptor.streaming,
                tool_calls=descriptor.tool_use,
                structured_output=descriptor.structured_output,
                reasoning=descriptor.reasoning,
                cancellation=True,
            ),
        )

    @property
    def profile(self) -> LocalEndpointProfile:
        """Return the local endpoint profile for this adapter."""
        return self._profile

    @property
    def model_capabilities(self) -> ModelCapabilities:
        """Return canonical capabilities for the configured local model."""
        return self._descriptor.to_capabilities()

    def _build_chat_params(self, request: ProviderRequest) -> dict[str, Any]:
        params = super()._build_chat_params(request)
        # Local endpoints cannot be assumed to support OpenAI reasoning_effort.
        if not self._capabilities.reasoning:
            params.pop("reasoning_effort", None)
        return params
