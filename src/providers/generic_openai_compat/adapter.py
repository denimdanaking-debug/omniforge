"""Configurable generic OpenAI-compatible endpoint adapter.

This adapter allows OmniForge to talk to arbitrary OpenAI-compatible endpoints
(custom enterprise endpoints, new cloud providers, self-hosted servers, temporary
vendors, development test endpoints) without source-code edits. Capabilities are
explicitly configured and default to conservative/unknown rather than assuming
full OpenAI parity.
"""

from __future__ import annotations

from typing import Any

from src.providers._models import ModelDescriptor
from src.providers._openai_compat import OpenAICompatibleAdapter
from src.providers.adapter import ProviderAdapterCapabilities
from src.providers.identity import ProviderIdentity
from src.providers.request import ProviderRequest
from src.routing.capabilities import DeploymentMode, ModelCapabilities
from src.routing.inference_route import InferenceRouteIdentity
from src.routing.model_identity import ModelIdentity


class GenericOpenAICompatibleAdapter(OpenAICompatibleAdapter):
    """Generic adapter for an arbitrary OpenAI-compatible endpoint.

    Identity, route, base URL, capabilities, and model metadata are supplied by
    configuration. The configured model identity survives request -> transport
    -> response -> telemetry without collapsing into the OpenAI provider identity.
    """

    def __init__(
        self,
        *,
        provider_identity: ProviderIdentity,
        model_identity: ModelIdentity,
        route_identity: InferenceRouteIdentity | None = None,
        base_url: str,
        api_key: str | None = None,
        capabilities: ProviderAdapterCapabilities | None = None,
        model_descriptor: ModelDescriptor | None = None,
    ) -> None:
        self._descriptor = model_descriptor or ModelDescriptor(
            model_id=model_identity.model_id,
            family=model_identity.family,
            context_tokens=4096,
            structured_output=False,
            tool_use=False,
            streaming=False,
            reasoning=False,
            code_generation=False,
            multimodal=False,
            deployment_mode=DeploymentMode(
                model_identity.capability_metadata.get(
                    "deployment_mode", DeploymentMode.CLOUD.value
                )
            ),
        )
        super().__init__(
            provider_id=provider_identity,
            model_id=model_identity,
            route_id=route_identity,
            api_key=api_key,
            base_url=base_url,
            capabilities=capabilities
            or ProviderAdapterCapabilities(
                streaming=False,
                tool_calls=False,
                structured_output=False,
                reasoning=False,
                cancellation=True,
            ),
        )

    @property
    def model_capabilities(self) -> ModelCapabilities:
        """Return canonical capabilities for the configured model."""
        return self._descriptor.to_capabilities()

    def _build_chat_params(self, request: ProviderRequest) -> dict[str, Any]:
        params = super()._build_chat_params(request)
        # Generic endpoints cannot be assumed to support OpenAI reasoning_effort.
        if not self._capabilities.reasoning:
            params.pop("reasoning_effort", None)
        return params
