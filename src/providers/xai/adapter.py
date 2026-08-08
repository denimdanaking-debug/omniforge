"""OpenAI-compatible adapter for xAI."""

from __future__ import annotations

from typing import Any

from src.providers._models import ModelDescriptor
from src.providers._openai_compat import OpenAICompatibleAdapter
from src.providers.identity import ProviderIdentity
from src.routing.model_identity import ModelLifecycle

DEFAULT_BASE_URL = "https://api.x.ai/v1"

_PROVIDER_IDENTITY = ProviderIdentity("xai", "xAI", "x.ai")

_DEFAULT_MODEL = ModelDescriptor(
    model_id="grok-3-latest",
    family="grok",
    lifecycle=ModelLifecycle.NORMAL,
    context_tokens=128_000,
).to_identity()


class XAIAdapter(OpenAICompatibleAdapter):
    """OpenAI-compatible adapter for xAI."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        route_id: Any = None,
        capabilities: Any = None,
    ) -> None:
        super().__init__(
            provider_id=_PROVIDER_IDENTITY,
            model_id=_DEFAULT_MODEL,
            route_id=route_id,
            api_key=api_key,
            base_url=base_url or DEFAULT_BASE_URL,
            capabilities=capabilities,
        )
