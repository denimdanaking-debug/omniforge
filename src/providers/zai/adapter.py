"""OpenAI-compatible adapter for Z.AI / GLM."""

from __future__ import annotations

from typing import Any

from src.providers._models import ModelDescriptor
from src.providers._openai_compat import OpenAICompatibleAdapter
from src.providers.identity import ProviderIdentity
from src.routing.model_identity import ModelLifecycle

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

_PROVIDER_IDENTITY = ProviderIdentity("zai", "Z.AI / GLM", "z.ai")

_DEFAULT_MODEL = ModelDescriptor(
    model_id="glm-4-plus",
    family="glm",
    lifecycle=ModelLifecycle.NORMAL,
    context_tokens=128_000,
).to_identity()


class ZAIAdapter(OpenAICompatibleAdapter):
    """OpenAI-compatible adapter for Z.AI / GLM."""

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
