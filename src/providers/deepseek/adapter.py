"""OpenAI-compatible adapter for DeepSeek."""

from __future__ import annotations

from typing import Any

from src.providers._models import ModelDescriptor
from src.providers._openai_compat import OpenAICompatibleAdapter
from src.providers.identity import ProviderIdentity
from src.routing.model_identity import ModelIdentity, ModelLifecycle

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

_PROVIDER_IDENTITY = ProviderIdentity("deepseek", "DeepSeek", "deepseek.com")

_DEEPSEEK_CHAT = ModelDescriptor(
    model_id="deepseek-chat",
    family="deepseek",
    lifecycle=ModelLifecycle.NORMAL,
    context_tokens=64_000,
).to_identity()

_DEEPSEEK_REASONER = ModelDescriptor(
    model_id="deepseek-reasoner",
    family="deepseek",
    lifecycle=ModelLifecycle.HIGH_RISK,
    reasoning=True,
    context_tokens=64_000,
).to_identity()


class DeepSeekAdapter(OpenAICompatibleAdapter):
    """OpenAI-compatible adapter for DeepSeek."""

    DEFAULT_MODEL: ModelIdentity = _DEEPSEEK_CHAT
    SUPPORTED_MODELS: tuple[ModelIdentity, ...] = (_DEEPSEEK_CHAT, _DEEPSEEK_REASONER)

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
            model_id=self.DEFAULT_MODEL,
            route_id=route_id,
            api_key=api_key,
            base_url=base_url or DEFAULT_BASE_URL,
            capabilities=capabilities,
        )
