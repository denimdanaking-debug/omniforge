"""OpenAI-compatible adapter for Qwen (Alibaba)."""

from __future__ import annotations

from typing import Any

from src.providers._models import ModelDescriptor
from src.providers._openai_compat import OpenAICompatibleAdapter
from src.providers.identity import ProviderIdentity
from src.routing.model_identity import ModelLifecycle

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_PROVIDER_IDENTITY = ProviderIdentity("qwen", "Qwen (Alibaba)", "aliyun.com")

_DEFAULT_MODEL = ModelDescriptor(
    model_id="qwen3.8-max",
    family="qwen",
    lifecycle=ModelLifecycle.HIGH_RISK,
    context_tokens=128_000,
).to_identity()


class QwenAdapter(OpenAICompatibleAdapter):
    """OpenAI-compatible adapter for Qwen (Alibaba)."""

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
