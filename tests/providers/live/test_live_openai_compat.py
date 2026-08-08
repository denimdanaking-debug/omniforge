"""Optional live integration tests for OpenAI-compatible providers.

These tests are skipped automatically when the corresponding API key environment
variable is not set. They are NOT run in ordinary CI.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from src.policy.risk import RiskLevel
from src.providers.request import Message, MessageRole, ProviderRequest
from src.routing.roles import ExecutionRole

pytestmark = [pytest.mark.live]

_LIVE_CONFIGS: list[tuple[str, str, str, str]] = [
    ("openai", "OPENAI_API_KEY", "src.providers.openai.adapter:OpenAIAdapter", "codex-mini-latest"),
    ("kimi", "KIMI_API_KEY", "src.providers.kimi.adapter:KimiAdapter", "kimi-k3-latest"),
    ("qwen", "QWEN_API_KEY", "src.providers.qwen.adapter:QwenAdapter", "qwen3.8-max"),
    (
        "deepseek",
        "DEEPSEEK_API_KEY",
        "src.providers.deepseek.adapter:DeepSeekAdapter",
        "deepseek-chat",
    ),
    ("xai", "XAI_API_KEY", "src.providers.xai.adapter:XAIAdapter", "grok-3-latest"),
    ("zai", "ZAI_API_KEY", "src.providers.zai.adapter:ZAIAdapter", "glm-4-plus"),
]


def _load_factory(dotted_path: str) -> Any:
    module_path, _, name = dotted_path.partition(":")
    module = __import__(module_path, fromlist=[name])
    return getattr(module, name)


@pytest.mark.parametrize("provider_id, env_var, dotted_path, default_model", _LIVE_CONFIGS)
async def test_live_openai_compat_simple_completion(
    provider_id: str, env_var: str, dotted_path: str, default_model: str
) -> None:
    api_key = os.environ.get(env_var)
    if not api_key:
        pytest.skip(f"Live test skipped: {env_var} not set")

    factory = _load_factory(dotted_path)
    adapter = factory(api_key=api_key, model_id=default_model)
    request = ProviderRequest(
        request_id=f"live-{provider_id}",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R3_HIGH,
        messages=[Message(role=MessageRole.USER, content="Say 'pong' and nothing else.")],
        max_output_tokens=20,
    )
    response = await adapter.submit(request)
    assert response.text is not None
    assert response.provider_id == adapter.identity
    assert response.model_id is not None
