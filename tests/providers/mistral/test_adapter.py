"""Mistral / Devstral-specific adapter unit tests."""

from __future__ import annotations

from typing import Any

import pytest

from src.policy.risk import RiskLevel
from src.providers.errors import ProviderErrorCode
from src.providers.identity import ProviderIdentity
from src.providers.mistral.adapter import DEFAULT_BASE_URL, MistralAdapter
from src.providers.request import Message, MessageRole, ProviderRequest
from src.providers.response import FinishReason, StreamingState
from src.routing.model_identity import ModelIdentity, ModelLifecycle
from src.routing.roles import ExecutionRole
from tests.providers._openai_compat_mocks import (
    FakeAuthError,
    FakeRateLimitError,
    build_mock_openai_client,
    make_stream_chunks,
    make_success_response,
)


@pytest.fixture
def adapter(monkeypatch: Any) -> MistralAdapter:
    client = build_mock_openai_client(
        response=make_success_response(content="Hello"),
        stream_chunks=make_stream_chunks(["Hello", " world"]),
    )
    adapter = MistralAdapter(api_key="test-key")
    monkeypatch.setattr(adapter, "_client", lambda: client)
    return adapter


async def test_identity(adapter: MistralAdapter) -> None:
    identity = adapter.identity
    assert identity == ProviderIdentity("mistral", "Mistral AI", "mistral.ai")
    assert identity.provider_id == "mistral"


async def test_default_model_identity(adapter: MistralAdapter) -> None:
    model = adapter.model_id
    assert model.model_id == "mistral-large-latest"
    assert model.family == "mistral"
    assert model.lifecycle is ModelLifecycle.HIGH_RISK


async def test_default_base_url() -> None:
    adapter = MistralAdapter(api_key="test-key")
    assert adapter._base_url == DEFAULT_BASE_URL


async def test_configurable_model_id() -> None:
    adapter = MistralAdapter(api_key="test-key", model_id="mistral-small-latest")
    assert adapter.model_id.model_id == "mistral-small-latest"
    assert adapter.model_id.family == "mistral"


async def test_devstral_model_retains_distinct_family() -> None:
    adapter = MistralAdapter(api_key="test-key", model_id="devstral-small-latest")
    assert adapter.model_id.model_id == "devstral-small-latest"
    assert adapter.model_id.family == "devstral"
    assert adapter.model_capabilities.code_generation is True


async def test_configurable_model_identity() -> None:
    custom = ModelIdentity(model_id="codestral-latest", family="mistral")
    adapter = MistralAdapter(api_key="test-key", model_identity=custom)
    assert adapter.model_id == custom


async def test_codestral_capabilities(adapter: MistralAdapter) -> None:
    adapter = MistralAdapter(api_key="test-key", model_id="codestral-latest")
    assert adapter.model_capabilities.code_generation is True


async def test_payload_translation_uses_default_model(adapter: MistralAdapter) -> None:
    request = ProviderRequest(
        request_id="req-1",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )
    params = adapter._build_chat_params(request)
    assert params["model"] == "mistral-large-latest"
    assert params["messages"] == [{"role": "user", "content": "Hi"}]


async def test_response_translation_text(adapter: MistralAdapter) -> None:
    request = ProviderRequest(
        request_id="req-2",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
    )
    response = await adapter.submit(request)
    assert response.request_id == "req-2"
    assert response.provider_id == adapter.identity
    assert response.model_id == adapter.model_id
    assert response.text == "Hello"
    assert response.finish_reason is FinishReason.STOP


async def test_response_translation_streaming(adapter: MistralAdapter) -> None:
    request = ProviderRequest(
        request_id="req-3",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
    )
    chunks = [chunk async for chunk in adapter.stream(request)]
    assert len(chunks) == 3
    assert chunks[-1].streaming_state is StreamingState.COMPLETE


async def test_error_normalization_rate_limited(monkeypatch: Any) -> None:
    client = build_mock_openai_client(exception=FakeRateLimitError)
    adapter = MistralAdapter(api_key="test-key")
    monkeypatch.setattr(adapter, "_client", lambda: client)
    request = ProviderRequest(
        request_id="req-4",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
    )
    response = await adapter.submit(request)
    assert response.error_reference == ProviderErrorCode.RATE_LIMITED.value


async def test_error_normalization_auth_failure(monkeypatch: Any) -> None:
    client = build_mock_openai_client(exception=FakeAuthError)
    adapter = MistralAdapter(api_key="test-key")
    monkeypatch.setattr(adapter, "_client", lambda: client)
    request = ProviderRequest(
        request_id="req-5",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
    )
    response = await adapter.submit(request)
    assert response.error_reference == ProviderErrorCode.AUTH_FAILURE.value


async def test_supported_roles_are_canonical() -> None:
    adapter = MistralAdapter(api_key="test-key")
    caps = adapter.model_capabilities
    assert ExecutionRole.PLANNING.value in caps.supported_roles
    assert ExecutionRole.HIGH_RISK_REVIEW.value in caps.supported_roles
