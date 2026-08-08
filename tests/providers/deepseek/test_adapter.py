"""DeepSeek adapter unit tests."""

from __future__ import annotations

from typing import Any

import pytest

from src.policy.risk import RiskLevel
from src.providers.adapter import ProviderAdapterCapabilities
from src.providers.deepseek.adapter import DEFAULT_BASE_URL, DeepSeekAdapter
from src.providers.errors import ProviderErrorCode
from src.providers.identity import ProviderIdentity
from src.providers.request import (
    CapabilityRequirement,
    Message,
    MessageRole,
    ProviderRequest,
    ToolChoiceMode,
    ToolDefinition,
)
from src.providers.response import FinishReason, StreamingState
from src.routing.model_identity import ModelLifecycle
from src.routing.roles import ExecutionRole
from tests.providers._openai_compat_mocks import (
    FakeAuthError,
    FakeRateLimitError,
    build_mock_openai_client,
    make_stream_chunks,
    make_success_response,
)


@pytest.fixture
def adapter(monkeypatch: Any) -> DeepSeekAdapter:
    client = build_mock_openai_client(
        response=make_success_response(
            content="Hello",
            tool_calls=[
                {
                    "id": "tc-1",
                    "function": {"name": "read_file", "arguments": '{"path": "/tmp"}'},
                }
            ],
        ),
        stream_chunks=make_stream_chunks(["Hello", " world"]),
    )
    adapter = DeepSeekAdapter(api_key="test-key")
    monkeypatch.setattr(adapter, "_client", lambda: client)
    return adapter


async def test_identity(adapter: DeepSeekAdapter) -> None:
    identity = adapter.identity
    assert identity == ProviderIdentity("deepseek", "DeepSeek", "deepseek.com")
    assert identity.provider_id == "deepseek"


async def test_default_model_identity(adapter: DeepSeekAdapter) -> None:
    model = adapter.model_id
    assert model.model_id == "deepseek-chat"
    assert model.family == "deepseek"
    assert model.lifecycle is ModelLifecycle.NORMAL


async def test_supported_models_exposes_distinct_ids() -> None:
    model_ids = {m.model_id for m in DeepSeekAdapter.SUPPORTED_MODELS}
    assert model_ids == {"deepseek-chat", "deepseek-reasoner"}


async def test_default_base_url() -> None:
    adapter = DeepSeekAdapter(api_key="test-key")
    assert adapter._base_url == DEFAULT_BASE_URL


async def test_base_url_override() -> None:
    adapter = DeepSeekAdapter(api_key="test-key", base_url="https://custom.example/v1")
    assert adapter._base_url == "https://custom.example/v1"


async def test_default_capabilities() -> None:
    adapter = DeepSeekAdapter(api_key="test-key")
    caps = adapter.capabilities
    assert caps.streaming is True
    assert caps.tool_calls is True
    assert caps.structured_output is True
    assert caps.cancellation is True


async def test_payload_translation_uses_default_model(adapter: DeepSeekAdapter) -> None:
    request = ProviderRequest(
        request_id="req-1",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )
    params = adapter._build_chat_params(request)
    assert params["model"] == "deepseek-chat"
    assert params["messages"] == [{"role": "user", "content": "Hi"}]


async def test_payload_translation_reasoner_model(adapter: DeepSeekAdapter) -> None:
    reasoner = next(
        m for m in DeepSeekAdapter.SUPPORTED_MODELS if m.model_id == "deepseek-reasoner"
    )
    request = ProviderRequest(
        request_id="req-2",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
        target_model=reasoner,
    )
    params = adapter._build_chat_params(request)
    assert params["model"] == "deepseek-reasoner"


async def test_payload_translation_with_overrides(adapter: DeepSeekAdapter) -> None:
    request = ProviderRequest(
        request_id="req-3",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        system_instructions=["Be helpful"],
        messages=[Message(role=MessageRole.USER, content="Hi")],
        temperature=0.5,
        max_output_tokens=100,
        stop_sequences=["STOP"],
    )
    params = adapter._build_chat_params(request)
    assert params["messages"] == [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hi"},
    ]
    assert params["temperature"] == 0.5
    assert params["max_tokens"] == 100
    assert params["stop"] == ["STOP"]


async def test_payload_translation_tool_choice_required(adapter: DeepSeekAdapter) -> None:
    request = ProviderRequest(
        request_id="req-4",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Use tool")],
        tools=[ToolDefinition(name="read_file", description="Read a file")],
        tool_choice=ToolChoiceMode.REQUIRED,
    )
    params = adapter._build_chat_params(request)
    assert params["tool_choice"] == "required"
    assert params["tools"][0]["function"]["name"] == "read_file"


async def test_response_translation_text(adapter: DeepSeekAdapter) -> None:
    request = ProviderRequest(
        request_id="req-5",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
    )
    response = await adapter.submit(request)
    assert response.request_id == "req-5"
    assert response.provider_id == adapter.identity
    assert response.model_id == adapter.model_id
    assert response.text == "Hello"
    assert response.finish_reason is FinishReason.STOP
    assert response.latency_seconds is not None
    assert response.provider_request_id == "resp-test"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5


async def test_response_translation_tool_calls(adapter: DeepSeekAdapter) -> None:
    request = ProviderRequest(
        request_id="req-6",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
    )
    response = await adapter.submit(request)
    assert response.has_tool_calls() is True
    assert response.tool_calls[0].tool_name == "read_file"
    assert response.tool_calls[0].raw_arguments == {"path": "/tmp"}


async def test_response_translation_streaming(adapter: DeepSeekAdapter) -> None:
    request = ProviderRequest(
        request_id="req-7",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
    )
    chunks = [chunk async for chunk in adapter.stream(request)]
    assert len(chunks) == 3
    assert chunks[0].text == "Hello"
    assert chunks[0].streaming_state is StreamingState.IN_PROGRESS
    assert chunks[-1].streaming_state is StreamingState.COMPLETE


async def test_error_normalization_rate_limited(monkeypatch: Any) -> None:
    client = build_mock_openai_client(exception=FakeRateLimitError)
    adapter = DeepSeekAdapter(api_key="test-key")
    monkeypatch.setattr(adapter, "_client", lambda: client)
    request = ProviderRequest(
        request_id="req-8",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
    )
    response = await adapter.submit(request)
    assert response.error_reference == ProviderErrorCode.RATE_LIMITED.value
    error = response.metadata["error"]
    assert error.code is ProviderErrorCode.RATE_LIMITED
    assert error.provider_id == adapter.identity


async def test_error_normalization_auth_failure(monkeypatch: Any) -> None:
    client = build_mock_openai_client(exception=FakeAuthError)
    adapter = DeepSeekAdapter(api_key="test-key")
    monkeypatch.setattr(adapter, "_client", lambda: client)
    request = ProviderRequest(
        request_id="req-9",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
    )
    response = await adapter.submit(request)
    assert response.error_reference == ProviderErrorCode.AUTH_FAILURE.value
    error = response.metadata["error"]
    assert error.code is ProviderErrorCode.AUTH_FAILURE


async def test_unsupported_streaming_rejected() -> None:
    adapter = DeepSeekAdapter(
        api_key="test-key",
        capabilities=ProviderAdapterCapabilities(
            streaming=False,
            tool_calls=True,
            structured_output=True,
            cancellation=True,
        ),
    )
    request = ProviderRequest(
        request_id="req-10",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        capability_requirements=[CapabilityRequirement(feature="streaming", required=True)],
    )
    can_serve, error = adapter.can_serve(request)
    assert can_serve is False
    assert error is not None
    assert error.code is ProviderErrorCode.UNSUPPORTED_CAPABILITY
