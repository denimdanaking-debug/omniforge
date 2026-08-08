"""Provider-specific unit tests for OpenAIAdapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.policy.risk import RiskLevel
from src.providers.errors import ProviderErrorCode
from src.providers.openai.adapter import OpenAIAdapter
from src.providers.request import (
    Message,
    MessageRole,
    ProviderRequest,
    ReasoningMode,
    StructuredOutputRequirement,
    ToolChoiceMode,
    ToolDefinition,
    ToolParameter,
)
from src.providers.response import FinishReason, StreamingState
from src.routing.model_identity import ModelLifecycle
from src.routing.roles import ExecutionRole


def _make_capture_client(response: dict[str, Any]) -> tuple[MagicMock, list[dict[str, Any]]]:
    """Return a mock client and a list that captures every create() call."""
    calls: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return response

    client = MagicMock()
    client.chat.completions.create = create
    return client, calls


def _make_streaming_client(chunks: list[Any]) -> MagicMock:
    async def create(**kwargs: Any) -> Any:
        async def gen() -> AsyncIterator[Any]:
            for chunk in chunks:
                yield chunk

        return gen()

    client = MagicMock()
    client.chat.completions.create = create
    return client


def _make_adapter_with_client(client: Any, **kwargs: Any) -> OpenAIAdapter:
    adapter = OpenAIAdapter(**kwargs)
    adapter._client = lambda: client  # type: ignore[method-assign]
    return adapter


@pytest.fixture
def base_request() -> ProviderRequest:
    return ProviderRequest(
        request_id="req-1",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
    )


@pytest.mark.asyncio
async def test_identity_and_capabilities() -> None:
    adapter = OpenAIAdapter(api_key="sk-test")
    assert adapter.identity.provider_id == "openai"
    assert adapter.identity.display_name == "OpenAI"
    assert adapter.identity.failure_domain == "openai.com"

    model = adapter.model_id
    assert model.model_id == "codex-mini-latest"
    assert model.family == "codex"
    assert model.lifecycle is ModelLifecycle.HIGH_RISK

    caps = adapter.model_capabilities
    assert caps.structured_output is True
    assert caps.tool_use is True
    assert caps.streaming is True
    assert caps.reasoning is True
    assert caps.code_generation is True


@pytest.mark.asyncio
async def test_configurable_model_id() -> None:
    adapter = OpenAIAdapter(model_id="codex-mini-2025-06-01", api_key="sk-test")
    assert adapter.model_id.model_id == "codex-mini-2025-06-01"
    assert adapter.model_id.family == "codex"


@pytest.mark.asyncio
async def test_payload_translation_messages(base_request: ProviderRequest) -> None:
    request = ProviderRequest(
        request_id="msg-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        system_instructions=["You are a coding assistant."],
        messages=[
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there"),
            Message(role=MessageRole.TOOL, content="result", tool_call_id="tc-1"),
        ],
    )
    response = {"id": "r", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    client, calls = _make_capture_client(response)
    adapter = _make_adapter_with_client(client, api_key="sk-test")

    await adapter.submit(request)

    params = calls[0]
    assert params["model"] == "codex-mini-latest"
    assert params["messages"] == [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "tool", "content": "result", "tool_call_id": "tc-1"},
    ]


@pytest.mark.asyncio
async def test_payload_translation_tools_and_tool_choice() -> None:
    request = ProviderRequest(
        request_id="tool-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Use tool")],
        tools=[
            ToolDefinition(
                name="read_file",
                description="Read a file",
                parameters=[ToolParameter(name="path", schema={"type": "string"}, required=True)],
            )
        ],
        tool_choice=ToolChoiceMode.REQUIRED,
    )
    response = {"id": "r", "choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
    client, calls = _make_capture_client(response)
    adapter = _make_adapter_with_client(client, api_key="sk-test")

    await adapter.submit(request)

    params = calls[0]
    assert params["tool_choice"] == "required"
    assert params["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_payload_translation_tool_choice_forbidden() -> None:
    request = ProviderRequest(
        request_id="forbidden-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="No tools")],
        tools=[ToolDefinition(name="read_file", description="Read a file")],
        tool_choice=ToolChoiceMode.FORBIDDEN,
    )
    response = {"id": "r", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    client, calls = _make_capture_client(response)
    adapter = _make_adapter_with_client(client, api_key="sk-test")

    await adapter.submit(request)

    assert calls[0]["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_payload_translation_structured_output() -> None:
    request = ProviderRequest(
        request_id="structured-req",
        execution_role=ExecutionRole.PLANNING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Plan")],
        structured_output=StructuredOutputRequirement(
            schema={"type": "object", "properties": {"steps": {"type": "array"}}},
            name="plan",
        ),
    )
    response = {"id": "r", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    client, calls = _make_capture_client(response)
    adapter = _make_adapter_with_client(client, api_key="sk-test")

    await adapter.submit(request)

    params = calls[0]
    assert params["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "plan",
            "schema": {"type": "object", "properties": {"steps": {"type": "array"}}},
            "strict": True,
        },
    }


@pytest.mark.asyncio
async def test_payload_translation_parameters() -> None:
    request = ProviderRequest(
        request_id="param-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
        max_output_tokens=100,
        temperature=0.5,
        stop_sequences=["STOP", "END"],
    )
    response = {"id": "r", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    client, calls = _make_capture_client(response)
    adapter = _make_adapter_with_client(client, api_key="sk-test")

    await adapter.submit(request)

    params = calls[0]
    assert params["max_tokens"] == 100
    assert params["temperature"] == 0.5
    assert params["stop"] == ["STOP", "END"]


@pytest.mark.asyncio
async def test_payload_translation_reasoning_effort() -> None:
    request = ProviderRequest(
        request_id="reasoning-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Think")],
        reasoning=ReasoningMode.EFFORT_HIGH,
    )
    response = {"id": "r", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    client, calls = _make_capture_client(response)
    adapter = _make_adapter_with_client(client, api_key="sk-test")

    await adapter.submit(request)

    assert calls[0]["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_response_translation_text() -> None:
    request = ProviderRequest(
        request_id="resp-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )
    response = {
        "id": "resp-123",
        "choices": [
            {
                "message": {"content": "Hello back"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10,
        },
    }
    client, _ = _make_capture_client(response)
    adapter = _make_adapter_with_client(client, api_key="sk-test")

    result = await adapter.submit(request)

    assert result.text == "Hello back"
    assert result.finish_reason is FinishReason.STOP
    assert result.provider_request_id == "resp-123"
    assert result.usage.input_tokens == 8
    assert result.usage.output_tokens == 2
    assert result.usage.total_tokens == 10
    assert result.latency_seconds is not None
    assert result.latency_seconds >= 0.0


@pytest.mark.asyncio
async def test_response_translation_tool_calls() -> None:
    request = ProviderRequest(
        request_id="tool-resp-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Use tool")],
    )
    response = {
        "id": "resp-tool",
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc-1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "/tmp/file.txt"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    client, _ = _make_capture_client(response)
    adapter = _make_adapter_with_client(client, api_key="sk-test")

    result = await adapter.submit(request)

    assert result.text is None
    assert result.finish_reason is FinishReason.TOOL_CALLS
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "tc-1"
    assert call.tool_name == "read_file"
    assert call.raw_arguments == {"path": "/tmp/file.txt"}
    assert any(arg.name == "path" and arg.value == "/tmp/file.txt" for arg in call.arguments)


@pytest.mark.asyncio
async def test_response_translation_usage_cached_and_reasoning() -> None:
    request = ProviderRequest(
        request_id="usage-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )
    response = {
        "id": "resp-usage",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 50},
            "reasoning_tokens": 10,
        },
    }
    client, _ = _make_capture_client(response)
    adapter = _make_adapter_with_client(client, api_key="sk-test")

    result = await adapter.submit(request)

    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 20
    assert result.usage.total_tokens == 120
    assert result.usage.cached_tokens == 50
    assert result.usage.reasoning_tokens == 10


@pytest.mark.asyncio
async def test_response_translation_structured_output() -> None:
    request = ProviderRequest(
        request_id="structured-resp-req",
        execution_role=ExecutionRole.PLANNING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Plan")],
        structured_output=StructuredOutputRequirement(
            schema={"type": "object", "properties": {"steps": {"type": "array"}}},
            name="plan",
        ),
    )
    response = {
        "id": "resp-structured",
        "choices": [
            {
                "message": {"content": '{"steps": ["a", "b"]}'},
                "finish_reason": "stop",
            }
        ],
    }
    client, _ = _make_capture_client(response)
    adapter = _make_adapter_with_client(client, api_key="sk-test")

    result = await adapter.submit(request)

    assert result.text is None
    assert result.structured_result == {"steps": ["a", "b"]}


@pytest.mark.asyncio
async def test_error_normalization_rate_limit() -> None:
    request = ProviderRequest(
        request_id="err-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )
    client = MagicMock()
    client.chat.completions.create = MagicMock(side_effect=Exception("rate limit exceeded"))
    adapter = _make_adapter_with_client(client, api_key="sk-test")

    result = await adapter.submit(request)

    assert result.error_reference == ProviderErrorCode.RATE_LIMITED.value
    error = result.metadata["error"]
    assert error.code is ProviderErrorCode.RATE_LIMITED
    assert error.retryable is True


@pytest.mark.asyncio
async def test_error_normalization_auth() -> None:
    request = ProviderRequest(
        request_id="auth-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )
    client = MagicMock()
    client.chat.completions.create = MagicMock(
        side_effect=Exception("Authentication failed: invalid api key")
    )
    adapter = _make_adapter_with_client(client, api_key="sk-test")

    result = await adapter.submit(request)

    error = result.metadata["error"]
    assert error.code is ProviderErrorCode.AUTH_FAILURE
    assert error.retryable is False


@pytest.mark.asyncio
async def test_error_normalization_network() -> None:
    request = ProviderRequest(
        request_id="net-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )
    client = MagicMock()
    client.chat.completions.create = MagicMock(side_effect=TimeoutError("connection timed out"))
    adapter = _make_adapter_with_client(client, api_key="sk-test")

    result = await adapter.submit(request)

    error = result.metadata["error"]
    assert error.code is ProviderErrorCode.TRANSIENT_TRANSPORT
    assert error.retryable is True


@pytest.mark.asyncio
async def test_credential_redaction() -> None:
    request = ProviderRequest(
        request_id="secret-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )
    secret_key = "sk-thisissecretandshouldnotleak"
    client = MagicMock()
    client.chat.completions.create = MagicMock(
        side_effect=Exception(f"Auth failed for key {secret_key}")
    )
    adapter = _make_adapter_with_client(client, api_key=secret_key)

    result = await adapter.submit(request)

    error = result.metadata["error"]
    assert "sk-thisissecretandshouldnotleak" not in error.message
    assert "sk-***" in error.message
    response_text = str(result.metadata).lower()
    assert "api_key" not in response_text
    assert "secret" not in response_text
    assert "token" not in response_text


@pytest.mark.asyncio
async def test_streaming() -> None:
    request = ProviderRequest(
        request_id="stream-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Count")],
    )
    chunks = [
        SimpleNamespace(
            id="s1",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="One "), finish_reason=None)],
        ),
        SimpleNamespace(
            id="s2",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="two"), finish_reason="stop")],
        ),
    ]
    client = _make_streaming_client(chunks)
    adapter = _make_adapter_with_client(client, api_key="sk-test")

    result_chunks = [chunk async for chunk in adapter.stream(request)]

    assert len(result_chunks) == 3
    assert result_chunks[0].text == "One "
    assert result_chunks[0].streaming_state is StreamingState.IN_PROGRESS
    assert result_chunks[1].text == "two"
    assert result_chunks[2].streaming_state is StreamingState.COMPLETE
    assert result_chunks[2].finish_reason is FinishReason.STOP
