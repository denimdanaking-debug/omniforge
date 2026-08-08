"""Anthropic-specific unit tests for the AnthropicAdapter."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from src.policy.risk import RiskLevel
from src.providers.anthropic import AnthropicAdapter
from src.providers.errors import ProviderErrorCode
from src.providers.identity import ProviderHealth
from src.providers.request import (
    Message,
    MessageRole,
    ProviderRequest,
    StructuredOutputRequirement,
    ToolChoiceMode,
    ToolDefinition,
    ToolParameter,
)
from src.providers.response import FinishReason, StreamingState
from src.routing.model_identity import ModelIdentity, ModelLifecycle
from src.routing.roles import ExecutionRole


class FakeTextBlock:
    type = "text"
    text = "Hello from Claude"


class FakeToolUseBlock:
    type = "tool_use"
    id = "tu_01Test"
    name = "read_file"
    input = {"path": "README.md"}


class FakeUsage:
    input_tokens = 12
    output_tokens = 6


class FakeMessage:
    id = "msg_01Test"
    content: list[Any] = [FakeTextBlock()]
    stop_reason = "end_turn"
    usage = FakeUsage()


class FakeToolMessage(FakeMessage):
    content = [FakeToolUseBlock()]
    stop_reason = "tool_use"


class FakeJsonBlock:
    type = "text"
    text = '{"steps": []}'


class FakeStructuredMessage(FakeMessage):
    content = [FakeJsonBlock()]


@dataclass
class FakeTextDelta:
    type = "text_delta"
    text: str


@dataclass
class FakeContentBlockDeltaEvent:
    type = "content_block_delta"
    delta: Any


class FakeStopDelta:
    stop_reason = "end_turn"


@dataclass
class FakeMessageDeltaEvent:
    type = "message_delta"
    delta: Any = FakeStopDelta()
    usage: Any = FakeUsage()


@dataclass
class FakeMessageStopEvent:
    type = "message_stop"


@dataclass
class FakeMessageStartEvent:
    type = "message_start"
    message: Any = FakeMessage()


async def _fake_stream() -> AsyncIterator[Any]:
    yield FakeMessageStartEvent()
    yield FakeContentBlockDeltaEvent(delta=FakeTextDelta(text="Hello"))
    yield FakeContentBlockDeltaEvent(delta=FakeTextDelta(text=" world"))
    yield FakeMessageDeltaEvent()
    yield FakeMessageStopEvent()


class CapturingFakeMessages:
    def __init__(self, response: Any | None = None) -> None:
        self._response = response if response is not None else FakeMessage()
        self.last_call: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.last_call = kwargs
        return self._response


class CapturingFakeClient:
    def __init__(self, response: Any | None = None) -> None:
        self.messages = CapturingFakeMessages(response)


class RaisingFakeMessages:
    def __init__(self, exception: BaseException) -> None:
        self._exception = exception

    async def create(self, **kwargs: Any) -> Any:
        raise self._exception


class RaisingFakeClient:
    def __init__(self, exception: BaseException) -> None:
        self.messages = RaisingFakeMessages(exception)


class StreamingFakeMessages:
    async def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return _fake_stream()
        return FakeMessage()


class StreamingFakeClient:
    def __init__(self) -> None:
        self.messages = StreamingFakeMessages()


def _make_request(**overrides: Any) -> ProviderRequest:
    defaults: dict[str, Any] = {
        "request_id": "req-1",
        "execution_role": ExecutionRole.CODING,
        "risk_level": RiskLevel.R2_NORMAL,
        "messages": [Message(role=MessageRole.USER, content="Hello")],
    }
    defaults.update(overrides)
    return ProviderRequest(**defaults)


def _make_adapter(monkeypatch: pytest.MonkeyPatch, fake_client: Any) -> AnthropicAdapter:
    monkeypatch.setattr(AnthropicAdapter, "_client", lambda self: fake_client)
    return AnthropicAdapter()


class FakeAnthropicErrorModule:
    """Minimal stand-in for the ``anthropic`` SDK exception namespace."""

    class APIStatusError(Exception):
        def __init__(self, message: str, *, response: Any | None = None, body: Any | None = None):
            super().__init__(message)
            self.response = response
            self.status_code = response.status_code if response is not None else None

    class AuthenticationError(APIStatusError):
        pass

    class RateLimitError(APIStatusError):
        pass

    class BadRequestError(APIStatusError):
        pass

    class InternalServerError(APIStatusError):
        pass

    class APIConnectionError(Exception):
        def __init__(self, message: str, *, request: Any | None = None):
            super().__init__(message)
            self.request = request


@pytest.fixture(autouse=True)
def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", FakeAnthropicErrorModule())


@pytest.mark.asyncio
async def test_payload_translation_messages_system_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    client = CapturingFakeClient()
    adapter = _make_adapter(monkeypatch, client)
    request = _make_request(
        system_instructions=["Be helpful", "Be concise"],
        messages=[
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi"),
            Message(role=MessageRole.TOOL, content="result", tool_call_id="tu_1"),
        ],
        tools=[
            ToolDefinition(
                name="read_file",
                description="Read a file",
                parameters=[
                    ToolParameter(
                        name="path",
                        schema={"type": "string"},
                        required=True,
                    )
                ],
            )
        ],
        tool_choice=ToolChoiceMode.REQUIRED,
        max_output_tokens=256,
        temperature=0.5,
        stop_sequences=["STOP"],
    )

    response = await adapter.submit(request)
    assert response.error_reference is None

    params = client.messages.last_call
    assert params is not None
    assert params["model"] == adapter._model_id.model_id
    assert params["max_tokens"] == 256
    assert params["temperature"] == 0.5
    assert params["stop_sequences"] == ["STOP"]
    assert params["system"] == "Be helpful\n\nBe concise"

    messages = params["messages"]
    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": "Hello"}
    assert messages[1] == {"role": "assistant", "content": "Hi"}
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": "result"}
    ]

    tools = params["tools"]
    assert tools == [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]
    assert params["tool_choice"] == {"type": "any"}


@pytest.mark.asyncio
async def test_payload_translation_tool_choice_none_and_forbidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for mode in (ToolChoiceMode.NONE, ToolChoiceMode.FORBIDDEN):
        client = CapturingFakeClient()
        adapter = _make_adapter(monkeypatch, client)
        request = _make_request(
            tools=[ToolDefinition(name="read_file", description="Read a file")],
            tool_choice=mode,
        )
        await adapter.submit(request)
        last_call = client.messages.last_call
        assert last_call is not None
        assert last_call["tool_choice"] == {"type": "none"}


@pytest.mark.asyncio
async def test_payload_translation_structured_output_json_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = CapturingFakeClient(FakeStructuredMessage())
    adapter = _make_adapter(monkeypatch, client)
    request = _make_request(
        structured_output=StructuredOutputRequirement(
            schema={"type": "object", "properties": {"steps": {"type": "array"}}},
            name="plan",
        ),
    )
    response = await adapter.submit(request)
    assert response.structured_result == {"steps": []}
    assert response.text is None


@pytest.mark.asyncio
async def test_response_translation_text_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    client = CapturingFakeClient()
    adapter = _make_adapter(monkeypatch, client)
    request = _make_request()
    response = await adapter.submit(request)
    assert response.request_id == request.request_id
    assert response.provider_id == adapter.identity
    assert response.model_id == adapter._model_id
    assert response.text == "Hello from Claude"
    assert response.finish_reason is FinishReason.STOP
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 6
    assert response.provider_request_id == "msg_01Test"
    assert response.latency_seconds is not None


@pytest.mark.asyncio
async def test_response_translation_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    client = CapturingFakeClient(FakeToolMessage())
    adapter = _make_adapter(monkeypatch, client)
    request = _make_request(tools=[ToolDefinition(name="read_file", description="Read")])
    response = await adapter.submit(request)
    assert response.has_tool_calls()
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call.id == "tu_01Test"
    assert tool_call.tool_name == "read_file"
    assert tool_call.raw_arguments == {"path": "README.md"}
    assert any(arg.name == "path" and arg.value == "README.md" for arg in tool_call.arguments)
    assert response.finish_reason is FinishReason.TOOL_CALLS


@pytest.mark.asyncio
async def test_streaming_response(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch, StreamingFakeClient())
    request = _make_request()
    chunks = []
    async for chunk in adapter.stream(request):
        chunks.append(chunk)
    assert len(chunks) > 0
    assert chunks[-1].streaming_state is StreamingState.COMPLETE
    assert chunks[-1].finish_reason is FinishReason.STOP
    assert chunks[-1].provider_request_id == "msg_01Test"
    assert "".join(chunk.text or "" for chunk in chunks[:-1]) == "Hello world"


@pytest.mark.asyncio
async def test_error_normalization_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    request_obj = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request_obj, text="Unauthorized")
    exc = FakeAnthropicErrorModule.AuthenticationError(
        "Invalid API key", response=response, body=None
    )
    adapter = _make_adapter(monkeypatch, RaisingFakeClient(exc))
    response_obj = await adapter.submit(_make_request())
    assert response_obj.error_reference == ProviderErrorCode.AUTH_FAILURE.value


@pytest.mark.asyncio
async def test_error_normalization_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    request_obj = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(
        429,
        request=request_obj,
        headers={"retry-after": "42"},
        text='{"error":{"type":"rate_limit_error"}}',
    )
    exc = FakeAnthropicErrorModule.RateLimitError(
        "Rate limited", response=response, body={"error": {"type": "rate_limit_error"}}
    )
    adapter = _make_adapter(monkeypatch, RaisingFakeClient(exc))
    response_obj = await adapter.submit(_make_request())
    assert response_obj.error_reference == ProviderErrorCode.RATE_LIMITED.value
    error = response_obj.metadata["error"]
    assert error.retry_after_seconds == 42


@pytest.mark.asyncio
async def test_error_normalization_api_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    request_obj = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    exc = FakeAnthropicErrorModule.APIConnectionError("Connection reset", request=request_obj)
    adapter = _make_adapter(monkeypatch, RaisingFakeClient(exc))
    response_obj = await adapter.submit(_make_request())
    assert response_obj.error_reference == ProviderErrorCode.TRANSIENT_TRANSPORT.value


@pytest.mark.asyncio
async def test_error_normalization_context_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    request_obj = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(
        400,
        request=request_obj,
        text='{"error":{"message":"prompt is too long"}}',
    )
    exc = FakeAnthropicErrorModule.BadRequestError(
        "prompt is too long", response=response, body={"error": {"message": "prompt is too long"}}
    )
    adapter = _make_adapter(monkeypatch, RaisingFakeClient(exc))
    response_obj = await adapter.submit(_make_request())
    assert response_obj.error_reference == ProviderErrorCode.CONTEXT_OVERFLOW.value


@pytest.mark.asyncio
async def test_error_normalization_api_status_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    request_obj = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(500, request=request_obj, text="Internal server error")
    exc = FakeAnthropicErrorModule.InternalServerError(
        "Internal error", response=response, body=None
    )
    adapter = _make_adapter(monkeypatch, RaisingFakeClient(exc))
    response_obj = await adapter.submit(_make_request())
    assert response_obj.error_reference == ProviderErrorCode.PROVIDER_UNAVAILABLE.value


@pytest.mark.asyncio
async def test_error_normalization_translate_exception_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch, RaisingFakeClient(RuntimeError("something failed")))
    response_obj = await adapter.submit(_make_request())
    assert response_obj.error_reference == ProviderErrorCode.UNKNOWN.value


@pytest.mark.asyncio
async def test_model_identity_and_capabilities() -> None:
    adapter = AnthropicAdapter()
    assert adapter.identity.provider_id == "anthropic"
    assert adapter.identity.display_name == "Anthropic"
    assert adapter._model_id.family == "claude"
    assert adapter._model_id.lifecycle == ModelLifecycle.HIGH_RISK
    assert adapter.capabilities.streaming
    assert adapter.capabilities.tool_calls
    assert adapter.capabilities.structured_output
    assert adapter.capabilities.cancellation


@pytest.mark.asyncio
async def test_configurable_model_id() -> None:
    custom_model = ModelIdentity(model_id="claude-opus-4", family="claude")
    adapter = AnthropicAdapter(model_id=custom_model)
    assert adapter._model_id == custom_model


@pytest.mark.asyncio
async def test_credential_redaction_in_error(monkeypatch: pytest.MonkeyPatch) -> None:
    request_obj = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request_obj, text="Unauthorized")
    exc = FakeAnthropicErrorModule.AuthenticationError(
        "Invalid API key sk-secret1234567890", response=response, body=None
    )
    adapter = _make_adapter(monkeypatch, RaisingFakeClient(exc))
    response_obj = await adapter.submit(_make_request())
    error = response_obj.metadata["error"]
    assert "sk-secret1234567890" not in error.message
    assert "sk-***" in error.message


@pytest.mark.asyncio
async def test_credential_redaction_api_key_not_in_response_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = CapturingFakeClient()
    adapter = AnthropicAdapter(api_key="sk-live-1234567890abcdef")
    monkeypatch.setattr(AnthropicAdapter, "_client", lambda self: client)
    response_obj = await adapter.submit(_make_request())
    metadata_text = str(response_obj.metadata).lower()
    assert "sk-live-1234567890abcdef" not in metadata_text
    assert "api_key" not in metadata_text


@pytest.mark.asyncio
async def test_cancellation_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch, CapturingFakeClient())
    request = _make_request(cancellation_id="cancel-1")
    await adapter.cancel("cancel-1")
    response_obj = await adapter.submit(request)
    assert response_obj.error_reference == ProviderErrorCode.CANCELLED.value


@pytest.mark.asyncio
async def test_cancellation_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch, StreamingFakeClient())
    request = _make_request(request_id="stream-req")
    await adapter.cancel("stream-req")
    chunks = [chunk async for chunk in adapter.stream(request)]
    assert len(chunks) == 1
    assert chunks[0].error_reference == ProviderErrorCode.CANCELLED.value


@pytest.mark.asyncio
async def test_health_and_quota() -> None:
    adapter = AnthropicAdapter()
    health = await adapter.health()
    # Phase 3: locally configured adapters do not fabricate external health.
    assert health.health is ProviderHealth.DEGRADED
    assert health.reason is not None
    assert "no external health observation" in health.reason
    quota = await adapter.quota()
    assert quota is not None
    assert quota.provider_signal.value == "unknown"


@pytest.mark.asyncio
async def test_target_model_identity_preserved_in_response(monkeypatch: pytest.MonkeyPatch) -> None:
    default_adapter = AnthropicAdapter()
    default_model = default_adapter._model_id
    target_model = ModelIdentity(model_id="claude-opus-4", family="claude")
    client = CapturingFakeClient()
    test_adapter = AnthropicAdapter(model_id=target_model)
    monkeypatch.setattr(AnthropicAdapter, "_client", lambda self: client)
    request = _make_request(
        request_id="target-req",
        target_model=target_model,
    )
    response = await test_adapter.submit(request)

    assert client.messages.last_call is not None
    assert client.messages.last_call["model"] == target_model.model_id
    assert response.model_id == target_model
    assert response.model_id != default_model


@pytest.mark.asyncio
async def test_target_model_identity_preserved_in_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    target_model = ModelIdentity(model_id="claude-opus-4", family="claude")
    test_adapter = AnthropicAdapter(model_id=target_model)
    monkeypatch.setattr(AnthropicAdapter, "_client", lambda self: StreamingFakeClient())
    request = _make_request(
        request_id="target-stream-req",
        target_model=target_model,
    )
    chunks = [chunk async for chunk in test_adapter.stream(request)]

    assert all(chunk.model_id == target_model for chunk in chunks)
    assert chunks[-1].streaming_state is StreamingState.COMPLETE


@pytest.mark.asyncio
async def test_target_model_identity_preserved_in_error_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_model = ModelIdentity(model_id="claude-opus-4", family="claude")
    test_adapter = AnthropicAdapter(model_id=target_model)
    monkeypatch.setattr(
        AnthropicAdapter,
        "_client",
        lambda self: RaisingFakeClient(RuntimeError("boom")),
    )
    request = _make_request(
        request_id="target-err-req",
        target_model=target_model,
    )
    response = await test_adapter.submit(request)

    assert response.error_reference is not None
    assert response.model_id == target_model


@pytest.mark.asyncio
async def test_target_model_identity_preserved_in_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_model = ModelIdentity(model_id="claude-opus-4", family="claude")
    test_adapter = AnthropicAdapter(model_id=target_model)
    monkeypatch.setattr(AnthropicAdapter, "_client", lambda self: CapturingFakeClient())
    request = _make_request(
        request_id="target-cancel-req",
        cancellation_id="cancel-1",
        target_model=target_model,
    )
    await test_adapter.cancel("cancel-1")
    response = await test_adapter.submit(request)

    assert response.error_reference == ProviderErrorCode.CANCELLED.value
    assert response.model_id == target_model


@pytest.mark.asyncio
async def test_supported_roles_are_canonical() -> None:
    from src.routing.capabilities import CapabilityRequirement, match_capabilities

    adapter = AnthropicAdapter()
    caps = adapter.model_capabilities
    assert ExecutionRole.PLANNING.value in caps.supported_roles
    assert ExecutionRole.HIGH_RISK_REVIEW.value in caps.supported_roles
    assert ExecutionRole.INTEGRATION_ANALYSIS.value in caps.supported_roles

    requirement = CapabilityRequirement(
        min_context_tokens=1,
        required_roles=frozenset(
            {
                ExecutionRole.PLANNING.value,
                ExecutionRole.ARCHITECTURE.value,
                ExecutionRole.CODING.value,
                ExecutionRole.DEBUGGING.value,
                ExecutionRole.REPAIR.value,
                ExecutionRole.REVIEW.value,
                ExecutionRole.HIGH_RISK_REVIEW.value,
                ExecutionRole.ARBITRATION.value,
                ExecutionRole.CONTEXT_ANALYSIS.value,
                ExecutionRole.INTEGRATION_ANALYSIS.value,
            }
        ),
    )
    match = match_capabilities(caps, requirement)
    assert match.eligible is True
    assert match.missing == ()
