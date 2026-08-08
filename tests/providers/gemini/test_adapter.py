"""Provider-specific tests for the Google Gemini adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from google.genai import types
from google.genai.errors import APIError

from src.policy.risk import RiskLevel
from src.providers.errors import ProviderErrorCode
from src.providers.gemini.adapter import GeminiAdapter
from src.providers.identity import ProviderIdentity
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
from src.routing.model_identity import ModelIdentity
from src.routing.roles import ExecutionRole


class _FakeModels:
    def __init__(self) -> None:
        self.generate_content_calls: list[dict[str, Any]] = []
        self.generate_content_stream_calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.generate_content_calls.append(kwargs)
        config = kwargs.get("config")
        if config and getattr(config, "tools", None):
            return types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part(
                                    function_call=types.FunctionCall(
                                        id="fc-1",
                                        name="read_file",
                                        args={"path": "/etc/hosts"},
                                    )
                                )
                            ],
                        ),
                        finish_reason=types.FinishReason.STOP,
                    )
                ],
                response_id="gemini-req-tool",
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=12,
                    candidates_token_count=7,
                    total_token_count=19,
                ),
            )
        if config and getattr(config, "response_mime_type", None) == "application/json":
            return types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(
                            role="model",
                            parts=[types.Part(text='{"valid": true}')],
                        ),
                        finish_reason=types.FinishReason.STOP,
                    )
                ],
                response_id="gemini-req-structured",
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=8,
                    candidates_token_count=6,
                    total_token_count=14,
                ),
            )
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="Hello from Gemini")],
                    ),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
            response_id="gemini-req-1",
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10,
                candidates_token_count=5,
                total_token_count=15,
            ),
        )

    async def generate_content_stream(self, **kwargs: Any) -> AsyncIterator[Any]:
        self.generate_content_stream_calls.append(kwargs)

        async def _gen() -> AsyncIterator[Any]:
            yield types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(role="model", parts=[types.Part(text="Hello")]),
                    )
                ],
                response_id="gemini-req-stream",
            )
            yield types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(role="model", parts=[types.Part(text=" Gemini")]),
                        finish_reason=types.FinishReason.STOP,
                    )
                ],
                response_id="gemini-req-stream",
            )

        return _gen()


class _FakeAio:
    def __init__(self) -> None:
        self.models = _FakeModels()


class _FakeClient:
    def __init__(self) -> None:
        self.aio = _FakeAio()


@pytest.fixture
def adapter() -> GeminiAdapter:
    gemini = GeminiAdapter(api_key="fake-api-key")
    fake_client = _FakeClient()
    gemini._client = lambda: fake_client  # type: ignore[method-assign]
    return gemini


@pytest.fixture
def sample_request() -> ProviderRequest:
    return ProviderRequest(
        request_id="test-req-1",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        system_instructions=["You are a helpful assistant."],
        messages=[Message(role=MessageRole.USER, content="Say hello")],
        temperature=0.5,
        max_output_tokens=100,
        stop_sequences=["DONE"],
    )


@pytest.mark.asyncio
async def test_identity(adapter: GeminiAdapter) -> None:
    identity = adapter.identity
    assert identity == ProviderIdentity(
        provider_id="gemini",
        display_name="Google Gemini",
        failure_domain="googleapis.com",
    )


@pytest.mark.asyncio
async def test_default_model_identity(adapter: GeminiAdapter) -> None:
    assert adapter.model_id == ModelIdentity(model_id="gemini-2.5-pro", family="gemini")


@pytest.mark.asyncio
async def test_configurable_model_id() -> None:
    gemini = GeminiAdapter(model_id="gemini-2.0-flash")
    assert gemini.model_id == ModelIdentity(model_id="gemini-2.0-flash", family="gemini")


@pytest.mark.asyncio
async def test_payload_translation(adapter: GeminiAdapter, sample_request: ProviderRequest) -> None:
    response = await adapter.submit(sample_request)
    assert response.request_id == sample_request.request_id
    assert response.provider_id == adapter.identity
    assert response.model_id == adapter.model_id
    assert response.text == "Hello from Gemini"
    assert response.finish_reason is FinishReason.STOP
    assert response.provider_request_id == "gemini-req-1"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 15
    assert response.latency_seconds is not None

    calls = adapter._client().aio.models.generate_content_calls
    assert len(calls) == 1
    call = calls[0]
    assert call["model"] == "gemini-2.5-pro"
    contents = call["contents"]
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Say hello"

    config = call["config"]
    assert config.system_instruction == "You are a helpful assistant."
    assert config.temperature == 0.5
    assert config.max_output_tokens == 100
    assert config.stop_sequences == ["DONE"]


@pytest.mark.asyncio
async def test_tool_payload_translation(adapter: GeminiAdapter) -> None:
    request = ProviderRequest(
        request_id="tool-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Use the tool")],
        tools=[
            ToolDefinition(
                name="read_file",
                description="Read a file",
                parameters=[
                    ToolParameter(
                        name="path",
                        schema={"type": "string"},
                        description="File path",
                    )
                ],
            )
        ],
        tool_choice=ToolChoiceMode.REQUIRED,
    )
    response = await adapter.submit(request)
    assert response.has_tool_calls() is True
    assert response.tool_calls[0].tool_name == "read_file"
    assert response.tool_calls[0].id == "fc-1"
    assert any(
        arg.name == "path" and arg.value == "/etc/hosts" for arg in response.tool_calls[0].arguments
    )

    calls = adapter._client().aio.models.generate_content_calls
    config = calls[-1]["config"]
    assert config.tools is not None
    assert len(config.tools[0].function_declarations) == 1
    decl = config.tools[0].function_declarations[0]
    assert decl.name == "read_file"
    assert decl.parameters_json_schema["properties"]["path"]["type"] == "string"
    assert config.tool_config.function_calling_config.mode.name == "ANY"


@pytest.mark.asyncio
async def test_structured_output_translation(adapter: GeminiAdapter) -> None:
    request = ProviderRequest(
        request_id="structured-req",
        execution_role=ExecutionRole.PLANNING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Return JSON")],
        structured_output=StructuredOutputRequirement(
            schema={"type": "object", "properties": {"valid": {"type": "boolean"}}},
            name="result",
        ),
    )
    response = await adapter.submit(request)
    assert response.structured_result == {"valid": True}
    assert response.text is None

    calls = adapter._client().aio.models.generate_content_calls
    config = calls[-1]["config"]
    assert config.response_mime_type == "application/json"
    assert request.structured_output is not None
    assert config.response_json_schema == request.structured_output.schema


@pytest.mark.asyncio
async def test_streaming_translation(adapter: GeminiAdapter) -> None:
    request = ProviderRequest(
        request_id="stream-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Stream")],
    )
    chunks = []
    async for chunk in adapter.stream(request):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0].text == "Hello"
    assert chunks[0].streaming_state is StreamingState.IN_PROGRESS
    assert chunks[1].text == " Gemini"
    assert chunks[2].text == ""
    assert chunks[2].streaming_state is StreamingState.COMPLETE
    assert chunks[2].finish_reason is FinishReason.STOP

    stream_calls = adapter._client().aio.models.generate_content_stream_calls
    assert len(stream_calls) == 1


@pytest.mark.asyncio
async def test_error_normalization_rate_limited(adapter: GeminiAdapter) -> None:
    class _FailingModels:
        async def generate_content(self, **kwargs: Any) -> Any:
            raise APIError(
                code=429,
                response_json={"error": {"message": "rate limit exceeded"}},
            )

        async def generate_content_stream(self, **kwargs: Any) -> AsyncIterator[Any]:
            raise APIError(
                code=429,
                response_json={"error": {"message": "rate limit exceeded"}},
            )

    failing_client = _FakeClient()
    failing_client.aio.models = _FailingModels()  # type: ignore[assignment]
    adapter._client = lambda: failing_client  # type: ignore[method-assign]

    request = ProviderRequest(
        request_id="error-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="fail")],
    )
    response = await adapter.submit(request)
    assert response.error_reference == ProviderErrorCode.RATE_LIMITED.value
    error = response.metadata["error"]
    assert error.code is ProviderErrorCode.RATE_LIMITED
    assert error.retryable is True


@pytest.mark.asyncio
async def test_error_normalization_auth_failure(adapter: GeminiAdapter) -> None:
    class _FailingModels:
        async def generate_content(self, **kwargs: Any) -> Any:
            raise APIError(
                code=401,
                response_json={"error": {"message": "invalid api key"}},
            )

        async def generate_content_stream(self, **kwargs: Any) -> AsyncIterator[Any]:
            raise APIError(
                code=401,
                response_json={"error": {"message": "invalid api key"}},
            )

    failing_client = _FakeClient()
    failing_client.aio.models = _FailingModels()  # type: ignore[assignment]
    adapter._client = lambda: failing_client  # type: ignore[method-assign]

    request = ProviderRequest(
        request_id="auth-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="fail")],
    )
    response = await adapter.submit(request)
    assert response.error_reference == ProviderErrorCode.AUTH_FAILURE.value
    error = response.metadata["error"]
    assert error.code is ProviderErrorCode.AUTH_FAILURE
    assert error.retryable is False


@pytest.mark.asyncio
async def test_error_normalization_server_error(adapter: GeminiAdapter) -> None:
    class _FailingModels:
        async def generate_content(self, **kwargs: Any) -> Any:
            raise APIError(
                code=503,
                response_json={"error": {"message": "service unavailable"}},
            )

        async def generate_content_stream(self, **kwargs: Any) -> AsyncIterator[Any]:
            raise APIError(
                code=503,
                response_json={"error": {"message": "service unavailable"}},
            )

    failing_client = _FakeClient()
    failing_client.aio.models = _FailingModels()  # type: ignore[assignment]
    adapter._client = lambda: failing_client  # type: ignore[method-assign]

    request = ProviderRequest(
        request_id="server-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="fail")],
    )
    response = await adapter.submit(request)
    assert response.error_reference == ProviderErrorCode.PROVIDER_UNAVAILABLE.value
    error = response.metadata["error"]
    assert error.code is ProviderErrorCode.PROVIDER_UNAVAILABLE
    assert error.retryable is True


@pytest.mark.asyncio
async def test_translate_error_redacts_credentials(adapter: GeminiAdapter) -> None:
    raw = APIError(
        code=400,
        response_json={"error": {"message": "Invalid API key: sk-1234567890abcdef"}},
    )
    error = adapter.translate_error(raw)
    assert "sk-1234567890abcdef" not in error.message
    assert "sk-***" in error.message


@pytest.mark.asyncio
async def test_no_credential_leakage_in_metadata(adapter: GeminiAdapter) -> None:
    request = ProviderRequest(
        request_id="leak-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="hello")],
    )
    response = await adapter.submit(request)
    metadata_text = str(response.metadata).lower()
    assert "api_key" not in metadata_text
    assert "secret" not in metadata_text
    assert "token" not in metadata_text


@pytest.mark.asyncio
async def test_cancel_tracks_request_id(adapter: GeminiAdapter) -> None:
    await adapter.cancel("cancel-req")
    request = ProviderRequest(
        request_id="cancel-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        cancellation_id="cancel-req",
    )
    response = await adapter.submit(request)
    assert response.error_reference == ProviderErrorCode.CANCELLED.value
