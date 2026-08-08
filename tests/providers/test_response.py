"""Unit tests for the normalized response contract (Step 2.3)."""

from __future__ import annotations

import pytest

from src.providers.identity import ProviderIdentity
from src.providers.response import (
    FinishReason,
    ProviderResponse,
    StreamingState,
    ToolCall,
    ToolCallArgument,
    Usage,
)
from src.routing.model_identity import ModelIdentity


@pytest.fixture
def provider() -> ProviderIdentity:
    return ProviderIdentity("stub", "Stub Provider", "stub.example")


@pytest.fixture
def model() -> ModelIdentity:
    return ModelIdentity(model_id="stub-model", family="stub", version="1.0", revision="rev-a")


def test_response_requires_request_id(provider: ProviderIdentity, model: ModelIdentity) -> None:
    with pytest.raises(ValueError):
        ProviderResponse(request_id="", provider_id=provider, model_id=model)


def test_complete_response_fields(provider: ProviderIdentity, model: ModelIdentity) -> None:
    response = ProviderResponse(
        request_id="req-1",
        provider_id=provider,
        model_id=model,
        text="Hello",
        structured_result={"answer": 42},
        tool_calls=[
            ToolCall(id="tc1", tool_name="read", arguments=[ToolCallArgument("path", "/x")])
        ],
        finish_reason=FinishReason.STOP,
        streaming_state=StreamingState.NOT_STREAMING,
        usage=Usage(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=10,
            reasoning_tokens=5,
            total_tokens=155,
        ),
        latency_seconds=0.123,
        provider_request_id="pr-1",
        warnings=["minor"],
        metadata={"key": "value"},
    )

    assert response.request_id == "req-1"
    assert response.provider_id is provider
    assert response.model_id is model
    assert response.text == "Hello"
    assert response.structured_result == {"answer": 42}
    assert response.has_tool_calls() is True
    assert response.finish_reason is FinishReason.STOP
    assert response.streaming_state is StreamingState.NOT_STREAMING
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 50
    assert response.usage.cached_tokens == 10
    assert response.usage.reasoning_tokens == 5
    assert response.usage.total_tokens == 155
    assert response.latency_seconds == 0.123
    assert response.provider_request_id == "pr-1"
    assert response.warnings == ["minor"]


def test_partial_response_does_not_fabricate_zeros(
    provider: ProviderIdentity, model: ModelIdentity
) -> None:
    response = ProviderResponse(
        request_id="req-2",
        provider_id=provider,
        model_id=model,
        text="Partial",
    )
    assert response.usage.input_tokens is None
    assert response.usage.output_tokens is None
    assert response.usage.cached_tokens is None
    assert response.usage.reasoning_tokens is None
    assert response.usage.total_tokens is None
    assert response.usage.has_any_known() is False
    assert response.latency_seconds is None
    assert response.provider_request_id is None
