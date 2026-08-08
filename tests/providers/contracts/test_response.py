"""Unit tests for the normalized response contract (Step 2.3)."""

from __future__ import annotations

import pytest

from providers.contracts.identity import ModelId, ProviderId, RouteId
from providers.contracts.response import (
    FinishReason,
    ProviderResponse,
    StreamingState,
    ToolCall,
    ToolCallArgument,
    Usage,
)


def test_response_requires_request_id() -> None:
    with pytest.raises(ValueError):
        ProviderResponse(
            request_id="",
            provider_id=ProviderId("stub"),
            model_id=ModelId("model"),
        )


def test_complete_response_fields() -> None:
    response = ProviderResponse(
        request_id="req-1",
        provider_id=ProviderId("stub"),
        model_id=ModelId("family", "v1", "rev-a"),
        route_id=RouteId("direct"),
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
        model_version="v1",
        model_revision="rev-a",
        metadata={"key": "value"},
    )

    assert response.request_id == "req-1"
    assert response.provider_id == ProviderId("stub")
    assert response.model_id == ModelId("family", "v1", "rev-a")
    assert response.route_id == RouteId("direct")
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
    assert response.model_version == "v1"
    assert response.model_revision == "rev-a"


def test_partial_response_does_not_fabricate_zeros() -> None:
    response = ProviderResponse(
        request_id="req-2",
        provider_id=ProviderId("stub"),
        model_id=ModelId("model"),
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
    assert response.model_version is None


def test_tool_call_argument_value_objects() -> None:
    a1 = ToolCallArgument("path", "/x")
    a2 = ToolCallArgument("path", "/x")
    assert a1 == a2
    assert hash(a1) == hash(a2)
