"""Reusable provider-adapter compliance suite (Step 2.7)."""

from __future__ import annotations

from typing import Any

import pytest

from src.policy.risk import RiskLevel
from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.errors import ErrorCategory, ProviderError, ProviderErrorCode
from src.providers.identity import (
    ProviderHealth,
    ProviderIdentity,
    ProviderOperationalState,
    ProviderQuotaState,
    QuotaSignal,
)
from src.providers.request import (
    CapabilityRequirement,
    Message,
    MessageRole,
    ProviderRequest,
    StructuredOutputRequirement,
    ToolChoiceMode,
    ToolDefinition,
)
from src.providers.response import FinishReason, StreamingState
from src.providers.stub_adapter import StubAdapterConfig, StubProviderAdapter
from src.routing.roles import ExecutionRole


@pytest.fixture
def healthy_adapter() -> StubProviderAdapter:
    return StubProviderAdapter()


@pytest.fixture
def sample_request() -> ProviderRequest:
    return ProviderRequest(
        request_id="contract-req-1",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        system_instructions=["You are a helpful coding assistant."],
        messages=[Message(role=MessageRole.USER, content="Write a function")],
    )


@pytest.mark.contract
async def test_stable_provider_identity(healthy_adapter: StubProviderAdapter) -> None:
    identity = healthy_adapter.identity
    assert isinstance(identity, ProviderIdentity)
    assert identity.provider_id == "stub"


@pytest.mark.contract
async def test_declared_capabilities(healthy_adapter: StubProviderAdapter) -> None:
    caps = healthy_adapter.capabilities
    assert caps.streaming is True
    assert caps.tool_calls is True
    assert caps.structured_output is True
    assert caps.cancellation is True


@pytest.mark.contract
async def test_request_translation_boundary(
    healthy_adapter: StubProviderAdapter, sample_request: ProviderRequest
) -> None:
    response = await healthy_adapter.submit(sample_request)
    assert response.request_id == sample_request.request_id
    assert response.provider_id == healthy_adapter.identity
    assert response.model_id == healthy_adapter.config.model


@pytest.mark.contract
async def test_normalized_response(
    healthy_adapter: StubProviderAdapter, sample_request: ProviderRequest
) -> None:
    response = await healthy_adapter.submit(sample_request)
    assert response.request_id == sample_request.request_id
    assert response.text is not None
    assert response.finish_reason is FinishReason.STOP
    assert response.usage.input_tokens is not None
    assert response.usage.output_tokens is not None
    assert response.latency_seconds is not None
    assert response.provider_request_id is not None


@pytest.mark.contract
async def test_streaming_semantics(
    healthy_adapter: StubProviderAdapter, sample_request: ProviderRequest
) -> None:
    chunks: list[Any] = []
    async for chunk in healthy_adapter.stream(sample_request):
        chunks.append(chunk)
    assert len(chunks) > 0
    assert chunks[-1].streaming_state is StreamingState.COMPLETE


@pytest.mark.contract
async def test_structured_output(
    healthy_adapter: StubProviderAdapter, sample_request: ProviderRequest
) -> None:
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
    response = await healthy_adapter.submit(request)
    assert response.structured_result is not None
    assert "request_id" in response.structured_result


@pytest.mark.contract
async def test_tool_calls(
    healthy_adapter: StubProviderAdapter, sample_request: ProviderRequest
) -> None:
    request = ProviderRequest(
        request_id="tool-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Use a tool")],
        tools=[ToolDefinition(name="read_file", description="Read a file")],
        tool_choice=ToolChoiceMode.AUTO,
    )
    response = await healthy_adapter.submit(request)
    assert response.has_tool_calls() is True
    assert response.tool_calls[0].tool_name == "read_file"


@pytest.mark.contract
async def test_cancellation(healthy_adapter: StubProviderAdapter) -> None:
    request = ProviderRequest(
        request_id="cancel-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        cancellation_id="cancel-1",
    )
    await healthy_adapter.cancel("cancel-1")
    response = await healthy_adapter.submit(request)
    assert response.error_reference == ProviderErrorCode.CANCELLED.value


@pytest.mark.contract
async def test_health(healthy_adapter: StubProviderAdapter) -> None:
    health = await healthy_adapter.health()
    assert isinstance(health, ProviderOperationalState)
    assert health.health is ProviderHealth.HEALTHY
    assert health.reason == "stub-configured"


@pytest.mark.contract
async def test_quota(healthy_adapter: StubProviderAdapter) -> None:
    quota = await healthy_adapter.quota()
    assert isinstance(quota, ProviderQuotaState)
    assert quota.provider_signal is QuotaSignal.AVAILABLE


@pytest.mark.contract
async def test_normalized_errors(sample_request: ProviderRequest) -> None:
    error = ProviderError(
        code=ProviderErrorCode.RATE_LIMITED,
        message="Too many requests",
        retryable=True,
        provider_id=ProviderIdentity("stub", "Stub", "stub.example"),
    )
    adapter = StubProviderAdapter(config=StubAdapterConfig(fail_with_error=error))
    response = await adapter.submit(sample_request)
    assert response.error_reference == ProviderErrorCode.RATE_LIMITED.value


@pytest.mark.contract
async def test_unsupported_capability_behavior(sample_request: ProviderRequest) -> None:
    caps = ProviderAdapterCapabilities(
        streaming=False,
        tool_calls=False,
        structured_output=False,
        cancellation=True,
    )
    adapter = StubProviderAdapter(config=StubAdapterConfig(capabilities=caps))
    request = ProviderRequest(
        request_id="unsupported-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        capability_requirements=[CapabilityRequirement("streaming", required=True)],
    )
    can_serve, err = adapter.can_serve(request)
    assert can_serve is False
    assert err is not None
    assert err.code is ProviderErrorCode.UNSUPPORTED_CAPABILITY
    assert err.category is ErrorCategory.CAPABILITY


@pytest.mark.contract
async def test_clean_lifecycle(healthy_adapter: StubProviderAdapter) -> None:
    health = await healthy_adapter.health()
    quota = await healthy_adapter.quota()
    assert health.is_available() is True
    assert quota.is_exhausted() is False


@pytest.mark.contract
async def test_no_credential_leakage(
    healthy_adapter: StubProviderAdapter, sample_request: ProviderRequest
) -> None:
    response = await healthy_adapter.submit(sample_request)
    assert "api_key" not in str(response.metadata).lower()
    assert "secret" not in str(response.metadata).lower()


def test_all_adapters_expose_identity() -> None:
    adapter: ProviderAdapter = StubProviderAdapter()
    assert adapter.identity.provider_id is not None
