"""Reusable provider-adapter compliance suite (Step 2.7).

Every concrete provider adapter must be capable of running against this same
suite. This implementation uses a deterministic stub adapter and fixtures; no
live external API credentials are required.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from providers.adapters.stub import StubAdapterConfig, StubProviderAdapter
from providers.contracts.adapter import ProviderAdapter
from providers.contracts.capabilities import (
    CapabilitySupport,
    ExecutionRole,
    ModelCapabilities,
    RiskLevel,
)
from providers.contracts.errors import (
    ErrorCategory,
    ProviderError,
    ProviderErrorCode,
)
from providers.contracts.health import HealthStatus, ProviderHealth
from providers.contracts.identity import ProviderId
from providers.contracts.quota import ProviderQuota, QuotaSignal
from providers.contracts.request import (
    CapabilityRequirement,
    Message,
    MessageRole,
    ProviderRequest,
    StructuredOutputRequirement,
    ToolChoiceMode,
    ToolDefinition,
)
from providers.contracts.response import FinishReason, StreamingState


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
    assert identity.provider_id == ProviderId("stub")
    assert len(identity.supported_models) > 0
    assert identity.capabilities.context_size > 0


@pytest.mark.contract
async def test_declared_capabilities(healthy_adapter: StubProviderAdapter) -> None:
    caps = healthy_adapter.identity.capabilities
    assert caps.supports_role(ExecutionRole.CODING) is True
    assert caps.supports_streaming in {
        CapabilitySupport.SUPPORTED,
        CapabilitySupport.PREFERRED,
        CapabilitySupport.REQUIRED,
    }


@pytest.mark.contract
async def test_request_translation_boundary(
    healthy_adapter: StubProviderAdapter, sample_request: ProviderRequest
) -> None:
    response = await healthy_adapter.submit(sample_request)
    assert response.request_id == sample_request.request_id
    assert response.provider_id == healthy_adapter.identity.provider_id
    assert response.model_id in healthy_adapter.identity.supported_models


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
    request = sample_request
    chunks: list[Any] = []
    async for chunk in healthy_adapter.stream(request):
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
    cancelled = await healthy_adapter.cancel("cancel-1")
    assert cancelled is True
    response = await healthy_adapter.submit(request)
    assert response.error_reference == ProviderErrorCode.CANCELLED.value


@pytest.mark.contract
async def test_health(healthy_adapter: StubProviderAdapter) -> None:
    health = await healthy_adapter.health()
    assert isinstance(health, ProviderHealth)
    assert health.status is HealthStatus.HEALTHY
    assert health.provider_id == healthy_adapter.identity.provider_id


@pytest.mark.contract
async def test_quota(healthy_adapter: StubProviderAdapter) -> None:
    quota = await healthy_adapter.quota()
    assert isinstance(quota, ProviderQuota)
    assert quota.provider_id == healthy_adapter.identity.provider_id
    assert quota.provider_signal is QuotaSignal.AVAILABLE


@pytest.mark.contract
async def test_normalized_errors(
    sample_request: ProviderRequest,
) -> None:
    error = ProviderError(
        code=ProviderErrorCode.RATE_LIMITED,
        message="Too many requests",
        retryable=True,
        provider_id=ProviderId("stub"),
    )
    adapter = StubProviderAdapter(config=StubAdapterConfig(fail_with_error=error))
    response = await adapter.submit(sample_request)
    assert response.error_reference == ProviderErrorCode.RATE_LIMITED.value


@pytest.mark.contract
async def test_unsupported_capability_behavior(
    sample_request: ProviderRequest,
) -> None:
    caps = ModelCapabilities(
        context_size=128_000,
        supports_streaming=CapabilitySupport.UNSUPPORTED,
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
    assert health.provider_id == quota.provider_id


@pytest.mark.contract
async def test_no_credential_leakage(
    healthy_adapter: StubProviderAdapter, sample_request: ProviderRequest
) -> None:
    # Adapters must not place credentials in normalized responses or errors.
    response = await healthy_adapter.submit(sample_request)
    assert "api_key" not in str(response.metadata).lower()
    assert "secret" not in str(response.metadata).lower()


@pytest.mark.contract
async def test_adapter_identity_is_immutable(
    healthy_adapter: StubProviderAdapter,
) -> None:
    identity = healthy_adapter.identity
    # Frozen dataclasses do not allow mutation via normal assignment.
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        identity.provider_id = ProviderId("other")  # type: ignore[misc]


def test_all_adapters_expose_identity() -> None:
    adapter: ProviderAdapter = StubProviderAdapter()
    assert adapter.identity.provider_id is not None
