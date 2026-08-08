"""Reusable provider-adapter contract test suite (Step 2.7).

Subclasses provide factory fixtures and inherit the provider-neutral assertions.
Concrete provider adapters added in later phases can run the same suite by
overriding the abstract factory fixtures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import pytest

from src.policy.risk import RiskLevel
from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.errors import ErrorCategory, ProviderErrorCode
from src.providers.identity import (
    ProviderHealth,
    ProviderIdentity,
    ProviderOperationalState,
    ProviderQuotaState,
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
from src.routing.roles import ExecutionRole


class AdapterContractSuite(ABC):
    """Provider-neutral contract assertions. Subclasses supply adapter factories."""

    @pytest.fixture
    @abstractmethod
    def adapter_factory(self) -> Callable[[], ProviderAdapter]:
        """Return a factory that creates a healthy, capable adapter under test."""

    @pytest.fixture
    @abstractmethod
    def limited_adapter_factory(self) -> Callable[[], ProviderAdapter]:
        """Return a factory that creates an adapter with streaming/tools/structured disabled."""

    @pytest.fixture
    @abstractmethod
    def failing_adapter_factory(self) -> Callable[[], ProviderAdapter]:
        """Return a factory that creates an adapter configured to fail with a normalized error."""

    @pytest.fixture
    def sample_request(self) -> ProviderRequest:
        return ProviderRequest(
            request_id="contract-req-1",
            execution_role=ExecutionRole.CODING,
            risk_level=RiskLevel.R2_NORMAL,
            system_instructions=["You are a helpful coding assistant."],
            messages=[Message(role=MessageRole.USER, content="Write a function")],
        )

    @pytest.fixture
    def structured_output_request(self) -> ProviderRequest:
        return ProviderRequest(
            request_id="structured-req",
            execution_role=ExecutionRole.PLANNING,
            risk_level=RiskLevel.R2_NORMAL,
            messages=[Message(role=MessageRole.USER, content="Plan")],
            structured_output=StructuredOutputRequirement(
                schema={"type": "object", "properties": {"steps": {"type": "array"}}},
                name="plan",
            ),
        )

    @pytest.fixture
    def tool_request(self) -> ProviderRequest:
        return ProviderRequest(
            request_id="tool-req",
            execution_role=ExecutionRole.CODING,
            risk_level=RiskLevel.R2_NORMAL,
            messages=[Message(role=MessageRole.USER, content="Use a tool")],
            tools=[ToolDefinition(name="read_file", description="Read a file")],
            tool_choice=ToolChoiceMode.AUTO,
        )

    @pytest.fixture
    def cancellation_request(self) -> ProviderRequest:
        return ProviderRequest(
            request_id="cancel-req",
            execution_role=ExecutionRole.CODING,
            risk_level=RiskLevel.R2_NORMAL,
            cancellation_id="cancel-1",
        )

    @pytest.mark.contract
    async def test_stable_provider_identity(
        self, adapter_factory: Callable[[], ProviderAdapter]
    ) -> None:
        adapter = adapter_factory()
        identity = adapter.identity
        assert isinstance(identity, ProviderIdentity)
        assert identity.provider_id

    @pytest.mark.contract
    async def test_declared_capabilities(
        self, adapter_factory: Callable[[], ProviderAdapter]
    ) -> None:
        adapter = adapter_factory()
        caps = adapter.capabilities
        assert isinstance(caps, ProviderAdapterCapabilities)

    @pytest.mark.contract
    async def test_request_translation_boundary(
        self,
        adapter_factory: Callable[[], ProviderAdapter],
        sample_request: ProviderRequest,
    ) -> None:
        adapter = adapter_factory()
        response = await adapter.submit(sample_request)
        assert response.request_id == sample_request.request_id
        assert response.provider_id == adapter.identity
        assert response.model_id is not None

    @pytest.mark.contract
    async def test_normalized_response(
        self,
        adapter_factory: Callable[[], ProviderAdapter],
        sample_request: ProviderRequest,
    ) -> None:
        adapter = adapter_factory()
        response = await adapter.submit(sample_request)
        assert response.request_id == sample_request.request_id
        assert response.finish_reason is FinishReason.STOP
        assert response.usage is not None
        assert response.latency_seconds is not None
        assert response.provider_request_id is not None

    @pytest.mark.contract
    async def test_streaming_semantics(
        self,
        adapter_factory: Callable[[], ProviderAdapter],
        sample_request: ProviderRequest,
    ) -> None:
        adapter = adapter_factory()
        chunks: list[Any] = []
        async for chunk in adapter.stream(sample_request):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert chunks[-1].streaming_state is StreamingState.COMPLETE

    @pytest.mark.contract
    async def test_structured_output(
        self,
        adapter_factory: Callable[[], ProviderAdapter],
        structured_output_request: ProviderRequest,
    ) -> None:
        adapter = adapter_factory()
        if not adapter.capabilities.structured_output:
            pytest.skip("adapter does not advertise structured output support")
        response = await adapter.submit(structured_output_request)
        assert response.structured_result is not None

    @pytest.mark.contract
    async def test_tool_calls(
        self,
        adapter_factory: Callable[[], ProviderAdapter],
        tool_request: ProviderRequest,
    ) -> None:
        adapter = adapter_factory()
        if not adapter.capabilities.tool_calls:
            pytest.skip("adapter does not advertise tool call support")
        response = await adapter.submit(tool_request)
        assert response.has_tool_calls() is True
        assert response.tool_calls[0].tool_name == "read_file"

    @pytest.mark.contract
    async def test_cancellation(
        self,
        adapter_factory: Callable[[], ProviderAdapter],
        cancellation_request: ProviderRequest,
    ) -> None:
        adapter = adapter_factory()
        if not adapter.capabilities.cancellation:
            pytest.skip("adapter does not advertise cancellation support")
        cancellation_id = cancellation_request.cancellation_id or cancellation_request.request_id
        await adapter.cancel(cancellation_id)
        response = await adapter.submit(cancellation_request)
        assert response.error_reference == ProviderErrorCode.CANCELLED.value

    @pytest.mark.contract
    async def test_health(self, adapter_factory: Callable[[], ProviderAdapter]) -> None:
        adapter = adapter_factory()
        health = await adapter.health()
        assert isinstance(health, ProviderOperationalState)
        assert health.health in ProviderHealth

    @pytest.mark.contract
    async def test_quota(self, adapter_factory: Callable[[], ProviderAdapter]) -> None:
        adapter = adapter_factory()
        quota = await adapter.quota()
        assert quota is not None
        assert isinstance(quota, ProviderQuotaState)

    @pytest.mark.contract
    async def test_normalized_errors(
        self,
        failing_adapter_factory: Callable[[], ProviderAdapter],
        sample_request: ProviderRequest,
    ) -> None:
        adapter = failing_adapter_factory()
        response = await adapter.submit(sample_request)
        assert response.error_reference is not None
        assert response.error_reference in {code.value for code in ProviderErrorCode}

    @pytest.mark.contract
    async def test_unsupported_capability_behavior(
        self,
        limited_adapter_factory: Callable[[], ProviderAdapter],
    ) -> None:
        adapter = limited_adapter_factory()
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
    async def test_clean_lifecycle(self, adapter_factory: Callable[[], ProviderAdapter]) -> None:
        adapter = adapter_factory()
        health = await adapter.health()
        quota = await adapter.quota()
        assert isinstance(health, ProviderOperationalState)
        assert isinstance(quota, ProviderQuotaState)
        if health.is_available():
            assert quota.is_exhausted() is False

    @pytest.mark.contract
    async def test_no_credential_leakage(
        self,
        adapter_factory: Callable[[], ProviderAdapter],
        sample_request: ProviderRequest,
    ) -> None:
        adapter = adapter_factory()
        response = await adapter.submit(sample_request)
        response_text = str(response.metadata).lower()
        assert "api_key" not in response_text
        assert "secret" not in response_text
        assert "token" not in response_text

    @pytest.mark.contract
    def test_all_adapters_expose_identity(
        self, adapter_factory: Callable[[], ProviderAdapter]
    ) -> None:
        adapter = adapter_factory()
        assert adapter.identity.provider_id is not None
