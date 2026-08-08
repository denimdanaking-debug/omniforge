"""Stub adapter contract compliance using the shared suite (Step 2.7)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.errors import ProviderError, ProviderErrorCode
from src.providers.identity import ProviderIdentity
from src.providers.stub_adapter import StubAdapterConfig, StubProviderAdapter
from tests.providers.adapter_contract_suite import AdapterContractSuite


class TestStubAdapterContract(AdapterContractSuite):
    """Run the provider-neutral contract suite against StubProviderAdapter."""

    @pytest.fixture
    def adapter_factory(self) -> Callable[[], ProviderAdapter]:
        return lambda: StubProviderAdapter()

    @pytest.fixture
    def limited_adapter_factory(self) -> Callable[[], ProviderAdapter]:
        return lambda: StubProviderAdapter(
            config=StubAdapterConfig(
                capabilities=ProviderAdapterCapabilities(
                    streaming=False,
                    tool_calls=False,
                    structured_output=False,
                    cancellation=True,
                )
            )
        )

    @pytest.fixture
    def failing_adapter_factory(self) -> Callable[[], ProviderAdapter]:
        return lambda: StubProviderAdapter(
            config=StubAdapterConfig(
                fail_with_error=ProviderError(
                    code=ProviderErrorCode.RATE_LIMITED,
                    message="Too many requests",
                    provider_id=ProviderIdentity("stub", "Stub", "stub.example"),
                )
            )
        )
