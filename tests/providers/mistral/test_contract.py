"""Adapter contract tests for MistralAdapter."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.mistral.adapter import MistralAdapter
from tests.providers._openai_compat_mocks import (
    build_mock_openai_client,
    make_stream_chunks,
    make_success_response,
)
from tests.providers.adapter_contract_suite import AdapterContractSuite


class MistralContractSuite(AdapterContractSuite):
    __test__ = True  # pytest must collect this abstract-suite subclass

    @pytest.fixture
    def adapter_factory(self) -> Callable[[], ProviderAdapter]:
        def factory() -> ProviderAdapter:
            adapter = MistralAdapter(api_key="test-key")
            adapter._client = lambda: build_mock_openai_client(  # type: ignore[method-assign]
                response=make_success_response(
                    content='{"steps": []}',
                    tool_calls=[
                        {
                            "id": "tc-1",
                            "function": {"name": "read_file", "arguments": '{"path": "/tmp"}'},
                        }
                    ],
                ),
                stream_chunks=make_stream_chunks(["Hello", " world"]),
            )
            return adapter

        return factory

    @pytest.fixture
    def limited_adapter_factory(self) -> Callable[[], ProviderAdapter]:
        def factory() -> ProviderAdapter:
            adapter = MistralAdapter(
                api_key="test-key",
                capabilities=ProviderAdapterCapabilities(
                    streaming=False,
                    tool_calls=False,
                    structured_output=False,
                    cancellation=True,
                ),
            )
            adapter._client = lambda: build_mock_openai_client(response=make_success_response())  # type: ignore[method-assign]
            return adapter

        return factory

    @pytest.fixture
    def failing_adapter_factory(self) -> Callable[[], ProviderAdapter]:
        def factory() -> ProviderAdapter:
            adapter = MistralAdapter(api_key="test-key")
            adapter._client = lambda: build_mock_openai_client(  # type: ignore[method-assign]
                exception=Exception("rate limit exceeded")
            )
            return adapter

        return factory
