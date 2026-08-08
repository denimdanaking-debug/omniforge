"""Qwen adapter contract tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.qwen.adapter import DEFAULT_BASE_URL, QwenAdapter
from tests.providers._openai_compat_mocks import (
    FakeRateLimitError,
    build_mock_openai_client,
    make_stream_chunks,
    make_success_response,
)
from tests.providers.adapter_contract_suite import AdapterContractSuite


class TestQwenAdapterContract(AdapterContractSuite):
    """Run the provider-neutral contract suite against QwenAdapter."""

    @pytest.fixture
    def adapter_factory(self, monkeypatch: Any) -> Callable[[], ProviderAdapter]:
        def _factory() -> ProviderAdapter:
            client = build_mock_openai_client(
                response=make_success_response(
                    content='{"steps": []}',
                    tool_calls=[
                        {
                            "id": "tc-1",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "/tmp"}',
                            },
                        }
                    ],
                ),
                stream_chunks=make_stream_chunks(["Hello", " world"]),
            )
            adapter = QwenAdapter(api_key="test-key")
            monkeypatch.setattr(adapter, "_client", lambda: client)
            return adapter

        return _factory

    @pytest.fixture
    def limited_adapter_factory(self, monkeypatch: Any) -> Callable[[], ProviderAdapter]:
        def _factory() -> ProviderAdapter:
            client = build_mock_openai_client(response=make_success_response())
            adapter = QwenAdapter(
                api_key="test-key",
                capabilities=ProviderAdapterCapabilities(
                    streaming=False,
                    tool_calls=False,
                    structured_output=False,
                    cancellation=True,
                ),
            )
            monkeypatch.setattr(adapter, "_client", lambda: client)
            return adapter

        return _factory

    @pytest.fixture
    def failing_adapter_factory(self, monkeypatch: Any) -> Callable[[], ProviderAdapter]:
        def _factory() -> ProviderAdapter:
            client = build_mock_openai_client(exception=FakeRateLimitError)
            adapter = QwenAdapter(api_key="test-key")
            monkeypatch.setattr(adapter, "_client", lambda: client)
            return adapter

        return _factory


@pytest.mark.contract
async def test_qwen_adapter_uses_default_base_url() -> None:
    adapter = QwenAdapter(api_key="test-key")
    assert adapter._base_url == DEFAULT_BASE_URL
