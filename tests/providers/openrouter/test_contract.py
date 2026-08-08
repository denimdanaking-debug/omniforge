"""Adapter contract tests for OpenRouterAdapter routing Claude."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.identity import ProviderIdentity
from src.providers.openrouter.adapter import OpenRouterAdapter
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity, ModelLifecycle
from tests.providers._openai_compat_mocks import (
    build_mock_openai_client,
    make_stream_chunks,
    make_success_response,
)
from tests.providers.adapter_contract_suite import AdapterContractSuite


def _claude_identity() -> ModelIdentity:
    return ModelIdentity(
        model_id="claude-sonnet-4-20250514",
        family="claude",
        lifecycle=ModelLifecycle.HIGH_RISK,
    )


def _openrouter_route() -> InferenceRouteIdentity:
    return InferenceRouteIdentity(
        route_id="openrouter-claude",
        provider_id="openrouter",
        route_type=RouteType.GATEWAY,
        endpoint_key="openrouter://anthropic/claude-sonnet-4-20250514",
        failure_domain="openrouter.ai",
    )


class OpenRouterContractSuite(AdapterContractSuite):
    __test__ = True  # pytest must collect this abstract-suite subclass

    @pytest.fixture
    def adapter_factory(self) -> Callable[[], ProviderAdapter]:
        def factory() -> ProviderAdapter:
            adapter = OpenRouterAdapter(
                provider_identity=ProviderIdentity("anthropic", "Anthropic", "anthropic.com"),
                model_identity=_claude_identity(),
                route_identity=_openrouter_route(),
                api_key="test-key",
            )
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
            adapter = OpenRouterAdapter(
                provider_identity=ProviderIdentity("anthropic", "Anthropic", "anthropic.com"),
                model_identity=_claude_identity(),
                route_identity=_openrouter_route(),
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
            adapter = OpenRouterAdapter(
                provider_identity=ProviderIdentity("anthropic", "Anthropic", "anthropic.com"),
                model_identity=_claude_identity(),
                route_identity=_openrouter_route(),
                api_key="test-key",
            )
            adapter._client = lambda: build_mock_openai_client(  # type: ignore[method-assign]
                exception=Exception("rate limit exceeded")
            )
            return adapter

        return factory
