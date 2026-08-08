"""Adapter contract tests for LocalEndpointAdapter."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.local.adapter import LocalEndpointAdapter
from src.providers.local.profile import LocalEndpointProfile, LocalModelConfig, LocalRuntimeKind
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from tests.providers._openai_compat_mocks import (
    build_mock_openai_client,
    make_stream_chunks,
    make_success_response,
)
from tests.providers.adapter_contract_suite import AdapterContractSuite


def _profile() -> LocalEndpointProfile:
    return LocalEndpointProfile(
        runtime_kind=LocalRuntimeKind.OLLAMA,
        base_url="http://localhost:11434/v1",
        route_identity=InferenceRouteIdentity(
            route_id="ollama-contract",
            provider_id="local-ollama",
            route_type=RouteType.LOCAL,
            endpoint_key="http://localhost:11434/v1",
            failure_domain="localhost:11434",
        ),
        failure_domain="localhost:11434",
    )


def _model_config() -> LocalModelConfig:
    return LocalModelConfig(
        model_id="qwen2.5:7b",
        family="qwen",
        profile=_profile(),
        explicit_capabilities={
            "streaming": True,
            "tool_use": True,
            "structured_output": True,
        },
    )


class LocalEndpointContractSuite(AdapterContractSuite):
    __test__ = True  # pytest must collect this abstract-suite subclass

    @pytest.fixture
    def adapter_factory(self) -> Callable[[], ProviderAdapter]:
        def factory() -> ProviderAdapter:
            adapter = LocalEndpointAdapter(model_config=_model_config())
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
            adapter = LocalEndpointAdapter(
                model_config=LocalModelConfig(
                    model_id="qwen2.5:7b",
                    family="qwen",
                    profile=_profile(),
                    explicit_capabilities={},
                ),
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
            adapter = LocalEndpointAdapter(model_config=_model_config())
            adapter._client = lambda: build_mock_openai_client(  # type: ignore[method-assign]
                exception=Exception("rate limit exceeded")
            )
            return adapter

        return factory
