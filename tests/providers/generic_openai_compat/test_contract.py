"""Adapter contract tests for GenericOpenAICompatibleAdapter."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.generic_openai_compat.adapter import GenericOpenAICompatibleAdapter
from src.providers.identity import ProviderIdentity
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity
from tests.providers._openai_compat_mocks import (
    build_mock_openai_client,
    make_stream_chunks,
    make_success_response,
)
from tests.providers.adapter_contract_suite import AdapterContractSuite


def _custom_identity() -> ProviderIdentity:
    return ProviderIdentity("acme-corp", "Acme Corp Inference", "acme.example")


def _custom_model() -> ModelIdentity:
    return ModelIdentity(model_id="acme-model-v1", family="acme")


def _custom_route() -> InferenceRouteIdentity:
    return InferenceRouteIdentity(
        route_id="acme-direct",
        provider_id="acme-corp",
        route_type=RouteType.DIRECT,
        endpoint_key="https://inference.acme.example/v1",
        failure_domain="acme.example",
    )


class GenericOpenAICompatibleContractSuite(AdapterContractSuite):
    __test__ = True  # pytest must collect this abstract-suite subclass

    @pytest.fixture
    def adapter_factory(self) -> Callable[[], ProviderAdapter]:
        def factory() -> ProviderAdapter:
            adapter = GenericOpenAICompatibleAdapter(
                provider_identity=_custom_identity(),
                model_identity=_custom_model(),
                route_identity=_custom_route(),
                base_url="https://inference.acme.example/v1",
                api_key="test-key",
                capabilities=ProviderAdapterCapabilities(
                    streaming=True,
                    tool_calls=True,
                    structured_output=True,
                    cancellation=True,
                ),
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
            adapter = GenericOpenAICompatibleAdapter(
                provider_identity=_custom_identity(),
                model_identity=_custom_model(),
                route_identity=_custom_route(),
                base_url="https://inference.acme.example/v1",
                api_key="test-key",
            )
            adapter._client = lambda: build_mock_openai_client(response=make_success_response())  # type: ignore[method-assign]
            return adapter

        return factory

    @pytest.fixture
    def failing_adapter_factory(self) -> Callable[[], ProviderAdapter]:
        def factory() -> ProviderAdapter:
            adapter = GenericOpenAICompatibleAdapter(
                provider_identity=_custom_identity(),
                model_identity=_custom_model(),
                route_identity=_custom_route(),
                base_url="https://inference.acme.example/v1",
                api_key="test-key",
            )
            adapter._client = lambda: build_mock_openai_client(  # type: ignore[method-assign]
                exception=Exception("rate limit exceeded")
            )
            return adapter

        return factory
