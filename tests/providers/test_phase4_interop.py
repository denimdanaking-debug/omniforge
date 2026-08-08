"""Phase 4 interoperability tests for gateways, local endpoints, and enterprise routes.

These tests verify identity separation across direct, gateway, local, and
enterprise inference routes. Provider-specific SDK objects must never escape
normalized response boundaries.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.policy.risk import RiskLevel
from src.providers.adapter import ProviderAdapterCapabilities
from src.providers.errors import ProviderErrorCode
from src.providers.generic_openai_compat.adapter import GenericOpenAICompatibleAdapter
from src.providers.identity import ProviderIdentity
from src.providers.local.adapter import LocalEndpointAdapter
from src.providers.local.profile import LocalEndpointProfile, LocalModelConfig, LocalRuntimeKind
from src.providers.minimax.adapter import MiniMaxAdapter
from src.providers.mistral.adapter import MistralAdapter
from src.providers.openrouter.adapter import OpenRouterAdapter
from src.providers.request import Message, MessageRole, ProviderRequest
from src.providers.response import ProviderResponse
from src.routing.enterprise import EnterprisePlatform, EnterpriseRouteConfig
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity, ModelLifecycle
from src.routing.roles import ExecutionRole
from tests.providers._openai_compat_mocks import (
    build_mock_openai_client,
    make_stream_chunks,
    make_success_response,
)


def _claude_identity() -> ModelIdentity:
    return ModelIdentity(
        model_id="claude-sonnet-4-20250514",
        family="claude",
        lifecycle=ModelLifecycle.HIGH_RISK,
    )


def _anthropic_identity() -> ProviderIdentity:
    return ProviderIdentity("anthropic", "Anthropic", "anthropic.com")


def _direct_route() -> InferenceRouteIdentity:
    return InferenceRouteIdentity(
        route_id="anthropic-direct-claude",
        provider_id="anthropic",
        route_type=RouteType.DIRECT,
        endpoint_key="https://api.anthropic.com/v1",
        failure_domain="anthropic.com",
    )


def _openrouter_route() -> InferenceRouteIdentity:
    return InferenceRouteIdentity(
        route_id="openrouter-claude",
        provider_id="openrouter",
        route_type=RouteType.GATEWAY,
        endpoint_key="openrouter://anthropic/claude-sonnet-4-20250514",
        failure_domain="openrouter.ai",
    )


def _ollama_profile() -> LocalEndpointProfile:
    return LocalEndpointProfile(
        runtime_kind=LocalRuntimeKind.OLLAMA,
        base_url="http://localhost:11434/v1",
        route_identity=InferenceRouteIdentity(
            route_id="ollama-qwen",
            provider_id="local-ollama",
            route_type=RouteType.LOCAL,
            endpoint_key="http://localhost:11434/v1",
            failure_domain="localhost:11434",
        ),
        failure_domain="localhost:11434",
    )


def _vllm_profile() -> LocalEndpointProfile:
    return LocalEndpointProfile(
        runtime_kind=LocalRuntimeKind.VLLM,
        base_url="http://localhost:8000/v1",
        route_identity=InferenceRouteIdentity(
            route_id="vllm-qwen",
            provider_id="local-vllm",
            route_type=RouteType.LOCAL,
            endpoint_key="http://localhost:8000/v1",
            failure_domain="localhost:8000",
        ),
        failure_domain="localhost:8000",
    )


def _openai_compat_request() -> ProviderRequest:
    return ProviderRequest(
        request_id="interop-req",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )


def _mock_adapter(adapter: Any) -> Any:
    adapter._client = lambda: build_mock_openai_client(
        response=make_success_response(content="Hello"),
        stream_chunks=make_stream_chunks(["Hello", " world"]),
    )
    return adapter


@pytest.mark.fleet
async def test_same_model_via_direct_and_openrouter() -> None:
    direct = OpenRouterAdapter(
        provider_identity=_anthropic_identity(),
        model_identity=_claude_identity(),
        route_identity=_direct_route(),
        api_key="test-key",
    )
    gateway = OpenRouterAdapter(
        provider_identity=_anthropic_identity(),
        model_identity=_claude_identity(),
        route_identity=_openrouter_route(),
        api_key="test-key",
    )
    direct = _mock_adapter(direct)
    gateway = _mock_adapter(gateway)

    request = _openai_compat_request()
    direct_response = await direct.submit(request)
    gateway_response = await gateway.submit(request)

    assert direct_response.model_id == gateway_response.model_id == _claude_identity()
    assert direct_response.route_id == _direct_route()
    assert gateway_response.route_id == _openrouter_route()
    assert direct_response.provider_id == gateway_response.provider_id == _anthropic_identity()


@pytest.mark.fleet
async def test_same_model_via_two_local_endpoints() -> None:
    ollama = LocalEndpointAdapter(
        model_config=LocalModelConfig(
            model_id="qwen2.5:7b",
            family="qwen",
            profile=_ollama_profile(),
            explicit_capabilities={"streaming": True},
        )
    )
    vllm = LocalEndpointAdapter(
        model_config=LocalModelConfig(
            model_id="qwen2.5:7b",
            family="qwen",
            profile=_vllm_profile(),
            explicit_capabilities={"streaming": True},
        )
    )
    ollama = _mock_adapter(ollama)
    vllm = _mock_adapter(vllm)

    request = _openai_compat_request()
    ollama_response = await ollama.submit(request)
    vllm_response = await vllm.submit(request)

    assert ollama_response.model_id == vllm_response.model_id
    assert ollama_response.model_id.model_id == "qwen2.5:7b"
    assert ollama_response.route_id == _ollama_profile().route_identity
    assert vllm_response.route_id == _vllm_profile().route_identity
    assert ollama_response.route_id != vllm_response.route_id


@pytest.mark.fleet
async def test_generic_endpoint_preserves_custom_identity() -> None:
    custom_provider = ProviderIdentity("starship-ai", "Starship AI", "starship.example")
    custom_model = ModelIdentity(model_id="starship-mega", family="starship")
    custom_route = InferenceRouteIdentity(
        route_id="starship-direct",
        provider_id="starship-ai",
        route_type=RouteType.DIRECT,
        endpoint_key="https://api.starship.example/v1",
        failure_domain="starship.example",
    )
    adapter = GenericOpenAICompatibleAdapter(
        provider_identity=custom_provider,
        model_identity=custom_model,
        route_identity=custom_route,
        base_url="https://api.starship.example/v1",
        api_key="test-key",
        capabilities=ProviderAdapterCapabilities(streaming=True),
    )
    adapter = _mock_adapter(adapter)

    request = _openai_compat_request()
    response = await adapter.submit(request)

    assert response.provider_id == custom_provider
    assert response.model_id == custom_model
    assert response.route_id == custom_route
    assert response.provider_id.provider_id != "openai"
    assert response.model_id.family != "openai"


@pytest.mark.fleet
async def test_gateway_failure_does_not_corrupt_model_identity() -> None:
    gateway = OpenRouterAdapter(
        provider_identity=_anthropic_identity(),
        model_identity=_claude_identity(),
        route_identity=_openrouter_route(),
        api_key="test-key",
    )
    gateway._client = lambda: build_mock_openai_client(exception=Exception("rate limit exceeded"))  # type: ignore[method-assign]

    request = _openai_compat_request()
    response = await gateway.submit(request)

    assert response.error_reference == ProviderErrorCode.RATE_LIMITED.value
    assert response.model_id == _claude_identity()
    assert response.route_id == _openrouter_route()
    error = response.metadata["error"]
    assert error.route_id == _openrouter_route()


@pytest.mark.fleet
async def test_local_endpoint_failure_preserves_model_identity() -> None:
    local = LocalEndpointAdapter(
        model_config=LocalModelConfig(
            model_id="qwen2.5:7b",
            family="qwen",
            profile=_ollama_profile(),
            explicit_capabilities={"streaming": True},
        )
    )
    local._client = lambda: build_mock_openai_client(exception=Exception("connection refused"))  # type: ignore[method-assign]

    request = _openai_compat_request()
    response = await local.submit(request)

    assert response.model_id.model_id == "qwen2.5:7b"
    assert response.model_id.family == "qwen"
    assert response.route_id == _ollama_profile().route_identity
    assert response.error_reference is not None


@pytest.mark.architecture
def test_enterprise_route_coexists_with_other_route_types() -> None:
    enterprise_config = EnterpriseRouteConfig(
        route_identity=InferenceRouteIdentity(
            route_id="bedrock-claude",
            provider_id="aws",
            route_type=RouteType.ENTERPRISE,
            endpoint_key="bedrock://us-east-1/anthropic.claude-3-sonnet",
            failure_domain="us-east-1.bedrock.amazonaws.com",
        ),
        platform=EnterprisePlatform.AWS_BEDROCK,
        region="us-east-1",
        underlying_provider_id="anthropic",
        underlying_model_id="claude-sonnet-4-20250514",
    )
    direct_config = EnterpriseRouteConfig(
        route_identity=_direct_route(),
        platform=EnterprisePlatform.AZURE_AI,
        region="westus2",
    )
    assert enterprise_config.route_identity.route_type is RouteType.ENTERPRISE
    assert direct_config.route_identity.route_type is RouteType.DIRECT


@pytest.mark.architecture
def test_route_identities_are_unique_and_stable() -> None:
    routes = [
        _direct_route(),
        _openrouter_route(),
        _ollama_profile().route_identity,
        _vllm_profile().route_identity,
        InferenceRouteIdentity(
            route_id="bedrock-claude",
            provider_id="aws",
            route_type=RouteType.ENTERPRISE,
            endpoint_key="bedrock://us-east-1/anthropic.claude-3-sonnet",
            failure_domain="us-east-1.bedrock.amazonaws.com",
        ),
    ]
    ids = {r.route_id for r in routes}
    assert len(ids) == len(routes)


@pytest.mark.fleet
async def test_sdk_objects_do_not_escape_normalized_response() -> None:
    adapter = MiniMaxAdapter(api_key="test-key")
    adapter = _mock_adapter(adapter)
    request = _openai_compat_request()
    response = await adapter.submit(request)
    _assert_no_sdk_objects(response)

    chunks = [chunk async for chunk in adapter.stream(request)]
    for chunk in chunks:
        _assert_no_sdk_objects(chunk)


def _assert_no_sdk_objects(response: ProviderResponse) -> None:
    assert response.text is None or isinstance(response.text, str)
    assert response.structured_result is None or isinstance(response.structured_result, dict)
    for call in response.tool_calls:
        for arg in call.arguments:
            assert isinstance(arg.value, (str, int, float, bool, dict, list)) or arg.value is None
    for value in response.metadata.values():
        assert not type(value).__module__.startswith(("openai", "anthropic", "google"))


@pytest.mark.fleet
async def test_phase4_adapters_are_present_and_distinct() -> None:
    adapters = [
        MiniMaxAdapter(api_key="test-key"),
        MistralAdapter(api_key="test-key"),
        OpenRouterAdapter(
            provider_identity=_anthropic_identity(),
            model_identity=_claude_identity(),
            api_key="test-key",
        ),
        GenericOpenAICompatibleAdapter(
            provider_identity=ProviderIdentity("acme", "Acme", "acme.example"),
            model_identity=ModelIdentity(model_id="acme-model", family="acme"),
            base_url="https://acme.example/v1",
        ),
        LocalEndpointAdapter(
            model_config=LocalModelConfig(
                model_id="qwen2.5:7b",
                family="qwen",
                profile=_ollama_profile(),
            )
        ),
    ]
    ids = {a.identity.provider_id for a in adapters}
    assert ids == {"minimax", "mistral", "anthropic", "acme", "local-ollama"}
