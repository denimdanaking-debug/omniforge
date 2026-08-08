"""OpenRouter gateway-specific unit tests."""

from __future__ import annotations

from typing import Any

import pytest

from src.policy.risk import RiskLevel
from src.providers.errors import ProviderErrorCode
from src.providers.identity import ProviderIdentity
from src.providers.openrouter.adapter import DEFAULT_BASE_URL, OpenRouterAdapter
from src.providers.request import Message, MessageRole, ProviderRequest
from src.providers.response import FinishReason, StreamingState
from src.routing.capabilities import ModelCapabilities
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


def _openrouter_route() -> InferenceRouteIdentity:
    return InferenceRouteIdentity(
        route_id="openrouter-claude",
        provider_id="openrouter",
        route_type=RouteType.GATEWAY,
        endpoint_key="openrouter://anthropic/claude-sonnet-4-20250514",
        failure_domain="openrouter.ai",
    )


def _direct_route() -> InferenceRouteIdentity:
    return InferenceRouteIdentity(
        route_id="anthropic-direct-claude",
        provider_id="anthropic",
        route_type=RouteType.DIRECT,
        endpoint_key="https://api.anthropic.com/v1",
        failure_domain="anthropic.com",
    )


@pytest.fixture
def adapter(monkeypatch: Any) -> OpenRouterAdapter:
    client = build_mock_openai_client(
        response=make_success_response(content="Hello"),
        stream_chunks=make_stream_chunks(["Hello", " world"]),
    )
    adapter = OpenRouterAdapter(
        provider_identity=ProviderIdentity("anthropic", "Anthropic", "anthropic.com"),
        model_identity=_claude_identity(),
        route_identity=_openrouter_route(),
        api_key="test-key",
    )
    monkeypatch.setattr(adapter, "_client", lambda: client)
    return adapter


async def test_underlying_provider_identity(adapter: OpenRouterAdapter) -> None:
    assert adapter.identity.provider_id == "anthropic"
    assert adapter.identity.display_name == "Anthropic"


async def test_openrouter_route_identity(adapter: OpenRouterAdapter) -> None:
    route = adapter.route_id
    assert route is not None
    assert route.route_id == "openrouter-claude"
    assert route.route_type is RouteType.GATEWAY
    assert route.failure_domain == "openrouter.ai"


async def test_underlying_model_identity(adapter: OpenRouterAdapter) -> None:
    model = adapter.model_id
    assert model.model_id == "claude-sonnet-4-20250514"
    assert model.family == "claude"


async def test_default_base_url() -> None:
    adapter = OpenRouterAdapter(
        provider_identity=ProviderIdentity("anthropic", "Anthropic", "anthropic.com"),
        model_identity=_claude_identity(),
        api_key="test-key",
    )
    assert adapter._base_url == DEFAULT_BASE_URL


async def test_default_capabilities_are_conservative() -> None:
    adapter = OpenRouterAdapter(
        provider_identity=ProviderIdentity("anthropic", "Anthropic", "anthropic.com"),
        model_identity=_claude_identity(),
        api_key="test-key",
    )
    caps = adapter.model_capabilities
    assert caps.context_tokens == 4096
    assert caps.structured_output is False
    assert caps.tool_use is False
    assert caps.streaming is False
    assert caps.reasoning is False
    assert caps.code_generation is False
    assert caps.multimodal is False


async def test_explicit_model_capabilities_are_preserved() -> None:
    explicit_caps = ModelCapabilities(
        context_tokens=200_000,
        structured_output=True,
        tool_use=True,
        streaming=True,
        code_generation=True,
    )
    adapter = OpenRouterAdapter(
        provider_identity=ProviderIdentity("anthropic", "Anthropic", "anthropic.com"),
        model_identity=_claude_identity(),
        route_identity=_openrouter_route(),
        api_key="test-key",
        model_capabilities=explicit_caps,
    )
    assert adapter.model_capabilities == explicit_caps


async def test_response_preserves_model_and_route(adapter: OpenRouterAdapter) -> None:
    request = ProviderRequest(
        request_id="req-1",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )
    response = await adapter.submit(request)
    assert response.provider_id.provider_id == "anthropic"
    assert response.model_id == _claude_identity()
    assert response.route_id == _openrouter_route()
    assert response.text == "Hello"
    assert response.finish_reason is FinishReason.STOP


async def test_gateway_rate_limit_attributed_to_route(
    adapter: OpenRouterAdapter, monkeypatch: Any
) -> None:
    client = build_mock_openai_client(exception=Exception("rate limit exceeded"))
    monkeypatch.setattr(adapter, "_client", lambda: client)
    request = ProviderRequest(
        request_id="req-2",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
    )
    response = await adapter.submit(request)
    assert response.error_reference == ProviderErrorCode.RATE_LIMITED.value
    error = response.metadata["error"]
    assert error.provider_id.provider_id == "anthropic"
    assert error.route_id == _openrouter_route()


async def test_streaming_preserves_identities(adapter: OpenRouterAdapter) -> None:
    request = ProviderRequest(
        request_id="req-3",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
    )
    chunks = [chunk async for chunk in adapter.stream(request)]
    assert all(chunk.model_id == _claude_identity() for chunk in chunks)
    assert all(chunk.route_id == _openrouter_route() for chunk in chunks)
    assert chunks[-1].streaming_state is StreamingState.COMPLETE


async def test_direct_route_is_rejected() -> None:
    with pytest.raises(ValueError, match="OpenRouter route must be RouteType.GATEWAY"):
        OpenRouterAdapter(
            provider_identity=ProviderIdentity("anthropic", "Anthropic", "anthropic.com"),
            model_identity=_claude_identity(),
            route_identity=_direct_route(),
            api_key="test-key",
        )


async def test_local_route_is_rejected() -> None:
    local_route = InferenceRouteIdentity(
        route_id="openrouter-local",
        provider_id="openrouter",
        route_type=RouteType.LOCAL,
        endpoint_key="http://localhost:8000/v1",
        failure_domain="localhost",
    )
    with pytest.raises(ValueError, match="OpenRouter route must be RouteType.GATEWAY"):
        OpenRouterAdapter(
            provider_identity=ProviderIdentity("anthropic", "Anthropic", "anthropic.com"),
            model_identity=_claude_identity(),
            route_identity=local_route,
            api_key="test-key",
        )


async def test_enterprise_route_is_rejected() -> None:
    enterprise_route = InferenceRouteIdentity(
        route_id="openrouter-enterprise",
        provider_id="openrouter",
        route_type=RouteType.ENTERPRISE,
        endpoint_key="bedrock://us-east-1/claude",
        failure_domain="amazonaws.com",
    )
    with pytest.raises(ValueError, match="OpenRouter route must be RouteType.GATEWAY"):
        OpenRouterAdapter(
            provider_identity=ProviderIdentity("anthropic", "Anthropic", "anthropic.com"),
            model_identity=_claude_identity(),
            route_identity=enterprise_route,
            api_key="test-key",
        )


async def test_gateway_route_is_accepted() -> None:
    route = _openrouter_route()
    adapter = OpenRouterAdapter(
        provider_identity=ProviderIdentity("anthropic", "Anthropic", "anthropic.com"),
        model_identity=_claude_identity(),
        route_identity=route,
        api_key="test-key",
    )
    assert adapter.route_id == route
