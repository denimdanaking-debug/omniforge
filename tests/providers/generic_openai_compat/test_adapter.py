"""Unit tests for the configurable generic OpenAI-compatible endpoint adapter."""

from __future__ import annotations

from typing import Any

import pytest

from src.policy.risk import RiskLevel
from src.providers.adapter import ProviderAdapterCapabilities
from src.providers.errors import ProviderErrorCode
from src.providers.generic_openai_compat.adapter import GenericOpenAICompatibleAdapter
from src.providers.identity import ProviderHealth, ProviderIdentity
from src.providers.request import Message, MessageRole, ProviderRequest
from src.providers.response import FinishReason, StreamingState
from src.routing.capabilities import CapabilityRequirement, DeploymentMode, match_capabilities
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity
from src.routing.roles import ExecutionRole
from tests.providers._openai_compat_mocks import (
    FakeAuthError,
    FakeRateLimitError,
    build_mock_openai_client,
    make_stream_chunks,
    make_success_response,
)


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


@pytest.fixture
def adapter(monkeypatch: Any) -> GenericOpenAICompatibleAdapter:
    client = build_mock_openai_client(
        response=make_success_response(content="Hello"),
        stream_chunks=make_stream_chunks(["Hello", " world"]),
    )
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
    monkeypatch.setattr(adapter, "_client", lambda: client)
    return adapter


async def test_identity_is_configured_provider(adapter: GenericOpenAICompatibleAdapter) -> None:
    assert adapter.identity == _custom_identity()
    assert adapter.identity.provider_id == "acme-corp"


async def test_model_identity_is_configured_model(adapter: GenericOpenAICompatibleAdapter) -> None:
    assert adapter.model_id == _custom_model()
    assert adapter.model_id.family == "acme"


async def test_route_identity_is_configured_route(adapter: GenericOpenAICompatibleAdapter) -> None:
    route = adapter.route_id
    assert route is not None
    assert route.route_id == "acme-direct"
    assert route.route_type is RouteType.DIRECT


async def test_default_capabilities_are_conservative() -> None:
    adapter = GenericOpenAICompatibleAdapter(
        provider_identity=_custom_identity(),
        model_identity=_custom_model(),
        base_url="https://inference.acme.example/v1",
    )
    caps = adapter.capabilities
    assert caps.streaming is False
    assert caps.tool_calls is False
    assert caps.structured_output is False
    assert caps.reasoning is False
    assert caps.cancellation is True


async def test_model_capabilities_are_conservative_by_default() -> None:
    adapter = GenericOpenAICompatibleAdapter(
        provider_identity=_custom_identity(),
        model_identity=_custom_model(),
        base_url="https://inference.acme.example/v1",
    )
    model_caps = adapter.model_capabilities
    assert model_caps.structured_output is False
    assert model_caps.tool_use is False
    assert model_caps.streaming is False
    assert model_caps.reasoning is False
    assert model_caps.code_generation is False
    assert model_caps.multimodal is False
    assert model_caps.deployment_mode is DeploymentMode.CLOUD


async def test_payload_uses_configured_model(adapter: GenericOpenAICompatibleAdapter) -> None:
    request = ProviderRequest(
        request_id="req-1",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )
    params = adapter._build_chat_params(request)
    assert params["model"] == "acme-model-v1"


async def test_response_preserves_configured_identities(
    adapter: GenericOpenAICompatibleAdapter,
) -> None:
    request = ProviderRequest(
        request_id="req-2",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )
    response = await adapter.submit(request)
    assert response.request_id == "req-2"
    assert response.provider_id == _custom_identity()
    assert response.model_id == _custom_model()
    assert response.route_id == _custom_route()
    assert response.text == "Hello"
    assert response.finish_reason is FinishReason.STOP
    assert response.error_reference is None


async def test_streaming_preserves_configured_identities(
    adapter: GenericOpenAICompatibleAdapter,
) -> None:
    request = ProviderRequest(
        request_id="req-3",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )
    chunks = [chunk async for chunk in adapter.stream(request)]
    assert len(chunks) == 3
    assert all(chunk.provider_id == _custom_identity() for chunk in chunks)
    assert all(chunk.model_id == _custom_model() for chunk in chunks)
    assert all(chunk.route_id == _custom_route() for chunk in chunks)
    assert chunks[-1].streaming_state is StreamingState.COMPLETE
    assert chunks[-1].error_reference is None


async def test_target_model_survives_response(adapter: GenericOpenAICompatibleAdapter) -> None:
    target = ModelIdentity(model_id="acme-target", family="acme")
    request = ProviderRequest(
        request_id="req-4",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
        target_model=target,
    )
    response = await adapter.submit(request)
    assert response.model_id == target
    params = adapter._build_chat_params(request)
    assert params["model"] == "acme-target"


async def test_no_openai_identity_collapse() -> None:
    adapter = GenericOpenAICompatibleAdapter(
        provider_identity=_custom_identity(),
        model_identity=_custom_model(),
        base_url="https://inference.acme.example/v1",
    )
    assert adapter.identity.provider_id != "openai"
    assert adapter.model_id.model_id != "gpt-4"


async def test_health_is_conservative(adapter: GenericOpenAICompatibleAdapter) -> None:
    health = await adapter.health()
    assert health.health is not ProviderHealth.HEALTHY
    assert health.reason is not None
    assert "observation" in health.reason.lower() or "initialized" in health.reason.lower()


async def test_quota_is_unknown(adapter: GenericOpenAICompatibleAdapter) -> None:
    quota = await adapter.quota()
    assert quota.is_exhausted() is False


async def test_error_normalization_rate_limited(monkeypatch: Any) -> None:
    client = build_mock_openai_client(exception=FakeRateLimitError)
    adapter = GenericOpenAICompatibleAdapter(
        provider_identity=_custom_identity(),
        model_identity=_custom_model(),
        base_url="https://inference.acme.example/v1",
        api_key="test-key",
    )
    monkeypatch.setattr(adapter, "_client", lambda: client)
    request = ProviderRequest(
        request_id="req-5",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
    )
    response = await adapter.submit(request)
    assert response.error_reference == ProviderErrorCode.RATE_LIMITED.value
    error = response.metadata["error"]
    assert error.provider_id == _custom_identity()


async def test_error_normalization_auth_failure(monkeypatch: Any) -> None:
    client = build_mock_openai_client(exception=FakeAuthError)
    adapter = GenericOpenAICompatibleAdapter(
        provider_identity=_custom_identity(),
        model_identity=_custom_model(),
        base_url="https://inference.acme.example/v1",
        api_key="test-key",
    )
    monkeypatch.setattr(adapter, "_client", lambda: client)
    request = ProviderRequest(
        request_id="req-6",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
    )
    response = await adapter.submit(request)
    assert response.error_reference == ProviderErrorCode.AUTH_FAILURE.value


async def test_capability_requirements_match_model_capabilities() -> None:
    adapter = GenericOpenAICompatibleAdapter(
        provider_identity=_custom_identity(),
        model_identity=_custom_model(),
        base_url="https://inference.acme.example/v1",
        capabilities=ProviderAdapterCapabilities(
            streaming=True,
            tool_calls=True,
            structured_output=True,
        ),
    )
    model_caps = adapter.model_capabilities
    req = CapabilityRequirement(
        min_context_tokens=1,
        structured_output=True,
        required_roles=frozenset({ExecutionRole.CODING.value}),
    )
    match = match_capabilities(model_caps, req)
    assert not match.eligible
    assert any("role:" in missing for missing in match.missing)
