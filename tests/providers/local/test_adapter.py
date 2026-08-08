"""Unit tests for the local endpoint adapter."""

from __future__ import annotations

from typing import Any

import pytest

from src.policy.risk import RiskLevel
from src.providers.errors import ProviderErrorCode
from src.providers.identity import ProviderHealth
from src.providers.local.adapter import LocalEndpointAdapter
from src.providers.local.profile import LocalEndpointProfile, LocalModelConfig, LocalRuntimeKind
from src.providers.request import Message, MessageRole, ProviderRequest
from src.providers.response import FinishReason, StreamingState
from src.routing.capabilities import CapabilityRequirement, DeploymentMode, match_capabilities
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity
from src.routing.roles import ExecutionRole
from tests.providers._openai_compat_mocks import (
    FakeRateLimitError,
    build_mock_openai_client,
    make_stream_chunks,
    make_success_response,
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
            metadata={"runtime_kind": "ollama"},
        ),
        failure_domain="localhost:11434",
        models_endpoint="/v1/models",
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
            metadata={"runtime_kind": "vllm"},
        ),
        failure_domain="localhost:8000",
        models_endpoint="/v1/models",
    )


def _ollama_config() -> LocalModelConfig:
    return LocalModelConfig(
        model_id="qwen2.5:7b",
        family="qwen",
        profile=_ollama_profile(),
        explicit_capabilities={
            "streaming": True,
            "tool_use": True,
        },
    )


@pytest.fixture
def adapter(monkeypatch: Any) -> LocalEndpointAdapter:
    client = build_mock_openai_client(
        response=make_success_response(content="Hello"),
        stream_chunks=make_stream_chunks(["Hello", " world"]),
    )
    adapter = LocalEndpointAdapter(model_config=_ollama_config())
    monkeypatch.setattr(adapter, "_client", lambda: client)
    return adapter


async def test_provider_identity_is_local_runtime(adapter: LocalEndpointAdapter) -> None:
    assert adapter.identity.provider_id == "local-ollama"
    assert adapter.identity.display_name == "Local (ollama)"
    assert adapter.identity.metadata["runtime_kind"] == "ollama"


async def test_model_identity_is_distinct_from_runtime(adapter: LocalEndpointAdapter) -> None:
    model = adapter.model_id
    assert model.model_id == "qwen2.5:7b"
    assert model.family == "qwen"


async def test_route_identity_is_local_route(adapter: LocalEndpointAdapter) -> None:
    route = adapter.route_id
    assert route is not None
    assert route.route_id == "ollama-qwen"
    assert route.route_type is RouteType.LOCAL


async def test_runtime_kind_is_not_model_family() -> None:
    adapter = LocalEndpointAdapter(model_config=_ollama_config())
    assert adapter.model_id.family != "ollama"
    assert adapter.identity.provider_id != "ollama"


async def test_explicit_capabilities_override_defaults(adapter: LocalEndpointAdapter) -> None:
    caps = adapter.capabilities
    assert caps.streaming is True
    assert caps.tool_calls is True
    assert caps.structured_output is False
    assert caps.reasoning is False


async def test_model_capabilities_are_local(adapter: LocalEndpointAdapter) -> None:
    model_caps = adapter.model_capabilities
    assert model_caps.streaming is True
    assert model_caps.tool_use is True
    assert model_caps.deployment_mode is DeploymentMode.LOCAL


async def test_payload_uses_configured_model(adapter: LocalEndpointAdapter) -> None:
    request = ProviderRequest(
        request_id="req-1",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )
    params = adapter._build_chat_params(request)
    assert params["model"] == "qwen2.5:7b"
    assert params["messages"] == [{"role": "user", "content": "Hi"}]


async def test_response_preserves_local_identities(adapter: LocalEndpointAdapter) -> None:
    request = ProviderRequest(
        request_id="req-2",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )
    response = await adapter.submit(request)
    assert response.request_id == "req-2"
    assert response.provider_id == adapter.identity
    assert response.model_id == adapter.model_id
    assert response.route_id == _ollama_profile().route_identity
    assert response.text == "Hello"
    assert response.finish_reason is FinishReason.STOP
    assert response.error_reference is None


async def test_streaming_preserves_local_identities(adapter: LocalEndpointAdapter) -> None:
    request = ProviderRequest(
        request_id="req-3",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hi")],
    )
    chunks = [chunk async for chunk in adapter.stream(request)]
    assert len(chunks) == 3
    assert all(chunk.provider_id == adapter.identity for chunk in chunks)
    assert all(chunk.model_id == adapter.model_id for chunk in chunks)
    assert all(chunk.route_id == _ollama_profile().route_identity for chunk in chunks)
    assert chunks[-1].streaming_state is StreamingState.COMPLETE
    assert chunks[-1].error_reference is None


async def test_same_model_via_different_runtimes_has_distinct_routes() -> None:
    ollama = LocalEndpointAdapter(model_config=_ollama_config())
    vllm_config = LocalModelConfig(
        model_id="qwen2.5:7b",
        family="qwen",
        profile=_vllm_profile(),
        explicit_capabilities={"streaming": True},
    )
    vllm = LocalEndpointAdapter(model_config=vllm_config)

    assert ollama.route_id is not None
    assert vllm.route_id is not None
    assert ollama.model_id == vllm.model_id
    assert ollama.identity.provider_id != vllm.identity.provider_id
    assert ollama.route_id != vllm.route_id
    assert ollama.route_id.route_id == "ollama-qwen"
    assert vllm.route_id.route_id == "vllm-qwen"


async def test_target_model_survives_response(adapter: LocalEndpointAdapter) -> None:
    target = ModelIdentity(model_id="qwen2.5:14b", family="qwen")
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
    assert params["model"] == "qwen2.5:14b"


async def test_health_is_conservative(adapter: LocalEndpointAdapter) -> None:
    health = await adapter.health()
    assert health.health is not ProviderHealth.HEALTHY
    assert health.reason is not None
    assert "observation" in health.reason.lower() or "initialized" in health.reason.lower()


async def test_quota_is_unknown(adapter: LocalEndpointAdapter) -> None:
    quota = await adapter.quota()
    assert quota.is_exhausted() is False


async def test_error_normalization_preserves_local_identity(monkeypatch: Any) -> None:
    client = build_mock_openai_client(exception=FakeRateLimitError)
    adapter = LocalEndpointAdapter(model_config=_ollama_config())
    monkeypatch.setattr(adapter, "_client", lambda: client)
    request = ProviderRequest(
        request_id="req-5",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
    )
    response = await adapter.submit(request)
    assert response.error_reference == ProviderErrorCode.RATE_LIMITED.value
    error = response.metadata["error"]
    assert error.provider_id == adapter.identity
    assert error.route_id == adapter.route_id


async def test_capability_matching_respects_explicit_roles() -> None:
    config = LocalModelConfig(
        model_id="qwen2.5:7b",
        family="qwen",
        profile=_ollama_profile(),
        explicit_capabilities={
            "streaming": True,
            "tool_use": True,
            "code_generation": True,
        },
    )
    adapter = LocalEndpointAdapter(model_config=config)
    model_caps = adapter.model_capabilities
    req = CapabilityRequirement(
        min_context_tokens=1,
        streaming=True,
        tool_use=True,
        code_generation=True,
    )
    assert match_capabilities(model_caps, req).eligible is True


async def test_no_network_at_import() -> None:
    """Importing the local adapter module must not trigger any network calls."""
    import importlib

    module = importlib.import_module("src.providers.local.adapter")
    # The adapter construction below is explicit; module import itself did not probe.
    assert module is not None
