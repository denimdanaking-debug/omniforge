"""Unit tests for local-model capability probing."""

from __future__ import annotations

from typing import Any

from src.providers.errors import ProviderErrorCode
from src.providers.local.probe import (
    CapabilityProbeResult,
    CapabilitySource,
    LocalCapabilityProber,
)
from src.providers.local.profile import LocalEndpointProfile, LocalModelConfig, LocalRuntimeKind
from src.routing.capabilities import DeploymentMode, ModelCapabilities
from src.routing.inference_route import InferenceRouteIdentity, RouteType


def _profile() -> LocalEndpointProfile:
    return LocalEndpointProfile(
        runtime_kind=LocalRuntimeKind.VLLM,
        base_url="http://localhost:8000/v1",
        route_identity=InferenceRouteIdentity(
            route_id="vllm-test",
            provider_id="local-vllm",
            route_type=RouteType.LOCAL,
            endpoint_key="http://localhost:8000/v1",
            failure_domain="localhost:8000",
        ),
        failure_domain="localhost:8000",
        models_endpoint="/v1/models",
    )


def _model_config(**explicit: Any) -> LocalModelConfig:
    return LocalModelConfig(
        model_id="qwen2.5:7b",
        family="qwen",
        profile=_profile(),
        explicit_capabilities=explicit,
    )


class _FakeHttpClient:
    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.urls: list[str] = []

    async def get(self, url: str) -> Any:
        self.urls.append(url)
        return self._payload


class _FailingHttpClient:
    def __init__(self, exception: Exception) -> None:
        self._exception = exception

    async def get(self, url: str) -> dict[str, Any] | None:
        raise self._exception


def _model_payload(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "qwen2.5:7b",
        "context_length": 32768,
        "supports_tool_calls": True,
        "supports_streaming": True,
        "supports_structured_output": False,
        **overrides,
    }


def _evidence_by_capability(result: CapabilityProbeResult) -> dict[str, Any]:
    return {e.capability: e for e in result.evidence}


async def test_explicit_config_overrides_probe() -> None:
    client = _FakeHttpClient({"data": [_model_payload()]})
    prober = LocalCapabilityProber(http_client=client)
    result = await prober.probe_model_config(
        _model_config(
            streaming=False,
            tool_use=False,
        )
    )
    caps = result.capabilities
    assert caps.streaming is False
    assert caps.tool_use is False

    evidence = _evidence_by_capability(result)
    assert evidence["streaming"].source is CapabilitySource.EXPLICIT_CONFIG
    assert evidence["tool_use"].source is CapabilitySource.EXPLICIT_CONFIG


async def test_probe_result_records_model_listed() -> None:
    client = _FakeHttpClient({"data": [_model_payload()]})
    prober = LocalCapabilityProber(http_client=client)
    result = await prober.probe_model_config(_model_config())
    evidence = _evidence_by_capability(result)
    assert evidence["model_listed"].supported is True
    assert evidence["model_listed"].source is CapabilitySource.PROBE_RESULT


async def test_provider_metadata_maps_into_canonical_capabilities() -> None:
    client = _FakeHttpClient({"data": [_model_payload()]})
    prober = LocalCapabilityProber(http_client=client)
    result = await prober.probe_model_config(_model_config())
    caps = result.capabilities
    assert isinstance(caps, ModelCapabilities)
    assert caps.context_tokens == 32768
    assert caps.tool_use is True
    assert caps.streaming is True
    assert caps.structured_output is False
    assert caps.deployment_mode is DeploymentMode.LOCAL
    assert result.model_identity.model_id == "qwen2.5:7b"

    evidence = _evidence_by_capability(result)
    assert evidence["tool_use"].source is CapabilitySource.PROVIDER_METADATA
    assert evidence["streaming"].source is CapabilitySource.PROVIDER_METADATA
    assert evidence["structured_output"].source is CapabilitySource.PROVIDER_METADATA


async def test_absent_metadata_does_not_fabricate_support() -> None:
    client = _FakeHttpClient({"data": [{}]})
    prober = LocalCapabilityProber(http_client=client)
    result = await prober.probe_model_config(_model_config())
    caps = result.capabilities
    assert caps.tool_use is False
    assert caps.streaming is False
    assert caps.structured_output is False
    assert caps.reasoning is False
    assert caps.code_generation is False
    assert caps.multimodal is False

    evidence = _evidence_by_capability(result)
    for capability in (
        "tool_use",
        "streaming",
        "structured_output",
        "reasoning",
        "code_generation",
        "multimodal",
    ):
        assert evidence[capability].supported is None
        assert evidence[capability].source is CapabilitySource.UNKNOWN


async def test_unknown_capability_never_defaults_optimistically() -> None:
    """code_generation must not become True just because evidence is absent."""
    client = _FakeHttpClient({"data": [{}]})
    prober = LocalCapabilityProber(http_client=client)
    result = await prober.probe_model_config(_model_config())
    assert result.capabilities.code_generation is False
    evidence = _evidence_by_capability(result)
    assert evidence["code_generation"].supported is None
    assert evidence["code_generation"].source is CapabilitySource.UNKNOWN


async def test_context_fallback_is_marked_unverified() -> None:
    client = _FakeHttpClient({"data": [{}]})
    prober = LocalCapabilityProber(http_client=client)
    result = await prober.probe_model_config(_model_config())
    assert result.capabilities.context_tokens == 4096
    evidence = _evidence_by_capability(result)
    assert evidence["context_tokens"].source is CapabilitySource.UNKNOWN
    assert "not verified" in evidence["context_tokens"].metadata["reason"]


async def test_probe_transport_failure_is_normalized() -> None:
    client = _FailingHttpClient(ConnectionError("connection refused"))
    prober = LocalCapabilityProber(http_client=client)
    result = await prober.probe_model_config(_model_config())

    assert len(result.errors) == 1
    error = result.errors[0]
    assert error.code is ProviderErrorCode.TRANSIENT_TRANSPORT
    assert error.route_id == _profile().route_identity
    assert error.provider_id is not None
    assert error.provider_id.provider_id == "local-vllm"

    # Failure does not mark capabilities as verified unsupported.
    evidence = _evidence_by_capability(result)
    assert evidence["model_listed"].supported is False
    assert evidence["model_listed"].source is CapabilitySource.PROBE_RESULT
    assert evidence["tool_use"].source is CapabilitySource.UNKNOWN


async def test_probe_invalid_metadata_response_is_normalized() -> None:
    client = _FakeHttpClient("not-json")
    prober = LocalCapabilityProber(http_client=client)
    result = await prober.probe_model_config(_model_config())

    assert len(result.errors) == 1
    assert result.errors[0].code is ProviderErrorCode.UNKNOWN


async def test_probe_missing_http_client_is_normalized() -> None:
    prober = LocalCapabilityProber(http_client=None)
    result = await prober.probe_model_config(_model_config())

    assert len(result.errors) == 1
    assert result.errors[0].code is ProviderErrorCode.UNKNOWN
    evidence = _evidence_by_capability(result)
    assert evidence["tool_use"].source is CapabilitySource.UNKNOWN


async def test_probe_failure_does_not_mark_capabilities_unsupported_as_fact() -> None:
    client = _FailingHttpClient(ConnectionError("connection refused"))
    prober = LocalCapabilityProber(http_client=client)
    result = await prober.probe_model_config(_model_config())

    evidence = _evidence_by_capability(result)
    for capability in ("tool_use", "streaming", "structured_output"):
        assert evidence[capability].supported is None
        assert evidence[capability].source is CapabilitySource.UNKNOWN


async def test_different_local_routes_for_same_model_remain_separate() -> None:
    ollama_profile = LocalEndpointProfile(
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
    ollama_config = LocalModelConfig(
        model_id="qwen2.5:7b",
        family="qwen",
        profile=ollama_profile,
        explicit_capabilities={"streaming": True},
    )
    vllm_config = _model_config(streaming=True)

    prober = LocalCapabilityProber()
    ollama_result = await prober.probe_model_config(ollama_config)
    vllm_result = await prober.probe_model_config(vllm_config)

    assert ollama_result.model_identity == vllm_result.model_identity
    assert ollama_result.capabilities.deployment_mode is DeploymentMode.LOCAL
    assert vllm_result.capabilities.deployment_mode is DeploymentMode.LOCAL


async def test_no_destructive_prompts_sent() -> None:
    client = _FakeHttpClient({"data": [_model_payload()]})
    prober = LocalCapabilityProber(http_client=client)
    await prober.probe_model_config(_model_config())
    assert len(client.urls) == 1
    assert "/v1/models" in client.urls[0]
    assert "chat/completions" not in client.urls[0]
