"""Fleet interoperability smoke tests (Phase 3.10).

These tests verify that every core provider adapter can consume normalized
requests and return normalized responses.  Deterministic mocks are used for all
external SDK clients so no live credentials or network calls are required.

A provider adapter that cannot be imported or instantiated is a hard failure;
the required Phase 3 core fleet must never silently disappear from the matrix.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from src.policy.risk import RiskLevel
from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.errors import ProviderErrorCode
from src.providers.generic_openai_compat.adapter import GenericOpenAICompatibleAdapter
from src.providers.identity import ProviderIdentity
from src.providers.local.adapter import LocalEndpointAdapter
from src.providers.local.profile import LocalEndpointProfile, LocalModelConfig, LocalRuntimeKind
from src.providers.minimax.adapter import MiniMaxAdapter
from src.providers.mistral.adapter import MistralAdapter
from src.providers.openrouter.adapter import OpenRouterAdapter
from src.providers.request import (
    CapabilityRequirement,
    Message,
    MessageRole,
    ProviderRequest,
    StructuredOutputRequirement,
    ToolChoiceMode,
    ToolDefinition,
    ToolParameter,
)
from src.providers.response import FinishReason, ProviderResponse, StreamingState
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity, ModelLifecycle
from src.routing.roles import ExecutionRole
from tests.providers._openai_compat_mocks import (
    build_mock_openai_client,
    make_stream_chunks,
    make_success_response,
)

FleetAdapterFactory = Callable[[], ProviderAdapter]


def _local_profile() -> LocalEndpointProfile:
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


def _openrouter_route() -> InferenceRouteIdentity:
    return InferenceRouteIdentity(
        route_id="openrouter-claude",
        provider_id="openrouter",
        route_type=RouteType.GATEWAY,
        endpoint_key="openrouter://anthropic/claude-sonnet-4-20250514",
        failure_domain="openrouter.ai",
    )


_FLEET_ADAPTERS: dict[str, str | FleetAdapterFactory] = {
    "anthropic": "src.providers.anthropic.adapter:AnthropicAdapter",
    "openai": "src.providers.openai.adapter:OpenAIAdapter",
    "kimi": "src.providers.kimi.adapter:KimiAdapter",
    "qwen": "src.providers.qwen.adapter:QwenAdapter",
    "deepseek": "src.providers.deepseek.adapter:DeepSeekAdapter",
    "gemini": "src.providers.gemini.adapter:GeminiAdapter",
    "xai": "src.providers.xai.adapter:XAIAdapter",
    "zai": "src.providers.zai.adapter:ZAIAdapter",
    "cursor": "src.providers.cursor.adapter:CursorRouteAdapter",
    "minimax": lambda: MiniMaxAdapter(api_key="test-key"),
    "mistral": lambda: MistralAdapter(api_key="test-key"),
    "openrouter": lambda: OpenRouterAdapter(
        provider_identity=ProviderIdentity("anthropic", "Anthropic", "anthropic.com"),
        model_identity=ModelIdentity(
            model_id="claude-sonnet-4-20250514",
            family="claude",
            lifecycle=ModelLifecycle.HIGH_RISK,
        ),
        route_identity=_openrouter_route(),
        api_key="test-key",
    ),
    "generic_openai_compat": lambda: GenericOpenAICompatibleAdapter(
        provider_identity=ProviderIdentity("acme", "Acme", "acme.example"),
        model_identity=ModelIdentity(model_id="acme-model", family="acme"),
        base_url="https://acme.example/v1",
        api_key="test-key",
        capabilities=ProviderAdapterCapabilities(
            streaming=True, tool_calls=True, structured_output=True
        ),
    ),
    "local_endpoint": lambda: LocalEndpointAdapter(
        model_config=LocalModelConfig(
            model_id="qwen2.5:7b",
            family="qwen",
            profile=_local_profile(),
            explicit_capabilities={"streaming": True, "tool_use": True, "structured_output": True},
        )
    ),
}

_REQUIRED_PROVIDER_IDS: frozenset[str] = frozenset(_FLEET_ADAPTERS)
_EXECUTABLE_PROVIDER_IDS: frozenset[str] = _REQUIRED_PROVIDER_IDS - {"cursor"}

_ROLES = {
    "planning": ExecutionRole.PLANNING,
    "coding": ExecutionRole.CODING,
    "review": ExecutionRole.REVIEW,
}


class _FakeAnthropicBlock:
    type = "text"
    text = "Fleet result"


class _FakeAnthropicUsage:
    input_tokens = 10
    output_tokens = 5


class _FakeAnthropicMessage:
    id = "msg-fleet"
    content = [_FakeAnthropicBlock()]
    stop_reason = "end_turn"
    usage = _FakeAnthropicUsage()


class _FakeAnthropicTextDelta:
    type = "text_delta"
    text = "Fleet result"


class _FakeAnthropicContentBlockDelta:
    type = "content_block_delta"
    delta = _FakeAnthropicTextDelta()


class _FakeAnthropicStopDelta:
    stop_reason = "end_turn"


class _FakeAnthropicMessageDelta:
    type = "message_delta"
    delta = _FakeAnthropicStopDelta()
    usage = _FakeAnthropicUsage()


class _FakeAnthropicStream:
    def __init__(self) -> None:
        self._events = [
            type(
                "_MessageStart", (), {"type": "message_start", "message": _FakeAnthropicMessage()}
            )(),
            _FakeAnthropicContentBlockDelta(),
            _FakeAnthropicMessageDelta(),
        ]
        self._index = 0

    def __aiter__(self) -> _FakeAnthropicStream:
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event


class _FakeAnthropicMessages:
    async def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return _FakeAnthropicStream()

        return _FakeAnthropicMessage()


class _FakeAsyncAnthropic:
    def __init__(self, **kwargs: Any) -> None:
        self.messages = _FakeAnthropicMessages()


class _FakeGeminiModels:
    async def generate_content(self, **kwargs: Any) -> Any:
        return _make_gemini_response("Fleet result", finish=True)

    async def generate_content_stream(self, **kwargs: Any) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            yield _make_gemini_response("Fleet")
            yield _make_gemini_response(" result", finish=True)

        return _gen()


class _FakeGeminiAio:
    def __init__(self) -> None:
        self.models = _FakeGeminiModels()


class _FakeGeminiClient:
    def __init__(self) -> None:
        self.aio = _FakeGeminiAio()


def _make_gemini_response(text: str, finish: bool = False) -> Any:
    from google.genai import types

    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                finish_reason=types.FinishReason.STOP if finish else None,
            )
        ],
        response_id="gemini-fleet",
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        ),
    )


def _mock_openai_compat_adapter(adapter: ProviderAdapter) -> ProviderAdapter:
    """Attach a mock OpenAI-compatible client to an adapter instance."""
    client = build_mock_openai_client(
        response=make_success_response(
            content='{"steps": []}',
            tool_calls=[
                {
                    "id": "tc-1",
                    "function": {"name": "read_file", "arguments": '{"path": "/tmp"}'},
                }
            ],
        ),
        stream_chunks=make_stream_chunks(["Fleet", " result"]),
    )
    adapter._client = lambda: client  # type: ignore
    return adapter


def _mock_adapter(adapter: ProviderAdapter) -> ProviderAdapter:
    """Attach deterministic SDK mocks to ``adapter`` based on its provider."""
    provider_id = adapter.identity.provider_id
    # OpenRouter uses OpenAI-compatible transport even though its underlying
    # provider identity may be Anthropic/Qwen/etc.
    if isinstance(
        adapter, (OpenRouterAdapter, GenericOpenAICompatibleAdapter, LocalEndpointAdapter)
    ):
        return _mock_openai_compat_adapter(adapter)
    if provider_id in {
        "openai",
        "kimi",
        "qwen",
        "deepseek",
        "xai",
        "zai",
        "minimax",
        "mistral",
    }:
        return _mock_openai_compat_adapter(adapter)
    if provider_id == "anthropic":
        adapter._client = lambda: _FakeAsyncAnthropic()  # type: ignore
        return adapter
    if provider_id == "gemini":
        adapter._client = lambda: _FakeGeminiClient()  # type: ignore
        return adapter
    # Cursor and other route adapters need no SDK mocking.
    return adapter


def _import_factory(factory_source: str | FleetAdapterFactory) -> FleetAdapterFactory:
    """Resolve a fleet adapter factory from a dotted path or callable."""
    if callable(factory_source):
        return lambda: _mock_adapter(factory_source())

    dotted_path = factory_source
    module_path, _, name = dotted_path.partition(":")
    try:
        module = __import__(module_path, fromlist=[name])
    except Exception as exc:
        raise AssertionError(
            f"Required fleet adapter {module_path!r} cannot be imported: {exc}"
        ) from exc
    try:
        factory = getattr(module, name)
    except AttributeError as exc:
        raise AssertionError(
            f"Required fleet adapter {dotted_path!r} missing symbol {name!r}"
        ) from exc
    if not callable(factory):
        raise AssertionError(f"Required fleet adapter {dotted_path!r} is not callable")
    return lambda: _mock_adapter(factory())


def _all_factories() -> list[tuple[str, FleetAdapterFactory]]:
    result: list[tuple[str, FleetAdapterFactory]] = []
    for provider_id, factory_source in sorted(_FLEET_ADAPTERS.items()):
        factory = _import_factory(factory_source)
        result.append((provider_id, factory))
    return result


def _planning_request() -> ProviderRequest:
    return ProviderRequest(
        request_id="fleet-planning",
        execution_role=ExecutionRole.PLANNING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="plan task")],
        structured_output=StructuredOutputRequirement(
            schema={"type": "object", "properties": {"steps": {"type": "array"}}},
            name="plan",
        ),
        capability_requirements=[
            CapabilityRequirement(feature="structured_output", required=True),
            CapabilityRequirement(feature="streaming", required=True),
        ],
    )


def _coding_request() -> ProviderRequest:
    return ProviderRequest(
        request_id="fleet-coding",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="coding task")],
        tools=[
            ToolDefinition(
                name="read_file",
                description="Read a file",
                parameters=[
                    ToolParameter(
                        name="path",
                        schema={"type": "string"},
                        required=True,
                    )
                ],
            )
        ],
        tool_choice=ToolChoiceMode.REQUIRED,
        capability_requirements=[
            CapabilityRequirement(feature="tool_use", required=True),
            CapabilityRequirement(feature="streaming", required=True),
        ],
    )


def _review_request() -> ProviderRequest:
    return ProviderRequest(
        request_id="fleet-review",
        execution_role=ExecutionRole.REVIEW,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="review task")],
        capability_requirements=[
            CapabilityRequirement(feature="streaming", required=True),
        ],
    )


def _request_for_role(role: ExecutionRole) -> ProviderRequest:
    if role is ExecutionRole.PLANNING:
        return _planning_request()
    if role is ExecutionRole.CODING:
        return _coding_request()
    return _review_request()


def _assert_successful_executable_response(
    response: ProviderResponse,
    adapter: ProviderAdapter,
    request: ProviderRequest,
) -> None:
    """Validate a non-error response from an executable adapter."""
    assert response.request_id == request.request_id, "request id mismatch"
    assert response.provider_id == adapter.identity, "provider identity mismatch"
    assert response.model_id is not None, "model identity missing"
    assert response.error_reference is None, f"unexpected error: {response.error_reference}"
    assert response.finish_reason in {
        FinishReason.STOP,
        FinishReason.TOOL_CALLS,
    }, f"unexpected finish reason: {response.finish_reason}"
    assert response.provider_request_id is not None, "provider request id missing"
    # SDK-specific types must not escape normalization.
    assert response.text is None or isinstance(response.text, str)
    assert response.structured_result is None or isinstance(response.structured_result, dict)


def _assert_cursor_unsupported(response: ProviderResponse) -> None:
    """Cursor must report a normalized unsupported-capability response."""
    assert response.error_reference is not None
    assert response.error_reference == ProviderErrorCode.UNSUPPORTED_CAPABILITY.value


@pytest.mark.fleet
def test_required_fleet_adapters_are_all_present() -> None:
    """Every required Phase 3 adapter must import and instantiate."""
    loaded = {provider_id for provider_id, _factory in _all_factories()}
    missing = _REQUIRED_PROVIDER_IDS - loaded
    assert loaded == _REQUIRED_PROVIDER_IDS, f"missing required adapters: {sorted(missing)}"


@pytest.mark.fleet
@pytest.mark.parametrize("role_name, role", list(_ROLES.items()))
@pytest.mark.parametrize("provider_id, factory", _all_factories())
async def test_adapter_handles_normalized_role(
    provider_id: str, factory: FleetAdapterFactory, role_name: str, role: ExecutionRole
) -> None:
    adapter = factory()
    request = _request_for_role(role)
    response = await adapter.submit(request)

    if provider_id == "cursor":
        _assert_cursor_unsupported(response)
        return

    _assert_successful_executable_response(response, adapter, request)
    if role is ExecutionRole.PLANNING:
        assert response.text is not None or response.has_structured_result(), (
            "planning response should contain useful content"
        )
    elif role is ExecutionRole.CODING:
        assert response.text is not None or response.has_tool_calls(), (
            "coding response should contain useful content or tool calls"
        )
    else:
        assert response.text is not None, "review response should contain text"


@pytest.mark.fleet
@pytest.mark.parametrize("provider_id, factory", _all_factories())
async def test_adapter_identity_is_stable(provider_id: str, factory: FleetAdapterFactory) -> None:
    adapter = factory()
    assert adapter.identity.provider_id
    # Identity must remain the same across repeated access.
    assert adapter.identity == adapter.identity


@pytest.mark.fleet
@pytest.mark.parametrize("provider_id, factory", _all_factories())
async def test_adapter_response_preserves_route_when_set(
    provider_id: str, factory: FleetAdapterFactory
) -> None:
    adapter = factory()
    route = InferenceRouteIdentity(
        route_id="fleet-route",
        provider_id=provider_id,
        route_type=RouteType.DIRECT,
        endpoint_key=f"{provider_id}://fleet",
        failure_domain="fleet.test",
    )
    if provider_id == "cursor":
        # Cursor carries its own fixed route identity; verify it is preserved.
        request = _review_request()
        response = await adapter.submit(request)
        _assert_cursor_unsupported(response)
        assert response.route_id is not None
        assert response.route_id.route_id == "cursor-local"
        return

    adapter._route_id = route  # type: ignore[attr-defined]
    request = _review_request()
    response = await adapter.submit(request)

    _assert_successful_executable_response(response, adapter, request)
    assert response.route_id == route, "route identity must be preserved in response"


@pytest.mark.fleet
@pytest.mark.parametrize("provider_id, factory", _all_factories())
async def test_adapter_response_preserves_target_model(
    provider_id: str, factory: FleetAdapterFactory
) -> None:
    adapter = factory()
    if hasattr(adapter, "model_id"):
        family = adapter.model_id.family
    elif hasattr(adapter, "model_identity"):
        family = adapter.model_identity.family
    else:
        family = provider_id
    target_model = ModelIdentity(
        model_id=f"{provider_id}-target-model",
        family=family,
    )
    request = dataclasses.replace(_review_request(), target_model=target_model)
    response = await adapter.submit(request)

    if provider_id == "cursor":
        _assert_cursor_unsupported(response)
        return

    _assert_successful_executable_response(response, adapter, request)
    assert response.model_id == target_model, "target model identity must survive response"


@pytest.mark.fleet
def test_fleet_identities_are_unique() -> None:
    # A provider identity may legitimately appear through multiple routes
    # (e.g. Anthropic direct + Anthropic via OpenRouter), so uniqueness is
    # enforced on the (provider_id, route_id) pair.
    identities: set[tuple[str, str | None]] = set()
    for _provider_id, factory in _all_factories():
        adapter = factory()
        route = getattr(adapter, "route_id", None)
        route_id = route.route_id if route is not None else None
        key = (adapter.identity.provider_id, route_id)
        assert key not in identities, f"duplicate fleet identity/route pair: {key}"
        identities.add(key)


@pytest.mark.fleet
@pytest.mark.parametrize("provider_id, factory", _all_factories())
async def test_adapter_stream_does_not_crash(
    provider_id: str, factory: FleetAdapterFactory
) -> None:
    adapter = factory()
    request = _review_request()
    chunks: list[ProviderResponse] = []
    async for chunk in adapter.stream(request):
        chunks.append(chunk)

    if provider_id == "cursor":
        assert len(chunks) == 1
        assert chunks[0].error_reference == ProviderErrorCode.UNSUPPORTED_CAPABILITY.value
        assert chunks[0].streaming_state is StreamingState.FAILED
        return

    assert len(chunks) >= 1, "streaming adapters must emit at least one chunk"
    assert chunks[-1].streaming_state is StreamingState.COMPLETE, "final chunk must be COMPLETE"
    assert chunks[-1].error_reference is None, "successful stream final chunk must not be an error"
    assert chunks[-1].request_id == request.request_id
    assert chunks[-1].provider_id == adapter.identity
    assert chunks[-1].model_id is not None
    assert all(chunk.provider_id == adapter.identity for chunk in chunks)
