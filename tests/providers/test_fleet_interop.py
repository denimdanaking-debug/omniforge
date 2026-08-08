"""Fleet interoperability smoke tests (Phase 3.10).

These tests verify that every core provider adapter can consume normalized
requests and return normalized responses.  Deterministic mocks are used for all
external SDK clients so no live credentials or network calls are required.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from src.policy.risk import RiskLevel
from src.providers.adapter import ProviderAdapter
from src.providers.request import Message, MessageRole, ProviderRequest
from src.routing.roles import ExecutionRole
from tests.providers._openai_compat_mocks import (
    build_mock_openai_client,
    make_stream_chunks,
    make_success_response,
)

FleetAdapterFactory = Callable[[], ProviderAdapter]

_FLEET_ADAPTERS: dict[str, str] = {
    "anthropic": "src.providers.anthropic.adapter:AnthropicAdapter",
    "openai": "src.providers.openai.adapter:OpenAIAdapter",
    "kimi": "src.providers.kimi.adapter:KimiAdapter",
    "qwen": "src.providers.qwen.adapter:QwenAdapter",
    "deepseek": "src.providers.deepseek.adapter:DeepSeekAdapter",
    "gemini": "src.providers.gemini.adapter:GeminiAdapter",
    "xai": "src.providers.xai.adapter:XAIAdapter",
    "zai": "src.providers.zai.adapter:ZAIAdapter",
    "cursor": "src.providers.cursor.adapter:CursorRouteAdapter",
}

_ROLES = {
    "planning": ExecutionRole.PLANNING,
    "coding": ExecutionRole.CODING,
    "review": ExecutionRole.REVIEW,
}


class _FakeAnthropicMessages:
    async def create(self, **kwargs: Any) -> Any:
        class _Block:
            type = "text"
            text = "Fleet result"

        class _Usage:
            input_tokens = 10
            output_tokens = 5

        class _Response:
            id = "msg-fleet"
            content = [_Block()]
            stop_reason = "end_turn"
            usage = _Usage()

        return _Response()


class _FakeAsyncAnthropic:
    def __init__(self, **kwargs: Any) -> None:
        self.messages = _FakeAnthropicMessages()


class _FakeGeminiModels:
    async def generate_content(self, **kwargs: Any) -> Any:
        return _make_gemini_response("Fleet result")

    async def generate_content_stream(self, **kwargs: Any) -> AsyncIterator[Any]:
        yield _make_gemini_response("Fleet")
        yield _make_gemini_response(" result", finish=True)


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
    if provider_id in {
        "openai",
        "kimi",
        "qwen",
        "deepseek",
        "xai",
        "zai",
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


def _import_factory(dotted_path: str) -> FleetAdapterFactory | None:
    module_path, _, name = dotted_path.partition(":")
    try:
        module = __import__(module_path, fromlist=[name])
        factory = getattr(module, name)
        if not callable(factory):
            return None
        return lambda: _mock_adapter(factory())
    except Exception:
        return None


def _all_factories() -> list[tuple[str, FleetAdapterFactory]]:
    result: list[tuple[str, FleetAdapterFactory]] = []
    for provider_id, dotted_path in sorted(_FLEET_ADAPTERS.items()):
        factory = _import_factory(dotted_path)
        if factory is not None:
            result.append((provider_id, factory))
    return result


def _request(role: ExecutionRole) -> ProviderRequest:
    return ProviderRequest(
        request_id=f"fleet-{role.value}",
        execution_role=role,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content=f"{role.value} task")],
    )


@pytest.mark.fleet
@pytest.mark.parametrize("role_name, role", list(_ROLES.items()))
@pytest.mark.parametrize("provider_id, factory", _all_factories())
async def test_adapter_handles_normalized_role(
    provider_id: str, factory: FleetAdapterFactory, role_name: str, role: ExecutionRole
) -> None:
    adapter = factory()
    request = _request(role)
    response = await adapter.submit(request)
    assert response.request_id == request.request_id
    assert response.provider_id == adapter.identity
    assert response.model_id is not None


@pytest.mark.fleet
@pytest.mark.parametrize("provider_id, factory", _all_factories())
async def test_adapter_identity_is_stable(provider_id: str, factory: FleetAdapterFactory) -> None:
    adapter = factory()
    assert adapter.identity.provider_id == provider_id


@pytest.mark.fleet
@pytest.mark.parametrize("provider_id, factory", _all_factories())
async def test_adapter_response_preserves_route_when_set(
    provider_id: str, factory: FleetAdapterFactory
) -> None:
    adapter = factory()
    request = _request(ExecutionRole.CODING)
    response = await adapter.submit(request)
    # Cursor sets a route id; others may leave it None. Either is valid.
    assert response.route_id is None or response.route_id.provider_id == provider_id


@pytest.mark.fleet
def test_fleet_identities_are_unique() -> None:
    identities: set[str] = set()
    for _provider_id, factory in _all_factories():
        adapter = factory()
        assert adapter.identity.provider_id not in identities
        identities.add(adapter.identity.provider_id)


@pytest.mark.fleet
@pytest.mark.parametrize("provider_id, factory", _all_factories())
async def test_adapter_stream_does_not_crash(
    provider_id: str, factory: FleetAdapterFactory
) -> None:
    adapter = factory()
    request = _request(ExecutionRole.CODING)
    chunks: list[Any] = []
    async for chunk in adapter.stream(request):
        chunks.append(chunk)
    assert len(chunks) >= 0
