"""OpenAI adapter contract compliance using the shared suite."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.openai.adapter import OpenAIAdapter
from tests.providers.adapter_contract_suite import AdapterContractSuite


def _make_streaming_chunks(content: str = "Hello world") -> list[Any]:
    words = content.split()
    chunks: list[Any] = []
    for i, word in enumerate(words):
        is_last = i == len(words) - 1
        chunks.append(
            SimpleNamespace(
                id=f"stream-chunk-{i}",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=word + ("" if is_last else " ")),
                        finish_reason="stop" if is_last else None,
                    )
                ],
            )
        )
    return chunks


def _make_completion_response(
    content: str | None = "Hello",
    finish_reason: str = "stop",
    usage: dict[str, Any] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": "resp-openai-1",
        "choices": [
            {
                "message": {
                    "content": content,
                    "tool_calls": tool_calls,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _make_smart_client() -> MagicMock:
    """Return a mock OpenAI client that responds based on the request shape."""

    async def create(**kwargs: Any) -> Any:
        params = kwargs
        if params.get("stream"):
            chunks = _make_streaming_chunks()

            async def gen() -> AsyncIterator[Any]:
                for chunk in chunks:
                    yield chunk

            return gen()

        if params.get("tools"):
            return _make_completion_response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "tc-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            )

        if params.get("response_format"):
            return _make_completion_response(
                content='{"steps": ["analyze", "implement"]}',
                finish_reason="stop",
            )

        return _make_completion_response()

    client = MagicMock()
    client.chat.completions.create = create
    return client


def _make_failing_client() -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = MagicMock(side_effect=Exception("Rate limit exceeded"))
    return client


@pytest.fixture
def openai_adapter_factory() -> Callable[[], ProviderAdapter]:
    def factory() -> ProviderAdapter:
        adapter = OpenAIAdapter(api_key="sk-test")
        adapter._client = lambda: _make_smart_client()  # type: ignore[method-assign]
        return adapter

    return factory


@pytest.fixture
def limited_openai_adapter_factory() -> Callable[[], ProviderAdapter]:
    def factory() -> ProviderAdapter:
        adapter = OpenAIAdapter(
            api_key="sk-test",
            capabilities=ProviderAdapterCapabilities(
                streaming=False,
                tool_calls=False,
                structured_output=False,
                cancellation=True,
            ),
        )
        adapter._client = lambda: _make_smart_client()  # type: ignore[method-assign]
        return adapter

    return factory


@pytest.fixture
def failing_openai_adapter_factory() -> Callable[[], ProviderAdapter]:
    def factory() -> ProviderAdapter:
        adapter = OpenAIAdapter(api_key="sk-test")
        adapter._client = lambda: _make_failing_client()  # type: ignore[method-assign]
        return adapter

    return factory


class TestOpenAIAdapterContract(AdapterContractSuite):
    """Run the provider-neutral contract suite against OpenAIAdapter."""

    @pytest.fixture
    def adapter_factory(
        self, openai_adapter_factory: Callable[[], ProviderAdapter]
    ) -> Callable[[], ProviderAdapter]:
        return openai_adapter_factory

    @pytest.fixture
    def limited_adapter_factory(
        self, limited_openai_adapter_factory: Callable[[], ProviderAdapter]
    ) -> Callable[[], ProviderAdapter]:
        return limited_openai_adapter_factory

    @pytest.fixture
    def failing_adapter_factory(
        self, failing_openai_adapter_factory: Callable[[], ProviderAdapter]
    ) -> Callable[[], ProviderAdapter]:
        return failing_openai_adapter_factory
