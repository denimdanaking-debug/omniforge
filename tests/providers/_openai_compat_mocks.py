"""Shared mocks for OpenAI-compatible provider adapter tests.

OpenAI-compatible adapters lazily import the ``openai`` SDK inside their
``_client()`` method. Tests create a mock client and assign it to the adapter
instance via ``adapter._client = lambda: client``. This avoids import-path
shadowing issues and keeps tests deterministic with no network calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock


def make_success_response(
    content: str | None = "Hello",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    """Build a mocked OpenAI chat completion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.to_dict.return_value = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.id = "resp-test"
    return response


def make_stream_chunks(contents: list[str | None]) -> list[MagicMock]:
    """Build mocked OpenAI streaming chunks from text fragments."""
    chunks: list[MagicMock] = []
    for index, content in enumerate(contents):
        delta = MagicMock()
        delta.content = content

        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = "stop" if index == len(contents) - 1 else None

        chunk = MagicMock()
        chunk.choices = [choice]
        chunk.id = "chunk-test"
        chunks.append(chunk)
    return chunks


class FakeRateLimitError(Exception):
    """Exception name contains ``RateLimit`` so translation maps to RATE_LIMITED."""


class FakeAuthError(Exception):
    """Exception name contains ``Auth`` so translation maps to AUTH_FAILURE."""


def build_mock_openai_client(
    *,
    response: MagicMock | None = None,
    stream_chunks: list[MagicMock] | None = None,
    exception: type[Exception] | Exception | None = None,
) -> MagicMock:
    """Return a configured mock ``openai.AsyncOpenAI`` client.

    When ``stream_chunks`` is supplied alongside ``response``, the mock routes
    calls based on the ``stream`` keyword argument so the same client can serve
    both ``submit`` and ``stream`` adapter methods.
    """
    client = MagicMock()
    chat = MagicMock()
    completions = MagicMock()

    if exception is not None:
        exc = exception("boom") if isinstance(exception, type) else exception

        async def _create(**kwargs: Any) -> Any:
            raise exc
    elif stream_chunks is not None:
        stream_coro = _stream_generator(stream_chunks)

        async def _create(**kwargs: Any) -> Any:
            if kwargs.get("stream"):
                return stream_coro
            return response
    else:

        async def _create(**kwargs: Any) -> Any:
            return response

    completions.create = _create
    chat.completions = completions
    client.chat = chat
    return client


async def _stream_generator(chunks: list[MagicMock]) -> AsyncIterator[MagicMock]:
    for chunk in chunks:
        yield chunk
