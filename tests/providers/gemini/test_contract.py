"""Gemini adapter contract compliance using the shared suite."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from google.genai import types
from google.genai.errors import APIError

from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.gemini.adapter import GeminiAdapter
from tests.providers.adapter_contract_suite import AdapterContractSuite


def _text_response(text: str = "Hello world", response_id: str = "gemini-req-1") -> Any:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        response_id=response_id,
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        ),
    )


def _tool_response(response_id: str = "gemini-req-tool") -> Any:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id="fc-1",
                                name="read_file",
                                args={"path": "/etc/hosts"},
                            )
                        )
                    ],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        response_id=response_id,
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12,
            candidates_token_count=7,
            total_token_count=19,
        ),
    )


def _structured_response(response_id: str = "gemini-req-structured") -> Any:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text='{"steps": ["analyze", "plan"]}')],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        response_id=response_id,
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=8,
            candidates_token_count=6,
            total_token_count=14,
        ),
    )


def _stream_chunks(response_id: str = "gemini-req-stream") -> list[Any]:
    return [
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(role="model", parts=[types.Part(text="Hello")]),
                )
            ],
            response_id=response_id,
        ),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(role="model", parts=[types.Part(text=" world")]),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
            response_id=response_id,
        ),
    ]


class _FakeModels:
    def __init__(self) -> None:
        self.generate_content_calls: list[dict[str, Any]] = []
        self.generate_content_stream_calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.generate_content_calls.append(kwargs)
        config = kwargs.get("config")
        if config and getattr(config, "tools", None):
            return _tool_response()
        if config and getattr(config, "response_mime_type", None) == "application/json":
            return _structured_response()
        return _text_response()

    async def generate_content_stream(self, **kwargs: Any) -> AsyncIterator[Any]:
        self.generate_content_stream_calls.append(kwargs)

        async def _gen() -> AsyncIterator[Any]:
            for chunk in _stream_chunks():
                yield chunk

        return _gen()


class _FakeAio:
    def __init__(self) -> None:
        self.models = _FakeModels()


class _FakeClient:
    def __init__(self) -> None:
        self.aio = _FakeAio()


def _fake_client_factory() -> Any:
    return _FakeClient()


class TestGeminiAdapterContract(AdapterContractSuite):
    """Run the provider-neutral contract suite against ``GeminiAdapter``."""

    @pytest.fixture
    def adapter_factory(self) -> Callable[[], ProviderAdapter]:
        def _factory() -> ProviderAdapter:
            adapter = GeminiAdapter(api_key="fake-key")
            adapter._client = _fake_client_factory  # type: ignore[method-assign]
            return adapter

        return _factory

    @pytest.fixture
    def limited_adapter_factory(self) -> Callable[[], ProviderAdapter]:
        def _factory() -> ProviderAdapter:
            adapter = GeminiAdapter(
                api_key="fake-key",
                capabilities=ProviderAdapterCapabilities(
                    streaming=False,
                    tool_calls=False,
                    structured_output=False,
                    cancellation=True,
                ),
            )
            adapter._client = _fake_client_factory  # type: ignore[method-assign]
            return adapter

        return _factory

    @pytest.fixture
    def failing_adapter_factory(self) -> Callable[[], ProviderAdapter]:
        def _factory() -> ProviderAdapter:
            adapter = GeminiAdapter(api_key="fake-key")

            class _FailingFakeModels:
                async def generate_content(self, **kwargs: Any) -> Any:
                    raise APIError(
                        code=429,
                        response_json={"error": {"message": "rate limit exceeded"}},
                    )

                async def generate_content_stream(self, **kwargs: Any) -> AsyncIterator[Any]:
                    raise APIError(
                        code=429,
                        response_json={"error": {"message": "rate limit exceeded"}},
                    )

            failing_client = _FakeClient()
            failing_client.aio.models = _FailingFakeModels()  # type: ignore[assignment]
            adapter._client = lambda: failing_client  # type: ignore[method-assign]
            return adapter

        return _factory
