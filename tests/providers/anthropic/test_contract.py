"""Anthropic adapter contract compliance using the shared suite."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.anthropic import AnthropicAdapter
from tests.providers.adapter_contract_suite import AdapterContractSuite


class FakeAnthropicErrorModule:
    """Minimal stand-in for the ``anthropic`` SDK exception namespace."""

    class APIStatusError(Exception):
        def __init__(self, message: str, *, response: Any | None = None, body: Any | None = None):
            super().__init__(message)
            self.response = response
            self.status_code = response.status_code if response is not None else None

    class AuthenticationError(APIStatusError):
        pass

    class RateLimitError(APIStatusError):
        pass

    class BadRequestError(APIStatusError):
        pass

    class InternalServerError(APIStatusError):
        pass

    class APIConnectionError(Exception):
        def __init__(self, message: str, *, request: Any | None = None):
            super().__init__(message)
            self.request = request


@pytest.fixture(autouse=True)
def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", FakeAnthropicErrorModule())


class FakeTextBlock:
    type = "text"
    text = "Hello from Claude"


class FakeJsonBlock:
    type = "text"
    text = '{"steps": []}'


class FakeToolUseBlock:
    type = "tool_use"
    id = "tu_01Test"
    name = "read_file"
    input = {"path": "README.md"}


class FakeUsage:
    input_tokens = 12
    output_tokens = 6


class FakeMessage:
    id = "msg_01Test"
    content: list[Any] = [FakeTextBlock()]
    stop_reason = "end_turn"
    usage = FakeUsage()


class FakeToolMessage(FakeMessage):
    content = [FakeToolUseBlock()]
    stop_reason = "tool_use"


class FakeStructuredMessage(FakeMessage):
    content = [FakeJsonBlock()]


@dataclass
class FakeTextDelta:
    type = "text_delta"
    text: str


@dataclass
class FakeContentBlockDeltaEvent:
    type = "content_block_delta"
    delta: Any


class FakeStopDelta:
    stop_reason = "end_turn"


@dataclass
class FakeMessageDeltaEvent:
    type = "message_delta"
    delta: Any = FakeStopDelta()
    usage: Any = FakeUsage()


@dataclass
class FakeMessageStopEvent:
    type = "message_stop"


@dataclass
class FakeMessageStartEvent:
    type = "message_start"
    message: Any = FakeMessage()


async def _fake_stream() -> Any:
    yield FakeMessageStartEvent()
    yield FakeContentBlockDeltaEvent(delta=FakeTextDelta(text="Hello"))
    yield FakeContentBlockDeltaEvent(delta=FakeTextDelta(text=" world"))
    yield FakeMessageDeltaEvent()
    yield FakeMessageStopEvent()


class FakeMessages:
    def __init__(self, response: Any | None = None) -> None:
        self._response = response if response is not None else FakeMessage()

    async def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return _fake_stream()
        if kwargs.get("tools"):
            return FakeToolMessage()
        return FakeStructuredMessage()


class FakeClient:
    def __init__(self, response: Any | None = None) -> None:
        self.messages = FakeMessages(response)


class FailingFakeMessages:
    async def create(self, **kwargs: Any) -> Any:
        raise RuntimeError("Rate limit exceeded")


class FailingFakeClient:
    def __init__(self) -> None:
        self.messages = FailingFakeMessages()


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: Any) -> None:
    monkeypatch.setattr(AnthropicAdapter, "_client", lambda self: client)


class TestAnthropicAdapterContract(AdapterContractSuite):
    """Run the provider-neutral contract suite against AnthropicAdapter."""

    @pytest.fixture
    def adapter_factory(self, monkeypatch: pytest.MonkeyPatch) -> Callable[[], ProviderAdapter]:
        _patch_client(monkeypatch, FakeClient())
        return lambda: AnthropicAdapter()

    @pytest.fixture
    def limited_adapter_factory(self) -> Callable[[], ProviderAdapter]:
        return lambda: AnthropicAdapter(
            capabilities=ProviderAdapterCapabilities(
                streaming=False,
                tool_calls=False,
                structured_output=False,
                cancellation=True,
            )
        )

    @pytest.fixture
    def failing_adapter_factory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> Callable[[], ProviderAdapter]:
        _patch_client(monkeypatch, FailingFakeClient())
        return lambda: AnthropicAdapter()
