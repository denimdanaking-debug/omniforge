from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator

from src.policy.risk import RiskLevel
from src.providers.adapter import (
    ProviderAdapter,
    ProviderAdapterCapabilities,
    inspect_adapter,
)
from src.providers.identity import (
    ProviderHealth,
    ProviderIdentity,
    ProviderOperationalState,
    ProviderQuotaState,
)
from src.providers.request import Message, MessageRole, ProviderRequest
from src.providers.response import FinishReason, ProviderResponse, StreamingState, Usage
from src.routing.model_identity import ModelIdentity
from src.routing.roles import ExecutionRole


class FakeAdapter(ProviderAdapter):
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.model = ModelIdentity(model_id="fake-model", family="fake")

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity("fake", "Fake Provider", "fake.example")

    @property
    def capabilities(self) -> ProviderAdapterCapabilities:
        return ProviderAdapterCapabilities(
            streaming=True,
            tool_calls=True,
            structured_output=True,
            cancellation=True,
        )

    async def submit(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            request_id=request.request_id,
            provider_id=self.identity,
            model_id=self.model,
            text=request.messages[0].content if request.messages else "",
            finish_reason=FinishReason.STOP,
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderResponse]:
        content = request.messages[0].content if request.messages else ""
        yield ProviderResponse(
            request_id=request.request_id,
            provider_id=self.identity,
            model_id=self.model,
            text=content[:2],
            streaming_state=StreamingState.IN_PROGRESS,
        )
        yield ProviderResponse(
            request_id=request.request_id,
            provider_id=self.identity,
            model_id=self.model,
            text=content[2:],
            streaming_state=StreamingState.COMPLETE,
            finish_reason=FinishReason.STOP,
        )

    async def cancel(self, request_id: str) -> None:
        self.cancelled.append(request_id)

    async def health(self) -> ProviderOperationalState:
        return ProviderOperationalState(health=ProviderHealth.HEALTHY)

    async def quota(self) -> ProviderQuotaState:
        return ProviderQuotaState(remaining_fraction=0.75, reset_at="2026-08-09T00:00:00Z")


class ProviderAdapterTests(unittest.TestCase):
    def test_incomplete_adapter_cannot_be_instantiated(self) -> None:
        class Incomplete(ProviderAdapter):
            pass

        with self.assertRaises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def _request(self, content: str) -> ProviderRequest:
        return ProviderRequest(
            request_id="req-1",
            execution_role=ExecutionRole.CODING,
            risk_level=RiskLevel.R2_NORMAL,
            messages=[Message(role=MessageRole.USER, content=content)],
        )

    def test_submit_contract(self) -> None:
        adapter = FakeAdapter()
        result = asyncio.run(adapter.submit(self._request("hello")))
        self.assertEqual("hello", result.text)
        self.assertEqual(FinishReason.STOP, result.finish_reason)

    def test_stream_contract(self) -> None:
        adapter = FakeAdapter()

        async def collect() -> list[ProviderResponse]:
            return [event async for event in adapter.stream(self._request("abcd"))]

        chunks = asyncio.run(collect())
        self.assertEqual("ab", chunks[0].text)
        self.assertEqual(StreamingState.IN_PROGRESS, chunks[0].streaming_state)
        self.assertEqual("cd", chunks[1].text)
        self.assertEqual(StreamingState.COMPLETE, chunks[1].streaming_state)

    def test_cancel_health_and_quota_contracts(self) -> None:
        adapter = FakeAdapter()
        asyncio.run(adapter.cancel("request-1"))
        self.assertEqual(["request-1"], adapter.cancelled)
        self.assertEqual(ProviderHealth.HEALTHY, asyncio.run(adapter.health()).health)
        self.assertEqual(0.75, asyncio.run(adapter.quota()).remaining_fraction)

    def test_adapter_declares_tool_structured_streaming_support(self) -> None:
        snapshot = inspect_adapter(FakeAdapter())
        self.assertEqual("fake", snapshot.provider_id)
        self.assertTrue(snapshot.streaming)
        self.assertTrue(snapshot.tool_calls)
        self.assertTrue(snapshot.structured_output)
        self.assertTrue(snapshot.cancellation)


if __name__ == "__main__":
    unittest.main()
