from __future__ import annotations

import asyncio
import unittest

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


class FakeAdapter(ProviderAdapter[dict, dict, str]):
    def __init__(self) -> None:
        self.cancelled: list[str] = []

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

    async def submit(self, request: dict) -> dict:
        return {"echo": request}

    async def stream(self, request: dict):
        yield "first"
        yield "second"

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
            Incomplete()

    def test_submit_contract(self) -> None:
        adapter = FakeAdapter()
        result = asyncio.run(adapter.submit({"task": "hello"}))
        self.assertEqual({"echo": {"task": "hello"}}, result)

    def test_stream_contract(self) -> None:
        adapter = FakeAdapter()

        async def collect() -> list[str]:
            return [event async for event in adapter.stream({})]

        self.assertEqual(["first", "second"], asyncio.run(collect()))

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
