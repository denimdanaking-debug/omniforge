"""Optional live integration test for Google Gemini."""

from __future__ import annotations

import os

import pytest

from src.policy.risk import RiskLevel
from src.providers.request import Message, MessageRole, ProviderRequest
from src.routing.roles import ExecutionRole

pytestmark = [pytest.mark.live]


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_live_gemini_simple_completion() -> None:
    from src.providers.gemini.adapter import GeminiAdapter

    adapter = GeminiAdapter(api_key=os.environ["GEMINI_API_KEY"])
    request = ProviderRequest(
        request_id="live-gemini",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R3_HIGH,
        messages=[Message(role=MessageRole.USER, content="Say 'pong' and nothing else.")],
        max_output_tokens=20,
    )
    response = await adapter.submit(request)
    assert response.text is not None
    assert response.provider_id == adapter.identity
