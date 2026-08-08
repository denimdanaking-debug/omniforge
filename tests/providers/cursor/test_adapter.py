"""Unit tests for the Cursor route adapter."""

from __future__ import annotations

import pytest

from src.policy.risk import RiskLevel
from src.providers.cursor.adapter import (
    CURSOR_MODEL,
    CURSOR_PROVIDER,
    CURSOR_ROUTE,
    CursorRouteAdapter,
)
from src.providers.errors import ProviderErrorCode
from src.providers.identity import ProviderHealth, QuotaSignal
from src.providers.request import Message, MessageRole, ProviderRequest
from src.providers.response import StreamingState
from src.routing.inference_route import RouteType
from src.routing.roles import ExecutionRole


@pytest.fixture
def adapter() -> CursorRouteAdapter:
    return CursorRouteAdapter()


@pytest.fixture
def sample_request() -> ProviderRequest:
    return ProviderRequest(
        request_id="cursor-test-1",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        messages=[Message(role=MessageRole.USER, content="Hello")],
    )


async def test_provider_identity(adapter: CursorRouteAdapter) -> None:
    assert adapter.identity == CURSOR_PROVIDER
    assert adapter.identity.provider_id == "cursor"
    assert adapter.identity.display_name == "Cursor Route"
    assert adapter.identity.failure_domain == "cursor.com"


async def test_route_identity_declared(adapter: CursorRouteAdapter) -> None:
    route = adapter.route_identity
    assert route == CURSOR_ROUTE
    assert route.route_id == "cursor-local"
    assert route.provider_id == "cursor"
    assert route.route_type is RouteType.LOCAL
    assert route.endpoint_key == "cursor://local"
    assert route.failure_domain == "cursor.com"


async def test_model_identity_declared(adapter: CursorRouteAdapter) -> None:
    model = adapter.model_identity
    assert model == CURSOR_MODEL
    assert model.model_id == "cursor-agent"
    assert model.family == "cursor"


async def test_capabilities_are_limited(adapter: CursorRouteAdapter) -> None:
    caps = adapter.capabilities
    assert caps.streaming is False
    assert caps.tool_calls is False
    assert caps.structured_output is False
    assert caps.cancellation is False


async def test_submit_returns_unsupported_capability_error(
    adapter: CursorRouteAdapter,
    sample_request: ProviderRequest,
) -> None:
    response = await adapter.submit(sample_request)
    assert response.request_id == sample_request.request_id
    assert response.provider_id == CURSOR_PROVIDER
    assert response.model_id == CURSOR_MODEL
    assert response.route_id == CURSOR_ROUTE
    assert response.error_reference == ProviderErrorCode.UNSUPPORTED_CAPABILITY.value
    assert "Cursor" in str(response.metadata.get("error", ""))


async def test_stream_returns_unsupported_capability_error(
    adapter: CursorRouteAdapter,
    sample_request: ProviderRequest,
) -> None:
    chunks = []
    async for chunk in adapter.stream(sample_request):
        chunks.append(chunk)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.request_id == sample_request.request_id
    assert chunk.provider_id == CURSOR_PROVIDER
    assert chunk.model_id == CURSOR_MODEL
    assert chunk.error_reference == ProviderErrorCode.UNSUPPORTED_CAPABILITY.value
    assert chunk.streaming_state is StreamingState.FAILED


async def test_health_reports_unavailable(adapter: CursorRouteAdapter) -> None:
    health = await adapter.health()
    assert health.health is ProviderHealth.UNAVAILABLE
    assert "Cursor remote automation not available" in (health.reason or "")


async def test_quota_returns_unknown(adapter: CursorRouteAdapter) -> None:
    quota = await adapter.quota()
    assert quota.provider_signal is QuotaSignal.UNKNOWN
    assert not quota.is_exhausted()


async def test_cancel_is_noop(adapter: CursorRouteAdapter) -> None:
    await adapter.cancel("req-1")


async def test_no_credential_leakage(
    adapter: CursorRouteAdapter,
    sample_request: ProviderRequest,
) -> None:
    response = await adapter.submit(sample_request)
    response_text = str(response.metadata).lower()
    assert "api_key" not in response_text
    assert "secret" not in response_text
    assert "token" not in response_text
