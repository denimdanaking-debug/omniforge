"""Optional live integration tests for OpenRouter gateway.

These tests are skipped unless credentials are present and are never run in CI.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.providers.identity import ProviderIdentity
from src.providers.openrouter.adapter import OpenRouterAdapter
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity, ModelLifecycle

pytestmark = [pytest.mark.live]


@pytest.mark.live
async def test_openrouter_gateway_identity(skip_without_api_key: Callable[[str], None]) -> None:
    skip_without_api_key("OPENROUTER_API_KEY")
    adapter = OpenRouterAdapter(
        provider_identity=ProviderIdentity("anthropic", "Anthropic", "anthropic.com"),
        model_identity=ModelIdentity(
            model_id="claude-sonnet-4-20250514",
            family="claude",
            lifecycle=ModelLifecycle.HIGH_RISK,
        ),
        route_identity=InferenceRouteIdentity(
            route_id="openrouter-claude",
            provider_id="openrouter",
            route_type=RouteType.GATEWAY,
            endpoint_key="openrouter://anthropic/claude-sonnet-4-20250514",
            failure_domain="openrouter.ai",
        ),
        api_key="resolved-from-env",
    )
    assert adapter.route_id is not None
    assert adapter.route_id.route_type is RouteType.GATEWAY
