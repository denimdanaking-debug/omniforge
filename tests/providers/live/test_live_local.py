"""Optional live integration tests for local endpoints.

These tests are skipped unless a local endpoint is configured and are never run
in CI.
"""

from __future__ import annotations

import os

import pytest

from src.providers.local.adapter import LocalEndpointAdapter
from src.providers.local.profile import LocalEndpointProfile, LocalModelConfig, LocalRuntimeKind
from src.routing.inference_route import InferenceRouteIdentity, RouteType

pytestmark = [pytest.mark.live]


def _local_configured() -> bool:
    return bool(os.environ.get("LOCAL_ENDPOINT_URL"))


@pytest.mark.live
async def test_local_endpoint_identity() -> None:
    if not _local_configured():
        pytest.skip("LOCAL_ENDPOINT_URL not set")
    base_url = os.environ["LOCAL_ENDPOINT_URL"]
    profile = LocalEndpointProfile(
        runtime_kind=LocalRuntimeKind.GENERIC,
        base_url=base_url,
        route_identity=InferenceRouteIdentity(
            route_id="local-generic",
            provider_id="local-generic",
            route_type=RouteType.LOCAL,
            endpoint_key=base_url,
            failure_domain="localhost",
        ),
        failure_domain="localhost",
    )
    adapter = LocalEndpointAdapter(
        model_config=LocalModelConfig(
            model_id="local-model",
            family="local",
            profile=profile,
        )
    )
    assert adapter.identity.provider_id == "local-generic"
