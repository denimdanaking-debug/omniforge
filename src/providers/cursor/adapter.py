"""Cursor route adapter for OmniForge Phase 3.

Cursor is a genuine execution route, not a model family. There is no stable
public remote Cursor API that can be faked as successful execution in Phase 3,
so this adapter declines all execution requests with a normalized unsupported
capability error while still exposing stable identity for routing and
diagnostics.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from src.providers.adapter import ProviderAdapter, ProviderAdapterCapabilities
from src.providers.errors import ProviderError, ProviderErrorCode
from src.providers.identity import (
    ProviderHealth,
    ProviderIdentity,
    ProviderOperationalState,
    ProviderQuotaState,
    QuotaSignal,
)
from src.providers.request import ProviderRequest
from src.providers.response import ProviderResponse, StreamingState, Usage
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity

CURSOR_PROVIDER = ProviderIdentity("cursor", "Cursor Route", "cursor.com")
CURSOR_ROUTE = InferenceRouteIdentity(
    route_id="cursor-local",
    provider_id="cursor",
    route_type=RouteType.LOCAL,
    endpoint_key="cursor://local",
    failure_domain="cursor.com",
)
CURSOR_MODEL = ModelIdentity(model_id="cursor-agent", family="cursor")

_CURSOR_UNAVAILABLE_MESSAGE = "Cursor remote execution is not available in Phase 3"
_CURSOR_UNAVAILABLE_REASON = "Cursor remote automation not available"


class CursorRouteAdapter(ProviderAdapter):
    """Route adapter for Cursor.

    Cursor remote automation is not available in Phase 3. All execution paths
    return a normalized unsupported-capability error.
    """

    @property
    def identity(self) -> ProviderIdentity:
        return CURSOR_PROVIDER

    @property
    def route_identity(self) -> InferenceRouteIdentity:
        return CURSOR_ROUTE

    @property
    def model_identity(self) -> ModelIdentity:
        return CURSOR_MODEL

    @property
    def capabilities(self) -> ProviderAdapterCapabilities:
        return ProviderAdapterCapabilities(
            streaming=False,
            tool_calls=False,
            structured_output=False,
            cancellation=False,
        )

    async def submit(self, request: ProviderRequest) -> ProviderResponse:
        error = ProviderError(
            code=ProviderErrorCode.UNSUPPORTED_CAPABILITY,
            message=_CURSOR_UNAVAILABLE_MESSAGE,
            provider_id=self.identity,
            route_id=self.route_identity,
            safe_diagnostic_message=_CURSOR_UNAVAILABLE_REASON,
        )
        return ProviderResponse(
            request_id=request.request_id,
            provider_id=self.identity,
            model_id=self.model_identity,
            route_id=self.route_identity,
            text=None,
            streaming_state=StreamingState.NOT_STREAMING,
            usage=Usage(),
            error_reference=ProviderErrorCode.UNSUPPORTED_CAPABILITY.value,
            metadata={"error": error},
        )

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderResponse]:
        # No streaming support; emit a single error chunk.
        yield ProviderResponse(
            request_id=request.request_id,
            provider_id=self.identity,
            model_id=self.model_identity,
            route_id=self.route_identity,
            text=None,
            streaming_state=StreamingState.FAILED,
            usage=Usage(),
            error_reference=ProviderErrorCode.UNSUPPORTED_CAPABILITY.value,
            metadata={"error_message": _CURSOR_UNAVAILABLE_MESSAGE},
        )

    async def cancel(self, request_id: str) -> None:
        """No-op: cancellation is unsupported."""
        return None

    async def health(self) -> ProviderOperationalState:
        return ProviderOperationalState(
            health=ProviderHealth.UNAVAILABLE,
            reason=_CURSOR_UNAVAILABLE_REASON,
        )

    async def quota(self) -> ProviderQuotaState:
        return ProviderQuotaState(provider_signal=QuotaSignal.UNKNOWN)
