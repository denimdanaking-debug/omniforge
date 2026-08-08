"""Generic provider adapter interface (Phase 2.1 foundation).

Core orchestration consumes this normalized contract. Provider-specific
translation logic lives in concrete adapter implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from providers.contracts.capabilities import ModelCapabilities
from providers.contracts.errors import ProviderError, ProviderErrorCode
from providers.contracts.health import ProviderHealth
from providers.contracts.identity import ModelId, ProviderId, RouteId
from providers.contracts.quota import ProviderQuota
from providers.contracts.request import ProviderRequest
from providers.contracts.response import ProviderResponse


@dataclass(frozen=True, slots=True)
class AdapterIdentity:
    """Stable identity declared by a provider adapter."""

    provider_id: ProviderId
    supported_models: tuple[ModelId, ...] = field(default_factory=tuple)
    supported_routes: tuple[RouteId, ...] = field(default_factory=tuple)
    capabilities: ModelCapabilities = field(
        default_factory=lambda: ModelCapabilities(context_size=0)
    )


class ProviderAdapter(ABC):
    """Normalized contract every provider adapter must satisfy.

    Core orchestration must call only these methods. Provider-specific payload
    construction, transport, and error translation are internal to adapters.
    """

    @property
    @abstractmethod
    def identity(self) -> AdapterIdentity:
        """Return stable provider identity and declared capabilities."""
        raise NotImplementedError

    @abstractmethod
    async def submit(self, request: ProviderRequest) -> ProviderResponse:
        """Submit a normalized request and return a complete response."""
        raise NotImplementedError

    @abstractmethod
    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderResponse]:
        """Stream normalized response chunks.

        The final chunk should set ``streaming_state`` to ``COMPLETE``.
        """
        raise NotImplementedError
        yield  # Make this an async generator for type checkers.

    @abstractmethod
    async def health(self) -> ProviderHealth:
        """Return current normalized health for this adapter's provider/route."""
        raise NotImplementedError

    @abstractmethod
    async def quota(self) -> ProviderQuota:
        """Return current normalized quota/capacity information."""
        raise NotImplementedError

    @abstractmethod
    async def cancel(self, cancellation_id: str) -> bool:
        """Attempt to cancel an in-flight request identified by cancellation_id."""
        raise NotImplementedError

    def translate_error(self, raw_error: Any) -> ProviderError:
        """Translate a raw provider error into the normalized taxonomy.

        Concrete adapters override this. The default returns UNKNOWN.
        """
        return ProviderError(
            code=ProviderErrorCode.UNKNOWN,
            message=str(raw_error),
            provider_id=self.identity.provider_id,
        )

    def can_serve(self, request: ProviderRequest) -> tuple[bool, ProviderError | None]:
        """Validate that this adapter can serve the request without provider specifics.

        Returns (True, None) if eligible, or (False, error) if a required
        capability is unsupported.
        """
        # This default implementation only checks capability requirements that
        # have been expressed through the normalized request. Concrete adapters
        # may extend this with provider-specific eligibility checks, but must
        # not bypass the normalized contract.
        if request.requires_streaming():
            cap = self.identity.capabilities.supports_streaming
            if cap.name == "UNSUPPORTED":
                return False, ProviderError(
                    code=ProviderErrorCode.UNSUPPORTED_CAPABILITY,
                    message="Streaming is required but unsupported by this adapter",
                    provider_id=self.identity.provider_id,
                    route_id=request.target_route,
                )
        if request.requires_tools():
            cap = self.identity.capabilities.supports_tool_use
            if cap.name == "UNSUPPORTED":
                return False, ProviderError(
                    code=ProviderErrorCode.UNSUPPORTED_CAPABILITY,
                    message="Tool use is required but unsupported by this adapter",
                    provider_id=self.identity.provider_id,
                    route_id=request.target_route,
                )
        if request.requires_structured_output():
            cap = self.identity.capabilities.supports_structured_output
            if cap.name == "UNSUPPORTED":
                return False, ProviderError(
                    code=ProviderErrorCode.UNSUPPORTED_CAPABILITY,
                    message="Structured output is required but unsupported by this adapter",
                    provider_id=self.identity.provider_id,
                    route_id=request.target_route,
                )
        return True, None
