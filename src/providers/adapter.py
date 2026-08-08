"""Provider-neutral adapter contract for all inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from src.providers.errors import ProviderError, ProviderErrorCode
from src.providers.identity import (
    ProviderIdentity,
    ProviderOperationalState,
    ProviderQuotaState,
)
from src.providers.request import ProviderRequest
from src.providers.response import ProviderResponse


@dataclass(frozen=True)
class ProviderAdapterCapabilities:
    streaming: bool = False
    tool_calls: bool = False
    structured_output: bool = False
    cancellation: bool = True


class ProviderAdapter(ABC):
    """Stable orchestration-facing provider contract.

    Concrete adapters translate normalized OmniForge request/response models into
    provider-specific APIs. Core orchestration depends only on this interface.
    """

    @property
    @abstractmethod
    def identity(self) -> ProviderIdentity:
        """Stable provider identity for this adapter."""

    @property
    @abstractmethod
    def capabilities(self) -> ProviderAdapterCapabilities:
        """Features the adapter can expose reliably."""

    @abstractmethod
    async def submit(self, request: ProviderRequest) -> ProviderResponse:
        """Submit one complete request and return one normalized response."""

    @abstractmethod
    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderResponse]:
        """Stream normalized response chunks for one request when supported."""
        if False:
            yield None

    @abstractmethod
    async def cancel(self, request_id: str) -> None:
        """Cancel in-flight work when cancellation is supported."""

    @abstractmethod
    async def health(self) -> ProviderOperationalState:
        """Return current provider/route operational health."""

    @abstractmethod
    async def quota(self) -> ProviderQuotaState:
        """Return quota/reset metadata when the provider exposes it."""

    async def close(self) -> None:
        """Optional lifecycle hook for transports or sessions."""
        return None

    def translate_error(self, raw_error: Any) -> ProviderError:
        """Translate a raw provider error into the normalized taxonomy."""
        return ProviderError(
            code=ProviderErrorCode.UNKNOWN,
            message=str(raw_error),
            provider_id=self.identity,
        )

    def can_serve(self, request: ProviderRequest) -> tuple[bool, ProviderError | None]:
        """Validate that this adapter can serve the request without provider specifics.

        Returns (True, None) if eligible, or (False, error) if a required
        capability is unsupported.
        """
        caps = self.capabilities
        if request.requires_streaming() and not caps.streaming:
            return False, ProviderError(
                code=ProviderErrorCode.UNSUPPORTED_CAPABILITY,
                message="Streaming is required but unsupported by this adapter",
                provider_id=self.identity,
                route_id=request.target_route,
            )
        if request.requires_tools() and not caps.tool_calls:
            return False, ProviderError(
                code=ProviderErrorCode.UNSUPPORTED_CAPABILITY,
                message="Tool use is required but unsupported by this adapter",
                provider_id=self.identity,
                route_id=request.target_route,
            )
        if request.requires_structured_output() and not caps.structured_output:
            return False, ProviderError(
                code=ProviderErrorCode.UNSUPPORTED_CAPABILITY,
                message="Structured output is required but unsupported by this adapter",
                provider_id=self.identity,
                route_id=request.target_route,
            )
        return True, None


@dataclass(frozen=True)
class AdapterContractSnapshot:
    provider_id: str
    streaming: bool
    tool_calls: bool
    structured_output: bool
    cancellation: bool


def inspect_adapter(adapter: ProviderAdapter) -> AdapterContractSnapshot:
    """Expose a small deterministic contract snapshot for diagnostics/tests."""

    capabilities = adapter.capabilities
    return AdapterContractSnapshot(
        provider_id=adapter.identity.provider_id,
        streaming=capabilities.streaming,
        tool_calls=capabilities.tool_calls,
        structured_output=capabilities.structured_output,
        cancellation=capabilities.cancellation,
    )
