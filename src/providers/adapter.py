"""Provider-neutral adapter contract for all inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from src.providers.identity import (
    ProviderIdentity,
    ProviderOperationalState,
    ProviderQuotaState,
)

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")
StreamEventT = TypeVar("StreamEventT")


@dataclass(frozen=True)
class ProviderAdapterCapabilities:
    streaming: bool = False
    tool_calls: bool = False
    structured_output: bool = False
    cancellation: bool = True


class ProviderAdapter(ABC, Generic[RequestT, ResponseT, StreamEventT]):
    """Stable orchestration-facing provider contract.

    Concrete adapters translate the normalized OmniForge request/response models
    introduced by later roadmap steps into provider-specific APIs. Core
    orchestration depends only on this interface.
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
    async def submit(self, request: RequestT) -> ResponseT:
        """Submit one complete request and return one normalized response."""

    @abstractmethod
    async def stream(self, request: RequestT) -> AsyncIterator[StreamEventT]:
        """Stream normalized events for one request when supported."""
        if False:
            yield None  # type: ignore[misc]

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


@dataclass(frozen=True)
class AdapterContractSnapshot:
    provider_id: str
    streaming: bool
    tool_calls: bool
    structured_output: bool
    cancellation: bool


def inspect_adapter(adapter: ProviderAdapter[Any, Any, Any]) -> AdapterContractSnapshot:
    """Expose a small deterministic contract snapshot for diagnostics/tests."""

    capabilities = adapter.capabilities
    return AdapterContractSnapshot(
        provider_id=adapter.identity.provider_id,
        streaming=capabilities.streaming,
        tool_calls=capabilities.tool_calls,
        structured_output=capabilities.structured_output,
        cancellation=capabilities.cancellation,
    )
