"""Dynamic routing candidate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.providers.identity import ProviderOperationalState, ProviderQuotaState
from src.recovery.state_machine import RouteRecoveryState
from src.routing.capabilities import ModelCapabilities
from src.routing.inference_route import InferenceRouteIdentity, RouteOperationalState
from src.routing.model_identity import ModelIdentity


@dataclass(frozen=True)
class PerformanceEvidence:
    """Empirical performance evidence for a candidate in a specific role."""

    attempts: int = 0
    successes: int = 0
    success_rate: float | None = None
    recent_attempts: int = 0
    recent_successes: int = 0
    recent_success_rate: float | None = None
    repair_rate: float | None = None
    retry_rate: float | None = None
    average_latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.attempts < 0
            or self.successes < 0
            or self.recent_attempts < 0
            or self.recent_successes < 0
        ):
            raise ValueError("performance counters must be non-negative")
        if self.successes > self.attempts:
            raise ValueError("successes cannot exceed attempts")
        if self.recent_successes > self.recent_attempts:
            raise ValueError("recent_successes cannot exceed recent_attempts")
        for name, value in (
            ("success_rate", self.success_rate),
            ("recent_success_rate", self.recent_success_rate),
            ("repair_rate", self.repair_rate),
            ("retry_rate", self.retry_rate),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.average_latency_ms is not None and self.average_latency_ms < 0:
            raise ValueError("average_latency_ms must be non-negative")


@dataclass(frozen=True)
class RoutingCandidate:
    """One provider/model/route candidate for dynamic routing."""

    provider_id: str
    model_id: str
    route_id: str
    model_identity: ModelIdentity
    route_identity: InferenceRouteIdentity
    capabilities: ModelCapabilities
    recovery_state: RouteRecoveryState | None = None
    quota_state: ProviderQuotaState | None = None
    operational_state: ProviderOperationalState | None = None
    route_cost_state: RouteOperationalState | None = None
    performance_evidence: PerformanceEvidence | None = None

    @property
    def identity_key(self) -> str:
        """Canonical lexical identity key for deterministic ordering."""
        return f"{self.provider_id}:{self.model_id}:{self.route_id}"
