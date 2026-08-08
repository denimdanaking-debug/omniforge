"""Inference route identity and route-specific operational measurements."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

_ROUTE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._:/-]{1,127}$")


class InferenceRouteError(ValueError):
    pass


class RouteType(StrEnum):
    DIRECT = "direct"
    GATEWAY = "gateway"
    LOCAL = "local"
    ENTERPRISE = "enterprise"


class RouteHealth(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


@dataclass(frozen=True)
class InferenceRouteIdentity:
    route_id: str
    provider_id: str
    route_type: RouteType
    endpoint_key: str
    failure_domain: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _ROUTE_ID_PATTERN.fullmatch(self.route_id):
            raise InferenceRouteError(
                "route_id must be 2-128 lowercase letters/digits plus '.', '_', ':', '/' or '-'"
            )
        if not self.provider_id.strip():
            raise InferenceRouteError("provider_id must be non-empty")
        if not self.endpoint_key.strip():
            raise InferenceRouteError("endpoint_key must be non-empty")
        if not self.failure_domain.strip():
            raise InferenceRouteError("failure_domain must be non-empty")


@dataclass(frozen=True)
class RouteOperationalState:
    health: RouteHealth = RouteHealth.HEALTHY
    rolling_latency_ms: float | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    error_count: int = 0
    request_count: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("rolling_latency_ms", self.rolling_latency_ms),
            ("input_cost_per_million", self.input_cost_per_million),
            ("output_cost_per_million", self.output_cost_per_million),
        ):
            if value is not None and value < 0:
                raise InferenceRouteError(f"{name} cannot be negative")
        if self.error_count < 0 or self.request_count < 0:
            raise InferenceRouteError("route counters cannot be negative")
        if self.error_count > self.request_count:
            raise InferenceRouteError("error_count cannot exceed request_count")


@dataclass(frozen=True)
class RouteRegistration:
    identity: InferenceRouteIdentity
    model_ids: frozenset[str] = frozenset()
    operational: RouteOperationalState = field(default_factory=RouteOperationalState)


class InferenceRouteRegistry:
    def __init__(self) -> None:
        self._routes: dict[str, RouteRegistration] = {}

    def register(self, identity: InferenceRouteIdentity) -> None:
        existing = self._routes.get(identity.route_id)
        if existing is not None and existing.identity != identity:
            raise InferenceRouteError(
                f"route_id {identity.route_id!r} is already bound to another route identity"
            )
        if existing is None:
            self._routes[identity.route_id] = RouteRegistration(identity=identity)

    def get(self, route_id: str) -> RouteRegistration:
        try:
            return self._routes[route_id]
        except KeyError as exc:
            raise InferenceRouteError(f"unknown route_id {route_id!r}") from exc

    def attach_model(self, route_id: str, model_id: str) -> None:
        if not model_id.strip():
            raise InferenceRouteError("model_id must be non-empty")
        registration = self.get(route_id)
        self._routes[route_id] = replace(
            registration, model_ids=registration.model_ids | {model_id}
        )

    def set_operational_state(self, route_id: str, state: RouteOperationalState) -> None:
        registration = self.get(route_id)
        self._routes[route_id] = replace(registration, operational=state)

    def routes_for_model(self, model_id: str) -> tuple[RouteRegistration, ...]:
        return tuple(
            self._routes[key]
            for key in sorted(self._routes)
            if model_id in self._routes[key].model_ids
        )

    def registrations(self) -> tuple[RouteRegistration, ...]:
        return tuple(self._routes[key] for key in sorted(self._routes))
