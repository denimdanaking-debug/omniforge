"""Failure-domain index for shared outage propagation.

Routes that share a failure domain are not independent fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.recovery.signals import ProviderSignal
from src.recovery.state_machine import HealthStateMachine, RouteRecoveryState


@dataclass
class FailureDomainIndex:
    """Maps failure-domain strings to the route IDs that share them."""

    _domains: dict[str, set[str]] = field(default_factory=dict)

    def register(self, route_id: str, failure_domain: str) -> None:
        if not route_id.strip():
            raise ValueError("route_id must be non-empty")
        if not failure_domain.strip():
            raise ValueError("failure_domain must be non-empty")
        self._domains.setdefault(failure_domain, set()).add(route_id)

    def routes_in_domain(self, failure_domain: str) -> frozenset[str]:
        return frozenset(self._domains.get(failure_domain, set()))

    def affected_routes(self, signal: ProviderSignal) -> frozenset[str]:
        """Return all registered routes in the signal's failure domain."""
        return self.routes_in_domain(signal.failure_domain)

    def mark_domain_affected(
        self,
        failure_domain: str,
        state_machine: HealthStateMachine,
        current_states: dict[str, RouteRecoveryState],
        reason: str,
    ) -> dict[str, RouteRecoveryState]:
        """Propagate an outage to all routes in the domain.

        Model quality is not mutated. Only route operational state changes.
        """
        from src.providers.identity import ProviderHealth, ProviderOperationalState
        from src.recovery.signals import signal_from_health_check

        updated: dict[str, RouteRecoveryState] = {}
        for route_id in self.routes_in_domain(failure_domain):
            state = current_states.get(route_id)
            if state is None:
                continue
            # Only propagate if the route is currently eligible.
            if not state.is_eligible():
                continue
            signal = signal_from_health_check(
                ProviderOperationalState(
                    health=ProviderHealth.UNAVAILABLE,
                    reason=reason,
                    failure_domain_id=failure_domain,
                ),
                provider_id="domain-propagation",
                route_id=route_id,
                failure_domain=failure_domain,
                clock=state_machine.clock,
            )
            updated[route_id] = state_machine.apply(state, signal)
        return updated

    def to_dict(self) -> dict[str, list[str]]:
        return {
            failure_domain: sorted(route_ids)
            for failure_domain, route_ids in sorted(self._domains.items())
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureDomainIndex:
        index = cls()
        for failure_domain, route_ids in data.items():
            for route_id in route_ids:
                index.register(route_id, failure_domain)
        return index
