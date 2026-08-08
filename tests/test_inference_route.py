from __future__ import annotations

import unittest

from src.routing.inference_route import (
    InferenceRouteIdentity,
    InferenceRouteRegistry,
    InferenceRouteError,
    RouteHealth,
    RouteOperationalState,
    RouteType,
)


class InferenceRouteTests(unittest.TestCase):
    def direct(self) -> InferenceRouteIdentity:
        return InferenceRouteIdentity(
            route_id="kimi:direct",
            provider_id="kimi",
            route_type=RouteType.DIRECT,
            endpoint_key="kimi-primary-api",
            failure_domain="api.moonshot.ai",
        )

    def gateway(self) -> InferenceRouteIdentity:
        return InferenceRouteIdentity(
            route_id="kimi:openrouter",
            provider_id="openrouter",
            route_type=RouteType.GATEWAY,
            endpoint_key="openrouter-primary",
            failure_domain="openrouter.ai",
        )

    def test_same_model_can_use_multiple_routes(self) -> None:
        registry = InferenceRouteRegistry()
        registry.register(self.direct())
        registry.register(self.gateway())
        registry.attach_model("kimi:direct", "kimi:k3-code")
        registry.attach_model("kimi:openrouter", "kimi:k3-code")
        routes = registry.routes_for_model("kimi:k3-code")
        self.assertEqual(["kimi:direct", "kimi:openrouter"], [r.identity.route_id for r in routes])

    def test_route_metrics_are_independent(self) -> None:
        registry = InferenceRouteRegistry()
        registry.register(self.direct())
        registry.register(self.gateway())
        registry.set_operational_state(
            "kimi:direct",
            RouteOperationalState(
                health=RouteHealth.DEGRADED,
                rolling_latency_ms=850.0,
                input_cost_per_million=1.0,
                output_cost_per_million=3.0,
                error_count=2,
                request_count=10,
            ),
        )
        self.assertEqual(RouteHealth.DEGRADED, registry.get("kimi:direct").operational.health)
        self.assertEqual(RouteHealth.HEALTHY, registry.get("kimi:openrouter").operational.health)

    def test_route_identity_is_not_mutated_by_operational_updates(self) -> None:
        registry = InferenceRouteRegistry()
        identity = self.direct()
        registry.register(identity)
        registry.set_operational_state(
            identity.route_id,
            RouteOperationalState(request_count=2, error_count=1),
        )
        self.assertEqual(identity, registry.get(identity.route_id).identity)

    def test_invalid_metrics_are_rejected(self) -> None:
        with self.assertRaises(InferenceRouteError):
            RouteOperationalState(request_count=1, error_count=2)
        with self.assertRaises(InferenceRouteError):
            RouteOperationalState(rolling_latency_ms=-1)

    def test_conflicting_route_identity_cannot_rebind_id(self) -> None:
        registry = InferenceRouteRegistry()
        registry.register(self.direct())
        with self.assertRaises(InferenceRouteError):
            registry.register(
                InferenceRouteIdentity(
                    route_id="kimi:direct",
                    provider_id="different",
                    route_type=RouteType.DIRECT,
                    endpoint_key="different",
                    failure_domain="different.example",
                )
            )


if __name__ == "__main__":
    unittest.main()
