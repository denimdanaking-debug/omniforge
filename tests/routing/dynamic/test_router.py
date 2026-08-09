from __future__ import annotations

from src.policy.risk import RiskLevel
from src.routing.dynamic.candidate import RoutingCandidate
from src.routing.dynamic.request import DynamicRoutingRequest
from src.routing.dynamic.router import RoutingCoordinator
from src.routing.roles import ExecutionRole


def _config(routing_mode: str = "legacy") -> dict:
    return {
        "schema_version": "1.2.0",
        "routing_mode": routing_mode,
        "exploration_enabled": False,
        "providers": {
            "openai": {
                "enabled": True,
                "models": {"gpt-4o": {"enabled": True}},
                "routes": {"openai-direct": {"enabled": True}},
            },
            "anthropic": {
                "enabled": True,
                "models": {"claude": {"enabled": True}},
                "routes": {"anthropic-direct": {"enabled": True}},
            },
        },
        "pins": {},
        "project_policies": {
            "project-a": {
                "prohibited_provider_ids": ["anthropic"],
                "minimum_review_independence": None,
            }
        },
        "router_config": {},
    }


def _request(role: ExecutionRole = ExecutionRole.CODING) -> DynamicRoutingRequest:
    return DynamicRoutingRequest(
        task_id="t1",
        project_id="project-a",
        role=role,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )


def test_legacy_mode_returns_first_eligible(healthy_candidate: RoutingCandidate) -> None:
    coordinator = RoutingCoordinator()
    decision = coordinator.route(
        _request(),
        (healthy_candidate,),
        _config("legacy"),
        state={"run_id": "run-1"},
    )
    assert decision.record.routing_mode == "legacy"
    assert decision.selected_candidate == healthy_candidate


def test_dynamic_mode_selects_winner(healthy_candidate: RoutingCandidate) -> None:
    coordinator = RoutingCoordinator()
    decision = coordinator.route(
        _request(),
        (healthy_candidate,),
        _config("dynamic"),
        state={"run_id": "run-1"},
    )
    assert decision.record.routing_mode == "dynamic"
    assert decision.selected_candidate == healthy_candidate
    assert not decision.emergency_fallback_used


def test_project_restriction_beats_pin(healthy_candidate: RoutingCandidate) -> None:
    from src.routing.policy import RoutingPin

    pin = RoutingPin(provider_id="anthropic")
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="project-a",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
        pin=pin,
    )
    # Make anthropic candidate.
    from src.routing.capabilities import ModelCapabilities
    from src.routing.inference_route import InferenceRouteIdentity, RouteType
    from src.routing.model_identity import ModelIdentity, ModelLifecycle

    anthropic = type(healthy_candidate)(
        provider_id="anthropic",
        model_id="claude",
        route_id="anthropic-direct",
        model_identity=ModelIdentity(
            model_id="claude", family="claude", lifecycle=ModelLifecycle.HIGH_RISK
        ),
        route_identity=InferenceRouteIdentity(
            route_id="anthropic-direct",
            provider_id="anthropic",
            route_type=RouteType.DIRECT,
            endpoint_key="https://anthropic.com",
            failure_domain="anthropic.com",
        ),
        capabilities=ModelCapabilities(
            context_tokens=128_000,
            code_generation=True,
            supported_roles=frozenset({ExecutionRole.CODING.value}),
        ),
        operational_state=healthy_candidate.operational_state,
        recovery_state=healthy_candidate.recovery_state,
        quota_state=healthy_candidate.quota_state,
    )
    coordinator = RoutingCoordinator()
    decision = coordinator.route(
        request,
        (anthropic,),
        _config("dynamic"),
        state={"run_id": "run-1"},
    )
    assert decision.selected_candidate is None
    assert any("project" in e.reason for e in decision.excluded)


def test_pin_narrows_dynamic_selection(healthy_candidate: RoutingCandidate) -> None:
    from src.routing.policy import RoutingPin

    pin = RoutingPin(provider_id="openai")
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="project-a",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
        pin=pin,
    )
    coordinator = RoutingCoordinator()
    decision = coordinator.route(
        request,
        (healthy_candidate,),
        _config("dynamic"),
        state={"run_id": "run-1"},
    )
    assert decision.selected_candidate == healthy_candidate


def test_dynamic_mode_records_exploration_metadata(healthy_candidate: RoutingCandidate) -> None:
    config = _config("dynamic")
    config["router_config"] = {"exploration_enabled": True}
    coordinator = RoutingCoordinator()
    decision = coordinator.route(
        _request(),
        (healthy_candidate,),
        config,
        state={"run_id": "run-1"},
    )
    assert decision.record.exploration_enabled is True
