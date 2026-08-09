from __future__ import annotations

from src.policy.risk import RiskLevel
from src.routing.dynamic.candidate import RoutingCandidate
from src.routing.dynamic.request import DynamicRoutingRequest
from src.routing.dynamic.router import RoutingCoordinator
from src.routing.model_identity import ModelIdentity, ModelLifecycle
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


def test_runtime_factor_weights_drive_winner(
    cheap_unreliable_candidate: RoutingCandidate,
    expensive_reliable_candidate: RoutingCandidate,
) -> None:
    """Passed runtime factor_weights must override constructor defaults."""
    request = _request()
    candidates = (cheap_unreliable_candidate, expensive_reliable_candidate)

    # Weight cost heavily. Because cost is expected total cost to accepted
    # integration, the failure-prone cheap model has a higher expected total
    # cost than the reliable expensive model, so reliability wins.
    cost_weighted_config = _config("dynamic")
    cost_weighted_config["router_config"] = {
        "factor_weights": {
            "expected_success": 0.0,
            "role_fit": 0.0,
            "risk_fit": 0.0,
            "empirical_reliability": 0.0,
            "context_suitability": 0.0,
            "recent_performance": 0.0,
            "provider_health": 0.0,
            "quota_pressure": 0.0,
            "cost": 1.0,
            "latency": 0.0,
            "affinity": 0.0,
            "diversity_reserve": 0.0,
        }
    }
    coordinator = RoutingCoordinator()
    decision = coordinator.route(request, candidates, cost_weighted_config)
    assert decision.selected_candidate == expensive_reliable_candidate

    # Weight empirical reliability heavily: reliable candidate should win.
    reliability_weighted_config = _config("dynamic")
    reliability_weighted_config["router_config"] = {
        "factor_weights": {
            "expected_success": 0.0,
            "role_fit": 0.0,
            "risk_fit": 0.0,
            "empirical_reliability": 1.0,
            "context_suitability": 0.0,
            "recent_performance": 0.0,
            "provider_health": 0.0,
            "quota_pressure": 0.0,
            "cost": 0.0,
            "latency": 0.0,
            "affinity": 0.0,
            "diversity_reserve": 0.0,
        }
    }
    decision = coordinator.route(request, candidates, reliability_weighted_config)
    assert decision.selected_candidate == expensive_reliable_candidate


def test_runtime_priors_affect_cold_start_score() -> None:
    """A prior supplied via runtime router_config must affect scoring."""
    from src.routing.capabilities import ModelCapabilities
    from src.routing.inference_route import InferenceRouteIdentity, RouteType

    model = ModelIdentity(
        model_id="custom-model", family="custom", lifecycle=ModelLifecycle.HIGH_RISK
    )
    route = InferenceRouteIdentity(
        route_id="rt",
        provider_id="pv",
        route_type=RouteType.DIRECT,
        endpoint_key="e",
        failure_domain="f",
    )
    caps = ModelCapabilities(
        context_tokens=10_000,
        supported_roles=frozenset({ExecutionRole.CODING.value}),
    )
    candidate = RoutingCandidate(
        provider_id="pv",
        model_id="custom-model",
        route_id="rt",
        model_identity=model,
        route_identity=route,
        capabilities=caps,
    )

    config_no_prior = _config("dynamic")
    config_with_prior = _config("dynamic")
    config_with_prior["router_config"] = {
        "priors": [
            {
                "model_id": "custom-model",
                "role": "coding",
                "factor_name": "expected_success",
                "prior_value": 0.99,
                "confidence": 10,
            }
        ]
    }

    coordinator = RoutingCoordinator()
    decision_no = coordinator.route(_request(), (candidate,), config_no_prior)
    decision_yes = coordinator.route(_request(), (candidate,), config_with_prior)

    score_no = next(
        wf for wf in decision_no.record.scores[0].weighted_factors if wf.name == "expected_success"
    )
    score_yes = next(
        wf for wf in decision_yes.record.scores[0].weighted_factors if wf.name == "expected_success"
    )
    assert score_yes.normalized_value > score_no.normalized_value


def test_runtime_fallback_order_affects_fallback(
    cheap_unreliable_candidate: RoutingCandidate,
) -> None:
    """Passed runtime emergency_fallback_orders must actually change fallback order."""
    request = _request()
    config = _config("dynamic")
    # Trigger the emergency fallback path by making scoring fail, while the
    # candidate remains eligible for fallback selection.
    config["router_config"] = {
        "emergency_fallback_orders": {"coding": ["tiny-provider:tiny-model:tiny-direct"]},
        "cost_metadata": {"invalid_kwarg_for_forced_fallback": 1.0},
    }

    coordinator = RoutingCoordinator()
    decision = coordinator.route(
        request,
        (cheap_unreliable_candidate,),
        config,
    )
    assert decision.emergency_fallback_used
    assert decision.selected_candidate == cheap_unreliable_candidate


def test_scoring_failure_fallback_is_auditable(
    cheap_unreliable_candidate: RoutingCandidate,
) -> None:
    """A scoring failure must produce a deterministic, fully audited fallback decision."""
    request = _request()
    config = _config("dynamic")
    config["router_config"] = {
        "cost_metadata": {"invalid_kwarg_for_forced_fallback": 1.0},
        "exploration_enabled": True,
    }

    coordinator = RoutingCoordinator()
    decision = coordinator.route(request, (cheap_unreliable_candidate,), config)

    assert decision.emergency_fallback_used
    assert decision.record.fallback_used is True
    assert len(decision.record.input_fingerprint) > 0
    assert "scoring_failed" in (decision.record.fallback_reason or "")
    assert decision.record.exploration_enabled is True
    assert decision.record.eligible_candidates == (cheap_unreliable_candidate,)
    assert decision.selected_candidate == cheap_unreliable_candidate


def test_runtime_safety_margin_affects_context_eligibility(
    healthy_candidate: RoutingCandidate,
) -> None:
    """Passed runtime default_safety_margin_fraction must affect context eligibility."""
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="project-a",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
        required_context_tokens=120_000,
    )
    # gpt-4o has 128k tokens. With 10% margin usable = 115,200 < 120k -> excluded.
    # With 1% margin usable = 126,720 >= 120k -> eligible.
    config_small_margin = _config("dynamic")
    config_small_margin["router_config"] = {"default_safety_margin_fraction": 0.01}
    config_large_margin = _config("dynamic")
    config_large_margin["router_config"] = {"default_safety_margin_fraction": 0.1}

    coordinator = RoutingCoordinator()
    decision_small = coordinator.route(request, (healthy_candidate,), config_small_margin)
    decision_large = coordinator.route(request, (healthy_candidate,), config_large_margin)

    assert decision_small.selected_candidate == healthy_candidate
    assert decision_large.selected_candidate is None
    assert any(e.reason == "insufficient_context" for e in decision_large.excluded)


def test_decision_fingerprint_reflects_effective_config(
    healthy_candidate: RoutingCandidate,
) -> None:
    """The decision fingerprint must change when runtime router config changes."""
    request = _request()
    candidates = (healthy_candidate,)
    config_a = _config("dynamic")
    config_a["router_config"] = {"factor_weights": {"cost": 1.0}}
    config_b = _config("dynamic")
    config_b["router_config"] = {"factor_weights": {"cost": 0.5}}

    coordinator = RoutingCoordinator()
    decision_a = coordinator.route(request, candidates, config_a)
    decision_b = coordinator.route(request, candidates, config_b)

    assert decision_a.record.input_fingerprint != decision_b.record.input_fingerprint


def test_fingerprint_uses_pre_decision_state(healthy_candidate: RoutingCandidate) -> None:
    """The recorded fingerprint must describe the state before the decision mutated it."""
    from src.persistence.configuration import extract_administrative_state
    from src.routing.dynamic.config import load_router_config
    from src.routing.dynamic.fingerprint import routing_input_fingerprint
    from src.routing.dynamic.scoring import ScoringState
    from src.routing.policy import ProjectRoutingPolicy, RoutingPolicyEngine

    request = _request()
    candidates = (healthy_candidate,)
    config = _config("dynamic")

    coordinator = RoutingCoordinator()
    assert coordinator.state.last_selected_key is None
    assert coordinator.state.failure_domain_counts == {}

    decision = coordinator.route(request, candidates, config)

    # Reconstruct the exact policy engine and router config the coordinator used.
    admin_state = extract_administrative_state(config)
    policies = admin_state.get("project_policies", {})
    project_policy = ProjectRoutingPolicy.from_dict(policies.get(request.project_id, {}))
    provider_status = admin_state.get("provider_status", {})
    model_status = admin_state.get("model_status", {})
    route_status = admin_state.get("route_status", {})
    policy_engine = RoutingPolicyEngine(
        provider_enabled={k: v.get("enabled", True) for k, v in provider_status.items()},
        model_enabled={k: v.get("enabled", True) for k, v in model_status.items()},
        route_enabled={k: v.get("enabled", True) for k, v in route_status.items()},
        project_policy=project_policy,
        pin=request.pin,
    )
    router_cfg = load_router_config(config.get("router_config", {}))

    # The record fingerprint must match the empty pre-decision state, not the
    # post-decision state that now contains affinity/diversity updates.
    pre_decision_state = ScoringState(
        last_selected_key=None,
        failure_domain_counts={},
    )
    expected_fingerprint = routing_input_fingerprint(
        request=request,
        candidates=candidates,
        policy_engine=policy_engine,
        router_config=router_cfg,
        scoring_state=pre_decision_state,
    )
    assert decision.record.input_fingerprint == expected_fingerprint

    # Coordinator state advanced after the decision.
    assert coordinator.state.last_selected_key == healthy_candidate.identity_key
    assert coordinator.state.failure_domain_counts["openai.com"] == 1

    # A second identical request now gets a different fingerprint because the
    # pre-decision state has legitimately changed.
    decision2 = coordinator.route(request, candidates, config)
    assert decision2.record.input_fingerprint != decision.record.input_fingerprint
