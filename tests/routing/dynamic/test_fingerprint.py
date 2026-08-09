from __future__ import annotations

from datetime import UTC, datetime

from src.policy.risk import RiskLevel
from src.providers.identity import (
    ProviderHealth,
    ProviderOperationalState,
    ProviderQuotaState,
    QuotaSignal,
)
from src.recovery.state_machine import RouteRecoveryState
from src.routing.capabilities import CostMetadata, ModelCapabilities
from src.routing.dynamic.candidate import PerformanceEvidence, RoutingCandidate
from src.routing.dynamic.config import RouterConfig, RoutingCoordinatorState
from src.routing.dynamic.fingerprint import input_fingerprint, routing_input_fingerprint
from src.routing.dynamic.request import DynamicRoutingRequest
from src.routing.inference_route import (
    InferenceRouteIdentity,
    RouteHealth,
    RouteOperationalState,
    RouteType,
)
from src.routing.model_identity import ModelIdentity, ModelLifecycle
from src.routing.policy import ProjectRoutingPolicy, RoutingPin, RoutingPolicyEngine
from src.routing.roles import ExecutionRole


def test_fingerprint_deterministic() -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )
    f1 = input_fingerprint(request)
    f2 = input_fingerprint(request)
    assert f1 == f2
    assert len(f1) == 64


def test_fingerprint_excludes_timestamp() -> None:
    request1 = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
        timestamp=datetime.now(UTC),
    )
    request2 = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
        timestamp=datetime(2020, 1, 1, tzinfo=UTC),
    )
    assert input_fingerprint(request1) == input_fingerprint(request2)


def test_fingerprint_changes_with_role() -> None:
    request1 = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )
    request2 = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.REVIEW,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )
    assert input_fingerprint(request1) != input_fingerprint(request2)


# ---------------------------------------------------------------------------
# Full routing-input fingerprint coverage
# ---------------------------------------------------------------------------


def _request(
    role: ExecutionRole = ExecutionRole.CODING,
    required_context_tokens: int | None = None,
    pin: RoutingPin | None = None,
) -> DynamicRoutingRequest:
    return DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=role,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
        required_context_tokens=required_context_tokens,
        pin=pin,
        expected_input_tokens=1000,
        expected_output_tokens=500,
    )


def _candidate(
    provider_id: str = "openai",
    model_id: str = "gpt-4o",
    route_id: str = "openai-direct",
    provider_health: ProviderHealth | None = ProviderHealth.HEALTHY,
    quota_signal: QuotaSignal = QuotaSignal.AVAILABLE,
    success_rate: float | None = None,
) -> RoutingCandidate:
    model = ModelIdentity(model_id=model_id, family="gpt", lifecycle=ModelLifecycle.HIGH_RISK)
    route = InferenceRouteIdentity(
        route_id=route_id,
        provider_id=provider_id,
        route_type=RouteType.DIRECT,
        endpoint_key="https://example.com",
        failure_domain=f"{provider_id}.com",
    )
    evidence = None
    if success_rate is not None:
        evidence = PerformanceEvidence(attempts=50, successes=40, success_rate=success_rate)
    return RoutingCandidate(
        provider_id=provider_id,
        model_id=model_id,
        route_id=route_id,
        model_identity=model,
        route_identity=route,
        capabilities=ModelCapabilities(
            context_tokens=128_000,
            code_generation=True,
            cost=CostMetadata(input_per_million=2.5, output_per_million=10.0),
            supported_roles=frozenset({ExecutionRole.CODING.value}),
        ),
        operational_state=None
        if provider_health is None
        else ProviderOperationalState(health=provider_health),
        quota_state=ProviderQuotaState(provider_signal=quota_signal),
        recovery_state=RouteRecoveryState(health=ProviderHealth.HEALTHY),
        route_cost_state=RouteOperationalState(
            health=RouteHealth.HEALTHY,
            input_cost_per_million=2.5,
            output_cost_per_million=10.0,
        ),
        performance_evidence=evidence,
    )


def _policy_engine(
    prohibited_provider_ids: frozenset[str] | None = None,
    pin: RoutingPin | None = None,
) -> RoutingPolicyEngine:
    return RoutingPolicyEngine(
        provider_enabled={"openai": True, "anthropic": True},
        model_enabled={"gpt-4o": True, "claude": True},
        route_enabled={"openai-direct": True, "anthropic-direct": True},
        project_policy=ProjectRoutingPolicy(
            prohibited_provider_ids=prohibited_provider_ids or frozenset()
        ),
        pin=pin,
    )


def _router_config(
    factor_weights: dict[str, float] | None = None,
    priors: list[dict] | None = None,
    safety_margin: float = 0.1,
) -> RouterConfig:
    from src.routing.dynamic.scoring import ScoringWeights

    weights = ScoringWeights(**factor_weights) if factor_weights else ScoringWeights()
    return RouterConfig(
        factor_weights=weights,
        priors=priors or [],
        default_safety_margin_fraction=safety_margin,
    )


def test_full_fingerprint_reorders_candidates_invariant() -> None:
    request = _request()
    c1 = _candidate()
    c2 = _candidate(provider_id="anthropic", model_id="claude", route_id="anthropic-direct")
    policy = _policy_engine()
    config = _router_config()
    state = RoutingCoordinatorState()

    f1 = routing_input_fingerprint(
        request=request,
        candidates=(c1, c2),
        policy_engine=policy,
        router_config=config,
        scoring_state=state,
    )
    f2 = routing_input_fingerprint(
        request=request,
        candidates=(c2, c1),
        policy_engine=policy,
        router_config=config,
        scoring_state=state,
    )
    assert f1 == f2


def test_full_fingerprint_changes_with_provider_health() -> None:
    request = _request()
    healthy = _candidate(provider_health=ProviderHealth.HEALTHY)
    degraded = _candidate(provider_health=ProviderHealth.DEGRADED)
    policy = _policy_engine()
    config = _router_config()
    state = RoutingCoordinatorState()

    f1 = routing_input_fingerprint(
        request=request,
        candidates=(healthy,),
        policy_engine=policy,
        router_config=config,
        scoring_state=state,
    )
    f2 = routing_input_fingerprint(
        request=request,
        candidates=(degraded,),
        policy_engine=policy,
        router_config=config,
        scoring_state=state,
    )
    assert f1 != f2


def test_full_fingerprint_changes_with_quota_pressure() -> None:
    request = _request()
    available = _candidate(quota_signal=QuotaSignal.AVAILABLE)
    limited = _candidate(quota_signal=QuotaSignal.LIMITED)
    policy = _policy_engine()
    config = _router_config()
    state = RoutingCoordinatorState()

    f1 = routing_input_fingerprint(
        request=request,
        candidates=(available,),
        policy_engine=policy,
        router_config=config,
        scoring_state=state,
    )
    f2 = routing_input_fingerprint(
        request=request,
        candidates=(limited,),
        policy_engine=policy,
        router_config=config,
        scoring_state=state,
    )
    assert f1 != f2


def test_full_fingerprint_changes_with_empirical_success_rate() -> None:
    request = _request()
    low_evidence = _candidate(success_rate=0.1)
    high_evidence = _candidate(success_rate=0.9)
    policy = _policy_engine()
    config = _router_config()
    state = RoutingCoordinatorState()

    f1 = routing_input_fingerprint(
        request=request,
        candidates=(low_evidence,),
        policy_engine=policy,
        router_config=config,
        scoring_state=state,
    )
    f2 = routing_input_fingerprint(
        request=request,
        candidates=(high_evidence,),
        policy_engine=policy,
        router_config=config,
        scoring_state=state,
    )
    assert f1 != f2


def test_full_fingerprint_changes_with_factor_weights() -> None:
    request = _request()
    candidate = _candidate()
    policy = _policy_engine()
    state = RoutingCoordinatorState()

    f1 = routing_input_fingerprint(
        request=request,
        candidates=(candidate,),
        policy_engine=policy,
        router_config=_router_config(factor_weights={"cost": 1.0}),
        scoring_state=state,
    )
    f2 = routing_input_fingerprint(
        request=request,
        candidates=(candidate,),
        policy_engine=policy,
        router_config=_router_config(factor_weights={"cost": 0.5}),
        scoring_state=state,
    )
    assert f1 != f2


def test_full_fingerprint_changes_with_seeded_prior() -> None:
    request = _request()
    candidate = _candidate()
    policy = _policy_engine()
    state = RoutingCoordinatorState()

    f1 = routing_input_fingerprint(
        request=request,
        candidates=(candidate,),
        policy_engine=policy,
        router_config=_router_config(),
        scoring_state=state,
    )
    f2 = routing_input_fingerprint(
        request=request,
        candidates=(candidate,),
        policy_engine=policy,
        router_config=_router_config(
            priors=[
                {
                    "model_id": "gpt-4o",
                    "role": "coding",
                    "factor_name": "expected_success",
                    "prior_value": 0.99,
                    "confidence": 10,
                }
            ]
        ),
        scoring_state=state,
    )
    assert f1 != f2


def test_full_fingerprint_changes_with_project_prohibition() -> None:
    request = _request()
    candidate = _candidate()
    config = _router_config()
    state = RoutingCoordinatorState()

    f1 = routing_input_fingerprint(
        request=request,
        candidates=(candidate,),
        policy_engine=_policy_engine(),
        router_config=config,
        scoring_state=state,
    )
    f2 = routing_input_fingerprint(
        request=request,
        candidates=(candidate,),
        policy_engine=_policy_engine(prohibited_provider_ids=frozenset({"openai"})),
        router_config=config,
        scoring_state=state,
    )
    assert f1 != f2


def test_full_fingerprint_changes_with_pin() -> None:
    request = _request()
    candidate = _candidate()
    policy = _policy_engine()
    config = _router_config()
    state = RoutingCoordinatorState()

    f1 = routing_input_fingerprint(
        request=request,
        candidates=(candidate,),
        policy_engine=policy,
        router_config=config,
        scoring_state=state,
    )
    pinned_request = _request(pin=RoutingPin(provider_id="openai"))
    f2 = routing_input_fingerprint(
        request=pinned_request,
        candidates=(candidate,),
        policy_engine=policy,
        router_config=config,
        scoring_state=state,
    )
    assert f1 != f2


def test_full_fingerprint_changes_with_failure_domain_concentration() -> None:
    request = _request()
    candidate = _candidate()
    policy = _policy_engine()
    config = _router_config()

    f1 = routing_input_fingerprint(
        request=request,
        candidates=(candidate,),
        policy_engine=policy,
        router_config=config,
        scoring_state=RoutingCoordinatorState(),
    )
    f2 = routing_input_fingerprint(
        request=request,
        candidates=(candidate,),
        policy_engine=policy,
        router_config=config,
        scoring_state=RoutingCoordinatorState(failure_domain_counts={"openai.com": 3}),
    )
    assert f1 != f2


def test_full_fingerprint_changes_with_context_requirement() -> None:
    request = _request()
    candidate = _candidate()
    policy = _policy_engine()
    config = _router_config()
    state = RoutingCoordinatorState()

    f1 = routing_input_fingerprint(
        request=request,
        candidates=(candidate,),
        policy_engine=policy,
        router_config=config,
        scoring_state=state,
    )
    large_request = _request(required_context_tokens=200_000)
    f2 = routing_input_fingerprint(
        request=large_request,
        candidates=(candidate,),
        policy_engine=policy,
        router_config=config,
        scoring_state=state,
    )
    assert f1 != f2
