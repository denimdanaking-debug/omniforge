from __future__ import annotations

import math

import pytest

from src.policy.risk import RiskLevel
from src.providers.identity import ProviderHealth, ProviderOperationalState
from src.routing.dynamic.candidate import PerformanceEvidence, RoutingCandidate
from src.routing.dynamic.request import DynamicRoutingRequest
from src.routing.dynamic.scoring import (
    DeterministicRouterScorer,
    RoutingScoringError,
    ScoringState,
    ScoringWeights,
)
from src.routing.model_identity import ModelIdentity, ModelLifecycle
from src.routing.roles import ExecutionRole


def _request(
    role: ExecutionRole = ExecutionRole.CODING,
    risk: RiskLevel = RiskLevel.R2_NORMAL,
) -> DynamicRoutingRequest:
    return DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=role,
        risk=risk,
        task_class="default",
    )


def test_score_determinism(healthy_candidate: RoutingCandidate) -> None:
    scorer = DeterministicRouterScorer()
    s1 = scorer.score(_request(), healthy_candidate)
    s2 = scorer.score(_request(), healthy_candidate)
    assert s1 == s2


def test_tie_break_key_lexical(healthy_candidate: RoutingCandidate) -> None:
    scorer = DeterministicRouterScorer()
    score = scorer.score(_request(), healthy_candidate)
    assert score.tie_break_key == "openai:gpt-4o:openai-direct"


def test_weighted_factors_sum_to_total(healthy_candidate: RoutingCandidate) -> None:
    scorer = DeterministicRouterScorer()
    score = scorer.score(_request(), healthy_candidate)
    total = sum(wf.contribution for wf in score.weighted_factors)
    assert math.isclose(total, score.total_score, rel_tol=1e-9)


def test_role_fit_zero_when_unsupported(healthy_candidate: RoutingCandidate) -> None:
    scorer = DeterministicRouterScorer()
    request = _request(role=ExecutionRole.REVIEW)
    score = scorer.score(request, healthy_candidate)
    role_factor = next(wf for wf in score.weighted_factors if wf.name == "role_fit")
    assert role_factor.normalized_value == 0.0


def test_role_fit_one_when_supported(healthy_candidate: RoutingCandidate) -> None:
    scorer = DeterministicRouterScorer()
    score = scorer.score(_request(), healthy_candidate)
    role_factor = next(wf for wf in score.weighted_factors if wf.name == "role_fit")
    assert role_factor.normalized_value == 1.0


def test_risk_fit_lifecycle_scale() -> None:
    from src.routing.capabilities import ModelCapabilities
    from src.routing.inference_route import InferenceRouteIdentity, RouteType

    model = ModelIdentity(model_id="low-model", family="low", lifecycle=ModelLifecycle.LOW_RISK)
    route = InferenceRouteIdentity(
        route_id="rt",
        provider_id="pv",
        route_type=RouteType.DIRECT,
        endpoint_key="e",
        failure_domain="f",
    )
    caps = ModelCapabilities(
        context_tokens=10_000, supported_roles=frozenset({ExecutionRole.CODING.value})
    )
    candidate = RoutingCandidate(
        provider_id="pv",
        model_id="low-model",
        route_id="rt",
        model_identity=model,
        route_identity=route,
        capabilities=caps,
        operational_state=ProviderOperationalState(health=ProviderHealth.HEALTHY),
    )
    scorer = DeterministicRouterScorer()
    score = scorer.score(_request(risk=RiskLevel.R2_NORMAL), candidate)
    risk_factor = next(wf for wf in score.weighted_factors if wf.name == "risk_fit")
    assert risk_factor.normalized_value < 1.0


def test_empirical_reliability_is_pure_observed_rate() -> None:
    """empirical_reliability must be the observed success rate, not blended with priors."""
    from src.routing.capabilities import ModelCapabilities
    from src.routing.inference_route import InferenceRouteIdentity, RouteType

    model = ModelIdentity(model_id="kimi-k3", family="kimi", lifecycle=ModelLifecycle.HIGH_RISK)
    route = InferenceRouteIdentity(
        route_id="rt",
        provider_id="pv",
        route_type=RouteType.DIRECT,
        endpoint_key="e",
        failure_domain="f",
    )
    caps = ModelCapabilities(
        context_tokens=10_000, supported_roles=frozenset({ExecutionRole.CODING.value})
    )
    evidence = PerformanceEvidence(attempts=100, successes=50, success_rate=0.5)
    candidate = RoutingCandidate(
        provider_id="pv",
        model_id="kimi-k3",
        route_id="rt",
        model_identity=model,
        route_identity=route,
        capabilities=caps,
        operational_state=ProviderOperationalState(health=ProviderHealth.HEALTHY),
        performance_evidence=evidence,
    )
    scorer = DeterministicRouterScorer()
    score = scorer.score(_request(), candidate)
    rel_factor = next(wf for wf in score.weighted_factors if wf.name == "empirical_reliability")
    assert rel_factor.normalized_value == 0.5

    # expected_success remains the forward-looking blended prediction.
    exp_factor = next(wf for wf in score.weighted_factors if wf.name == "expected_success")
    # Prior for kimi coding is 0.78; blended with 0.5 empirical at 100 attempts
    # should move toward 0.5 but stay above it.
    assert exp_factor.normalized_value < 0.78
    assert exp_factor.normalized_value > 0.5


def test_expected_success_differs_from_empirical_reliability_for_cold_start() -> None:
    """Cold-start candidate gets prior-based expected success despite no empirical evidence."""
    from src.routing.capabilities import ModelCapabilities
    from src.routing.inference_route import InferenceRouteIdentity, RouteType

    model = ModelIdentity(
        model_id="openai-model", family="openai", lifecycle=ModelLifecycle.HIGH_RISK
    )
    route = InferenceRouteIdentity(
        route_id="rt",
        provider_id="pv",
        route_type=RouteType.DIRECT,
        endpoint_key="e",
        failure_domain="f",
    )
    caps = ModelCapabilities(
        context_tokens=10_000, supported_roles=frozenset({ExecutionRole.CODING.value})
    )
    candidate = RoutingCandidate(
        provider_id="pv",
        model_id="openai-model",
        route_id="rt",
        model_identity=model,
        route_identity=route,
        capabilities=caps,
        operational_state=ProviderOperationalState(health=ProviderHealth.HEALTHY),
    )
    scorer = DeterministicRouterScorer()
    score = scorer.score(_request(), candidate)
    rel_factor = next(wf for wf in score.weighted_factors if wf.name == "empirical_reliability")
    exp_factor = next(wf for wf in score.weighted_factors if wf.name == "expected_success")
    # Cold start: empirical reliability is neutral; expected success uses seeded prior.
    assert rel_factor.normalized_value == 0.5
    assert exp_factor.normalized_value > 0.5


def test_no_brand_privilege_across_models() -> None:
    from src.routing.capabilities import ModelCapabilities
    from src.routing.inference_route import InferenceRouteIdentity, RouteType

    def make(model_id: str, family: str) -> RoutingCandidate:
        model = ModelIdentity(model_id=model_id, family=family, lifecycle=ModelLifecycle.HIGH_RISK)
        route = InferenceRouteIdentity(
            route_id="rt",
            provider_id="pv",
            route_type=RouteType.DIRECT,
            endpoint_key="e",
            failure_domain=f"{family}.com",
        )
        caps = ModelCapabilities(
            context_tokens=10_000,
            supported_roles=frozenset({ExecutionRole.CODING.value}),
        )
        return RoutingCandidate(
            provider_id="pv",
            model_id=model_id,
            route_id="rt",
            model_identity=model,
            route_identity=route,
            capabilities=caps,
            operational_state=ProviderOperationalState(health=ProviderHealth.HEALTHY),
        )

    scorer = DeterministicRouterScorer()
    kim = make("kimi-k3", "kimi")
    qwen = make("qwen3.8-max", "qwen")
    s1 = scorer.score(_request(), kim)
    s2 = scorer.score(_request(), qwen)
    # No hardcoded branch; both are scored by priors only. Difference should be
    # small and data-driven.
    assert abs(s1.total_score - s2.total_score) < 0.2


def test_nan_weight_rejected() -> None:
    with pytest.raises(RoutingScoringError):
        ScoringWeights(cost=float("nan"))


def test_infinite_weight_rejected() -> None:
    with pytest.raises(RoutingScoringError):
        ScoringWeights(cost=float("inf"))


def test_negative_weight_rejected() -> None:
    with pytest.raises(RoutingScoringError):
        ScoringWeights(cost=-0.1)


def test_all_zero_weights_rejected() -> None:
    with pytest.raises(RoutingScoringError):
        ScoringWeights(**dict.fromkeys(ScoringWeights.__dataclass_fields__, 0.0))


def test_context_suitability_insufficient_zero(healthy_candidate: RoutingCandidate) -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
        required_context_tokens=1_000_000,
    )
    scorer = DeterministicRouterScorer()
    score = scorer.score(request, healthy_candidate)
    ctx_factor = next(wf for wf in score.weighted_factors if wf.name == "context_suitability")
    assert ctx_factor.normalized_value == 0.0


def test_diversity_reserve_penalty_increases_with_concentration(
    healthy_candidate: RoutingCandidate,
) -> None:
    scorer = DeterministicRouterScorer()
    state = ScoringState(failure_domain_counts={"openai.com": 2})
    score = scorer.score(_request(), healthy_candidate, state)
    div_factor = next(wf for wf in score.weighted_factors if wf.name == "diversity_reserve")
    assert div_factor.normalized_value < 0.0


# ---------------------------------------------------------------------------
# Cost-to-accepted scoring integration
# ---------------------------------------------------------------------------


def test_cost_factor_uses_expected_total_not_direct_cost(
    cheap_unreliable_candidate: RoutingCandidate,
    expensive_reliable_candidate: RoutingCandidate,
) -> None:
    """Cheap failure-prone model loses when expected total accepted-task cost is higher."""
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
        expected_input_tokens=1000,
        expected_output_tokens=500,
    )
    scorer = DeterministicRouterScorer(
        weights=ScoringWeights(
            **dict.fromkeys(ScoringWeights.__dataclass_fields__, 0.0) | {"cost": 1.0}
        )
    )
    cheap_score = scorer.score(request, cheap_unreliable_candidate)
    reliable_score = scorer.score(request, expensive_reliable_candidate)

    cheap_cost = cheap_score.cost_estimate
    reliable_cost = reliable_score.cost_estimate
    assert cheap_cost is not None
    assert reliable_cost is not None
    assert cheap_cost.expected_total is not None
    assert reliable_cost.expected_total is not None
    # The failure-prone model's expected accepted-task cost is higher.
    assert cheap_cost.expected_total > reliable_cost.expected_total
    # Therefore the reliable model scores higher on the cost factor.
    cheap_factor = next(wf for wf in cheap_score.weighted_factors if wf.name == "cost")
    reliable_factor = next(wf for wf in reliable_score.weighted_factors if wf.name == "cost")
    assert reliable_factor.normalized_value > cheap_factor.normalized_value


def test_unknown_pricing_not_free_in_scoring(healthy_candidate: RoutingCandidate) -> None:
    """A candidate with unknown pricing receives a neutral cost score, not a free-lunch score."""
    from src.routing.capabilities import CostMetadata, ModelCapabilities

    caps = ModelCapabilities(
        context_tokens=healthy_candidate.capabilities.context_tokens,
        supported_roles=healthy_candidate.capabilities.supported_roles,
        cost=CostMetadata(),
    )
    candidate = RoutingCandidate(
        provider_id=healthy_candidate.provider_id,
        model_id=healthy_candidate.model_id,
        route_id=healthy_candidate.route_id,
        model_identity=healthy_candidate.model_identity,
        route_identity=healthy_candidate.route_identity,
        capabilities=caps,
        route_cost_state=None,
    )
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
        expected_input_tokens=1000,
        expected_output_tokens=500,
    )
    scorer = DeterministicRouterScorer()
    score = scorer.score(request, candidate)
    cost_factor = next(wf for wf in score.weighted_factors if wf.name == "cost")
    assert cost_factor.normalized_value == 0.5
    assert score.cost_estimate is not None
    assert score.cost_estimate.expected_total is None


# ---------------------------------------------------------------------------
# Health semantics
# ---------------------------------------------------------------------------


def test_unknown_provider_health_is_neutral_not_healthy() -> None:
    """A candidate with no operational state remains eligible but scores neutrally on health."""
    from src.routing.capabilities import ModelCapabilities
    from src.routing.inference_route import InferenceRouteIdentity, RouteType

    model = ModelIdentity(model_id="mystery", family="mystery", lifecycle=ModelLifecycle.HIGH_RISK)
    route = InferenceRouteIdentity(
        route_id="rt",
        provider_id="pv",
        route_type=RouteType.DIRECT,
        endpoint_key="e",
        failure_domain="f",
    )
    caps = ModelCapabilities(
        context_tokens=10_000, supported_roles=frozenset({ExecutionRole.CODING.value})
    )
    candidate = RoutingCandidate(
        provider_id="pv",
        model_id="mystery",
        route_id="rt",
        model_identity=model,
        route_identity=route,
        capabilities=caps,
        operational_state=None,
    )
    scorer = DeterministicRouterScorer()
    score = scorer.score(_request(), candidate)
    health_factor = next(wf for wf in score.weighted_factors if wf.name == "provider_health")
    assert health_factor.normalized_value == 0.5
    assert "unknown" in health_factor.provenance.lower()
