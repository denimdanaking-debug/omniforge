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


def test_empirical_reliability_overrides_prior() -> None:
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
    # Prior for kimi coding is 0.78; blended with 0.5 empirical at 100 attempts
    # should move toward 0.5.
    assert rel_factor.normalized_value < 0.78
    assert rel_factor.normalized_value > 0.5


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
