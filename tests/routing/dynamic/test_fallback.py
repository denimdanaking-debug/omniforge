from __future__ import annotations

from datetime import UTC, datetime

from src.policy.risk import RiskLevel
from src.routing.capabilities import ModelCapabilities
from src.routing.dynamic.candidate import RoutingCandidate
from src.routing.dynamic.eligibility import CandidateEligibilityPipeline
from src.routing.dynamic.fallback import EmergencyFallbackRouter
from src.routing.dynamic.request import DynamicRoutingRequest
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity, ModelLifecycle
from src.routing.policy import RoutingPolicyEngine
from src.routing.roles import ExecutionRole


def test_emergency_fallback_triggered_when_no_eligible(healthy_candidate: RoutingCandidate) -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )
    # Disable the only candidate.
    engine = RoutingPolicyEngine(provider_enabled={"openai": False})
    pipeline = CandidateEligibilityPipeline(engine)
    router = EmergencyFallbackRouter()
    decision = router.route(
        request, (healthy_candidate,), pipeline, "dec-1", timestamp=datetime.now(UTC)
    )
    assert decision.emergency_fallback_used
    assert decision.record.fallback_used
    assert decision.selected_candidate is None


def test_emergency_fallback_still_filters_ineligible(healthy_candidate: RoutingCandidate) -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )
    # Keep candidate eligible so fallback list can pick it.
    engine = RoutingPolicyEngine()
    pipeline = CandidateEligibilityPipeline(engine)
    router = EmergencyFallbackRouter()
    decision = router.route(
        request, (healthy_candidate,), pipeline, "dec-1", timestamp=datetime.now(UTC)
    )
    assert decision.emergency_fallback_used
    assert decision.selected_candidate is not None


def test_emergency_fallback_uses_role_specific_order() -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.REVIEW,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )
    c1 = RoutingCandidate(
        provider_id="openai",
        model_id="gpt-4o",
        route_id="openai-direct",
        model_identity=ModelIdentity(
            model_id="gpt-4o", family="gpt", lifecycle=ModelLifecycle.HIGH_RISK
        ),
        route_identity=InferenceRouteIdentity(
            route_id="openai-direct",
            provider_id="openai",
            route_type=RouteType.DIRECT,
            endpoint_key="e",
            failure_domain="openai.com",
        ),
        capabilities=ModelCapabilities(
            context_tokens=128_000, supported_roles=frozenset({ExecutionRole.REVIEW.value})
        ),
    )
    c2 = RoutingCandidate(
        provider_id="anthropic",
        model_id="claude-sonnet",
        route_id="anthropic-direct",
        model_identity=ModelIdentity(
            model_id="claude-sonnet", family="claude", lifecycle=ModelLifecycle.HIGH_RISK
        ),
        route_identity=InferenceRouteIdentity(
            route_id="anthropic-direct",
            provider_id="anthropic",
            route_type=RouteType.DIRECT,
            endpoint_key="e",
            failure_domain="anthropic.com",
        ),
        capabilities=ModelCapabilities(
            context_tokens=128_000, supported_roles=frozenset({ExecutionRole.REVIEW.value})
        ),
    )
    engine = RoutingPolicyEngine()
    pipeline = CandidateEligibilityPipeline(engine)
    router = EmergencyFallbackRouter()
    decision = router.route(request, (c1, c2), pipeline, "dec-1", timestamp=datetime.now(UTC))
    assert decision.selected_candidate == c2  # anthropic first for REVIEW


def test_fallback_record_contains_full_fingerprint(healthy_candidate: RoutingCandidate) -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )
    engine = RoutingPolicyEngine(provider_enabled={"openai": False})
    pipeline = CandidateEligibilityPipeline(engine)
    router = EmergencyFallbackRouter()
    decision = router.route(
        request,
        (healthy_candidate,),
        pipeline,
        "dec-1",
        input_fingerprint="sha256-deadbeef",
        timestamp=datetime.now(UTC),
    )
    assert decision.record.input_fingerprint == "sha256-deadbeef"
    assert len(decision.record.input_fingerprint) > 0


def test_fallback_preserves_exploration_metadata(healthy_candidate: RoutingCandidate) -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )
    engine = RoutingPolicyEngine(provider_enabled={"openai": False})
    pipeline = CandidateEligibilityPipeline(engine)
    router = EmergencyFallbackRouter()
    decision = router.route(
        request,
        (healthy_candidate,),
        pipeline,
        "dec-1",
        exploration_enabled=True,
        timestamp=datetime.now(UTC),
    )
    assert decision.record.exploration_enabled is True


def test_fallback_configured_order_deterministic_regardless_of_input_order() -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )

    def make(provider_id: str, model_id: str, route_id: str) -> RoutingCandidate:
        return RoutingCandidate(
            provider_id=provider_id,
            model_id=model_id,
            route_id=route_id,
            model_identity=ModelIdentity(
                model_id=model_id, family=provider_id, lifecycle=ModelLifecycle.HIGH_RISK
            ),
            route_identity=InferenceRouteIdentity(
                route_id=route_id,
                provider_id=provider_id,
                route_type=RouteType.DIRECT,
                endpoint_key="e",
                failure_domain=f"{provider_id}.com",
            ),
            capabilities=ModelCapabilities(
                context_tokens=128_000, supported_roles=frozenset({ExecutionRole.CODING.value})
            ),
        )

    a = make("pv", "model-a", "route-a")
    b = make("pv", "model-b", "route-b")

    router = EmergencyFallbackRouter(
        fallback_orders={ExecutionRole.CODING: ["pv:model-b:route-b", "pv:model-a:route-a"]}
    )
    engine = RoutingPolicyEngine()
    pipeline = CandidateEligibilityPipeline(engine)

    decision_ab = router.route(request, (a, b), pipeline, "dec-1", timestamp=datetime.now(UTC))
    decision_ba = router.route(request, (b, a), pipeline, "dec-2", timestamp=datetime.now(UTC))

    assert decision_ab.selected_candidate == b
    assert decision_ba.selected_candidate == b
    assert decision_ab.record.runner_up == a
    assert decision_ba.record.runner_up == a
    assert decision_ab.ranked_candidates == decision_ba.ranked_candidates


def test_fallback_unlisted_candidates_sorted_by_identity_key() -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )

    def make(provider_id: str, model_id: str, route_id: str) -> RoutingCandidate:
        return RoutingCandidate(
            provider_id=provider_id,
            model_id=model_id,
            route_id=route_id,
            model_identity=ModelIdentity(
                model_id=model_id, family=provider_id, lifecycle=ModelLifecycle.HIGH_RISK
            ),
            route_identity=InferenceRouteIdentity(
                route_id=route_id,
                provider_id=provider_id,
                route_type=RouteType.DIRECT,
                endpoint_key="e",
                failure_domain=f"{provider_id}.com",
            ),
            capabilities=ModelCapabilities(
                context_tokens=128_000, supported_roles=frozenset({ExecutionRole.CODING.value})
            ),
        )

    # Neither candidate appears in the configured fallback order.
    a = make("z", "model", "route")
    b = make("a", "model", "route")

    router = EmergencyFallbackRouter(fallback_orders={ExecutionRole.CODING: []})
    engine = RoutingPolicyEngine()
    pipeline = CandidateEligibilityPipeline(engine)

    decision_ab = router.route(request, (a, b), pipeline, "dec-1", timestamp=datetime.now(UTC))
    decision_ba = router.route(request, (b, a), pipeline, "dec-2", timestamp=datetime.now(UTC))

    # Canonical identity-key order is "a:model:route" before "z:model:route".
    assert decision_ab.selected_candidate == b
    assert decision_ba.selected_candidate == b
    assert decision_ab.ranked_candidates == decision_ba.ranked_candidates
    assert decision_ab.record.input_fingerprint == decision_ba.record.input_fingerprint


def test_fallback_no_eligible_candidates_returns_empty_result() -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )

    candidate = RoutingCandidate(
        provider_id="openai",
        model_id="gpt-4o",
        route_id="openai-direct",
        model_identity=ModelIdentity(
            model_id="gpt-4o", family="gpt", lifecycle=ModelLifecycle.HIGH_RISK
        ),
        route_identity=InferenceRouteIdentity(
            route_id="openai-direct",
            provider_id="openai",
            route_type=RouteType.DIRECT,
            endpoint_key="e",
            failure_domain="openai.com",
        ),
        capabilities=ModelCapabilities(
            context_tokens=128_000, supported_roles=frozenset({ExecutionRole.CODING.value})
        ),
    )
    engine = RoutingPolicyEngine(provider_enabled={"openai": False})
    pipeline = CandidateEligibilityPipeline(engine)
    router = EmergencyFallbackRouter()
    decision = router.route(
        request,
        (candidate,),
        pipeline,
        "dec-1",
        input_fingerprint="sha256-no-eligible",
        timestamp=datetime.now(UTC),
    )
    assert decision.selected_candidate is None
    assert decision.no_eligible_reason is not None
    assert decision.record.fallback_used is True
    assert decision.record.input_fingerprint == "sha256-no-eligible"
