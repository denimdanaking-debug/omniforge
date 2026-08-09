from __future__ import annotations

from datetime import UTC, datetime

from src.policy.risk import RiskLevel
from src.routing.dynamic.candidate import RoutingCandidate
from src.routing.dynamic.eligibility import CandidateEligibilityPipeline
from src.routing.dynamic.fallback import EmergencyFallbackRouter
from src.routing.dynamic.request import DynamicRoutingRequest
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
    from src.routing.capabilities import ModelCapabilities
    from src.routing.inference_route import InferenceRouteIdentity, RouteType
    from src.routing.model_identity import ModelIdentity, ModelLifecycle

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
