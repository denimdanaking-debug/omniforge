from __future__ import annotations

from src.policy.risk import RiskLevel
from src.routing.dynamic.candidate import RoutingCandidate
from src.routing.dynamic.cost import estimate_cost_to_accepted
from src.routing.dynamic.request import DynamicRoutingRequest
from src.routing.roles import ExecutionRole


def test_unknown_cost_returns_none_total(healthy_candidate: RoutingCandidate) -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
        expected_input_tokens=1000,
        expected_output_tokens=500,
    )
    # Clear cost info from both route state and capabilities.
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
    estimate = estimate_cost_to_accepted(request, candidate)
    assert estimate.expected_total is None
    assert estimate.confidence == "UNKNOWN"


def test_cheap_failure_prone_can_lose_to_reliable(
    cheap_unreliable_candidate: RoutingCandidate,
    expensive_reliable_candidate: RoutingCandidate,
) -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
        expected_input_tokens=1000,
        expected_output_tokens=500,
    )
    cheap = estimate_cost_to_accepted(request, cheap_unreliable_candidate)
    reliable = estimate_cost_to_accepted(request, expensive_reliable_candidate)
    assert cheap.expected_total is not None
    assert reliable.expected_total is not None
    assert reliable.expected_total < cheap.expected_total


def test_expected_total_includes_retry_and_repair(
    cheap_unreliable_candidate: RoutingCandidate,
) -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
        expected_input_tokens=1000,
        expected_output_tokens=500,
    )
    estimate = estimate_cost_to_accepted(request, cheap_unreliable_candidate)
    assert estimate.direct_attempt_cost is not None
    assert estimate.expected_retry_cost is not None
    assert estimate.expected_repair_cost is not None
    assert estimate.expected_total is not None
    assert estimate.expected_total > estimate.direct_attempt_cost
