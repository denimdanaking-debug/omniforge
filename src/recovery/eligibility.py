"""Canonical eligibility seam between Phase 10 recovery and Phase 8 routing.

Recovery rerouting must not resurrect candidates that the Phase 8 eligibility
pipeline would exclude. This adapter converts recovery candidates/inputs into the
same ``CandidateEligibilityPipeline`` used by dynamic routing, then maps the
result back to recovery candidates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.context.budget import estimate_tokens
from src.providers.identity import ProviderQuotaState
from src.recovery.failure_domain import FailureDomainIndex
from src.recovery.reserve import ReserveCapacityPolicy
from src.routing.dynamic.candidate import RoutingCandidate
from src.routing.dynamic.eligibility import CandidateEligibilityPipeline
from src.routing.dynamic.request import DynamicRoutingRequest
from src.routing.policy import ProjectRoutingPolicy, RoutingPolicyEngine

if TYPE_CHECKING:
    from src.recovery.recovery_coordinator import RecoveryCandidate, RecoveryCoordinatorInput


def _required_context_tokens(inputs: RecoveryCoordinatorInput) -> int | None:
    """Return canonical required context tokens, deriving from overflow if needed."""
    if inputs.required_context_tokens is not None:
        return inputs.required_context_tokens
    overflow = inputs.classifier_input.context_overflow
    if overflow is None:
        return None
    if overflow.required_context_tokens is not None:
        return overflow.required_context_tokens
    if overflow.estimated_input_tokens is not None:
        return overflow.estimated_input_tokens
    if overflow.estimated_input_chars is not None:
        return estimate_tokens(overflow.estimated_input_chars)
    return None


def _effective_quota(
    candidate: RecoveryCandidate,
    quota_domain_states: dict[str, ProviderQuotaState] | None,
) -> ProviderQuotaState | None:
    """Return the effective quota state, preferring shared domain state."""
    if candidate.quota_domain and quota_domain_states:
        domain_state = quota_domain_states.get(candidate.quota_domain)
        if domain_state is not None:
            return domain_state
    return candidate.quota


def _recovery_candidate_to_routing(
    candidate: RecoveryCandidate,
    quota_domain_states: dict[str, ProviderQuotaState] | None,
) -> RoutingCandidate:
    """Adapt a recovery candidate to the dynamic-routing candidate shape."""
    return RoutingCandidate(
        provider_id=candidate.provider_id,
        model_id=candidate.model_id,
        route_id=candidate.route_id,
        model_identity=candidate.model_identity,
        route_identity=candidate.route_identity,
        capabilities=candidate.capabilities,
        recovery_state=candidate.recovery_state,
        quota_state=_effective_quota(candidate, quota_domain_states),
        operational_state=candidate.operational_state,
        route_cost_state=candidate.route_cost_state,
    )


def _routing_candidate_to_recovery(
    routing_candidate: RoutingCandidate,
    recovery_candidates: tuple[RecoveryCandidate, ...],
) -> RecoveryCandidate:
    """Map a canonical routing candidate back to the original recovery candidate."""
    key = routing_candidate.identity_key
    lookup = {c.key: c for c in recovery_candidates}
    return lookup[key]


def evaluate_recovery_eligibility(
    inputs: RecoveryCoordinatorInput,
) -> tuple[tuple[RecoveryCandidate, ...], tuple[Any, ...]]:
    """Return recovery candidates that survive the Phase 8 eligibility pipeline.

    The adapter supplies conservative defaults so callers that only have minimal
    recovery state still get deterministic eligibility filtering. All hard gates
    (admin enable, project prohibitions, pins, capabilities, health, quota,
    context capacity, independence, reserve) are evaluated by the shared pipeline.
    """
    classifier_input = inputs.classifier_input
    role = inputs.role if inputs.role is not None else classifier_input.role

    request = DynamicRoutingRequest(
        task_id=classifier_input.task_id,
        project_id=inputs.project_id,
        role=role,
        risk=inputs.current_risk,
        task_class="recovery",
        capability_requirement=inputs.capability_requirement,
        required_context_tokens=_required_context_tokens(inputs),
        pin=inputs.pin,
        reviewer_identities=inputs.reviewer_identities,
        coder_identities=inputs.coder_identities,
    )

    policy_engine = RoutingPolicyEngine(
        provider_enabled=inputs.provider_enabled or {},
        model_enabled=inputs.model_enabled or {},
        route_enabled=inputs.route_enabled or {},
        project_policy=inputs.project_policy or ProjectRoutingPolicy(),
        pin=inputs.pin,
    )

    reserve_policy: ReserveCapacityPolicy | None = None
    if inputs.reserve_policy is not None and isinstance(
        inputs.reserve_policy, ReserveCapacityPolicy
    ):
        reserve_policy = inputs.reserve_policy

    failure_domain_index: FailureDomainIndex | None = inputs.failure_domain_index
    if failure_domain_index is None:
        failure_domain_index = FailureDomainIndex()

    routing_candidates = tuple(
        _recovery_candidate_to_routing(c, inputs.quota_domain_states) for c in inputs.candidates
    )

    pipeline = CandidateEligibilityPipeline(
        policy_engine=policy_engine,
        reserve_policy=reserve_policy,
        failure_domain_index=failure_domain_index,
    )
    result = pipeline.evaluate(request, routing_candidates)

    eligible = tuple(
        _routing_candidate_to_recovery(rc, inputs.candidates) for rc in result.candidates
    )
    return eligible, result.exclusions
