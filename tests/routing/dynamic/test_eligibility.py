from __future__ import annotations

from typing import Any

from src.policy.risk import RiskLevel
from src.providers.identity import (
    ProviderHealth,
    ProviderOperationalState,
    ProviderQuotaState,
    QuotaSignal,
)
from src.recovery.failure_domain import FailureDomainIndex
from src.recovery.reserve import ReserveCapacityPolicy
from src.recovery.state_machine import RouteRecoveryState
from src.routing.capabilities import CapabilityRequirement, ModelCapabilities
from src.routing.dynamic.candidate import RoutingCandidate
from src.routing.dynamic.eligibility import (
    CandidateEligibilityPipeline,
    ExclusionReason,
)
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


def _candidate(
    provider_id: str = "openai",
    model_id: str = "gpt-4o",
    route_id: str = "openai-direct",
    lifecycle: ModelLifecycle = ModelLifecycle.HIGH_RISK,
    context_tokens: int = 128_000,
    supported_roles: frozenset[str] | None = None,
    provider_health: ProviderHealth = ProviderHealth.HEALTHY,
    quota_signal: QuotaSignal = QuotaSignal.AVAILABLE,
    recovery_health: ProviderHealth = ProviderHealth.HEALTHY,
    route_health: RouteHealth = RouteHealth.HEALTHY,
    failure_domain: str = "openai.com",
) -> RoutingCandidate:
    supported = supported_roles or frozenset({ExecutionRole.CODING.value})
    return RoutingCandidate(
        provider_id=provider_id,
        model_id=model_id,
        route_id=route_id,
        model_identity=ModelIdentity(model_id=model_id, family="gpt", lifecycle=lifecycle),
        route_identity=InferenceRouteIdentity(
            route_id=route_id,
            provider_id=provider_id,
            route_type=RouteType.DIRECT,
            endpoint_key="https://example.com",
            failure_domain=failure_domain,
        ),
        capabilities=ModelCapabilities(
            context_tokens=context_tokens,
            code_generation=True,
            supported_roles=supported,
        ),
        operational_state=ProviderOperationalState(health=provider_health),
        quota_state=ProviderQuotaState(provider_signal=quota_signal),
        recovery_state=RouteRecoveryState(health=recovery_health),
        route_cost_state=RouteOperationalState(health=route_health),
    )


def _request(
    role: ExecutionRole = ExecutionRole.CODING,
    risk: RiskLevel = RiskLevel.R2_NORMAL,
    pin: RoutingPin | None = None,
    required_context_tokens: int | None = None,
    reviewer_identities: tuple[str, ...] = (),
    coder_identities: tuple[str, ...] = (),
) -> DynamicRoutingRequest:
    return DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=role,
        risk=risk,
        task_class="default",
        pin=pin,
        required_context_tokens=required_context_tokens,
        reviewer_identities=reviewer_identities,
        coder_identities=coder_identities,
    )


def _engine(**kwargs: Any) -> RoutingPolicyEngine:
    return RoutingPolicyEngine(**kwargs)


def test_provider_disabled_excluded() -> None:
    candidate = _candidate()
    engine = _engine(provider_enabled={"openai": False})
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(), (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.PROVIDER_DISABLED for e in result.exclusions)


def test_model_disabled_excluded() -> None:
    candidate = _candidate()
    engine = _engine(model_enabled={"gpt-4o": False})
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(), (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.MODEL_DISABLED for e in result.exclusions)


def test_route_disabled_excluded() -> None:
    candidate = _candidate()
    engine = _engine(route_enabled={"openai-direct": False})
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(), (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.ROUTE_DISABLED for e in result.exclusions)


def test_project_provider_prohibited() -> None:
    candidate = _candidate()
    policy = ProjectRoutingPolicy(prohibited_provider_ids=frozenset({"openai"}))
    engine = _engine(project_policy=policy)
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(), (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.PROJECT_PROVIDER_PROHIBITED for e in result.exclusions)


def test_project_model_prohibited() -> None:
    candidate = _candidate()
    policy = ProjectRoutingPolicy(prohibited_model_ids=frozenset({"gpt-4o"}))
    engine = _engine(project_policy=policy)
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(), (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.PROJECT_MODEL_PROHIBITED for e in result.exclusions)


def test_project_route_prohibited() -> None:
    candidate = _candidate()
    policy = ProjectRoutingPolicy(prohibited_route_ids=frozenset({"openai-direct"}))
    engine = _engine(project_policy=policy)
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(), (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.PROJECT_ROUTE_PROHIBITED for e in result.exclusions)


def test_project_prohibition_beats_pin() -> None:
    candidate = _candidate()
    pin = RoutingPin(provider_id="openai")
    policy = ProjectRoutingPolicy(prohibited_provider_ids=frozenset({"openai"}))
    engine = _engine(project_policy=policy)
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(pin=pin), (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.PROJECT_PROVIDER_PROHIBITED for e in result.exclusions)


def test_pin_mismatch_excluded() -> None:
    candidate = _candidate()
    pin = RoutingPin(provider_id="anthropic")
    engine = _engine()
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(pin=pin), (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.PIN_MISMATCH for e in result.exclusions)


def test_pin_match_kept() -> None:
    candidate = _candidate()
    pin = RoutingPin(provider_id="openai", model_id="gpt-4o")
    engine = _engine()
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(pin=pin), (candidate,))
    assert len(result.candidates) == 1


def test_role_unsupported_excluded() -> None:
    candidate = _candidate(supported_roles=frozenset({ExecutionRole.REVIEW.value}))
    engine = _engine()
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(role=ExecutionRole.CODING), (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.ROLE_UNSUPPORTED for e in result.exclusions)


def test_capability_mismatch_excluded() -> None:
    candidate = _candidate()
    engine = _engine()
    req = CapabilityRequirement(min_context_tokens=256_000, structured_output=True)
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(
        DynamicRoutingRequest(
            task_id="t1",
            project_id="p1",
            role=ExecutionRole.CODING,
            risk=RiskLevel.R2_NORMAL,
            task_class="default",
            capability_requirement=req,
        ),
        (candidate,),
    )
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.CAPABILITY_MISMATCH for e in result.exclusions)


def test_risk_lifecycle_excluded() -> None:
    candidate = _candidate(lifecycle=ModelLifecycle.LOW_RISK)
    engine = _engine()
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(risk=RiskLevel.R3_HIGH), (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.RISK_INELIGIBLE for e in result.exclusions)


def test_provider_unhealthy_excluded() -> None:
    candidate = _candidate(provider_health=ProviderHealth.UNAVAILABLE)
    engine = _engine()
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(), (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.PROVIDER_UNHEALTHY for e in result.exclusions)


def test_route_unhealthy_excluded() -> None:
    candidate = _candidate(recovery_health=ProviderHealth.UNAVAILABLE)
    engine = _engine()
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(), (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.ROUTE_UNHEALTHY for e in result.exclusions)


def test_quota_exhausted_excluded() -> None:
    candidate = _candidate(quota_signal=QuotaSignal.EXHAUSTED)
    engine = _engine()
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(), (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.QUOTA_EXHAUSTED for e in result.exclusions)


def test_insufficient_context_excluded() -> None:
    candidate = _candidate(context_tokens=10_000)
    engine = _engine()
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(required_context_tokens=100_000), (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.INSUFFICIENT_CONTEXT for e in result.exclusions)


def test_independence_same_provider() -> None:
    candidate = _candidate(supported_roles=frozenset({ExecutionRole.REVIEW.value}))
    policy = ProjectRoutingPolicy(minimum_review_independence="same_provider")
    engine = _engine(project_policy=policy)
    pipeline = CandidateEligibilityPipeline(engine)
    request = _request(
        role=ExecutionRole.REVIEW,
        coder_identities=("openai:gpt-4o:openai-direct",),
    )
    result = pipeline.evaluate(request, (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.INDEPENDENCE_VIOLATION for e in result.exclusions)


def test_independence_same_model() -> None:
    candidate = _candidate(
        provider_id="openai",
        model_id="gpt-4o",
        supported_roles=frozenset({ExecutionRole.REVIEW.value}),
    )
    other_provider = _candidate(
        provider_id="azure",
        model_id="gpt-4o",
        route_id="azure-openai",
        supported_roles=frozenset({ExecutionRole.REVIEW.value}),
    )
    policy = ProjectRoutingPolicy(minimum_review_independence="same_model")
    engine = _engine(project_policy=policy)
    pipeline = CandidateEligibilityPipeline(engine)
    request = _request(
        role=ExecutionRole.REVIEW,
        coder_identities=("openai:gpt-4o:openai-direct",),
    )
    result = pipeline.evaluate(request, (candidate, other_provider))
    assert candidate not in result.candidates
    assert other_provider not in result.candidates
    assert (
        len([e for e in result.exclusions if e.reason == ExclusionReason.INDEPENDENCE_VIOLATION])
        == 2
    )


def test_independence_independent_failure_domain() -> None:
    candidate = _candidate(
        failure_domain="openai.com",
        supported_roles=frozenset({ExecutionRole.REVIEW.value}),
    )
    policy = ProjectRoutingPolicy(minimum_review_independence="independent")
    engine = _engine(project_policy=policy)
    fdi = FailureDomainIndex()
    fdi.register("openai-direct", "openai.com")
    fdi.register("other-route", "openai.com")
    pipeline = CandidateEligibilityPipeline(engine, failure_domain_index=fdi)
    request = _request(
        role=ExecutionRole.REVIEW,
        coder_identities=("other-provider:other-model:other-route",),
    )
    result = pipeline.evaluate(request, (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.INDEPENDENCE_VIOLATION for e in result.exclusions)


def test_reserve_capacity_protected() -> None:
    candidate = _candidate()
    engine = _engine()
    reserve = ReserveCapacityPolicy(
        reserved_provider_ids=frozenset({"openai"}),
        reserved_roles=frozenset({ExecutionRole.HIGH_RISK_REVIEW.value}),
    )
    pipeline = CandidateEligibilityPipeline(engine, reserve_policy=reserve)
    request = _request(role=ExecutionRole.CODING)
    result = pipeline.evaluate(request, (candidate,))
    assert len(result.candidates) == 0
    assert any(e.reason == ExclusionReason.RESERVE_CAPACITY_PROTECTED for e in result.exclusions)


def test_eligible_candidate_passes_all_filters() -> None:
    candidate = _candidate()
    engine = _engine()
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(), (candidate,))
    assert len(result.candidates) == 1
    assert candidate in result.candidates
    assert len(result.exclusions) == 0


def test_multiple_exclusions_recorded() -> None:
    c1 = _candidate()
    c2 = _candidate(provider_id="anthropic", model_id="claude", route_id="anthropic-direct")
    engine = _engine(
        model_enabled={"gpt-4o": False},
        provider_enabled={"anthropic": False},
    )
    pipeline = CandidateEligibilityPipeline(engine)
    result = pipeline.evaluate(_request(), (c1, c2))
    assert len(result.candidates) == 0
    assert len(result.exclusions) == 2
