"""Tests that recovery rerouting reuses canonical Phase 8 eligibility gates."""

from __future__ import annotations

from src.policy.risk import RiskLevel
from src.providers.identity import ProviderHealth, ProviderQuotaState, QuotaSignal
from src.recovery.eligibility import evaluate_recovery_eligibility
from src.recovery.failure_classification import FailureClassifierInput
from src.recovery.recovery_coordinator import RecoveryCandidate, RecoveryCoordinatorInput
from src.recovery.reserve import ReserveCapacityPolicy
from src.recovery.retry_policy import FailureRecoveryPolicy
from src.recovery.retry_state import RetryLedger
from src.recovery.state_machine import RouteRecoveryState
from src.routing.capabilities import CapabilityRequirement, ModelCapabilities
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity
from src.routing.policy import ProjectRoutingPolicy, RoutingPin
from src.routing.roles import ExecutionRole


def _candidate(
    provider_id: str,
    model_id: str,
    route_id: str,
    *,
    health: ProviderHealth = ProviderHealth.HEALTHY,
    context_tokens: int = 1000,
    failure_domain: str = "",
    quota: ProviderQuotaState | None = None,
    supported_roles: frozenset[str] | None = None,
    code_generation: bool = False,
) -> RecoveryCandidate:
    if supported_roles is None:
        supported_roles = frozenset({ExecutionRole.CODING.value})
    return RecoveryCandidate(
        provider_id=provider_id,
        model_id=model_id,
        route_id=route_id,
        model_identity=ModelIdentity(model_id=model_id, family=model_id),
        route_identity=InferenceRouteIdentity(
            route_id=route_id,
            provider_id=provider_id,
            route_type=RouteType.DIRECT,
            endpoint_key=route_id,
            failure_domain=failure_domain or provider_id,
        ),
        capabilities=ModelCapabilities(
            context_tokens=context_tokens,
            supported_roles=supported_roles,
            code_generation=code_generation,
        ),
        recovery_state=RouteRecoveryState(health=health),
        quota=quota,
        failure_domain=failure_domain or provider_id,
    )


def _make_input(
    candidates: tuple[RecoveryCandidate, ...],
    **kwargs: object,
) -> RecoveryCoordinatorInput:
    classifier_input = FailureClassifierInput(
        task_id="task-1",
        role=ExecutionRole.CODING,
        provider_id="openai",
        model_id="gpt-4o",
        route_id="openai-direct",
    )
    defaults: dict[str, object] = {
        "classifier_input": classifier_input,
        "candidates": candidates,
        "ledger": RetryLedger("task-1"),
        "policy": FailureRecoveryPolicy(),
        "current_risk": RiskLevel.R2_NORMAL,
        "role": ExecutionRole.CODING,
    }
    defaults.update(kwargs)
    return RecoveryCoordinatorInput(**defaults)  # type: ignore[arg-type]


class TestCanonicalEligibilityExclusions:
    def test_disabled_provider_excluded(self) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        inputs = _make_input(candidates, provider_enabled={"openai": False})
        eligible, _ = evaluate_recovery_eligibility(inputs)
        assert [c.provider_id for c in eligible] == ["anthropic"]

    def test_disabled_model_excluded(self) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        inputs = _make_input(candidates, model_enabled={"gpt-4o": False})
        eligible, _ = evaluate_recovery_eligibility(inputs)
        assert [c.model_id for c in eligible] == ["claude"]

    def test_disabled_route_excluded(self) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("openai", "gpt-4o", "openai-fallback"),
        )
        inputs = _make_input(candidates, route_enabled={"openai-direct": False})
        eligible, _ = evaluate_recovery_eligibility(inputs)
        assert [c.route_id for c in eligible] == ["openai-fallback"]

    def test_project_prohibited_provider_excluded(self) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        policy = ProjectRoutingPolicy(prohibited_provider_ids=frozenset({"openai"}))
        inputs = _make_input(candidates, project_policy=policy)
        eligible, _ = evaluate_recovery_eligibility(inputs)
        assert [c.provider_id for c in eligible] == ["anthropic"]

    def test_project_prohibited_model_excluded(self) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        project_policy = ProjectRoutingPolicy(prohibited_model_ids=frozenset({"gpt-4o"}))
        inputs = _make_input(candidates, project_policy=project_policy)
        eligible, _ = evaluate_recovery_eligibility(inputs)
        assert [c.model_id for c in eligible] == ["claude"]

    def test_project_prohibited_route_excluded(self) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("openai", "gpt-4o", "openai-fallback"),
        )
        project_policy = ProjectRoutingPolicy(prohibited_route_ids=frozenset({"openai-direct"}))
        inputs = _make_input(candidates, project_policy=project_policy)
        eligible, _ = evaluate_recovery_eligibility(inputs)
        assert [c.route_id for c in eligible] == ["openai-fallback"]

    def test_capability_incompatible_candidate_excluded(self) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct", code_generation=True),
        )
        req = CapabilityRequirement(code_generation=True)
        inputs = _make_input(candidates, capability_requirement=req)
        eligible, _ = evaluate_recovery_eligibility(inputs)
        assert [c.model_id for c in eligible] == ["claude"]

    def test_quota_exhausted_alternate_excluded(self) -> None:
        candidates = (
            _candidate(
                "openai",
                "gpt-4o",
                "openai-direct",
                quota=ProviderQuotaState(provider_signal=QuotaSignal.EXHAUSTED),
            ),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        inputs = _make_input(candidates)
        eligible, _ = evaluate_recovery_eligibility(inputs)
        assert [c.provider_id for c in eligible] == ["anthropic"]

    def test_reserve_protected_candidate_excluded(self) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        reserve = ReserveCapacityPolicy(
            reserved_provider_ids=frozenset({"anthropic"}),
            reserved_roles=frozenset({"review"}),
        )
        inputs = _make_input(candidates, reserve_policy=reserve)
        eligible, _ = evaluate_recovery_eligibility(inputs)
        assert [c.provider_id for c in eligible] == ["openai"]

    def test_context_ineligible_candidate_excluded(self) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct", context_tokens=1000),
            _candidate("anthropic", "claude", "anthropic-direct", context_tokens=10000),
        )
        inputs = _make_input(candidates, required_context_tokens=5000)
        eligible, _ = evaluate_recovery_eligibility(inputs)
        assert [c.model_id for c in eligible] == ["claude"]

    def test_independence_violating_candidate_excluded(self) -> None:
        review_roles = frozenset({ExecutionRole.REVIEW.value})
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct", supported_roles=review_roles),
            _candidate("anthropic", "claude", "anthropic-direct", supported_roles=review_roles),
        )
        project_policy = ProjectRoutingPolicy(minimum_review_independence="independent")
        inputs = _make_input(
            candidates,
            role=ExecutionRole.REVIEW,
            project_policy=project_policy,
            coder_identities=("openai:gpt-4o:openai-direct",),
        )
        eligible, _ = evaluate_recovery_eligibility(inputs)
        assert [c.provider_id for c in eligible] == ["anthropic"]

    def test_pin_mismatched_candidate_excluded(self) -> None:
        candidates = (
            _candidate("openai", "gpt-4o", "openai-direct"),
            _candidate("anthropic", "claude", "anthropic-direct"),
        )
        pin = RoutingPin(provider_id="anthropic")
        inputs = _make_input(candidates, pin=pin)
        eligible, _ = evaluate_recovery_eligibility(inputs)
        assert [c.provider_id for c in eligible] == ["anthropic"]

    def test_candidate_ordering_does_not_change_eligible_set(self) -> None:
        c1 = _candidate("openai", "gpt-4o", "openai-direct")
        c2 = _candidate("anthropic", "claude", "anthropic-direct")
        inputs1 = _make_input((c1, c2))
        inputs2 = _make_input((c2, c1))
        eligible1, _ = evaluate_recovery_eligibility(inputs1)
        eligible2, _ = evaluate_recovery_eligibility(inputs2)
        assert {c.key for c in eligible1} == {c.key for c in eligible2}
