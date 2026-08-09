"""Candidate eligibility pipeline for dynamic routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from src.context.budget import BudgetType, ContextBudget, compute_usable_budget
from src.policy.risk import lifecycle_eligible
from src.recovery.failure_domain import FailureDomainIndex
from src.recovery.reserve import ReserveCapacityPolicy, evaluate_reserve_eligibility
from src.routing.capabilities import match_capabilities
from src.routing.inference_route import RouteHealth
from src.routing.policy import RoutingPolicyEngine
from src.routing.roles import ExecutionRole

from .candidate import RoutingCandidate
from .decision import ExclusionRecord
from .request import DynamicRoutingRequest


class ExclusionReason(StrEnum):
    """Reason codes for candidate exclusion."""

    PROVIDER_DISABLED = "provider_disabled"
    MODEL_DISABLED = "model_disabled"
    ROUTE_DISABLED = "route_disabled"
    PROJECT_PROVIDER_PROHIBITED = "project_provider_prohibited"
    PROJECT_MODEL_PROHIBITED = "project_model_prohibited"
    PROJECT_ROUTE_PROHIBITED = "project_route_prohibited"
    PIN_MISMATCH = "pin_mismatch"
    ROLE_UNSUPPORTED = "role_unsupported"
    CAPABILITY_MISMATCH = "capability_mismatch"
    RISK_INELIGIBLE = "risk_ineligible"
    PROVIDER_UNHEALTHY = "provider_unhealthy"
    ROUTE_UNHEALTHY = "route_unhealthy"
    QUOTA_EXHAUSTED = "quota_exhausted"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    INDEPENDENCE_VIOLATION = "independence_violation"
    RESERVE_CAPACITY_PROTECTED = "reserve_capacity_protected"
    UNSUPPORTED_ROUTE_PAIRING = "unsupported_route_pairing"


@dataclass(frozen=True)
class EligibilityResult:
    """Outcome of the eligibility pipeline."""

    candidates: tuple[RoutingCandidate, ...]
    exclusions: tuple[ExclusionRecord, ...]


class CandidateEligibilityPipeline:
    """Hard-filter pipeline producing eligible candidates and exclusion records."""

    def __init__(
        self,
        policy_engine: RoutingPolicyEngine,
        reserve_policy: ReserveCapacityPolicy | None = None,
        failure_domain_index: FailureDomainIndex | None = None,
        context_requirement: ContextBudget | None = None,
    ) -> None:
        self._policy_engine = policy_engine
        self._reserve_policy = reserve_policy
        self._failure_domain_index = failure_domain_index
        self._context_requirement = context_requirement

    def evaluate(
        self,
        request: DynamicRoutingRequest,
        candidates: tuple[RoutingCandidate, ...],
    ) -> EligibilityResult:
        """Run hard filters in order; scoring cannot resurrect ineligible candidates."""
        eligible = list(candidates)
        exclusions: list[ExclusionRecord] = []

        eligible, new_exclusions = self._filter_project_prohibitions(request, eligible)
        exclusions.extend(new_exclusions)

        eligible, new_exclusions = self._filter_admin_enable(request, eligible)
        exclusions.extend(new_exclusions)

        eligible, new_exclusions = self._filter_pin(request, eligible)
        exclusions.extend(new_exclusions)

        eligible, new_exclusions = self._filter_role_capability(request, eligible)
        exclusions.extend(new_exclusions)

        eligible, new_exclusions = self._filter_risk_lifecycle(request, eligible)
        exclusions.extend(new_exclusions)

        eligible, new_exclusions = self._filter_provider_health(request, eligible)
        exclusions.extend(new_exclusions)

        eligible, new_exclusions = self._filter_route_health(request, eligible)
        exclusions.extend(new_exclusions)

        eligible, new_exclusions = self._filter_quota_exhausted(request, eligible)
        exclusions.extend(new_exclusions)

        eligible, new_exclusions = self._filter_context_capacity(request, eligible)
        exclusions.extend(new_exclusions)

        eligible, new_exclusions = self._filter_independence(request, eligible)
        exclusions.extend(new_exclusions)

        eligible, new_exclusions = self._filter_reserve(request, eligible)
        exclusions.extend(new_exclusions)

        return EligibilityResult(
            candidates=tuple(eligible),
            exclusions=tuple(exclusions),
        )

    def _exclude(
        self,
        candidate: RoutingCandidate,
        reason: ExclusionReason,
        detail: str,
    ) -> ExclusionRecord:
        return ExclusionRecord(
            provider_id=candidate.provider_id,
            model_id=candidate.model_id,
            route_id=candidate.route_id,
            reason=reason.value,
            detail=detail,
        )

    def _filter_project_prohibitions(
        self, request: DynamicRoutingRequest, candidates: list[RoutingCandidate]
    ) -> tuple[list[RoutingCandidate], list[ExclusionRecord]]:
        eligible: list[RoutingCandidate] = []
        exclusions: list[ExclusionRecord] = []
        policy = self._policy_engine._project_policy
        for candidate in candidates:
            if candidate.provider_id in policy.prohibited_provider_ids:
                detail = (
                    f"provider {candidate.provider_id} prohibited by project {request.project_id}"
                )
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.PROJECT_PROVIDER_PROHIBITED,
                        detail,
                    )
                )
                continue
            if candidate.model_id in policy.prohibited_model_ids:
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.PROJECT_MODEL_PROHIBITED,
                        f"model {candidate.model_id} prohibited by project {request.project_id}",
                    )
                )
                continue
            if candidate.route_id in policy.prohibited_route_ids:
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.PROJECT_ROUTE_PROHIBITED,
                        f"route {candidate.route_id} prohibited by project {request.project_id}",
                    )
                )
                continue
            eligible.append(candidate)
        return eligible, exclusions

    def _filter_admin_enable(
        self, request: DynamicRoutingRequest, candidates: list[RoutingCandidate]
    ) -> tuple[list[RoutingCandidate], list[ExclusionRecord]]:
        eligible: list[RoutingCandidate] = []
        exclusions: list[ExclusionRecord] = []
        for candidate in candidates:
            if not self._policy_engine._provider_enabled.get(candidate.provider_id, True):
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.PROVIDER_DISABLED,
                        f"provider {candidate.provider_id} disabled",
                    )
                )
                continue
            if not self._policy_engine._model_enabled.get(candidate.model_id, True):
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.MODEL_DISABLED,
                        f"model {candidate.model_id} disabled",
                    )
                )
                continue
            if not self._policy_engine._route_enabled.get(candidate.route_id, True):
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.ROUTE_DISABLED,
                        f"route {candidate.route_id} disabled",
                    )
                )
                continue
            eligible.append(candidate)
        return eligible, exclusions

    def _filter_pin(
        self, request: DynamicRoutingRequest, candidates: list[RoutingCandidate]
    ) -> tuple[list[RoutingCandidate], list[ExclusionRecord]]:
        pin = request.pin
        if pin is None:
            return candidates, []
        eligible: list[RoutingCandidate] = []
        exclusions: list[ExclusionRecord] = []
        for candidate in candidates:
            mismatches: list[str] = []
            if pin.provider_id is not None and pin.provider_id != candidate.provider_id:
                mismatches.append("provider")
            if pin.model_id is not None and pin.model_id != candidate.model_id:
                mismatches.append("model")
            if pin.route_id is not None and pin.route_id != candidate.route_id:
                mismatches.append("route")
            if mismatches:
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.PIN_MISMATCH,
                        f"pin mismatch on {','.join(mismatches)}",
                    )
                )
                continue
            eligible.append(candidate)
        return eligible, exclusions

    def _filter_role_capability(
        self, request: DynamicRoutingRequest, candidates: list[RoutingCandidate]
    ) -> tuple[list[RoutingCandidate], list[ExclusionRecord]]:
        eligible: list[RoutingCandidate] = []
        exclusions: list[ExclusionRecord] = []
        role_value = request.role.value
        for candidate in candidates:
            supported = candidate.capabilities.supported_roles
            if supported and role_value not in supported:
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.ROLE_UNSUPPORTED,
                        f"role {role_value} not supported",
                    )
                )
                continue
            req = request.capability_requirement
            if req is not None:
                match = match_capabilities(candidate.capabilities, req)
                if not match.eligible:
                    exclusions.append(
                        self._exclude(
                            candidate,
                            ExclusionReason.CAPABILITY_MISMATCH,
                            f"missing: {','.join(match.missing)}",
                        )
                    )
                    continue
            eligible.append(candidate)
        return eligible, exclusions

    def _filter_risk_lifecycle(
        self, request: DynamicRoutingRequest, candidates: list[RoutingCandidate]
    ) -> tuple[list[RoutingCandidate], list[ExclusionRecord]]:
        eligible: list[RoutingCandidate] = []
        exclusions: list[ExclusionRecord] = []
        for candidate in candidates:
            lifecycle = candidate.model_identity.lifecycle
            if not lifecycle_eligible(lifecycle, request.risk):
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.RISK_INELIGIBLE,
                        f"lifecycle {lifecycle.value} ineligible for {request.risk.name}",
                    )
                )
                continue
            eligible.append(candidate)
        return eligible, exclusions

    def _filter_provider_health(
        self, request: DynamicRoutingRequest, candidates: list[RoutingCandidate]
    ) -> tuple[list[RoutingCandidate], list[ExclusionRecord]]:
        eligible: list[RoutingCandidate] = []
        exclusions: list[ExclusionRecord] = []
        for candidate in candidates:
            state = candidate.operational_state
            if state is None:
                eligible.append(candidate)
                continue
            if not state.is_available():
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.PROVIDER_UNHEALTHY,
                        f"provider health {state.health.value}",
                    )
                )
                continue
            eligible.append(candidate)
        return eligible, exclusions

    def _filter_route_health(
        self, request: DynamicRoutingRequest, candidates: list[RoutingCandidate]
    ) -> tuple[list[RoutingCandidate], list[ExclusionRecord]]:
        eligible: list[RoutingCandidate] = []
        exclusions: list[ExclusionRecord] = []
        for candidate in candidates:
            recovery = candidate.recovery_state
            if recovery is not None and not recovery.is_eligible():
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.ROUTE_UNHEALTHY,
                        f"route health {recovery.health.value}",
                    )
                )
                continue
            route_state = candidate.route_cost_state
            # RATE_LIMITED means the route cannot presently dispatch; it must be
            # excluded until the recovery state machine marks it eligible again.
            if route_state is not None and route_state.health not in {
                RouteHealth.HEALTHY,
                RouteHealth.DEGRADED,
            }:
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.ROUTE_UNHEALTHY,
                        f"route operational health {route_state.health.value}",
                    )
                )
                continue
            eligible.append(candidate)
        return eligible, exclusions

    def _filter_quota_exhausted(
        self, request: DynamicRoutingRequest, candidates: list[RoutingCandidate]
    ) -> tuple[list[RoutingCandidate], list[ExclusionRecord]]:
        eligible: list[RoutingCandidate] = []
        exclusions: list[ExclusionRecord] = []
        for candidate in candidates:
            quota = candidate.quota_state
            if quota is not None and quota.is_exhausted():
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.QUOTA_EXHAUSTED,
                        "quota exhausted",
                    )
                )
                continue
            eligible.append(candidate)
        return eligible, exclusions

    def _filter_context_capacity(
        self, request: DynamicRoutingRequest, candidates: list[RoutingCandidate]
    ) -> tuple[list[RoutingCandidate], list[ExclusionRecord]]:
        required = request.required_context_tokens
        if required is None:
            return candidates, []
        budget = self._context_requirement or ContextBudget(
            primary_budget=required,
            budget_type=BudgetType.TOKENS_ESTIMATE,
            safety_margin_fraction=0.1,
        )
        eligible: list[RoutingCandidate] = []
        exclusions: list[ExclusionRecord] = []
        for candidate in candidates:
            result = compute_usable_budget(candidate.capabilities.context_tokens, budget)
            if result.usable < required:
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.INSUFFICIENT_CONTEXT,
                        f"usable context {result.usable} < required {required}",
                    )
                )
                continue
            eligible.append(candidate)
        return eligible, exclusions

    def _filter_independence(
        self, request: DynamicRoutingRequest, candidates: list[RoutingCandidate]
    ) -> tuple[list[RoutingCandidate], list[ExclusionRecord]]:
        policy = self._policy_engine._project_policy
        level = policy.minimum_review_independence
        if level is None:
            return candidates, []
        if request.role not in {ExecutionRole.REVIEW, ExecutionRole.HIGH_RISK_REVIEW}:
            return candidates, []

        def _parse_identity(identity: str) -> tuple[str, str, str]:
            parts = identity.split(":", 2)
            if len(parts) == 3:
                return parts[0], parts[1], parts[2]
            if len(parts) == 2:
                return parts[0], parts[1], ""
            return identity, "", ""

        coder_providers = {_parse_identity(rid)[0] for rid in request.coder_identities}
        coder_models = {_parse_identity(rid)[1] for rid in request.coder_identities}
        coder_route_ids = {_parse_identity(rid)[2] for rid in request.coder_identities}

        eligible: list[RoutingCandidate] = []
        exclusions: list[ExclusionRecord] = []
        for candidate in candidates:
            if level == "same_provider" and candidate.provider_id in coder_providers:
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.INDEPENDENCE_VIOLATION,
                        f"shares provider {candidate.provider_id} with coder",
                    )
                )
                continue
            if level in {"same_model", "independent"} and candidate.model_id in coder_models:
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.INDEPENDENCE_VIOLATION,
                        f"shares model {candidate.model_id} with coder",
                    )
                )
                continue
            if level == "independent":
                if candidate.provider_id in coder_providers:
                    exclusions.append(
                        self._exclude(
                            candidate,
                            ExclusionReason.INDEPENDENCE_VIOLATION,
                            f"shares provider {candidate.provider_id} with coder",
                        )
                    )
                    continue
                if candidate.route_id in coder_route_ids:
                    exclusions.append(
                        self._exclude(
                            candidate,
                            ExclusionReason.INDEPENDENCE_VIOLATION,
                            f"shares route {candidate.route_id} with coder",
                        )
                    )
                    continue
                domain = candidate.route_identity.failure_domain
                if self._failure_domain_index is not None:
                    coder_domains: set[str] = set()
                    for fd, routes in self._failure_domain_index.to_dict().items():
                        if coder_route_ids & frozenset(routes):
                            coder_domains.add(fd)
                    if domain in coder_domains:
                        exclusions.append(
                            self._exclude(
                                candidate,
                                ExclusionReason.INDEPENDENCE_VIOLATION,
                                f"shares failure domain {domain} with coder",
                            )
                        )
                        continue
            eligible.append(candidate)
        return eligible, exclusions

    def _filter_reserve(
        self, request: DynamicRoutingRequest, candidates: list[RoutingCandidate]
    ) -> tuple[list[RoutingCandidate], list[ExclusionRecord]]:
        if self._reserve_policy is None:
            return candidates, []
        eligible: list[RoutingCandidate] = []
        exclusions: list[ExclusionRecord] = []
        for candidate in candidates:
            result = evaluate_reserve_eligibility(
                role=request.role,
                provider_id=candidate.provider_id,
                model_id=candidate.model_id,
                route_id=candidate.route_id,
                quota_state=candidate.quota_state,
                policy=self._reserve_policy,
            )
            if not result.eligible:
                exclusions.append(
                    self._exclude(
                        candidate,
                        ExclusionReason.RESERVE_CAPACITY_PROTECTED,
                        result.reason,
                    )
                )
                continue
            eligible.append(candidate)
        return eligible, exclusions
