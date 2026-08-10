"""Failure-type-aware recovery coordinator.

Turns a FailureClassification, retry history, risk assessment, and eligible
candidates into a deterministic RecoveryDecision. Recovery rerouting reuses the
Phase 8 ``CandidateEligibilityPipeline`` so ineligible candidates cannot be
resurrected. No provider calls are made inside the coordinator.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.context.budget import BudgetType, ContextBudget, compute_usable_budget, estimate_tokens
from src.context.schema import AuthorityPresence, ContextPacket
from src.context.validation import ContextPacketValidator, ValidationIssue
from src.policy.risk import RiskLevel
from src.providers.identity import ProviderOperationalState, ProviderQuotaState
from src.recovery.backoff import BackoffPolicy
from src.recovery.clock import Clock
from src.recovery.eligibility import (
    _effective_quota,
    _required_context_tokens,
    evaluate_recovery_eligibility,
)
from src.recovery.failure_classification import (
    FailureCategory,
    FailureClassification,
    FailureClassifier,
    FailureClassifierInput,
)
from src.recovery.fingerprint import recovery_input_fingerprint
from src.recovery.retry_policy import FailureRecoveryPolicy
from src.recovery.retry_state import FailureAttemptRecord, RetryLedger, RetryType, WaitState
from src.recovery.state_machine import RouteRecoveryState
from src.risk.context_policy import RiskContextPolicy, RiskContextRequirements
from src.routing.capabilities import CapabilityRequirement, ModelCapabilities
from src.routing.inference_route import InferenceRouteIdentity, RouteOperationalState
from src.routing.model_identity import ModelIdentity
from src.routing.policy import RoutingPin
from src.routing.roles import ExecutionRole


class RecoveryAction(StrEnum):
    """Typed recovery actions."""

    RETRY_SAME_ROUTE = "RETRY_SAME_ROUTE"
    RETRY_ALTERNATE_ROUTE = "RETRY_ALTERNATE_ROUTE"
    REROUTE_PROVIDER = "REROUTE_PROVIDER"
    REROUTE_MODEL = "REROUTE_MODEL"
    CONSTRAINED_OUTPUT_RETRY = "CONSTRAINED_OUTPUT_RETRY"
    REPLAN = "REPLAN"
    REPAIR_WITH_EVIDENCE = "REPAIR_WITH_EVIDENCE"
    CROSS_MODEL_REPAIR = "CROSS_MODEL_REPAIR"
    REBUILD_CONTEXT = "REBUILD_CONTEXT"
    WAIT_FOR_PROVIDER = "WAIT_FOR_PROVIDER"
    BLOCK = "BLOCK"
    CANCEL = "CANCEL"


@dataclass(frozen=True)
class RecoveryCandidate:
    """One candidate under consideration by the recovery coordinator."""

    provider_id: str
    model_id: str
    route_id: str
    model_identity: ModelIdentity
    route_identity: InferenceRouteIdentity
    capabilities: ModelCapabilities
    recovery_state: RouteRecoveryState
    quota: ProviderQuotaState | None = None
    quota_domain: str | None = None
    failure_domain: str = ""
    operational_state: ProviderOperationalState | None = None
    route_cost_state: RouteOperationalState | None = None

    @property
    def key(self) -> str:
        return f"{self.provider_id}:{self.model_id}:{self.route_id}"


@dataclass(frozen=True)
class RecoveryDecision:
    """Deterministic recovery decision."""

    action: RecoveryAction
    classification: FailureClassification
    failure_signature: str
    deterministic_input_fingerprint: str
    retry_allowed: bool
    selected_candidate: RecoveryCandidate | None
    require_reroute: bool
    require_context_rebuild: bool
    require_risk_escalation: bool
    wait_reason: str
    retry_after: datetime.datetime | None
    evidence_packet: dict[str, Any]
    attempt_counters: dict[str, int]
    terminal: bool
    explanation: str
    transition_fingerprint: str = ""


@dataclass(frozen=True)
class RecoveryCoordinatorInput:
    """Normalized inputs for the recovery coordinator."""

    classifier_input: FailureClassifierInput
    candidates: tuple[RecoveryCandidate, ...]
    ledger: RetryLedger
    policy: FailureRecoveryPolicy
    current_risk: RiskLevel
    project_id: str = "default"
    project_policy: Any | None = None
    provider_enabled: dict[str, bool] | None = None
    model_enabled: dict[str, bool] | None = None
    route_enabled: dict[str, bool] | None = None
    capability_requirement: CapabilityRequirement | None = None
    required_context_tokens: int | None = None
    pin: RoutingPin | None = None
    reserve_policy: Any | None = None
    failure_domain_index: Any | None = None
    quota_domain_states: dict[str, ProviderQuotaState] | None = None
    reviewer_identities: tuple[str, ...] = ()
    coder_identities: tuple[str, ...] = ()
    role: ExecutionRole | None = None
    context_packet: ContextPacket | None = None
    risk_context_requirements: RiskContextRequirements | None = None
    risk_context_policy: RiskContextPolicy | None = None


class RecoveryCoordinator:
    """Deterministic recovery decision engine."""

    def __init__(
        self,
        classifier: FailureClassifier | None = None,
        backoff: BackoffPolicy | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._classifier = classifier or FailureClassifier()
        self._backoff = backoff or BackoffPolicy()
        self._clock = clock

    def decide(self, inputs: RecoveryCoordinatorInput) -> RecoveryDecision:
        """Return a deterministic recovery decision."""
        classification = self._classifier.classify(inputs.classifier_input)
        signature = classification.deterministic_fingerprint
        ledger = inputs.ledger

        # Canonical Phase 8 eligibility is applied to the candidate fleet first.
        eligible_candidates, exclusions = evaluate_recovery_eligibility(inputs)
        fingerprint = recovery_input_fingerprint(
            inputs, classification, eligible_candidates, exclusions
        )

        # Global bounds.
        if ledger.attempt_count >= inputs.policy.max_total_attempts:
            return self._terminal(
                inputs,
                classification,
                signature,
                fingerprint,
                RecoveryAction.BLOCK,
                "max_total_attempts reached",
                attempt_counters=self._counters(ledger),
            )

        same_signature = ledger.signature_count(signature)
        if same_signature >= inputs.policy.max_same_signature_attempts:
            return self._terminal(
                inputs,
                classification,
                signature,
                fingerprint,
                RecoveryAction.BLOCK,
                "max_same_signature_attempts reached",
                attempt_counters=self._counters(ledger),
            )

        if classification.category is FailureCategory.CANCELLED:
            return self._terminal(
                inputs,
                classification,
                signature,
                fingerprint,
                RecoveryAction.CANCEL,
                "cancelled: no auto-retry",
                attempt_counters=self._counters(ledger),
            )

        if classification.category is FailureCategory.AUTHORITY_VIOLATION:
            return self._handle_authority_violation(inputs, classification, signature, fingerprint)

        if classification.category is FailureCategory.CONTEXT_CAPACITY:
            return self._handle_context_overflow(inputs, classification, signature, fingerprint)

        if classification.category in {
            FailureCategory.INFRASTRUCTURE_TRANSIENT,
            FailureCategory.INFRASTRUCTURE_UNAVAILABLE,
        }:
            return self._handle_infrastructure_transient(
                inputs, classification, signature, fingerprint, eligible_candidates
            )

        if classification.category is FailureCategory.INFRASTRUCTURE_QUOTA:
            return self._handle_quota_exhaustion(
                inputs, classification, signature, fingerprint, eligible_candidates
            )

        if classification.category is FailureCategory.INFRASTRUCTURE_AUTH:
            return self._handle_auth_failure(
                inputs, classification, signature, fingerprint, eligible_candidates
            )

        if classification.category is FailureCategory.CAPABILITY_MISMATCH:
            return self._handle_capability_mismatch(
                inputs, classification, signature, fingerprint, eligible_candidates
            )

        if classification.category is FailureCategory.STRUCTURED_OUTPUT_INVALID:
            return self._handle_structured_output_invalid(
                inputs, classification, signature, fingerprint, eligible_candidates
            )

        if classification.category is FailureCategory.PLANNING_OUTPUT_INVALID:
            return self._handle_planning_output_invalid(
                inputs, classification, signature, fingerprint, eligible_candidates
            )

        if classification.category is FailureCategory.IMPLEMENTATION_DETERMINISTIC:
            return self._handle_deterministic_implementation(
                inputs, classification, signature, fingerprint, eligible_candidates
            )

        if classification.category is FailureCategory.IMPLEMENTATION_CONCEPTUAL:
            return self._handle_conceptual_implementation(
                inputs, classification, signature, fingerprint, eligible_candidates
            )

        if classification.category is FailureCategory.INTEGRATION_FAILURE:
            return self._terminal(
                inputs,
                classification,
                signature,
                fingerprint,
                RecoveryAction.BLOCK,
                "integration anomaly: block for review",
                require_risk_escalation=True,
                attempt_counters=self._counters(ledger),
            )

        return self._handle_unknown(
            inputs, classification, signature, fingerprint, eligible_candidates
        )

    def decide_and_record(
        self,
        inputs: RecoveryCoordinatorInput,
        timestamp: datetime.datetime | None = None,
    ) -> RecoveryDecision:
        """Return a decision and atomically record the retry-state transition."""
        decision = self.decide(inputs)
        return self.apply_decision(inputs, decision, timestamp)

    def apply_decision(
        self,
        inputs: RecoveryCoordinatorInput,
        decision: RecoveryDecision,
        timestamp: datetime.datetime | None = None,
    ) -> RecoveryDecision:
        """Record a decision into the ledger exactly once."""
        ledger = inputs.ledger
        transition_fingerprint = decision.deterministic_input_fingerprint
        if any(r.transition_fingerprint == transition_fingerprint for r in ledger.records):
            # Already committed this exact transition.
            return decision

        now = timestamp or self._now()
        retry_type = self._retry_type_for_action(decision.action)

        candidate = decision.selected_candidate or self._current_candidate(inputs)
        provider_id = candidate.provider_id if candidate is not None else None
        model_id = candidate.model_id if candidate is not None else None
        route_id = candidate.route_id if candidate is not None else None

        record = FailureAttemptRecord(
            attempt_index=ledger.total_attempt_index,
            failure_category=decision.classification.category.value,
            failure_subtype=decision.classification.subtype.value,
            failure_signature=decision.failure_signature,
            provider_id=provider_id,
            model_id=model_id,
            route_id=route_id,
            action_taken=decision.action.value,
            retry_type=retry_type,
            timestamp=now,
            retry_after=decision.retry_after,
            context_rebuild_number=ledger.context_rebuild_count(),
            repair_number=ledger.repair_count(),
            transition_fingerprint=transition_fingerprint,
        )
        ledger.records.append(record)

        if decision.action is RecoveryAction.REBUILD_CONTEXT:
            ledger.current_context_rebuild = dict(self._context_evidence(inputs))
            ledger.current_context_rebuild["rebuild_number"] = ledger.context_rebuild_count()

        if decision.terminal or self._retry_path_exhausted(decision, inputs):
            ledger.mark_exhausted_path(decision.failure_signature, provider_id, model_id)

        if decision.action is not RecoveryAction.WAIT_FOR_PROVIDER:
            ledger.clear_wait()

        object.__setattr__(decision, "transition_fingerprint", transition_fingerprint)
        return decision

    def _retry_path_exhausted(
        self, decision: RecoveryDecision, inputs: RecoveryCoordinatorInput
    ) -> bool:
        """Return True when a bounded local retry path has reached its limit."""
        policy = inputs.policy
        ledger = inputs.ledger
        action = decision.action
        if action is RecoveryAction.BLOCK:
            return False
        if action in {RecoveryAction.REROUTE_MODEL, RecoveryAction.REROUTE_PROVIDER}:
            # Switching away means the prior local path is exhausted for this signature.
            return True
        if action is RecoveryAction.CROSS_MODEL_REPAIR:
            return True
        if action is RecoveryAction.WAIT_FOR_PROVIDER:
            return False
        if action is RecoveryAction.REBUILD_CONTEXT:
            return ledger.context_rebuild_count() >= policy.max_context_rebuilds
        if action is RecoveryAction.CONSTRAINED_OUTPUT_RETRY:
            return ledger.constrained_output_retry_count() >= policy.max_structured_output_retries
        if action is RecoveryAction.REPLAN:
            return ledger.planning_retry_count() >= policy.max_planning_retries
        if action is RecoveryAction.REPAIR_WITH_EVIDENCE:
            return ledger.repair_count() >= policy.max_same_model_repairs
        return False

    def _retry_type_for_action(self, action: RecoveryAction) -> RetryType:
        mapping = {
            RecoveryAction.RETRY_SAME_ROUTE: RetryType.TRANSIENT_RETRY,
            RecoveryAction.RETRY_ALTERNATE_ROUTE: RetryType.REROUTE_ROUTE,
            RecoveryAction.REROUTE_PROVIDER: RetryType.REROUTE_PROVIDER,
            RecoveryAction.REROUTE_MODEL: RetryType.REROUTE_MODEL,
            RecoveryAction.CONSTRAINED_OUTPUT_RETRY: RetryType.CONSTRAINED_OUTPUT_RETRY,
            RecoveryAction.REPLAN: RetryType.REPLAN,
            RecoveryAction.REPAIR_WITH_EVIDENCE: RetryType.REPAIR,
            RecoveryAction.CROSS_MODEL_REPAIR: RetryType.CROSS_MODEL_REPAIR,
            RecoveryAction.REBUILD_CONTEXT: RetryType.REBUILD_CONTEXT,
            RecoveryAction.WAIT_FOR_PROVIDER: RetryType.WAIT_FOR_PROVIDER,
            RecoveryAction.BLOCK: RetryType.BLOCK,
            RecoveryAction.CANCEL: RetryType.CANCEL,
        }
        return mapping[action]

    def _handle_infrastructure_transient(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        fingerprint: str,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryDecision:
        ledger = inputs.ledger
        if ledger.transient_retry_count() >= inputs.policy.max_transient_retries:
            return self._try_reroute_or_wait(
                inputs,
                classification,
                signature,
                fingerprint,
                eligible_candidates,
                "transient retry budget exhausted",
            )

        cross_provider_threshold = inputs.policy.require_cross_provider_after_same_signature
        if ledger.signature_count(signature) >= cross_provider_threshold:
            candidate = self._select_cross_provider_candidate(inputs, eligible_candidates)
            if candidate is not None:
                return self._reroute(
                    inputs,
                    classification,
                    signature,
                    fingerprint,
                    candidate,
                    RecoveryAction.REROUTE_PROVIDER,
                    "repeated transient signature: cross-provider reroute",
                )
            return self._wait(
                inputs,
                classification,
                signature,
                fingerprint,
                "no_cross_provider_alternative_for_repeated_transient",
            )

        # Same-route consecutive infrastructure retry bound.
        consecutive = self._consecutive_same_route_infrastructure_retries(inputs, signature)
        if consecutive >= inputs.policy.max_consecutive_infrastructure_retries:
            candidate = self._select_alternate_route(inputs, eligible_candidates)
            if candidate is not None:
                return self._reroute(
                    inputs,
                    classification,
                    signature,
                    fingerprint,
                    candidate,
                    RecoveryAction.RETRY_ALTERNATE_ROUTE,
                    "consecutive same-route retries exceeded: alternate route",
                )
            return self._wait(
                inputs,
                classification,
                signature,
                fingerprint,
                "no_alternate_route_for_consecutive_retries",
            )

        alternate = self._select_alternate_route(inputs, eligible_candidates)
        if alternate is not None:
            return self._reroute(
                inputs,
                classification,
                signature,
                fingerprint,
                alternate,
                RecoveryAction.RETRY_ALTERNATE_ROUTE,
                "alternate eligible route available",
            )

        if classification.category is FailureCategory.INFRASTRUCTURE_UNAVAILABLE:
            return self._wait(
                inputs,
                classification,
                signature,
                fingerprint,
                "no_alternate_route_available",
            )

        selected = self._select_current_or_first_eligible(inputs, eligible_candidates)
        if selected is None:
            return self._wait(
                inputs,
                classification,
                signature,
                fingerprint,
                "current_route_ineligible_no_alternative",
            )

        attempt = ledger.transient_retry_count()
        delay = self._backoff.delay_for_attempt(attempt)
        now = self._now()
        action = (
            RecoveryAction.RETRY_SAME_ROUTE
            if selected == self._current_candidate(inputs)
            else RecoveryAction.RETRY_ALTERNATE_ROUTE
        )
        return RecoveryDecision(
            action=action,
            classification=classification,
            failure_signature=signature,
            deterministic_input_fingerprint=fingerprint,
            retry_allowed=True,
            selected_candidate=selected,
            require_reroute=action is not RecoveryAction.RETRY_SAME_ROUTE,
            require_context_rebuild=False,
            require_risk_escalation=False,
            wait_reason="bounded transient retry with backoff",
            retry_after=now + datetime.timedelta(seconds=delay),
            evidence_packet={"backoff_seconds": delay, "transient_attempt": attempt + 1},
            attempt_counters=self._counters(ledger),
            terminal=False,
            explanation=(
                "Transient infrastructure failure: bounded retry on eligible route "
                f"after {delay}s backoff; model quality unaffected."
            ),
        )

    def _consecutive_same_route_infrastructure_retries(
        self, inputs: RecoveryCoordinatorInput, signature: str
    ) -> int:
        count = 0
        for record in reversed(inputs.ledger.records):
            if record.failure_signature != signature:
                break
            if record.retry_type is RetryType.TRANSIENT_RETRY:
                count += 1
            else:
                break
        return count

    def _handle_quota_exhaustion(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        fingerprint: str,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryDecision:
        if inputs.pin is not None and self._is_pinned_capacity_exhausted(
            inputs, eligible_candidates
        ):
            return self._wait(
                inputs,
                classification,
                signature,
                fingerprint,
                "PINNED_CAPACITY_UNAVAILABLE",
            )

        candidate = self._select_eligible_candidate(inputs, eligible_candidates)
        if candidate is not None:
            return self._reroute(
                inputs,
                classification,
                signature,
                fingerprint,
                candidate,
                RecoveryAction.REROUTE_PROVIDER,
                "alternate eligible capacity available",
            )

        return self._wait(
            inputs,
            classification,
            signature,
            fingerprint,
            "no_eligible_capacity",
        )

    def _handle_auth_failure(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        fingerprint: str,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryDecision:
        candidate = self._select_alternate_route_auth(inputs, eligible_candidates)
        if candidate is None:
            candidate = self._select_different_provider_candidate(inputs, eligible_candidates)
        if candidate is None:
            candidate = self._select_different_model_candidate(inputs, eligible_candidates)
        if candidate is not None:
            return self._reroute(
                inputs,
                classification,
                signature,
                fingerprint,
                candidate,
                RecoveryAction.REROUTE_PROVIDER,
                "alternate configured credential/route available",
            )

        return self._terminal(
            inputs,
            classification,
            signature,
            fingerprint,
            RecoveryAction.BLOCK,
            "auth failure: no alternate credential/route configured",
            attempt_counters=self._counters(inputs.ledger),
        )

    def _handle_capability_mismatch(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        fingerprint: str,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryDecision:
        candidate = self._select_eligible_candidate(inputs, eligible_candidates)
        if candidate is not None:
            return self._reroute(
                inputs,
                classification,
                signature,
                fingerprint,
                candidate,
                RecoveryAction.REROUTE_MODEL,
                "capable candidate available",
            )

        return self._terminal(
            inputs,
            classification,
            signature,
            fingerprint,
            RecoveryAction.BLOCK,
            "no capable candidate available",
            attempt_counters=self._counters(inputs.ledger),
        )

    def _handle_structured_output_invalid(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        fingerprint: str,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryDecision:
        ledger = inputs.ledger
        if ledger.constrained_output_retry_count() >= inputs.policy.max_structured_output_retries:
            candidate = self._select_different_model_candidate(inputs, eligible_candidates)
            if candidate is not None:
                return self._reroute(
                    inputs,
                    classification,
                    signature,
                    fingerprint,
                    candidate,
                    RecoveryAction.REROUTE_MODEL,
                    "structured output retries exhausted: switch model",
                )
            return self._terminal(
                inputs,
                classification,
                signature,
                fingerprint,
                RecoveryAction.BLOCK,
                "structured output retries exhausted: no alternative",
                attempt_counters=self._counters(ledger),
            )

        selected = self._select_current_or_first_eligible(inputs, eligible_candidates)
        if selected is None:
            return self._terminal(
                inputs,
                classification,
                signature,
                fingerprint,
                RecoveryAction.BLOCK,
                "structured output retry: current candidate no longer eligible",
                attempt_counters=self._counters(ledger),
            )

        current = self._current_candidate(inputs)
        if selected != current:
            return self._reroute(
                inputs,
                classification,
                signature,
                fingerprint,
                selected,
                RecoveryAction.REROUTE_MODEL,
                "structured output retry: current candidate ineligible, switching model",
                evidence_packet=self._structured_output_evidence(inputs),
            )

        return RecoveryDecision(
            action=RecoveryAction.CONSTRAINED_OUTPUT_RETRY,
            classification=classification,
            failure_signature=signature,
            deterministic_input_fingerprint=fingerprint,
            retry_allowed=True,
            selected_candidate=selected,
            require_reroute=False,
            require_context_rebuild=False,
            require_risk_escalation=False,
            wait_reason="",
            retry_after=None,
            evidence_packet=self._structured_output_evidence(inputs),
            attempt_counters=self._counters(ledger),
            terminal=False,
            explanation=(
                "Structured output failed validation: bounded constrained retry "
                "with explicit validation feedback."
            ),
        )

    def _handle_planning_output_invalid(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        fingerprint: str,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryDecision:
        ledger = inputs.ledger
        if ledger.planning_retry_count() >= inputs.policy.max_planning_retries:
            candidate = self._select_different_model_candidate(inputs, eligible_candidates)
            if candidate is not None:
                return self._reroute(
                    inputs,
                    classification,
                    signature,
                    fingerprint,
                    candidate,
                    RecoveryAction.REROUTE_MODEL,
                    "planning retries exhausted: switch planner",
                )
            return self._terminal(
                inputs,
                classification,
                signature,
                fingerprint,
                RecoveryAction.BLOCK,
                "planning retries exhausted: no alternative",
                attempt_counters=self._counters(ledger),
            )

        selected = self._select_current_or_first_eligible(inputs, eligible_candidates)
        if selected is None:
            return self._terminal(
                inputs,
                classification,
                signature,
                fingerprint,
                RecoveryAction.BLOCK,
                "replan: current candidate no longer eligible",
                attempt_counters=self._counters(ledger),
            )

        current = self._current_candidate(inputs)
        if selected != current:
            return self._reroute(
                inputs,
                classification,
                signature,
                fingerprint,
                selected,
                RecoveryAction.REROUTE_MODEL,
                "replan: current planner ineligible, switching planner",
                evidence_packet=self._planning_evidence(inputs),
            )

        return RecoveryDecision(
            action=RecoveryAction.REPLAN,
            classification=classification,
            failure_signature=signature,
            deterministic_input_fingerprint=fingerprint,
            retry_allowed=True,
            selected_candidate=selected,
            require_reroute=False,
            require_context_rebuild=False,
            require_risk_escalation=False,
            wait_reason="",
            retry_after=None,
            evidence_packet=self._planning_evidence(inputs),
            attempt_counters=self._counters(ledger),
            terminal=False,
            explanation=(
                "Planning output failed deterministic validation: bounded replan "
                "with preserved rejection evidence."
            ),
        )

    def _handle_deterministic_implementation(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        fingerprint: str,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryDecision:
        ledger = inputs.ledger
        same_signature = ledger.signature_count(signature)

        if same_signature >= inputs.policy.require_cross_provider_after_same_signature:
            candidate = self._select_cross_model_provider_candidate(inputs, eligible_candidates)
            if candidate is not None:
                return self._reroute(
                    inputs,
                    classification,
                    signature,
                    fingerprint,
                    candidate,
                    RecoveryAction.CROSS_MODEL_REPAIR,
                    "repeated identical implementation signature: cross-model escalation",
                )
            return self._terminal(
                inputs,
                classification,
                signature,
                fingerprint,
                RecoveryAction.BLOCK,
                "repeated implementation signature: no cross-model alternative",
                require_risk_escalation=True,
                attempt_counters=self._counters(ledger),
            )

        if ledger.repair_count() >= inputs.policy.max_same_model_repairs:
            candidate = self._select_different_model_candidate(inputs, eligible_candidates)
            if candidate is not None:
                return self._reroute(
                    inputs,
                    classification,
                    signature,
                    fingerprint,
                    candidate,
                    RecoveryAction.CROSS_MODEL_REPAIR,
                    "same-model repair budget exhausted",
                )
            return self._terminal(
                inputs,
                classification,
                signature,
                fingerprint,
                RecoveryAction.BLOCK,
                "same-model repair budget exhausted: no alternative",
                require_risk_escalation=True,
                attempt_counters=self._counters(ledger),
            )

        selected = self._select_current_or_first_eligible(inputs, eligible_candidates)
        if selected is None:
            return self._terminal(
                inputs,
                classification,
                signature,
                fingerprint,
                RecoveryAction.BLOCK,
                "repair: current candidate no longer eligible",
                attempt_counters=self._counters(ledger),
            )

        current = self._current_candidate(inputs)
        if selected != current:
            action = (
                RecoveryAction.CROSS_MODEL_REPAIR
                if selected.provider_id != current.provider_id
                else RecoveryAction.REROUTE_MODEL
            )
            return self._reroute(
                inputs,
                classification,
                signature,
                fingerprint,
                selected,
                action,
                "repair: current model ineligible, switching model",
                evidence_packet=self._implementation_evidence(inputs),
            )

        return RecoveryDecision(
            action=RecoveryAction.REPAIR_WITH_EVIDENCE,
            classification=classification,
            failure_signature=signature,
            deterministic_input_fingerprint=fingerprint,
            retry_allowed=True,
            selected_candidate=selected,
            require_reroute=False,
            require_context_rebuild=False,
            require_risk_escalation=False,
            wait_reason="",
            retry_after=None,
            evidence_packet=self._implementation_evidence(inputs),
            attempt_counters=self._counters(ledger),
            terminal=False,
            explanation="Deterministic implementation failure: repair with validation evidence.",
        )

    def _handle_conceptual_implementation(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        fingerprint: str,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryDecision:
        candidate = self._select_cross_model_provider_candidate(inputs, eligible_candidates)
        if candidate is not None:
            return self._reroute(
                inputs,
                classification,
                signature,
                fingerprint,
                candidate,
                RecoveryAction.CROSS_MODEL_REPAIR,
                "conceptual failure: cross-model/provider escalation",
                require_risk_escalation=True,
            )
        return self._terminal(
            inputs,
            classification,
            signature,
            fingerprint,
            RecoveryAction.BLOCK,
            "conceptual failure: no cross-model alternative",
            require_risk_escalation=True,
            attempt_counters=self._counters(inputs.ledger),
        )

    def _handle_context_overflow(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        fingerprint: str,
    ) -> RecoveryDecision:
        ledger = inputs.ledger
        eligible, _ = evaluate_recovery_eligibility(inputs)

        if not self._context_authority_safe(inputs):
            return self._terminal(
                inputs,
                classification,
                signature,
                fingerprint,
                RecoveryAction.BLOCK,
                "context overflow: authority validation failed",
                attempt_counters=self._counters(ledger),
            )

        if ledger.context_rebuild_count() >= inputs.policy.max_context_rebuilds:
            candidate = self._select_larger_context_candidate(inputs, eligible)
            if candidate is not None:
                return self._reroute(
                    inputs,
                    classification,
                    signature,
                    fingerprint,
                    candidate,
                    RecoveryAction.REROUTE_MODEL,
                    "context rebuild budget exhausted: choose larger-context model",
                )
            return self._terminal(
                inputs,
                classification,
                signature,
                fingerprint,
                RecoveryAction.BLOCK,
                "context rebuild budget exhausted: authority cannot safely fit",
                attempt_counters=self._counters(ledger),
            )

        return RecoveryDecision(
            action=RecoveryAction.REBUILD_CONTEXT,
            classification=classification,
            failure_signature=signature,
            deterministic_input_fingerprint=fingerprint,
            retry_allowed=True,
            selected_candidate=self._select_current_or_first_eligible(inputs, eligible),
            require_reroute=False,
            require_context_rebuild=True,
            require_risk_escalation=False,
            wait_reason="",
            retry_after=None,
            evidence_packet=self._context_evidence(inputs),
            attempt_counters=self._counters(ledger),
            terminal=False,
            explanation=(
                "Context overflow: rebuild context using more compact strategy "
                "while preserving required authority."
            ),
        )

    def _handle_authority_violation(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        fingerprint: str,
    ) -> RecoveryDecision:
        eligible, _ = evaluate_recovery_eligibility(inputs)
        candidate = self._select_different_model_candidate(inputs, eligible)
        if candidate is not None:
            return self._reroute(
                inputs,
                classification,
                signature,
                fingerprint,
                candidate,
                RecoveryAction.CROSS_MODEL_REPAIR,
                "authority violation: require different model/provider for repair",
                require_risk_escalation=True,
            )
        return self._terminal(
            inputs,
            classification,
            signature,
            fingerprint,
            RecoveryAction.BLOCK,
            "authority violation: no safe repair alternative",
            require_risk_escalation=True,
            attempt_counters=self._counters(inputs.ledger),
        )

    def _handle_unknown(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        fingerprint: str,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryDecision:
        if inputs.ledger.attempt_count < inputs.policy.max_total_attempts // 2:
            selected = self._select_current_or_first_eligible(inputs, eligible_candidates)
            if selected is None:
                return self._terminal(
                    inputs,
                    classification,
                    signature,
                    fingerprint,
                    RecoveryAction.BLOCK,
                    "unknown failure: no eligible candidate for retry",
                    attempt_counters=self._counters(inputs.ledger),
                )
            action = (
                RecoveryAction.RETRY_SAME_ROUTE
                if selected == self._current_candidate(inputs)
                else RecoveryAction.RETRY_ALTERNATE_ROUTE
            )
            return RecoveryDecision(
                action=action,
                classification=classification,
                failure_signature=signature,
                deterministic_input_fingerprint=fingerprint,
                retry_allowed=True,
                selected_candidate=selected,
                require_reroute=action is not RecoveryAction.RETRY_SAME_ROUTE,
                require_context_rebuild=False,
                require_risk_escalation=False,
                wait_reason="bounded unknown retry",
                retry_after=None,
                evidence_packet={"unknown": True},
                attempt_counters=self._counters(inputs.ledger),
                terminal=False,
                explanation="Unknown failure: small bounded retry permitted by policy.",
            )
        return self._terminal(
            inputs,
            classification,
            signature,
            fingerprint,
            RecoveryAction.BLOCK,
            "unknown failure: bounded retry exhausted",
            attempt_counters=self._counters(inputs.ledger),
        )

    def _try_reroute_or_wait(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        fingerprint: str,
        eligible_candidates: tuple[RecoveryCandidate, ...],
        reason: str,
    ) -> RecoveryDecision:
        candidate = self._select_eligible_candidate(inputs, eligible_candidates)
        if candidate is not None:
            return self._reroute(
                inputs,
                classification,
                signature,
                fingerprint,
                candidate,
                RecoveryAction.REROUTE_PROVIDER,
                f"{reason}: alternate eligible route available",
            )
        return self._wait(inputs, classification, signature, fingerprint, f"{reason}: no alternate")

    def _reroute(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        fingerprint: str,
        candidate: RecoveryCandidate,
        action: RecoveryAction,
        reason: str,
        require_risk_escalation: bool = False,
        evidence_packet: dict[str, Any] | None = None,
    ) -> RecoveryDecision:
        packet: dict[str, Any] = {"reroute_reason": reason, "selected_candidate": candidate.key}
        if evidence_packet is not None:
            packet["original_evidence"] = evidence_packet
        return RecoveryDecision(
            action=action,
            classification=classification,
            failure_signature=signature,
            deterministic_input_fingerprint=fingerprint,
            retry_allowed=True,
            selected_candidate=candidate,
            require_reroute=True,
            require_context_rebuild=False,
            require_risk_escalation=require_risk_escalation,
            wait_reason="",
            retry_after=None,
            evidence_packet=packet,
            attempt_counters=self._counters(inputs.ledger),
            terminal=False,
            explanation=f"{reason}; model quality unaffected for infrastructure failures.",
        )

    def _wait(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        fingerprint: str,
        reason: str,
    ) -> RecoveryDecision:
        now = self._now()
        delay = self._backoff.delay_for_attempt(inputs.ledger.transient_retry_count())
        retry_after = now + datetime.timedelta(seconds=delay)
        domains = {c.failure_domain for c in inputs.candidates if c.failure_domain}
        wait = WaitState(
            reason=reason,
            next_recheck_at=retry_after,
            entered_at=now,
            affected_failure_domains=frozenset(domains),
        )
        inputs.ledger.set_wait(wait)
        return RecoveryDecision(
            action=RecoveryAction.WAIT_FOR_PROVIDER,
            classification=classification,
            failure_signature=signature,
            deterministic_input_fingerprint=fingerprint,
            retry_allowed=False,
            selected_candidate=None,
            require_reroute=False,
            require_context_rebuild=False,
            require_risk_escalation=False,
            wait_reason=reason,
            retry_after=retry_after,
            evidence_packet={"wait_reason": reason, "affected_failure_domains": sorted(domains)},
            attempt_counters=self._counters(inputs.ledger),
            terminal=False,
            explanation=f"No eligible alternative: enter WAITING_FOR_PROVIDER ({reason}).",
        )

    def _terminal(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        fingerprint: str,
        action: RecoveryAction,
        reason: str,
        require_risk_escalation: bool = False,
        attempt_counters: dict[str, int] | None = None,
    ) -> RecoveryDecision:
        return RecoveryDecision(
            action=action,
            classification=classification,
            failure_signature=signature,
            deterministic_input_fingerprint=fingerprint,
            retry_allowed=False,
            selected_candidate=None,
            require_reroute=False,
            require_context_rebuild=False,
            require_risk_escalation=require_risk_escalation,
            wait_reason="",
            retry_after=None,
            evidence_packet={"terminal_reason": reason},
            attempt_counters=attempt_counters or self._counters(inputs.ledger),
            terminal=True,
            explanation=f"Terminal recovery state: {reason}.",
        )

    def _signature(self, inputs: RecoveryCoordinatorInput) -> str:
        return self._classifier.classify(inputs.classifier_input).deterministic_fingerprint

    def _select_eligible_candidate(
        self,
        inputs: RecoveryCoordinatorInput,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryCandidate | None:
        signature = self._signature(inputs)
        current = self._current_candidate(inputs)
        for candidate in self._ordered_candidates_from(eligible_candidates):
            if self._ledger_has_exhausted_path(signature, inputs.ledger, candidate):
                continue
            if candidate == current and self._retry_path_exhausted_for_signature(
                signature, inputs.ledger, candidate
            ):
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_alternate_route(
        self,
        inputs: RecoveryCoordinatorInput,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryCandidate | None:
        signature = self._signature(inputs)
        current = self._current_candidate(inputs)
        for candidate in self._ordered_candidates_from(eligible_candidates):
            if candidate == current:
                continue
            if candidate.route_id == current.route_id:
                continue
            if candidate.failure_domain == current.failure_domain and current.failure_domain:
                continue
            if self._ledger_has_exhausted_path(signature, inputs.ledger, candidate):
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_alternate_route_auth(
        self,
        inputs: RecoveryCoordinatorInput,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryCandidate | None:
        signature = self._signature(inputs)
        current = self._current_candidate(inputs)
        for candidate in self._ordered_candidates_from(eligible_candidates):
            if candidate == current:
                continue
            if candidate.route_id == current.route_id:
                continue
            if self._ledger_has_exhausted_path(signature, inputs.ledger, candidate):
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_different_provider_candidate(
        self,
        inputs: RecoveryCoordinatorInput,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryCandidate | None:
        if not self._provider_switch_allowed(inputs):
            return None
        signature = self._signature(inputs)
        current = self._current_candidate(inputs)
        for candidate in self._ordered_candidates_from(eligible_candidates):
            if candidate.provider_id == current.provider_id:
                continue
            if self._ledger_has_exhausted_path(signature, inputs.ledger, candidate):
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_different_model_candidate(
        self,
        inputs: RecoveryCoordinatorInput,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryCandidate | None:
        if not self._model_switch_allowed(inputs):
            return None
        signature = self._signature(inputs)
        current = self._current_candidate(inputs)
        for candidate in self._ordered_candidates_from(eligible_candidates):
            if candidate.model_id == current.model_id:
                continue
            if self._ledger_has_exhausted_path(signature, inputs.ledger, candidate):
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_cross_provider_candidate(
        self,
        inputs: RecoveryCoordinatorInput,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryCandidate | None:
        if not self._provider_switch_allowed(inputs):
            return None
        signature = self._signature(inputs)
        current = self._current_candidate(inputs)
        for candidate in self._ordered_candidates_from(eligible_candidates):
            if candidate.provider_id == current.provider_id:
                continue
            if candidate.failure_domain == current.failure_domain and current.failure_domain:
                continue
            if self._ledger_has_exhausted_path(signature, inputs.ledger, candidate):
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_cross_model_provider_candidate(
        self,
        inputs: RecoveryCoordinatorInput,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryCandidate | None:
        if not (self._provider_switch_allowed(inputs) and self._model_switch_allowed(inputs)):
            return None
        signature = self._signature(inputs)
        current = self._current_candidate(inputs)
        for candidate in self._ordered_candidates_from(eligible_candidates):
            if candidate.model_id == current.model_id:
                continue
            if candidate.provider_id == current.provider_id:
                continue
            if self._ledger_has_exhausted_path(signature, inputs.ledger, candidate):
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_larger_context_candidate(
        self,
        inputs: RecoveryCoordinatorInput,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryCandidate | None:
        if not self._model_switch_allowed(inputs):
            return None
        required = _required_context_tokens(inputs)
        if required is None:
            return None
        budget = ContextBudget(
            primary_budget=required,
            budget_type=BudgetType.TOKENS_ESTIMATE,
            safety_margin_fraction=0.1,
        )
        current = self._current_candidate(inputs)
        for candidate in self._ordered_candidates_from(eligible_candidates):
            if candidate.model_id == current.model_id:
                continue
            usable = compute_usable_budget(candidate.capabilities.context_tokens, budget).usable
            if usable < required:
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_current_or_first_eligible(
        self,
        inputs: RecoveryCoordinatorInput,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> RecoveryCandidate | None:
        current = self._current_candidate(inputs)
        if current in eligible_candidates:
            return current
        ordered = self._ordered_candidates_from(eligible_candidates)
        return ordered[0] if ordered else None

    def _ordered_candidates_from(
        self, candidates: tuple[RecoveryCandidate, ...]
    ) -> tuple[RecoveryCandidate, ...]:
        return tuple(sorted(candidates, key=lambda c: (c.provider_id, c.model_id, c.route_id)))

    def _ordered_candidates(
        self, inputs: RecoveryCoordinatorInput
    ) -> tuple[RecoveryCandidate, ...]:
        return self._ordered_candidates_from(inputs.candidates)

    def _current_candidate(self, inputs: RecoveryCoordinatorInput) -> RecoveryCandidate:
        classifier_input = inputs.classifier_input
        pin = inputs.pin
        for candidate in inputs.candidates:
            provider_match = (
                classifier_input.provider_id is None
                or candidate.provider_id == classifier_input.provider_id
            )
            model_match = (
                classifier_input.model_id is None or candidate.model_id == classifier_input.model_id
            )
            route_match = (
                classifier_input.route_id is None or candidate.route_id == classifier_input.route_id
            )
            pin_match = self._matches_pin(candidate, pin)
            if provider_match and model_match and route_match and pin_match:
                return candidate
        if inputs.candidates:
            return self._ordered_candidates(inputs)[0]
        raise ValueError("no current candidate available")

    def _ledger_has_exhausted_path(
        self,
        signature: str,
        ledger: RetryLedger,
        candidate: RecoveryCandidate,
    ) -> bool:
        return ledger.is_exhausted_path(signature, candidate.provider_id, candidate.model_id)

    def _retry_path_exhausted_for_signature(
        self,
        signature: str,
        ledger: RetryLedger,
        candidate: RecoveryCandidate,
    ) -> bool:
        return ledger.is_exhausted_path(signature, candidate.provider_id, candidate.model_id)

    def _provider_switch_allowed(self, inputs: RecoveryCoordinatorInput) -> bool:
        return inputs.ledger.provider_switch_count() < inputs.policy.max_provider_switches

    def _model_switch_allowed(self, inputs: RecoveryCoordinatorInput) -> bool:
        return inputs.ledger.model_switch_count() < inputs.policy.max_model_switches

    def _is_pinned_capacity_exhausted(
        self,
        inputs: RecoveryCoordinatorInput,
        eligible_candidates: tuple[RecoveryCandidate, ...],
    ) -> bool:
        pin = inputs.pin
        if pin is None:
            return False
        for candidate in inputs.candidates:
            if not self._matches_pin(candidate, pin):
                continue
            if candidate not in eligible_candidates:
                return True
            effective = _effective_quota(candidate, inputs.quota_domain_states)
            if effective is not None and effective.is_exhausted():
                return True
        return False

    def _matches_pin(
        self,
        candidate: RecoveryCandidate,
        pin: RoutingPin | None,
    ) -> bool:
        if pin is None:
            return True
        if pin.provider_id is not None and candidate.provider_id != pin.provider_id:
            return False
        if pin.model_id is not None and candidate.model_id != pin.model_id:
            return False
        return pin.route_id is None or candidate.route_id == pin.route_id

    def _structured_output_evidence(self, inputs: RecoveryCoordinatorInput) -> dict[str, Any]:
        validation = inputs.classifier_input.structured_output_validation
        if validation is None:
            return {}
        return {
            "missing_required_fields": sorted(validation.missing_required_fields),
            "invalid_enum_values": sorted(validation.invalid_enum_values),
            "schema_errors": sorted(validation.schema_errors),
            "has_parse_error": validation.parse_error is not None,
        }

    def _planning_evidence(self, inputs: RecoveryCoordinatorInput) -> dict[str, Any]:
        validation = inputs.classifier_input.planning_validation
        if validation is None:
            return {}
        return {
            "missing_steps": sorted(validation.missing_steps),
            "authority_violations": sorted(validation.authority_violations),
            "schema_errors": sorted(validation.schema_errors),
        }

    def _implementation_evidence(self, inputs: RecoveryCoordinatorInput) -> dict[str, Any]:
        validation = inputs.classifier_input.deterministic_validation
        if validation is None:
            return {}
        return {
            "command": validation.validator,
            "exit_status": validation.exit_status,
            "failing_check_names": sorted(validation.failing_check_names),
            "affected_files": sorted(validation.affected_files),
            "error_excerpts": list(validation.error_excerpts),
        }

    def _context_evidence(self, inputs: RecoveryCoordinatorInput) -> dict[str, Any]:
        meta = inputs.classifier_input.context_overflow
        if meta is None:
            return {}
        estimated_tokens = None
        if meta.estimated_input_chars is not None:
            estimated_tokens = estimate_tokens(meta.estimated_input_chars)
        evidence: dict[str, Any] = {
            "estimated_input_chars": meta.estimated_input_chars,
            "estimated_input_tokens": estimated_tokens,
            "required_context_tokens": _required_context_tokens(inputs),
            "model_context_tokens": meta.model_context_tokens,
            "authority_required": meta.authority_required,
            "authority_items_present": meta.authority_items_present,
            "authority_items_raw": meta.authority_items_raw,
            "rebuild_attempts": meta.rebuild_attempts,
        }
        validation = self._context_authority_validation(inputs)
        if validation is not None:
            evidence["authority_validation_issues"] = validation["issues"]
            evidence["authority_validation_safe"] = validation["safe"]
        return evidence

    def _risk_context_requirements(
        self, inputs: RecoveryCoordinatorInput
    ) -> RiskContextRequirements | None:
        if inputs.risk_context_requirements is not None:
            return inputs.risk_context_requirements
        if inputs.risk_context_policy is not None:
            return inputs.risk_context_policy.requirements_for(inputs.current_risk)
        return RiskContextPolicy.default().requirements_for(inputs.current_risk)

    def _context_authority_validation(
        self, inputs: RecoveryCoordinatorInput
    ) -> dict[str, Any] | None:
        packet = inputs.context_packet
        if packet is None:
            return None
        requirements = self._risk_context_requirements(inputs)
        issues = list(ContextPacketValidator().validate(packet))
        safe = True
        if requirements is not None:
            if (
                requirements.require_raw_authority
                and packet.authority_presence is not AuthorityPresence.RAW_INCLUDED
            ):
                safe = False
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="REQUIRES_RAW_INCLUDED_AUTHORITY",
                        message="Risk policy requires RAW_INCLUDED authority.",
                    )
                )
            if (
                requirements.authority_required
                and packet.authority_presence is AuthorityPresence.NOT_REQUIRED
            ):
                safe = False
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="REQUIRES_AUTHORITY_BUT_NOT_REQUIRED",
                        message="Risk policy requires authority but packet marks it not required.",
                    )
                )
        issue_dicts = [
            {"severity": issue.severity, "code": issue.code, "message": issue.message}
            for issue in issues
        ]
        if any(issue["severity"] == "error" for issue in issue_dicts):
            safe = False
        return {"issues": issue_dicts, "safe": safe}

    def _context_authority_safe(self, inputs: RecoveryCoordinatorInput) -> bool:
        packet = inputs.context_packet
        requirements = self._risk_context_requirements(inputs)
        authority_required = requirements.authority_required if requirements is not None else False
        if packet is None:
            # Legacy non-authority contexts may proceed without a typed packet.
            # Any context where authority is required must supply a Phase 7 packet.
            if not authority_required:
                meta = inputs.classifier_input.context_overflow
                if meta is None:
                    return True
                return not (meta.authority_required and meta.authority_items_raw == 0)
            return False
        validation = self._context_authority_validation(inputs)
        return validation is not None and validation["safe"]

    def _counters(self, ledger: RetryLedger) -> dict[str, int]:
        return {
            "attempt_count": ledger.attempt_count,
            "transient_retry_count": ledger.transient_retry_count(),
            "constrained_output_retry_count": ledger.constrained_output_retry_count(),
            "planning_retry_count": ledger.planning_retry_count(),
            "repair_count": ledger.repair_count(),
            "context_rebuild_count": ledger.context_rebuild_count(),
            "provider_switch_count": ledger.provider_switch_count(),
            "model_switch_count": ledger.model_switch_count(),
        }

    def _now(self) -> datetime.datetime:
        if self._clock is None:
            return datetime.datetime.now(tz=datetime.UTC)
        return self._clock.now()
