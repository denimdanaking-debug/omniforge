"""Failure-type-aware recovery coordinator.

Turns a FailureClassification, retry history, risk assessment, and eligible
candidates into a deterministic RecoveryDecision. No provider calls are made
inside the coordinator; it consumes normalized state only.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.policy.risk import RiskLevel, lifecycle_eligible
from src.providers.identity import ProviderQuotaState
from src.recovery.backoff import BackoffPolicy
from src.recovery.clock import Clock
from src.recovery.failure_classification import (
    FailureCategory,
    FailureClassification,
    FailureClassifier,
    FailureClassifierInput,
)
from src.recovery.retry_policy import FailureRecoveryPolicy
from src.recovery.retry_state import RetryLedger, WaitState
from src.recovery.state_machine import RouteRecoveryState
from src.routing.capabilities import CapabilityRequirement, ModelCapabilities, match_capabilities
from src.routing.inference_route import InferenceRouteIdentity
from src.routing.model_identity import ModelIdentity
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

    @property
    def key(self) -> str:
        return f"{self.provider_id}:{self.model_id}:{self.route_id}"


@dataclass(frozen=True)
class RecoveryDecision:
    """Deterministic recovery decision."""

    action: RecoveryAction
    classification: FailureClassification
    failure_signature: str
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
    deterministic_input_fingerprint: str


@dataclass(frozen=True)
class RecoveryCoordinatorInput:
    """Normalized inputs for the recovery coordinator."""

    classifier_input: FailureClassifierInput
    candidates: tuple[RecoveryCandidate, ...]
    ledger: RetryLedger
    policy: FailureRecoveryPolicy
    current_risk: RiskLevel
    pin: RecoveryCandidate | None = None
    reserve_policy: Any | None = None
    quota_domain_states: dict[str, ProviderQuotaState] | None = None
    role: ExecutionRole | None = None


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

        # Check global bounds first.
        total = ledger.attempt_count
        if total >= inputs.policy.max_total_attempts:
            return self._terminal(
                inputs,
                classification,
                signature,
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
                RecoveryAction.BLOCK,
                "max_same_signature_attempts reached",
                attempt_counters=self._counters(ledger),
            )

        # Category-specific dispatch.
        if classification.category is FailureCategory.CANCELLED:
            return self._terminal(
                inputs,
                classification,
                signature,
                RecoveryAction.CANCEL,
                "cancelled: no auto-retry",
                attempt_counters=self._counters(ledger),
            )

        if classification.category is FailureCategory.AUTHORITY_VIOLATION:
            return self._handle_authority_violation(inputs, classification, signature)

        if classification.category is FailureCategory.CONTEXT_CAPACITY:
            return self._handle_context_overflow(inputs, classification, signature)

        if classification.category in {
            FailureCategory.INFRASTRUCTURE_TRANSIENT,
            FailureCategory.INFRASTRUCTURE_UNAVAILABLE,
        }:
            return self._handle_infrastructure_transient(inputs, classification, signature)

        if classification.category is FailureCategory.INFRASTRUCTURE_QUOTA:
            return self._handle_quota_exhaustion(inputs, classification, signature)

        if classification.category is FailureCategory.INFRASTRUCTURE_AUTH:
            return self._handle_auth_failure(inputs, classification, signature)

        if classification.category is FailureCategory.CAPABILITY_MISMATCH:
            return self._handle_capability_mismatch(inputs, classification, signature)

        if classification.category is FailureCategory.STRUCTURED_OUTPUT_INVALID:
            return self._handle_structured_output_invalid(inputs, classification, signature)

        if classification.category is FailureCategory.PLANNING_OUTPUT_INVALID:
            return self._handle_planning_output_invalid(inputs, classification, signature)

        if classification.category is FailureCategory.IMPLEMENTATION_DETERMINISTIC:
            return self._handle_deterministic_implementation(inputs, classification, signature)

        if classification.category is FailureCategory.IMPLEMENTATION_CONCEPTUAL:
            return self._handle_conceptual_implementation(inputs, classification, signature)

        if classification.category is FailureCategory.INTEGRATION_FAILURE:
            return self._terminal(
                inputs,
                classification,
                signature,
                RecoveryAction.BLOCK,
                "integration anomaly: block for review",
                require_risk_escalation=True,
                attempt_counters=self._counters(ledger),
            )

        return self._handle_unknown(inputs, classification, signature)

    def _handle_infrastructure_transient(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
    ) -> RecoveryDecision:
        ledger = inputs.ledger
        if ledger.transient_retry_count() >= inputs.policy.max_transient_retries:
            return self._try_reroute_or_wait(
                inputs,
                classification,
                signature,
                "transient retry budget exhausted",
            )

        # If same signature repeated, prefer cross-provider reroute.
        cross_provider_threshold = inputs.policy.require_cross_provider_after_same_signature
        if ledger.signature_count(signature) >= cross_provider_threshold:
            candidate = self._select_cross_provider_candidate(inputs)
            if candidate is not None:
                return self._reroute(
                    inputs,
                    classification,
                    signature,
                    candidate,
                    RecoveryAction.REROUTE_PROVIDER,
                    "repeated transient signature: cross-provider reroute",
                )
            return self._wait(
                inputs,
                classification,
                signature,
                "no_cross_provider_alternative_for_repeated_transient",
            )

        # Prefer alternate route if available.
        alternate = self._select_alternate_route(inputs)
        if alternate is not None:
            return self._reroute(
                inputs,
                classification,
                signature,
                alternate,
                RecoveryAction.RETRY_ALTERNATE_ROUTE,
                "alternate eligible route available",
            )

        # Provider unavailable with no alternate should wait, not retry same route.
        if classification.category is FailureCategory.INFRASTRUCTURE_UNAVAILABLE:
            return self._wait(
                inputs,
                classification,
                signature,
                "no_alternate_route_available",
            )

        # Bounded retry on same route with backoff metadata.
        attempt = ledger.transient_retry_count()
        delay = self._backoff.delay_for_attempt(attempt)
        now = self._now()
        return RecoveryDecision(
            action=RecoveryAction.RETRY_SAME_ROUTE,
            classification=classification,
            failure_signature=signature,
            retry_allowed=True,
            selected_candidate=inputs.pin or self._current_candidate(inputs),
            require_reroute=False,
            require_context_rebuild=False,
            require_risk_escalation=False,
            wait_reason="bounded transient retry with backoff",
            retry_after=now + datetime.timedelta(seconds=delay),
            evidence_packet={"backoff_seconds": delay, "transient_attempt": attempt + 1},
            attempt_counters=self._counters(ledger),
            terminal=False,
            explanation=(
                "Transient infrastructure failure: bounded retry on same route "
                f"after {delay}s backoff; model quality unaffected."
            ),
            deterministic_input_fingerprint=signature,
        )

    def _handle_quota_exhaustion(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
    ) -> RecoveryDecision:
        # If pinned target is quota-exhausted, do not silently bypass.
        if inputs.pin is not None and self._is_pin_exhausted(inputs):
            return self._wait(
                inputs,
                classification,
                signature,
                "PINNED_CAPACITY_UNAVAILABLE",
            )

        candidate = self._select_eligible_candidate(inputs, exclude_exhausted=True)
        if candidate is not None:
            return self._reroute(
                inputs,
                classification,
                signature,
                candidate,
                RecoveryAction.REROUTE_PROVIDER,
                "alternate eligible capacity available",
            )

        return self._wait(
            inputs,
            classification,
            signature,
            "no_eligible_capacity",
        )

    def _handle_auth_failure(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
    ) -> RecoveryDecision:
        # Try another credential/route/model only if already configured and eligible.
        # Auth failures are credential/route-specific, so an alternate route on the
        # same provider/failure domain is acceptable.
        candidate = self._select_alternate_route_auth(inputs)
        if candidate is None:
            candidate = self._select_different_provider_candidate(inputs)
        if candidate is None:
            candidate = self._select_different_model_candidate(inputs)
        if candidate is not None:
            return self._reroute(
                inputs,
                classification,
                signature,
                candidate,
                RecoveryAction.REROUTE_PROVIDER,
                "alternate configured credential/route available",
            )

        return self._terminal(
            inputs,
            classification,
            signature,
            RecoveryAction.BLOCK,
            "auth failure: no alternate credential/route configured",
            attempt_counters=self._counters(inputs.ledger),
        )

    def _handle_capability_mismatch(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
    ) -> RecoveryDecision:
        candidate = self._select_capable_candidate(inputs)
        if candidate is not None:
            return self._reroute(
                inputs,
                classification,
                signature,
                candidate,
                RecoveryAction.REROUTE_MODEL,
                "capable candidate available",
            )

        return self._terminal(
            inputs,
            classification,
            signature,
            RecoveryAction.BLOCK,
            "no capable candidate available",
            attempt_counters=self._counters(inputs.ledger),
        )

    def _handle_structured_output_invalid(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
    ) -> RecoveryDecision:
        ledger = inputs.ledger
        if ledger.constrained_output_retry_count() >= inputs.policy.max_structured_output_retries:
            candidate = self._select_different_model_candidate(inputs)
            if candidate is not None:
                return self._reroute(
                    inputs,
                    classification,
                    signature,
                    candidate,
                    RecoveryAction.REROUTE_MODEL,
                    "structured output retries exhausted: switch model",
                )
            return self._terminal(
                inputs,
                classification,
                signature,
                RecoveryAction.BLOCK,
                "structured output retries exhausted: no alternative",
                attempt_counters=self._counters(ledger),
            )

        return RecoveryDecision(
            action=RecoveryAction.CONSTRAINED_OUTPUT_RETRY,
            classification=classification,
            failure_signature=signature,
            retry_allowed=True,
            selected_candidate=inputs.pin or self._current_candidate(inputs),
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
            deterministic_input_fingerprint=signature,
        )

    def _handle_planning_output_invalid(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
    ) -> RecoveryDecision:
        ledger = inputs.ledger
        if ledger.planning_retry_count() >= inputs.policy.max_planning_retries:
            candidate = self._select_different_model_candidate(inputs)
            if candidate is not None:
                return self._reroute(
                    inputs,
                    classification,
                    signature,
                    candidate,
                    RecoveryAction.REROUTE_MODEL,
                    "planning retries exhausted: switch planner",
                )
            return self._terminal(
                inputs,
                classification,
                signature,
                RecoveryAction.BLOCK,
                "planning retries exhausted: no alternative",
                attempt_counters=self._counters(ledger),
            )

        return RecoveryDecision(
            action=RecoveryAction.REPLAN,
            classification=classification,
            failure_signature=signature,
            retry_allowed=True,
            selected_candidate=inputs.pin or self._current_candidate(inputs),
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
            deterministic_input_fingerprint=signature,
        )

    def _handle_deterministic_implementation(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
    ) -> RecoveryDecision:
        ledger = inputs.ledger
        same_signature = ledger.signature_count(signature)

        if same_signature >= inputs.policy.require_cross_provider_after_same_signature:
            candidate = self._select_cross_model_provider_candidate(inputs)
            if candidate is not None:
                return self._reroute(
                    inputs,
                    classification,
                    signature,
                    candidate,
                    RecoveryAction.CROSS_MODEL_REPAIR,
                    "repeated identical implementation signature: cross-model escalation",
                )
            return self._terminal(
                inputs,
                classification,
                signature,
                RecoveryAction.BLOCK,
                "repeated implementation signature: no cross-model alternative",
                require_risk_escalation=True,
                attempt_counters=self._counters(ledger),
            )

        if ledger.repair_count() >= inputs.policy.max_same_model_repairs:
            candidate = self._select_different_model_candidate(inputs)
            if candidate is not None:
                return self._reroute(
                    inputs,
                    classification,
                    signature,
                    candidate,
                    RecoveryAction.CROSS_MODEL_REPAIR,
                    "same-model repair budget exhausted",
                )
            return self._terminal(
                inputs,
                classification,
                signature,
                RecoveryAction.BLOCK,
                "same-model repair budget exhausted: no alternative",
                require_risk_escalation=True,
                attempt_counters=self._counters(ledger),
            )

        return RecoveryDecision(
            action=RecoveryAction.REPAIR_WITH_EVIDENCE,
            classification=classification,
            failure_signature=signature,
            retry_allowed=True,
            selected_candidate=inputs.pin or self._current_candidate(inputs),
            require_reroute=False,
            require_context_rebuild=False,
            require_risk_escalation=False,
            wait_reason="",
            retry_after=None,
            evidence_packet=self._implementation_evidence(inputs),
            attempt_counters=self._counters(ledger),
            terminal=False,
            explanation=(
                "Deterministic implementation failure: repair with actual validation evidence."
            ),
            deterministic_input_fingerprint=signature,
        )

    def _handle_conceptual_implementation(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
    ) -> RecoveryDecision:
        candidate = self._select_cross_model_provider_candidate(inputs)
        if candidate is not None:
            return self._reroute(
                inputs,
                classification,
                signature,
                candidate,
                RecoveryAction.CROSS_MODEL_REPAIR,
                "conceptual failure: cross-model/provider escalation",
                require_risk_escalation=True,
            )
        return self._terminal(
            inputs,
            classification,
            signature,
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
    ) -> RecoveryDecision:
        ledger = inputs.ledger
        if ledger.context_rebuild_count() >= inputs.policy.max_context_rebuilds:
            # Try larger-context model.
            candidate = self._select_larger_context_candidate(inputs)
            if candidate is not None:
                return self._reroute(
                    inputs,
                    classification,
                    signature,
                    candidate,
                    RecoveryAction.REROUTE_MODEL,
                    "context rebuild budget exhausted: choose larger-context model",
                )
            return self._terminal(
                inputs,
                classification,
                signature,
                RecoveryAction.BLOCK,
                "context rebuild budget exhausted: authority cannot safely fit",
                attempt_counters=self._counters(ledger),
            )

        return RecoveryDecision(
            action=RecoveryAction.REBUILD_CONTEXT,
            classification=classification,
            failure_signature=signature,
            retry_allowed=True,
            selected_candidate=inputs.pin or self._current_candidate(inputs),
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
            deterministic_input_fingerprint=signature,
        )

    def _handle_authority_violation(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
    ) -> RecoveryDecision:
        ledger = inputs.ledger
        # Do not immediately reuse same model for authority repair when safer alternative exists.
        candidate = self._select_different_model_candidate(inputs)
        if candidate is not None:
            return self._reroute(
                inputs,
                classification,
                signature,
                candidate,
                RecoveryAction.CROSS_MODEL_REPAIR,
                "authority violation: require different model/provider for repair",
                require_risk_escalation=True,
            )
        return self._terminal(
            inputs,
            classification,
            signature,
            RecoveryAction.BLOCK,
            "authority violation: no safe repair alternative",
            require_risk_escalation=True,
            attempt_counters=self._counters(ledger),
        )

    def _handle_unknown(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
    ) -> RecoveryDecision:
        if inputs.ledger.attempt_count < inputs.policy.max_total_attempts // 2:
            return RecoveryDecision(
                action=RecoveryAction.RETRY_SAME_ROUTE,
                classification=classification,
                failure_signature=signature,
                retry_allowed=True,
                selected_candidate=inputs.pin or self._current_candidate(inputs),
                require_reroute=False,
                require_context_rebuild=False,
                require_risk_escalation=False,
                wait_reason="bounded unknown retry",
                retry_after=None,
                evidence_packet={"unknown": True},
                attempt_counters=self._counters(inputs.ledger),
                terminal=False,
                explanation="Unknown failure: small bounded retry permitted by policy.",
                deterministic_input_fingerprint=signature,
            )
        return self._terminal(
            inputs,
            classification,
            signature,
            RecoveryAction.BLOCK,
            "unknown failure: bounded retry exhausted",
            attempt_counters=self._counters(inputs.ledger),
        )

    def _try_reroute_or_wait(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        reason: str,
    ) -> RecoveryDecision:
        candidate = self._select_eligible_candidate(inputs)
        if candidate is not None:
            return self._reroute(
                inputs,
                classification,
                signature,
                candidate,
                RecoveryAction.REROUTE_PROVIDER,
                f"{reason}: alternate eligible route available",
            )
        return self._wait(inputs, classification, signature, f"{reason}: no alternate")

    def _reroute(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        candidate: RecoveryCandidate,
        action: RecoveryAction,
        reason: str,
        require_risk_escalation: bool = False,
    ) -> RecoveryDecision:
        return RecoveryDecision(
            action=action,
            classification=classification,
            failure_signature=signature,
            retry_allowed=True,
            selected_candidate=candidate,
            require_reroute=True,
            require_context_rebuild=False,
            require_risk_escalation=require_risk_escalation,
            wait_reason="",
            retry_after=None,
            evidence_packet={"reroute_reason": reason, "selected_candidate": candidate.key},
            attempt_counters=self._counters(inputs.ledger),
            terminal=False,
            explanation=f"{reason}; model quality unaffected for infrastructure failures.",
            deterministic_input_fingerprint=signature,
        )

    def _wait(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
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
            deterministic_input_fingerprint=signature,
        )

    def _terminal(
        self,
        inputs: RecoveryCoordinatorInput,
        classification: FailureClassification,
        signature: str,
        action: RecoveryAction,
        reason: str,
        require_risk_escalation: bool = False,
        attempt_counters: dict[str, int] | None = None,
    ) -> RecoveryDecision:
        return RecoveryDecision(
            action=action,
            classification=classification,
            failure_signature=signature,
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
            deterministic_input_fingerprint=signature,
        )

    def _select_eligible_candidate(
        self,
        inputs: RecoveryCoordinatorInput,
        exclude_exhausted: bool = False,
    ) -> RecoveryCandidate | None:
        for candidate in self._ordered_candidates(inputs):
            if not self._operationally_eligible(candidate, inputs):
                continue
            if exclude_exhausted and self._is_quota_exhausted(
                candidate, inputs.quota_domain_states
            ):
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_alternate_route(self, inputs: RecoveryCoordinatorInput) -> RecoveryCandidate | None:
        current = self._current_candidate(inputs)
        for candidate in self._ordered_candidates(inputs):
            if candidate == current:
                continue
            if not self._operationally_eligible(candidate, inputs):
                continue
            if candidate.route_id == current.route_id:
                continue
            if candidate.failure_domain == current.failure_domain and current.failure_domain:
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_alternate_route_auth(
        self,
        inputs: RecoveryCoordinatorInput,
    ) -> RecoveryCandidate | None:
        current = self._current_candidate(inputs)
        for candidate in self._ordered_candidates(inputs):
            if candidate == current:
                continue
            if not self._operationally_eligible(candidate, inputs):
                continue
            if candidate.route_id == current.route_id:
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_different_provider_candidate(
        self,
        inputs: RecoveryCoordinatorInput,
    ) -> RecoveryCandidate | None:
        if not self._provider_switch_allowed(inputs):
            return None
        current = self._current_candidate(inputs)
        for candidate in self._ordered_candidates(inputs):
            if candidate.provider_id == current.provider_id:
                continue
            if not self._operationally_eligible(candidate, inputs):
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_different_model_candidate(
        self,
        inputs: RecoveryCoordinatorInput,
    ) -> RecoveryCandidate | None:
        if not self._model_switch_allowed(inputs):
            return None
        current = self._current_candidate(inputs)
        for candidate in self._ordered_candidates(inputs):
            if candidate.model_id == current.model_id:
                continue
            if not self._operationally_eligible(candidate, inputs):
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_cross_provider_candidate(
        self,
        inputs: RecoveryCoordinatorInput,
    ) -> RecoveryCandidate | None:
        if not self._provider_switch_allowed(inputs):
            return None
        current = self._current_candidate(inputs)
        for candidate in self._ordered_candidates(inputs):
            if candidate.provider_id == current.provider_id:
                continue
            if candidate.failure_domain == current.failure_domain and current.failure_domain:
                continue
            if not self._operationally_eligible(candidate, inputs):
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_cross_model_provider_candidate(
        self,
        inputs: RecoveryCoordinatorInput,
    ) -> RecoveryCandidate | None:
        if not (self._provider_switch_allowed(inputs) and self._model_switch_allowed(inputs)):
            return None
        current = self._current_candidate(inputs)
        for candidate in self._ordered_candidates(inputs):
            if candidate.model_id == current.model_id:
                continue
            if candidate.provider_id == current.provider_id:
                continue
            if not self._operationally_eligible(candidate, inputs):
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_capable_candidate(
        self,
        inputs: RecoveryCoordinatorInput,
    ) -> RecoveryCandidate | None:
        role = inputs.role
        for candidate in self._ordered_candidates(inputs):
            if not self._operationally_eligible(candidate, inputs):
                continue
            if role is not None:
                req = CapabilityRequirement(required_roles=frozenset({role.value}))
                if not match_capabilities(candidate.capabilities, req).eligible:
                    continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _select_larger_context_candidate(
        self,
        inputs: RecoveryCoordinatorInput,
    ) -> RecoveryCandidate | None:
        if not self._model_switch_allowed(inputs):
            return None
        current = self._current_candidate(inputs)
        current_tokens = current.capabilities.context_tokens or 0
        for candidate in self._ordered_candidates(inputs):
            if candidate.model_id == current.model_id:
                continue
            tokens = candidate.capabilities.context_tokens or 0
            if tokens <= current_tokens:
                continue
            if not self._operationally_eligible(candidate, inputs):
                continue
            if self._matches_pin(candidate, inputs.pin):
                return candidate
        return None

    def _ordered_candidates(
        self,
        inputs: RecoveryCoordinatorInput,
    ) -> tuple[RecoveryCandidate, ...]:
        # Deterministic ordering: provider_id, model_id, route_id.
        return tuple(
            sorted(
                inputs.candidates,
                key=lambda c: (c.provider_id, c.model_id, c.route_id),
            )
        )

    def _operationally_eligible(
        self,
        candidate: RecoveryCandidate,
        inputs: RecoveryCoordinatorInput,
    ) -> bool:
        if not candidate.recovery_state.is_eligible():
            return False
        return lifecycle_eligible(candidate.model_identity.lifecycle, inputs.current_risk)

    def _provider_switch_allowed(self, inputs: RecoveryCoordinatorInput) -> bool:
        return inputs.ledger.provider_switch_count() < inputs.policy.max_provider_switches

    def _model_switch_allowed(self, inputs: RecoveryCoordinatorInput) -> bool:
        return inputs.ledger.model_switch_count() < inputs.policy.max_model_switches

    def _is_quota_exhausted(
        self,
        candidate: RecoveryCandidate,
        domain_states: dict[str, ProviderQuotaState] | None,
    ) -> bool:
        quota = candidate.quota
        if candidate.quota_domain and domain_states and candidate.quota_domain in domain_states:
            quota = domain_states[candidate.quota_domain]
        if quota is None:
            return False
        return quota.is_exhausted()

    def _is_pin_exhausted(self, inputs: RecoveryCoordinatorInput) -> bool:
        if inputs.pin is None:
            return False
        return self._is_quota_exhausted(inputs.pin, inputs.quota_domain_states)

    def _matches_pin(
        self,
        candidate: RecoveryCandidate,
        pin: RecoveryCandidate | None,
    ) -> bool:
        if pin is None:
            return True
        if pin.provider_id is not None and candidate.provider_id != pin.provider_id:
            return False
        if pin.model_id is not None and candidate.model_id != pin.model_id:
            return False
        return pin.route_id is None or candidate.route_id == pin.route_id

    def _current_candidate(self, inputs: RecoveryCoordinatorInput) -> RecoveryCandidate:
        if inputs.pin is not None:
            return inputs.pin
        classifier_input = inputs.classifier_input
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
            if provider_match and model_match and route_match:
                return candidate
        if inputs.candidates:
            return self._ordered_candidates(inputs)[0]
        raise ValueError("no current candidate available")

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
        return {
            "estimated_input_chars": meta.estimated_input_chars,
            "model_context_tokens": meta.model_context_tokens,
            "authority_required": meta.authority_required,
            "authority_items_present": meta.authority_items_present,
            "authority_items_raw": meta.authority_items_raw,
            "rebuild_attempts": meta.rebuild_attempts,
        }

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
