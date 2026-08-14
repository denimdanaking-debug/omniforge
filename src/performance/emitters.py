"""Pure helper builders that construct PerformanceEvent instances from existing
normalized outcomes, recovery decisions, and integration evidence.

These emitters are intentionally not wired into live orchestration; later phases
may call them at the appropriate lifecycle seams.
"""

from __future__ import annotations

import datetime
from typing import Any, cast

from src.context.outcomes import ContextOutcomeRecord
from src.orchestration.project_contract import AdvancementEvidence
from src.performance.attribution import (
    PerformanceAttribution,
    attribution_from_failure_classification,
    attribution_from_task_outcome,
)
from src.performance.event import (
    AcceptanceStatus,
    AuthorityAdherenceStatus,
    Cost,
    CostState,
    FindingDisposition,
    OutcomeCategory,
    PerformanceEvent,
    PerformanceEventType,
    RepairMetadata,
    Usage,
    event_identity,
)
from src.recovery.failure_classification import (
    AuthorityViolationData,
    FailureCategory,
    FailureClassification,
)
from src.recovery.recovery_coordinator import RecoveryDecision
from src.security.redaction import redact
from src.telemetry.outcomes import OutcomeKind, TaskOutcome

CURRENT_EVENT_SCHEMA = "1.0.0"


def _base_event(
    *,
    event_type: PerformanceEventType,
    project_id: str,
    task_id: str,
    run_id: str,
    timestamp: datetime.datetime,
    execution_role: str,
    task_class: str,
    risk: str,
    provider_id: str | None,
    model_id: str | None,
    route_id: str | None,
    sequence: int = 0,
    **kwargs: Any,
) -> PerformanceEvent:
    """Build a PerformanceEvent with deterministic identity and safe payload."""
    outcome_category = kwargs.get("outcome_category", OutcomeCategory.UNKNOWN)
    event_id = event_identity(
        event_type=event_type,
        task_id=task_id,
        run_id=run_id,
        timestamp=timestamp,
        model_id=model_id,
        provider_id=provider_id,
        route_id=route_id,
        outcome_category=outcome_category,
        sequence=sequence,
    )
    return PerformanceEvent(
        event_id=event_id,
        schema_version=CURRENT_EVENT_SCHEMA,
        timestamp=timestamp,
        event_type=event_type,
        project_id=project_id,
        task_id=task_id,
        run_id=run_id,
        execution_role=execution_role,
        task_class=task_class,
        risk=risk,
        provider_id=provider_id,
        model_id=model_id,
        route_id=route_id,
        **kwargs,
    )


def _outcome_category_from_task_outcome(outcome: TaskOutcome) -> OutcomeCategory:
    kind = outcome.kind
    if kind is OutcomeKind.SUCCESS:
        return OutcomeCategory.SUCCESS
    if kind is OutcomeKind.DETERMINISTIC_VALIDATION_FAILURE:
        return OutcomeCategory.DETERMINISTIC_VALIDATION_FAILURE
    if kind is OutcomeKind.AUTHORITY_VIOLATION:
        return OutcomeCategory.AUTHORITY_VIOLATION
    if kind is OutcomeKind.INVALID_MODEL_OUTPUT:
        return OutcomeCategory.STRUCTURED_OUTPUT_INVALID
    if kind is OutcomeKind.TASK_FAILURE:
        return OutcomeCategory.CONCEPTUAL_FAILURE
    if kind is OutcomeKind.PROVIDER_FAILURE:
        return OutcomeCategory.PROVIDER_FAILURE
    return OutcomeCategory.UNKNOWN


def _outcome_category_from_classification(
    classification: FailureClassification,
) -> OutcomeCategory:
    category = classification.category
    if category is FailureCategory.INFRASTRUCTURE_TRANSIENT:
        return OutcomeCategory.INFRASTRUCTURE_TRANSIENT
    if category is FailureCategory.INFRASTRUCTURE_QUOTA:
        return OutcomeCategory.QUOTA_EXHAUSTED
    if category is FailureCategory.INFRASTRUCTURE_AUTH:
        return OutcomeCategory.AUTH_FAILURE
    if category is FailureCategory.INFRASTRUCTURE_UNAVAILABLE:
        return OutcomeCategory.ROUTE_FAILURE
    if category is FailureCategory.CAPABILITY_MISMATCH:
        return OutcomeCategory.CAPABILITY_MISMATCH
    if category is FailureCategory.CONTEXT_CAPACITY:
        return OutcomeCategory.CONTEXT_CAPACITY
    if category is FailureCategory.STRUCTURED_OUTPUT_INVALID:
        return OutcomeCategory.STRUCTURED_OUTPUT_INVALID
    if category is FailureCategory.PLANNING_OUTPUT_INVALID:
        return OutcomeCategory.PLAN_INVALID
    if category is FailureCategory.IMPLEMENTATION_DETERMINISTIC:
        return OutcomeCategory.DETERMINISTIC_VALIDATION_FAILURE
    if category is FailureCategory.IMPLEMENTATION_CONCEPTUAL:
        return OutcomeCategory.CONCEPTUAL_FAILURE
    if category is FailureCategory.AUTHORITY_VIOLATION:
        return OutcomeCategory.AUTHORITY_VIOLATION
    if category is FailureCategory.CANCELLED:
        return OutcomeCategory.CANCELLED
    return OutcomeCategory.UNKNOWN


def _authority_status_from_violation(
    data: AuthorityViolationData | None,
) -> AuthorityAdherenceStatus | None:
    if data is None:
        return None
    if data.attempted_state_advancement:
        return AuthorityAdherenceStatus.INVALID_STATE_ADVANCEMENT
    if data.summary_substituted_for_raw:
        return AuthorityAdherenceStatus.SUMMARY_SUBSTITUTED_FOR_RAW
    if data.integration_state_mismatch:
        return AuthorityAdherenceStatus.INTEGRATION_STATE_MISMATCH
    if data.ignored_immutable_authority:
        return AuthorityAdherenceStatus.IGNORED_IMMUTABLE
    if data.touched_authority_paths:
        return AuthorityAdherenceStatus.ATTEMPTED_MUTATION
    return AuthorityAdherenceStatus.COMPLIANT


def emit_from_task_outcome(
    *,
    outcome: TaskOutcome,
    project_id: str,
    task_id: str,
    run_id: str,
    timestamp: datetime.datetime,
    execution_role: str,
    task_class: str,
    risk: str,
    provider_id: str | None,
    model_id: str | None,
    route_id: str | None,
    context_strategy: str | None = None,
    language_framework: str | None = None,
    first_pass: bool | None = None,
    latency_seconds: float | None = None,
    provider_wait_seconds: float | None = None,
    usage: Usage | None = None,
    direct_cost: Cost | None = None,
    evidence_refs: tuple[str, ...] = (),
    originating_ids: dict[str, str] | None = None,
    sequence: int = 0,
) -> PerformanceEvent:
    """Build a performance event from a normalized TaskOutcome.

    A successful task output records ``OutcomeCategory.SUCCESS`` but remains
    ``AcceptanceStatus.PENDING`` until canonical integration acceptance.
    """
    outcome_category = _outcome_category_from_task_outcome(outcome)
    # Model/task output success is not authoritative integration acceptance.
    acceptance = (
        AcceptanceStatus.PENDING
        if outcome.kind is OutcomeKind.SUCCESS
        else AcceptanceStatus.REJECTED
    )
    attribution = attribution_from_task_outcome(outcome).value
    return _base_event(
        event_type=PerformanceEventType.TASK_OUTCOME,
        project_id=project_id,
        task_id=task_id,
        run_id=run_id,
        timestamp=timestamp,
        execution_role=execution_role,
        task_class=task_class,
        risk=risk,
        provider_id=provider_id,
        model_id=model_id,
        route_id=route_id,
        sequence=sequence,
        outcome_category=outcome_category,
        acceptance_status=acceptance,
        first_pass=first_pass,
        context_strategy=context_strategy,
        language_framework=language_framework,
        latency_seconds=latency_seconds,
        provider_wait_seconds=provider_wait_seconds,
        usage=usage or Usage(),
        direct_cost=direct_cost or Cost(None, "USD", CostState.UNKNOWN),
        evidence_refs=evidence_refs,
        originating_ids=originating_ids or {},
        attribution=attribution,
    )


def emit_from_recovery_decision(
    *,
    decision: RecoveryDecision,
    project_id: str,
    task_id: str,
    run_id: str,
    timestamp: datetime.datetime,
    execution_role: str,
    task_class: str,
    risk: str,
    first_pass: bool = False,
    provider_wait_seconds: float | None = None,
    usage: Usage | None = None,
    direct_cost: Cost | None = None,
    evidence_refs: tuple[str, ...] = (),
    originating_ids: dict[str, str] | None = None,
    sequence: int = 0,
) -> PerformanceEvent:
    """Build a performance event from a Phase 10 recovery decision."""
    classification = decision.classification
    outcome_category = _outcome_category_from_classification(classification)
    # A recovery action is never acceptance; only canonical accepted-integration
    # evidence may finalize acceptance.
    acceptance = AcceptanceStatus.PENDING
    candidate = decision.selected_candidate
    provider_id = candidate.provider_id if candidate else None
    model_id = candidate.model_id if candidate else None
    route_id = candidate.route_id if candidate else None
    attribution = attribution_from_failure_classification(classification).value
    repair_meta = None
    if decision.action.value in {
        "REPAIR_WITH_EVIDENCE",
        "CROSS_MODEL_REPAIR",
    }:
        repair_meta = RepairMetadata(
            repair_model_id=model_id,
            original_model_id=originating_ids.get("original_model_id") if originating_ids else None,
            required_cross_model_escalation=decision.action.value == "CROSS_MODEL_REPAIR",
        )
    return _base_event(
        event_type=PerformanceEventType.RECOVERY_DECISION,
        project_id=project_id,
        task_id=task_id,
        run_id=run_id,
        timestamp=timestamp,
        execution_role=execution_role,
        task_class=task_class,
        risk=risk,
        provider_id=provider_id,
        model_id=model_id,
        route_id=route_id,
        sequence=sequence,
        outcome_category=outcome_category,
        acceptance_status=acceptance,
        first_pass=first_pass,
        repair_metadata=repair_meta,
        provider_wait_seconds=provider_wait_seconds,
        usage=usage or Usage(),
        direct_cost=direct_cost or Cost(None, "USD", CostState.UNKNOWN),
        evidence_refs=evidence_refs,
        originating_ids=originating_ids or {},
        attribution=attribution,
    )


def emit_from_advancement_evidence(
    *,
    evidence: AdvancementEvidence,
    project_id: str,
    task_id: str,
    run_id: str,
    timestamp: datetime.datetime,
    execution_role: str,
    task_class: str,
    risk: str,
    provider_id: str | None,
    model_id: str | None,
    route_id: str | None,
    context_strategy: str | None = None,
    language_framework: str | None = None,
    first_pass: bool | None = None,
    time_to_accepted_seconds: float | None = None,
    total_cost_to_accepted: Cost | None = None,
    evidence_refs: tuple[str, ...] = (),
    originating_ids: dict[str, str] | None = None,
    sequence: int = 0,
) -> PerformanceEvent:
    """Build an accepted-integration event from authoritative advancement evidence."""
    accepted = bool(
        evidence.implemented
        and evidence.deterministic_validation_passed
        and evidence.independent_review_satisfied
        and evidence.safely_integrated
    )
    return _base_event(
        event_type=PerformanceEventType.INTEGRATION_ACCEPTED
        if accepted
        else PerformanceEventType.INTEGRATION_REJECTED,
        project_id=project_id,
        task_id=task_id,
        run_id=run_id,
        timestamp=timestamp,
        execution_role=execution_role,
        task_class=task_class,
        risk=risk,
        provider_id=provider_id,
        model_id=model_id,
        route_id=route_id,
        sequence=sequence,
        outcome_category=OutcomeCategory.SUCCESS if accepted else OutcomeCategory.UNKNOWN,
        acceptance_status=AcceptanceStatus.ACCEPTED if accepted else AcceptanceStatus.REJECTED,
        first_pass=first_pass,
        context_strategy=context_strategy,
        language_framework=language_framework,
        time_to_accepted_seconds=time_to_accepted_seconds,
        total_cost_to_accepted=total_cost_to_accepted,
        evidence_refs=evidence_refs,
        originating_ids=originating_ids or {},
        attribution=PerformanceAttribution.TASK_PROCESS.value,
    )


def emit_from_context_outcome(
    *,
    outcome: ContextOutcomeRecord,
    project_id: str,
    task_id: str,
    run_id: str,
    timestamp: datetime.datetime,
    provider_id: str | None,
    route_id: str | None,
    latency_seconds: float | None = None,
    usage: Usage | None = None,
    evidence_refs: tuple[str, ...] = (),
    originating_ids: dict[str, str] | None = None,
    sequence: int = 0,
) -> PerformanceEvent:
    """Build a performance event from a context-outcome record."""
    authority = (
        AuthorityAdherenceStatus.COMPLIANT
        if outcome.accepted and not outcome.repair_required
        else None
    )
    return _base_event(
        event_type=PerformanceEventType.CONTEXT_OUTCOME,
        project_id=project_id,
        task_id=task_id,
        run_id=run_id,
        timestamp=timestamp,
        execution_role=outcome.role,
        task_class=outcome.task_class,
        risk=outcome.risk,
        provider_id=provider_id,
        model_id=outcome.model_id,
        route_id=route_id,
        sequence=sequence,
        context_strategy=outcome.strategy,
        outcome_category=OutcomeCategory(outcome.failure_category)
        if outcome.failure_category
        else OutcomeCategory.SUCCESS,
        # Context outcome acceptance is not authoritative integration acceptance.
        acceptance_status=AcceptanceStatus.PENDING
        if outcome.accepted
        else AcceptanceStatus.REJECTED,
        first_pass=not outcome.repair_required,
        authority_adherence=authority,
        latency_seconds=latency_seconds,
        usage=usage or Usage(),
        direct_cost=Cost(None, "USD", CostState.UNKNOWN),
        evidence_refs=evidence_refs,
        originating_ids=originating_ids or {},
        attribution=PerformanceAttribution.CONTEXT_STRATEGY.value,
    )


def emit_reviewer_finding_event(
    *,
    finding_id: str,
    disposition: FindingDisposition,
    reviewer_model_id: str,
    project_id: str,
    task_id: str,
    run_id: str,
    timestamp: datetime.datetime,
    risk: str,
    task_class: str,
    evidence_refs: tuple[str, ...] = (),
    originating_ids: dict[str, str] | None = None,
    sequence: int = 0,
) -> PerformanceEvent:
    """Build a reviewer-finding disposition event."""
    return _base_event(
        event_type=PerformanceEventType.REVIEWER_FINDING,
        project_id=project_id,
        task_id=task_id,
        run_id=run_id,
        timestamp=timestamp,
        execution_role="review",
        task_class=task_class,
        risk=risk,
        provider_id=None,
        model_id=reviewer_model_id,
        route_id=None,
        sequence=sequence,
        review_finding_dispositions={finding_id: disposition},
        evidence_refs=evidence_refs,
        originating_ids=originating_ids or {},
        attribution=PerformanceAttribution.REVIEW_QUALITY.value,
    )


def emit_reviewer_false_negative_event(
    *,
    original_review_event_id: str,
    defect_id: str,
    reviewer_model_id: str,
    project_id: str,
    task_id: str,
    run_id: str,
    timestamp: datetime.datetime,
    risk: str,
    task_class: str,
    evidence_refs: tuple[str, ...] = (),
    originating_ids: dict[str, str] | None = None,
    sequence: int = 0,
) -> PerformanceEvent:
    """Build a false-negative annotation event with required evidence."""
    return _base_event(
        event_type=PerformanceEventType.REVIEWER_FALSE_NEGATIVE,
        project_id=project_id,
        task_id=task_id,
        run_id=run_id,
        timestamp=timestamp,
        execution_role="review",
        task_class=task_class,
        risk=risk,
        provider_id=None,
        model_id=reviewer_model_id,
        route_id=None,
        sequence=sequence,
        outcome_category=OutcomeCategory.UNKNOWN,
        evidence_refs=evidence_refs,
        originating_ids={
            "original_review_event_id": original_review_event_id,
            "defect_id": defect_id,
            **(originating_ids or {}),
        },
        attribution=PerformanceAttribution.REVIEW_QUALITY.value,
    )


def emit_correction_event(
    *,
    corrected_event_id: str,
    correction_reason: str,
    project_id: str,
    task_id: str,
    run_id: str,
    timestamp: datetime.datetime,
    execution_role: str,
    task_class: str,
    risk: str,
    provider_id: str | None,
    model_id: str | None,
    route_id: str | None,
    evidence_refs: tuple[str, ...] = (),
    sequence: int = 0,
) -> PerformanceEvent:
    """Build an explicit correction event; never mutates the corrected event."""
    return _base_event(
        event_type=PerformanceEventType.CORRECTION,
        project_id=project_id,
        task_id=task_id,
        run_id=run_id,
        timestamp=timestamp,
        execution_role=execution_role,
        task_class=task_class,
        risk=risk,
        provider_id=provider_id,
        model_id=model_id,
        route_id=route_id,
        sequence=sequence,
        outcome_category=OutcomeCategory.UNKNOWN,
        evidence_refs=evidence_refs,
        originating_ids={
            "corrected_event_id": corrected_event_id,
            "correction_reason": correction_reason,
        },
        attribution=PerformanceAttribution.TASK_PROCESS.value,
    )


def safe_event_dict(event: PerformanceEvent) -> dict[str, Any]:
    """Redact an event dict before persistence or external display."""
    return cast(dict[str, Any], redact(event.to_dict()))
