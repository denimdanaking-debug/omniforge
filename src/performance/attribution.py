"""Performance-event attribution: decide whether an event reflects model quality,
provider/route health, task process, review quality, or context strategy."""

from __future__ import annotations

from enum import StrEnum

from src.recovery.failure_classification import FailureClassification
from src.telemetry.outcomes import OutcomeAttribution, TaskOutcome


class PerformanceAttribution(StrEnum):
    """Empirical dimension an event primarily informs."""

    MODEL_QUALITY = "model_quality"
    PROVIDER_ROUTE = "provider_route"
    TASK_PROCESS = "task_process"
    REVIEW_QUALITY = "review_quality"
    CONTEXT_STRATEGY = "context_strategy"
    UNKNOWN = "unknown"


def attribution_from_failure_classification(
    classification: FailureClassification,
) -> PerformanceAttribution:
    """Map a Phase 10 failure classification to a performance attribution."""
    if classification.model_quality_effect:
        return PerformanceAttribution.MODEL_QUALITY
    category = classification.category
    if category.value.startswith("INFRASTRUCTURE") or category.value == "ROUTE_FAILURE":
        return PerformanceAttribution.PROVIDER_ROUTE
    if category.value in {"CONTEXT_CAPACITY", "CAPABILITY_MISMATCH"}:
        return PerformanceAttribution.CONTEXT_STRATEGY
    if category.value in {"AUTHORITY_VIOLATION", "INTEGRATION_FAILURE"}:
        return PerformanceAttribution.TASK_PROCESS
    return PerformanceAttribution.UNKNOWN


def attribution_from_task_outcome(outcome: TaskOutcome) -> PerformanceAttribution:
    """Map a normalized task outcome to a performance attribution."""
    if outcome.attribution == OutcomeAttribution.MODEL:
        return PerformanceAttribution.MODEL_QUALITY
    if outcome.attribution == OutcomeAttribution.PROVIDER:
        return PerformanceAttribution.PROVIDER_ROUTE
    if outcome.attribution == OutcomeAttribution.SYSTEM:
        return PerformanceAttribution.TASK_PROCESS
    return PerformanceAttribution.UNKNOWN


def is_infrastructure_attribution(attribution: PerformanceAttribution) -> bool:
    return attribution is PerformanceAttribution.PROVIDER_ROUTE


def affects_model_quality(attribution: PerformanceAttribution) -> bool:
    return attribution is PerformanceAttribution.MODEL_QUALITY
