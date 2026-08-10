"""Tests for performance-event attribution."""

from __future__ import annotations

from typing import Any

from src.performance import (
    PerformanceAttribution,
    affects_model_quality,
    attribution_from_failure_classification,
    attribution_from_task_outcome,
    is_infrastructure_attribution,
)
from src.recovery.failure_classification import (
    FailureCategory,
    FailureClassification,
    FailureDomain,
    Retryability,
)
from src.telemetry.outcomes import (
    OutcomeAttribution,
    OutcomeKind,
    TaskOutcome,
    provider_failure,
    successful_outcome,
)


def _classification(category: FailureCategory) -> FailureClassification:
    return FailureClassification(
        category=category,
        subtype=_default_subtype(category),
        retryability=Retryability.BOUNDED,
        failure_domain=FailureDomain(provider_id="provider"),
        model_quality_effect=category
        in {
            FailureCategory.STRUCTURED_OUTPUT_INVALID,
            FailureCategory.PLANNING_OUTPUT_INVALID,
            FailureCategory.IMPLEMENTATION_DETERMINISTIC,
            FailureCategory.IMPLEMENTATION_CONCEPTUAL,
        },
        provider_health_effect=category.value.startswith("INFRASTRUCTURE"),
        route_health_effect=category.value.startswith("INFRASTRUCTURE"),
        recommended_action_class="BLOCK",
        evidence_refs=(),
        deterministic_fingerprint="fp",
    )


def _default_subtype(category: FailureCategory) -> Any:
    from src.recovery.failure_classification import FailureSubtype

    mapping = {
        FailureCategory.INFRASTRUCTURE_TRANSIENT: FailureSubtype.TRANSIENT_TRANSPORT,
        FailureCategory.INFRASTRUCTURE_QUOTA: FailureSubtype.QUOTA_EXHAUSTED,
        FailureCategory.INFRASTRUCTURE_AUTH: FailureSubtype.AUTH_FAILURE,
        FailureCategory.STRUCTURED_OUTPUT_INVALID: FailureSubtype.MISSING_REQUIRED_FIELDS,
        FailureCategory.PLANNING_OUTPUT_INVALID: FailureSubtype.MISSING_PLAN_STEPS,
        FailureCategory.IMPLEMENTATION_DETERMINISTIC: FailureSubtype.COMPILE_FAILURE,
        FailureCategory.AUTHORITY_VIOLATION: FailureSubtype.INTEGRATION_ANOMALY,
    }
    return mapping.get(category, FailureSubtype.UNKNOWN)


class TestAttributionFromTaskOutcome:
    def test_success_is_not_model_quality(self) -> None:
        outcome = successful_outcome()
        assert attribution_from_task_outcome(outcome) is PerformanceAttribution.UNKNOWN
        assert not affects_model_quality(attribution_from_task_outcome(outcome))

    def test_provider_failure_is_provider_route(self) -> None:
        outcome = provider_failure("outage")
        assert attribution_from_task_outcome(outcome) is PerformanceAttribution.PROVIDER_ROUTE
        assert is_infrastructure_attribution(attribution_from_task_outcome(outcome))

    def test_model_failure_is_model_quality(self) -> None:
        outcome = TaskOutcome(
            OutcomeKind.DETERMINISTIC_VALIDATION_FAILURE,
            OutcomeAttribution.MODEL,
        )
        assert attribution_from_task_outcome(outcome) is PerformanceAttribution.MODEL_QUALITY
        assert affects_model_quality(attribution_from_task_outcome(outcome))


class TestAttributionFromFailureClassification:
    def test_provider_outage_not_model_quality(self) -> None:
        classification = _classification(FailureCategory.INFRASTRUCTURE_TRANSIENT)
        assert (
            attribution_from_failure_classification(classification)
            is PerformanceAttribution.PROVIDER_ROUTE
        )
        assert not affects_model_quality(attribution_from_failure_classification(classification))

    def test_quota_exhausted_not_model_quality(self) -> None:
        classification = _classification(FailureCategory.INFRASTRUCTURE_QUOTA)
        assert (
            attribution_from_failure_classification(classification)
            is PerformanceAttribution.PROVIDER_ROUTE
        )

    def test_auth_failure_not_model_quality(self) -> None:
        classification = _classification(FailureCategory.INFRASTRUCTURE_AUTH)
        assert (
            attribution_from_failure_classification(classification)
            is PerformanceAttribution.PROVIDER_ROUTE
        )

    def test_invalid_structured_output_is_model_quality(self) -> None:
        classification = _classification(FailureCategory.STRUCTURED_OUTPUT_INVALID)
        assert (
            attribution_from_failure_classification(classification)
            is PerformanceAttribution.MODEL_QUALITY
        )

    def test_invalid_plan_is_model_quality(self) -> None:
        classification = _classification(FailureCategory.PLANNING_OUTPUT_INVALID)
        assert (
            attribution_from_failure_classification(classification)
            is PerformanceAttribution.MODEL_QUALITY
        )

    def test_deterministic_implementation_is_model_quality(self) -> None:
        classification = _classification(FailureCategory.IMPLEMENTATION_DETERMINISTIC)
        assert (
            attribution_from_failure_classification(classification)
            is PerformanceAttribution.MODEL_QUALITY
        )

    def test_authority_violation_is_task_process(self) -> None:
        classification = _classification(FailureCategory.AUTHORITY_VIOLATION)
        assert (
            attribution_from_failure_classification(classification)
            is PerformanceAttribution.TASK_PROCESS
        )
