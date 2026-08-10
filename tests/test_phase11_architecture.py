"""Architectural invariants for OmniForge Phase 11.

Guards against the anti-patterns listed in the Phase 11 spec:
- provider/quota/auth/route failures treated as model-quality failures
- one global model quality scalar
- mutable historical events
- aggregate-only persistence without raw events
- hidden brand rankings
- Phase 12 confidence/bandit implementation
- authority-state mutation
- direct PROJECT_STATE advancement
- secrets in event ledger
"""

from __future__ import annotations

import datetime

import pytest

from src.performance import (
    AcceptanceStatus,
    Cost,
    CostState,
    FindingDisposition,
    OutcomeCategory,
    PerformanceEvent,
    PerformanceEventType,
    PerformanceLedger,
    StatisticsBuilder,
    affects_model_quality,
    attribution_from_failure_classification,
)
from src.recovery.failure_classification import (
    FailureCategory,
    FailureClassification,
    FailureDomain,
    FailureSubtype,
    Retryability,
)
from src.security.redaction import contains_secret

SENTINEL = "OMNIFORGE_PHASE11_ARCH_SECRET_SENTINEL_999"


@pytest.fixture
def base_time() -> datetime.datetime:
    return datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)


def _classification(category: FailureCategory, model_quality_effect: bool) -> FailureClassification:
    return FailureClassification(
        category=category,
        subtype=FailureSubtype.UNKNOWN,
        retryability=Retryability.BOUNDED,
        failure_domain=FailureDomain(provider_id="provider"),
        model_quality_effect=model_quality_effect,
        provider_health_effect=category.value.startswith("INFRASTRUCTURE"),
        route_health_effect=category.value.startswith("INFRASTRUCTURE"),
        recommended_action_class="BLOCK",
        evidence_refs=(),
        deterministic_fingerprint="fp",
    )


def _event(
    *,
    event_id: str,
    timestamp: datetime.datetime,
    event_type: PerformanceEventType = PerformanceEventType.TASK_OUTCOME,
    model_id: str = "gpt-4o",
    provider_id: str = "openai",
    route_id: str = "openai-direct",
    role: str = "coding",
    outcome_category: OutcomeCategory = OutcomeCategory.SUCCESS,
    first_pass: bool = True,
    review_dispositions: dict[str, FindingDisposition] | None = None,
) -> PerformanceEvent:
    return PerformanceEvent(
        event_id=event_id,
        schema_version="1.0.0",
        timestamp=timestamp,
        event_type=event_type,
        project_id="omniforge",
        task_id="task-1",
        run_id="run-1",
        execution_role=role,
        task_class="feature",
        risk="R2_NORMAL",
        provider_id=provider_id,
        model_id=model_id,
        route_id=route_id,
        outcome_category=outcome_category,
        acceptance_status=AcceptanceStatus.ACCEPTED,
        first_pass=first_pass,
        review_finding_dispositions=review_dispositions or {},
        direct_cost=Cost(0.001, "USD", CostState.ACTUAL),
    )


class TestInfrastructureDoesNotAffectModelQuality:
    def test_transient_outage_not_model_quality(self) -> None:
        classification = _classification(FailureCategory.INFRASTRUCTURE_TRANSIENT, False)
        assert not affects_model_quality(attribution_from_failure_classification(classification))

    def test_quota_exhausted_not_model_quality(self) -> None:
        classification = _classification(FailureCategory.INFRASTRUCTURE_QUOTA, False)
        assert not affects_model_quality(attribution_from_failure_classification(classification))

    def test_auth_failure_not_model_quality(self) -> None:
        classification = _classification(FailureCategory.INFRASTRUCTURE_AUTH, False)
        assert not affects_model_quality(attribution_from_failure_classification(classification))

    def test_route_outage_not_model_quality(self) -> None:
        classification = _classification(FailureCategory.INFRASTRUCTURE_UNAVAILABLE, False)
        assert not affects_model_quality(attribution_from_failure_classification(classification))


class TestModelQualityFailures:
    def test_invalid_plan_is_model_quality(self) -> None:
        classification = _classification(FailureCategory.PLANNING_OUTPUT_INVALID, True)
        assert affects_model_quality(attribution_from_failure_classification(classification))

    def test_invalid_structured_output_is_model_quality(self) -> None:
        classification = _classification(FailureCategory.STRUCTURED_OUTPUT_INVALID, True)
        assert affects_model_quality(attribution_from_failure_classification(classification))

    def test_deterministic_implementation_is_model_quality(self) -> None:
        classification = _classification(FailureCategory.IMPLEMENTATION_DETERMINISTIC, True)
        assert affects_model_quality(attribution_from_failure_classification(classification))


class TestNoGlobalQualityScalar:
    def test_statistics_are_segmented(self, base_time: datetime.datetime) -> None:
        e1 = _event(event_id="e1", timestamp=base_time, model_id="gpt-4o", role="coding")
        e2 = _event(event_id="e2", timestamp=base_time, model_id="claude", role="coding")
        bundle = StatisticsBuilder().build(PerformanceLedger().append_all((e1, e2)))
        assert ("gpt-4o", "coding") in bundle.model_role
        assert ("claude", "coding") in bundle.model_role
        assert not hasattr(bundle, "global_model_quality_score")


class TestEventImmutability:
    def test_events_are_frozen(self, base_time: datetime.datetime) -> None:
        event = _event(event_id="e1", timestamp=base_time)
        with pytest.raises(AttributeError):
            event.acceptance_status = "rejected"  # type: ignore


class TestLedgerAppendOnly:
    def test_events_not_mutated_after_append(self, base_time: datetime.datetime) -> None:
        event = _event(event_id="e1", timestamp=base_time)
        ledger = PerformanceLedger().append(event)
        assert len(ledger.events) == 1
        with pytest.raises(AttributeError):
            ledger.events[0].model_id = "x"  # type: ignore[misc]


class TestNoPhase12Implementation:
    def test_no_bandit_router_exists(self) -> None:
        import src

        modules = [m for m in dir(src) if "bandit" in m.lower() or "confidence" in m.lower()]
        assert modules == []

    def test_no_global_winner_selection(self, base_time: datetime.datetime) -> None:
        bundle = StatisticsBuilder().build(PerformanceLedger())
        assert not hasattr(bundle, "selected_model")


class TestNoSecretsInLedger:
    def test_secret_sentinel_absent(self, base_time: datetime.datetime) -> None:
        event = PerformanceEvent.from_dict(
            {
                **_event(event_id="e1", timestamp=base_time).to_dict(),
                "originating_ids": {"api_key": SENTINEL},
            }
        )
        ledger = PerformanceLedger().append(event)
        safe = ledger.to_safe_dict()
        assert not contains_secret(safe, SENTINEL)
        assert SENTINEL not in str(safe)
