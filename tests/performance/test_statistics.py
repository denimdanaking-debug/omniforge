"""Tests for deterministic statistics rebuild from the performance ledger."""

from __future__ import annotations

import datetime
from typing import Any

import pytest

from src.performance import (
    AcceptanceStatus,
    AuthorityAdherenceStatus,
    Cost,
    CostState,
    FindingDisposition,
    OutcomeCategory,
    PerformanceEvent,
    PerformanceEventType,
    PerformanceLedger,
    StatisticsBuilder,
    Usage,
    safe_rate,
)


@pytest.fixture
def base_time() -> datetime.datetime:
    return datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)


def _task_event(
    *,
    event_id: str,
    timestamp: datetime.datetime,
    model_id: str = "gpt-4o",
    provider_id: str = "openai",
    route_id: str = "openai-direct",
    role: str = "coding",
    outcome_category: OutcomeCategory = OutcomeCategory.SUCCESS,
    acceptance_status: AcceptanceStatus = AcceptanceStatus.ACCEPTED,
    first_pass: bool = True,
    context_strategy: str | None = None,
    language_framework: str | None = None,
    risk: str = "R2_NORMAL",
    repair_metadata: Any | None = None,
    authority_adherence: AuthorityAdherenceStatus | None = None,
    direct_cost: Cost | None = None,
    latency_seconds: float | None = None,
    provider_wait_seconds: float | None = None,
    usage: Usage | None = None,
) -> PerformanceEvent:
    return PerformanceEvent(
        event_id=event_id,
        schema_version="1.0.0",
        timestamp=timestamp,
        event_type=PerformanceEventType.TASK_OUTCOME,
        project_id="omniforge",
        task_id="task-1",
        run_id="run-1",
        execution_role=role,
        task_class="feature",
        risk=risk,
        provider_id=provider_id,
        model_id=model_id,
        route_id=route_id,
        context_strategy=context_strategy,
        language_framework=language_framework,
        outcome_category=outcome_category,
        acceptance_status=acceptance_status,
        first_pass=first_pass,
        repair_metadata=repair_metadata,
        authority_adherence=authority_adherence,
        latency_seconds=latency_seconds,
        provider_wait_seconds=provider_wait_seconds,
        usage=usage or Usage(),
        direct_cost=direct_cost or Cost(None, "USD", CostState.UNKNOWN),
    )


class TestSafeRate:
    def test_no_data_returns_none(self) -> None:
        assert safe_rate(0, 0) is None

    def test_rate_computed(self) -> None:
        assert safe_rate(8, 10) == 0.8


class TestStatisticsBuilder:
    def test_role_success_tracked_separately(self, base_time: datetime.datetime) -> None:
        coding = _task_event(
            event_id="e1",
            timestamp=base_time,
            role="coding",
            model_id="gpt-4o",
            outcome_category=OutcomeCategory.SUCCESS,
            first_pass=True,
        )
        review = _task_event(
            event_id="e2",
            timestamp=base_time,
            role="review",
            model_id="gpt-4o",
            outcome_category=OutcomeCategory.SUCCESS,
            first_pass=True,
        )
        ledger = PerformanceLedger().append_all((coding, review))
        bundle = StatisticsBuilder().build(ledger)
        assert bundle.model_role[("gpt-4o", "coding")].attempts == 1
        assert bundle.model_role[("gpt-4o", "review")].attempts == 1

    def test_first_pass_tracked(self, base_time: datetime.datetime) -> None:
        first = _task_event(
            event_id="e1",
            timestamp=base_time,
            model_id="gpt-4o",
            first_pass=True,
        )
        retry = _task_event(
            event_id="e2",
            timestamp=base_time,
            model_id="gpt-4o",
            first_pass=False,
            acceptance_status=AcceptanceStatus.ACCEPTED,
        )
        ledger = PerformanceLedger().append_all((first, retry))
        bundle = StatisticsBuilder().build(ledger)
        stats = bundle.model_role[("gpt-4o", "coding")]
        assert stats.attempts == 2
        assert stats.first_pass_accepted == 1
        assert stats.first_pass_rate() == 0.5

    def test_invalid_plan_counts(self, base_time: datetime.datetime) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            role="planning",
            model_id="gpt-4o",
            outcome_category=OutcomeCategory.PLAN_INVALID,
            acceptance_status=AcceptanceStatus.REJECTED,
            first_pass=False,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        assert bundle.model_role[("gpt-4o", "planning")].invalid_plans == 1

    def test_deterministic_failure_counts(self, base_time: datetime.datetime) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            outcome_category=OutcomeCategory.DETERMINISTIC_VALIDATION_FAILURE,
            acceptance_status=AcceptanceStatus.REJECTED,
            first_pass=False,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        assert bundle.model_role[("gpt-4o", "coding")].deterministic_failures == 1

    def test_provider_outage_does_not_count_model_quality(
        self, base_time: datetime.datetime
    ) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            outcome_category=OutcomeCategory.INFRASTRUCTURE_TRANSIENT,
            acceptance_status=AcceptanceStatus.REJECTED,
            first_pass=False,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        stats = bundle.model_role[("gpt-4o", "coding")]
        assert stats.attempts == 1
        assert stats.accepted == 0
        route_stats = bundle.route["openai-direct"]
        assert route_stats.infrastructure_failures == 1
        assert route_stats.error_count == 1

    def test_quota_exhausted_route_stat(self, base_time: datetime.datetime) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            outcome_category=OutcomeCategory.QUOTA_EXHAUSTED,
            acceptance_status=AcceptanceStatus.REJECTED,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        assert bundle.route["openai-direct"].quota_failures == 1

    def test_repair_tracked(self, base_time: datetime.datetime) -> None:
        from src.performance import RepairMetadata

        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            model_id="claude",
            outcome_category=OutcomeCategory.SUCCESS,
            repair_metadata=RepairMetadata(
                repair_model_id="claude",
                original_model_id="gpt-4o",
                resolved=True,
            ),
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        stats = bundle.model_role[("claude", "coding")]
        assert stats.repairs_attempted == 1
        assert stats.repairs_resolved == 1

    def test_reviewer_findings_tracked(self, base_time: datetime.datetime) -> None:
        event = PerformanceEvent(
            event_id="e1",
            schema_version="1.0.0",
            timestamp=base_time,
            event_type=PerformanceEventType.REVIEWER_FINDING,
            project_id="omniforge",
            task_id="task-1",
            run_id="run-1",
            execution_role="review",
            task_class="feature",
            risk="R2_NORMAL",
            provider_id=None,
            model_id="reviewer-model",
            route_id=None,
            review_finding_dispositions={
                "f1": FindingDisposition.SUPPORTED,
                "f2": FindingDisposition.UNSUPPORTED,
            },
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        stats = bundle.reviewer["reviewer-model"]
        assert stats.findings_created == 1
        assert stats.supported == 1
        assert stats.unsupported == 1

    def test_false_negative_tracked(self, base_time: datetime.datetime) -> None:
        event = PerformanceEvent(
            event_id="e1",
            schema_version="1.0.0",
            timestamp=base_time,
            event_type=PerformanceEventType.REVIEWER_FALSE_NEGATIVE,
            project_id="omniforge",
            task_id="task-1",
            run_id="run-1",
            execution_role="review",
            task_class="feature",
            risk="R2_NORMAL",
            provider_id=None,
            model_id="reviewer-model",
            route_id=None,
            evidence_refs=("defect-evidence",),
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        assert bundle.reviewer["reviewer-model"].false_negatives == 1

    def test_context_strategy_tracked(self, base_time: datetime.datetime) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            context_strategy="hybrid",
            outcome_category=OutcomeCategory.SUCCESS,
            first_pass=True,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        assert bundle.context_strategy["hybrid"].attempts == 1
        assert bundle.context_strategy["hybrid"].first_pass_accepted == 1

    def test_risk_segmentation(self, base_time: datetime.datetime) -> None:
        r4 = _task_event(
            event_id="e1",
            timestamp=base_time,
            risk="R4_CRITICAL_AUTHORITY",
            outcome_category=OutcomeCategory.SUCCESS,
            first_pass=True,
        )
        r2 = _task_event(
            event_id="e2",
            timestamp=base_time,
            risk="R2_NORMAL",
            outcome_category=OutcomeCategory.SUCCESS,
            first_pass=True,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append_all((r4, r2)))
        assert bundle.risk["R4_CRITICAL_AUTHORITY"].attempts == 1
        assert bundle.risk["R2_NORMAL"].attempts == 1

    def test_language_framework_segmentation(self, base_time: datetime.datetime) -> None:
        py = _task_event(
            event_id="e1",
            timestamp=base_time,
            language_framework="python",
            outcome_category=OutcomeCategory.SUCCESS,
            first_pass=True,
        )
        ts = _task_event(
            event_id="e2",
            timestamp=base_time,
            language_framework="typescript",
            outcome_category=OutcomeCategory.SUCCESS,
            first_pass=True,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append_all((py, ts)))
        assert bundle.language_framework["python"].attempts == 1
        assert bundle.language_framework["typescript"].attempts == 1

    def test_project_statistics(self, base_time: datetime.datetime) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            direct_cost=Cost(0.5, "USD", CostState.ACTUAL),
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        assert bundle.project["omniforge"].total_cost_actual == 0.5

    def test_latency_and_wait_tracked(self, base_time: datetime.datetime) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            latency_seconds=1.5,
            provider_wait_seconds=3.0,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        stats = bundle.model_role[("gpt-4o", "coding")]
        assert stats.total_latency_seconds == 1.5
        assert stats.total_provider_wait_seconds == 3.0
        assert stats.average_latency_seconds() == 1.5

    def test_determinism_repeated_rebuild(self, base_time: datetime.datetime) -> None:
        events = tuple(
            _task_event(
                event_id=f"e{i}",
                timestamp=base_time + datetime.timedelta(seconds=i),
                model_id="gpt-4o" if i % 2 == 0 else "claude",
                provider_id="openai" if i % 2 == 0 else "anthropic",
                route_id="openai-direct" if i % 2 == 0 else "anthropic-direct",
                outcome_category=OutcomeCategory.SUCCESS
                if i % 3 == 0
                else OutcomeCategory.DETERMINISTIC_VALIDATION_FAILURE,
                acceptance_status=AcceptanceStatus.ACCEPTED
                if i % 3 == 0
                else AcceptanceStatus.REJECTED,
                first_pass=i % 3 == 0,
            )
            for i in range(20)
        )
        ledger = PerformanceLedger().append_all(events)
        builder = StatisticsBuilder()
        results = [builder.build(ledger).to_dict() for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_insertion_order_invariant_for_commutative_aggregates(
        self, base_time: datetime.datetime
    ) -> None:
        e1 = _task_event(event_id="e1", timestamp=base_time, model_id="gpt-4o")
        e2 = _task_event(event_id="e2", timestamp=base_time, model_id="claude")
        bundle_a = StatisticsBuilder().build(PerformanceLedger().append_all((e1, e2)))
        bundle_b = StatisticsBuilder().build(PerformanceLedger().append_all((e2, e1)))
        assert bundle_a.to_dict() == bundle_b.to_dict()
