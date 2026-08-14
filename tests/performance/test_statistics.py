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
    PerformanceAttribution,
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
    project_id: str = "omniforge",
    repair_metadata: Any | None = None,
    authority_adherence: AuthorityAdherenceStatus | None = None,
    direct_cost: Cost | None = None,
    latency_seconds: float | None = None,
    provider_wait_seconds: float | None = None,
    usage: Usage | None = None,
    attribution: str = PerformanceAttribution.MODEL_QUALITY.value,
) -> PerformanceEvent:
    return PerformanceEvent(
        event_id=event_id,
        schema_version="1.0.0",
        timestamp=timestamp,
        event_type=PerformanceEventType.TASK_OUTCOME,
        project_id=project_id,
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
        attribution=attribution,
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
            attribution=PerformanceAttribution.PROVIDER_ROUTE.value,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        stats = bundle.model_role[("gpt-4o", "coding")]
        assert stats.attempts == 0
        assert stats.calls == 1
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
            attribution=PerformanceAttribution.PROVIDER_ROUTE.value,
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
            attribution=PerformanceAttribution.REVIEW_QUALITY.value,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        key = ("reviewer-model", "review", "R2_NORMAL", "feature", "omniforge")
        stats = bundle.reviewer[key]
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
            attribution=PerformanceAttribution.REVIEW_QUALITY.value,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        key = ("reviewer-model", "review", "R2_NORMAL", "feature", "omniforge")
        assert bundle.reviewer[key].false_negatives == 1

    def test_context_strategy_tracked(self, base_time: datetime.datetime) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            context_strategy="hybrid",
            outcome_category=OutcomeCategory.SUCCESS,
            first_pass=True,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        key = ("hybrid", "gpt-4o", "coding", "R2_NORMAL", "feature", "omniforge")
        assert bundle.context_strategy[key].attempts == 1
        assert bundle.context_strategy[key].first_pass_accepted == 1

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
        py_key = ("python", "gpt-4o", "coding", "R2_NORMAL", "feature", "omniforge")
        ts_key = ("typescript", "gpt-4o", "coding", "R2_NORMAL", "feature", "omniforge")
        assert bundle.language_framework[py_key].attempts == 1
        assert bundle.language_framework[ts_key].attempts == 1

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


class TestAttributionGating:
    """Model-quality counters must be gated by canonical attribution."""

    def test_transient_transport_does_not_affect_model_quality(
        self, base_time: datetime.datetime
    ) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            outcome_category=OutcomeCategory.INFRASTRUCTURE_TRANSIENT,
            acceptance_status=AcceptanceStatus.REJECTED,
            first_pass=False,
            attribution=PerformanceAttribution.PROVIDER_ROUTE.value,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        stats = bundle.model_role[("gpt-4o", "coding")]
        assert stats.attempts == 0
        assert stats.accepted == 0
        assert stats.rejected == 0
        assert stats.calls == 1

    def test_quota_exhausted_does_not_affect_model_quality(
        self, base_time: datetime.datetime
    ) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            outcome_category=OutcomeCategory.QUOTA_EXHAUSTED,
            acceptance_status=AcceptanceStatus.REJECTED,
            first_pass=False,
            attribution=PerformanceAttribution.PROVIDER_ROUTE.value,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        stats = bundle.model_role[("gpt-4o", "coding")]
        assert stats.attempts == 0
        assert stats.calls == 1

    def test_auth_failure_does_not_affect_model_quality(self, base_time: datetime.datetime) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            outcome_category=OutcomeCategory.AUTH_FAILURE,
            acceptance_status=AcceptanceStatus.REJECTED,
            first_pass=False,
            attribution=PerformanceAttribution.PROVIDER_ROUTE.value,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        stats = bundle.model_role[("gpt-4o", "coding")]
        assert stats.attempts == 0

    def test_route_failure_does_not_affect_model_quality(
        self, base_time: datetime.datetime
    ) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            outcome_category=OutcomeCategory.ROUTE_FAILURE,
            acceptance_status=AcceptanceStatus.REJECTED,
            first_pass=False,
            attribution=PerformanceAttribution.PROVIDER_ROUTE.value,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        stats = bundle.model_role[("gpt-4o", "coding")]
        assert stats.attempts == 0

    def test_invalid_structured_output_affects_model_quality(
        self, base_time: datetime.datetime
    ) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            outcome_category=OutcomeCategory.STRUCTURED_OUTPUT_INVALID,
            acceptance_status=AcceptanceStatus.REJECTED,
            first_pass=False,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        assert bundle.model_role[("gpt-4o", "coding")].structured_output_invalid == 1

    def test_invalid_plan_affects_planner_quality(self, base_time: datetime.datetime) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            role="planning",
            outcome_category=OutcomeCategory.PLAN_INVALID,
            acceptance_status=AcceptanceStatus.REJECTED,
            first_pass=False,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        assert bundle.model_role[("gpt-4o", "planning")].invalid_plans == 1

    def test_deterministic_implementation_affects_model_quality(
        self, base_time: datetime.datetime
    ) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            outcome_category=OutcomeCategory.DETERMINISTIC_VALIDATION_FAILURE,
            acceptance_status=AcceptanceStatus.REJECTED,
            first_pass=False,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        assert bundle.model_role[("gpt-4o", "coding")].deterministic_failures == 1

    def test_same_model_two_routes_model_quality_unchanged_on_gateway_failure(
        self, base_time: datetime.datetime
    ) -> None:
        direct = _task_event(
            event_id="e1",
            timestamp=base_time,
            model_id="gpt-4o",
            provider_id="openai",
            route_id="openai-direct",
            outcome_category=OutcomeCategory.SUCCESS,
            first_pass=True,
        )
        gateway = _task_event(
            event_id="e2",
            timestamp=base_time,
            model_id="gpt-4o",
            provider_id="openrouter",
            route_id="openrouter-openai",
            outcome_category=OutcomeCategory.ROUTE_FAILURE,
            acceptance_status=AcceptanceStatus.REJECTED,
            first_pass=False,
            attribution=PerformanceAttribution.PROVIDER_ROUTE.value,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append_all((direct, gateway)))
        model_stats = bundle.model_role[("gpt-4o", "coding")]
        assert model_stats.attempts == 1
        assert model_stats.first_pass_accepted == 1
        assert bundle.route["openrouter-openai"].error_count == 1


class TestDimensionalStatistics:
    """Joint dimensions must remain independently queryable."""

    def test_reviewer_dimensions_separate_by_risk_and_project(
        self, base_time: datetime.datetime
    ) -> None:
        from src.performance.emitters import emit_reviewer_finding_event

        e1 = emit_reviewer_finding_event(
            finding_id="f1",
            disposition=FindingDisposition.SUPPORTED,
            reviewer_model_id="reviewer-x",
            project_id="project-a",
            task_id="task-1",
            run_id="run-1",
            timestamp=base_time,
            risk="R1_LOW",
            task_class="feature",
        )
        e2 = emit_reviewer_finding_event(
            finding_id="f2",
            disposition=FindingDisposition.UNSUPPORTED,
            reviewer_model_id="reviewer-x",
            project_id="project-b",
            task_id="task-2",
            run_id="run-1",
            timestamp=base_time,
            risk="R4_CRITICAL_AUTHORITY",
            task_class="feature",
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append_all((e1, e2)))
        key_a = ("reviewer-x", "review", "R1_LOW", "feature", "project-a")
        key_b = ("reviewer-x", "review", "R4_CRITICAL_AUTHORITY", "feature", "project-b")
        assert bundle.reviewer[key_a].supported == 1
        assert bundle.reviewer[key_b].unsupported == 1

    def test_context_strategy_dimensions_separate_by_model_and_role(
        self, base_time: datetime.datetime
    ) -> None:
        coding = _task_event(
            event_id="e1",
            timestamp=base_time,
            model_id="model-a",
            role="coding",
            context_strategy="hybrid",
            outcome_category=OutcomeCategory.SUCCESS,
            first_pass=True,
        )
        review = _task_event(
            event_id="e2",
            timestamp=base_time,
            model_id="model-b",
            role="review",
            context_strategy="hybrid",
            outcome_category=OutcomeCategory.SUCCESS,
            first_pass=True,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append_all((coding, review)))
        key_a = ("hybrid", "model-a", "coding", "R2_NORMAL", "feature", "omniforge")
        key_b = ("hybrid", "model-b", "review", "R2_NORMAL", "feature", "omniforge")
        assert bundle.context_strategy[key_a].attempts == 1
        assert bundle.context_strategy[key_b].attempts == 1

    def test_language_framework_dimensions_separate_by_project(
        self, base_time: datetime.datetime
    ) -> None:
        py = _task_event(
            event_id="e1",
            timestamp=base_time,
            project_id="project-a",
            language_framework="python",
            outcome_category=OutcomeCategory.SUCCESS,
            first_pass=True,
        )
        cs = _task_event(
            event_id="e2",
            timestamp=base_time,
            project_id="project-b",
            language_framework="csharp",
            outcome_category=OutcomeCategory.SUCCESS,
            first_pass=True,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append_all((py, cs)))
        py_key = ("python", "gpt-4o", "coding", "R2_NORMAL", "feature", "project-a")
        cs_key = ("csharp", "gpt-4o", "coding", "R2_NORMAL", "feature", "project-b")
        assert bundle.language_framework[py_key].attempts == 1
        assert bundle.language_framework[cs_key].attempts == 1


class TestTaskLifecycleStatistics:
    """Task-level cost and time aggregates are derived from the ledger."""

    def test_lifecycle_cost_aggregated_by_role(self, base_time: datetime.datetime) -> None:
        planner = _task_event(
            event_id="e1",
            timestamp=base_time,
            role="planning",
            direct_cost=Cost(1.0, "USD", CostState.ACTUAL),
        )
        coder = _task_event(
            event_id="e2",
            timestamp=base_time + datetime.timedelta(seconds=1),
            role="coding",
            direct_cost=Cost(4.0, "USD", CostState.ACTUAL),
        )
        reviewer = _task_event(
            event_id="e3",
            timestamp=base_time + datetime.timedelta(seconds=2),
            role="review",
            direct_cost=Cost(2.0, "USD", CostState.ACTUAL),
        )
        repair = _task_event(
            event_id="e4",
            timestamp=base_time + datetime.timedelta(seconds=3),
            role="repair",
            direct_cost=Cost(3.0, "USD", CostState.ACTUAL),
        )
        accepted = PerformanceEvent(
            event_id="e5",
            schema_version="1.0.0",
            timestamp=base_time + datetime.timedelta(seconds=4),
            event_type=PerformanceEventType.INTEGRATION_ACCEPTED,
            project_id="omniforge",
            task_id="task-1",
            run_id="run-1",
            execution_role="coding",
            task_class="feature",
            risk="R2_NORMAL",
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
            outcome_category=OutcomeCategory.SUCCESS,
            acceptance_status=AcceptanceStatus.ACCEPTED,
            first_pass=False,
            direct_cost=Cost(0.0, "USD", CostState.ACTUAL),
            attribution=PerformanceAttribution.TASK_PROCESS.value,
        )
        ledger = PerformanceLedger().append_all((planner, coder, reviewer, repair, accepted))
        bundle = StatisticsBuilder().build(ledger)
        lifecycle = bundle.task_lifecycle[("omniforge", "task-1", "run-1")]
        assert lifecycle.total_cost_actual == 10.0
        assert lifecycle.planning_cost_actual == 1.0
        assert lifecycle.implementation_cost_actual == 4.0
        assert lifecycle.review_cost_actual == 2.0
        assert lifecycle.repair_cost_actual == 3.0
        assert lifecycle.accepted is True

    def test_unknown_cost_preserved(self, base_time: datetime.datetime) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            direct_cost=Cost(None, "USD", CostState.UNKNOWN),
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        lifecycle = bundle.task_lifecycle[("omniforge", "task-1", "run-1")]
        assert lifecycle.unknown_cost_count == 1
        assert lifecycle.total_cost_actual == 0.0

    def test_abandoned_task_retains_cost(self, base_time: datetime.datetime) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            direct_cost=Cost(2.5, "USD", CostState.ACTUAL),
        )
        rejected = PerformanceEvent(
            event_id="e2",
            schema_version="1.0.0",
            timestamp=base_time + datetime.timedelta(seconds=1),
            event_type=PerformanceEventType.INTEGRATION_REJECTED,
            project_id="omniforge",
            task_id="task-1",
            run_id="run-1",
            execution_role="coding",
            task_class="feature",
            risk="R2_NORMAL",
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
            acceptance_status=AcceptanceStatus.ABANDONED,
            direct_cost=Cost(0.0, "USD", CostState.ACTUAL),
            attribution=PerformanceAttribution.TASK_PROCESS.value,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append_all((event, rejected)))
        lifecycle = bundle.task_lifecycle[("omniforge", "task-1", "run-1")]
        assert lifecycle.total_cost_actual == 2.5
        assert lifecycle.accepted is False
        assert lifecycle.abandoned is True
        assert lifecycle.time_to_accepted_seconds() is None

    def test_time_to_accepted_integration(self, base_time: datetime.datetime) -> None:
        planner = _task_event(
            event_id="e1",
            timestamp=base_time,
            role="planning",
        )
        accepted = PerformanceEvent(
            event_id="e2",
            schema_version="1.0.0",
            timestamp=base_time + datetime.timedelta(seconds=30),
            event_type=PerformanceEventType.INTEGRATION_ACCEPTED,
            project_id="omniforge",
            task_id="task-1",
            run_id="run-1",
            execution_role="coding",
            task_class="feature",
            risk="R2_NORMAL",
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
            outcome_category=OutcomeCategory.SUCCESS,
            acceptance_status=AcceptanceStatus.ACCEPTED,
            first_pass=True,
            direct_cost=Cost(0.0, "USD", CostState.ACTUAL),
            attribution=PerformanceAttribution.TASK_PROCESS.value,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append_all((planner, accepted)))
        lifecycle = bundle.task_lifecycle[("omniforge", "task-1", "run-1")]
        assert lifecycle.time_to_accepted_seconds() == 30.0

    def test_provider_wait_tracked_separately(self, base_time: datetime.datetime) -> None:
        event = _task_event(
            event_id="e1",
            timestamp=base_time,
            provider_wait_seconds=600.0,
        )
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        lifecycle = bundle.task_lifecycle[("omniforge", "task-1", "run-1")]
        assert lifecycle.provider_wait_seconds == 600.0
