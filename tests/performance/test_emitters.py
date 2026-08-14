"""Tests for performance-event emitters and the output-success vs integration boundary."""

from __future__ import annotations

import datetime

import pytest

from src.orchestration.project_contract import AdvancementEvidence
from src.performance import (
    AcceptanceStatus,
    OutcomeCategory,
    PerformanceEventType,
    PerformanceLedger,
    StatisticsBuilder,
)
from src.performance.emitters import (
    emit_from_advancement_evidence,
    emit_from_task_outcome,
)
from src.telemetry.outcomes import OutcomeAttribution, OutcomeKind, TaskOutcome


@pytest.fixture
def base_time() -> datetime.datetime:
    return datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)


class TestOutputSuccessVsIntegrationAcceptance:
    def test_successful_coding_task_is_not_accepted_integration(
        self, base_time: datetime.datetime
    ) -> None:
        event = emit_from_task_outcome(
            outcome=TaskOutcome(OutcomeKind.SUCCESS, OutcomeAttribution.NONE),
            project_id="omniforge",
            task_id="task-1",
            run_id="run-1",
            timestamp=base_time,
            execution_role="coding",
            task_class="feature",
            risk="R2_NORMAL",
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
            first_pass=True,
        )
        assert event.event_type is PerformanceEventType.TASK_OUTCOME
        assert event.outcome_category is OutcomeCategory.SUCCESS
        assert event.acceptance_status is AcceptanceStatus.PENDING

        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        stats = bundle.model_role[("gpt-4o", "coding")]
        assert stats.attempts == 1
        assert stats.successful_outputs == 1
        assert stats.first_pass_accepted == 1
        assert stats.accepted == 0
        lifecycle = bundle.task_lifecycle[("omniforge", "task-1", "run-1")]
        assert lifecycle.accepted is False

    def test_successful_planning_task_is_not_accepted_integration(
        self, base_time: datetime.datetime
    ) -> None:
        event = emit_from_task_outcome(
            outcome=TaskOutcome(OutcomeKind.SUCCESS, OutcomeAttribution.NONE),
            project_id="omniforge",
            task_id="task-1",
            run_id="run-1",
            timestamp=base_time,
            execution_role="planning",
            task_class="feature",
            risk="R2_NORMAL",
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
            first_pass=True,
        )
        assert event.outcome_category is OutcomeCategory.SUCCESS
        assert event.acceptance_status is AcceptanceStatus.PENDING

        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        stats = bundle.model_role[("gpt-4o", "planning")]
        assert stats.attempts == 1
        assert stats.successful_outputs == 1
        assert stats.accepted == 0

    def test_local_deterministic_validation_pass_is_not_final_acceptance(
        self, base_time: datetime.datetime
    ) -> None:
        event = emit_from_task_outcome(
            outcome=TaskOutcome(OutcomeKind.SUCCESS, OutcomeAttribution.NONE),
            project_id="omniforge",
            task_id="task-1",
            run_id="run-1",
            timestamp=base_time,
            execution_role="coding",
            task_class="feature",
            risk="R2_NORMAL",
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
            first_pass=True,
        )
        assert event.acceptance_status is not AcceptanceStatus.ACCEPTED
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        assert bundle.task_lifecycle[("omniforge", "task-1", "run-1")].accepted is False

    def test_feature_pr_without_safe_integration_is_not_accepted(
        self, base_time: datetime.datetime
    ) -> None:
        evidence = AdvancementEvidence(
            implemented=True,
            deterministic_validation_passed=True,
            independent_review_satisfied=True,
            safely_integrated=False,
            planner_declared_complete=True,
        )
        event = emit_from_advancement_evidence(
            evidence=evidence,
            project_id="omniforge",
            task_id="task-1",
            run_id="run-1",
            timestamp=base_time,
            execution_role="coding",
            task_class="feature",
            risk="R2_NORMAL",
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        assert event.event_type is PerformanceEventType.INTEGRATION_REJECTED
        assert event.acceptance_status is AcceptanceStatus.REJECTED
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        assert bundle.task_lifecycle[("omniforge", "task-1", "run-1")].accepted is False

    def test_canonical_advancement_evidence_finalizes_accepted_integration(
        self, base_time: datetime.datetime
    ) -> None:
        evidence = AdvancementEvidence(
            implemented=True,
            deterministic_validation_passed=True,
            independent_review_satisfied=True,
            safely_integrated=True,
            planner_declared_complete=True,
        )
        event = emit_from_advancement_evidence(
            evidence=evidence,
            project_id="omniforge",
            task_id="task-1",
            run_id="run-1",
            timestamp=base_time,
            execution_role="coding",
            task_class="feature",
            risk="R2_NORMAL",
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        assert event.event_type is PerformanceEventType.INTEGRATION_ACCEPTED
        assert event.acceptance_status is AcceptanceStatus.ACCEPTED
        bundle = StatisticsBuilder().build(PerformanceLedger().append(event))
        lifecycle = bundle.task_lifecycle[("omniforge", "task-1", "run-1")]
        assert lifecycle.accepted is True

    def test_first_pass_quality_preserved_through_accepted_integration(
        self, base_time: datetime.datetime
    ) -> None:
        coding_success = emit_from_task_outcome(
            outcome=TaskOutcome(OutcomeKind.SUCCESS, OutcomeAttribution.NONE),
            project_id="omniforge",
            task_id="task-1",
            run_id="run-1",
            timestamp=base_time,
            execution_role="coding",
            task_class="feature",
            risk="R2_NORMAL",
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
            first_pass=True,
        )
        integration = emit_from_advancement_evidence(
            evidence=AdvancementEvidence(
                implemented=True,
                deterministic_validation_passed=True,
                independent_review_satisfied=True,
                safely_integrated=True,
                planner_declared_complete=True,
            ),
            project_id="omniforge",
            task_id="task-1",
            run_id="run-1",
            timestamp=base_time + datetime.timedelta(seconds=10),
            execution_role="coding",
            task_class="feature",
            risk="R2_NORMAL",
            provider_id="openai",
            model_id="gpt-4o",
            route_id="openai-direct",
        )
        ledger = PerformanceLedger().append_all((coding_success, integration))
        bundle = StatisticsBuilder().build(ledger)
        model_stats = bundle.model_role[("gpt-4o", "coding")]
        assert model_stats.attempts == 1
        assert model_stats.successful_outputs == 1
        assert model_stats.first_pass_accepted == 1
        lifecycle = bundle.task_lifecycle[("omniforge", "task-1", "run-1")]
        assert lifecycle.accepted is True
        assert lifecycle.accepted_time is not None
