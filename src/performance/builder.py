"""Deterministic statistics builder that rebuilds aggregates from a ledger."""

from __future__ import annotations

from dataclasses import replace

from src.performance.event import (
    AcceptanceStatus,
    AuthorityAdherenceStatus,
    Cost,
    FindingDisposition,
    OutcomeCategory,
    PerformanceEvent,
    PerformanceEventType,
)
from src.performance.ledger import PerformanceLedger
from src.performance.statistics import (
    ContextStrategyStatistics,
    LanguageFrameworkStatistics,
    ModelRoleStatistics,
    PerformanceStatisticsBundle,
    ProjectPerformanceStatistics,
    ReviewerStatistics,
    RiskDifficultyStatistics,
    RouteStatistics,
)


class StatisticsBuilder:
    """Deterministically rebuild PerformanceStatisticsBundle from a ledger."""

    def build(self, ledger: PerformanceLedger) -> PerformanceStatisticsBundle:
        bundle = PerformanceStatisticsBundle(total_events=len(ledger.events))
        for event in ledger.events:
            bundle = self._apply_event(bundle, event)
        return bundle

    def _apply_event(
        self, bundle: PerformanceStatisticsBundle, event: PerformanceEvent
    ) -> PerformanceStatisticsBundle:
        bundle = self._update_model_role(bundle, event)
        bundle = self._update_route(bundle, event)
        bundle = self._update_reviewer(bundle, event)
        bundle = self._update_reviewer_false_negative(bundle, event)
        bundle = self._update_context_strategy(bundle, event)
        bundle = self._update_risk(bundle, event)
        bundle = self._update_language_framework(bundle, event)
        bundle = self._update_project(bundle, event)
        return bundle

    def _update_model_role(
        self, bundle: PerformanceStatisticsBundle, event: PerformanceEvent
    ) -> PerformanceStatisticsBundle:
        if not event.model_id or not event.execution_role:
            return bundle
        key = (event.model_id, event.execution_role)
        stats = bundle.model_role.get(key) or ModelRoleStatistics(
            model_id=event.model_id, role=event.execution_role
        )

        attempts = (
            1
            if event.event_type
            in {
                PerformanceEventType.TASK_OUTCOME,
                PerformanceEventType.RECOVERY_DECISION,
                PerformanceEventType.INTEGRATION_ACCEPTED,
                PerformanceEventType.INTEGRATION_REJECTED,
            }
            else 0
        )

        if attempts:
            stats = replace(stats, attempts=stats.attempts + 1)
            if event.acceptance_status is AcceptanceStatus.ACCEPTED:
                stats = replace(stats, accepted=stats.accepted + 1)
            elif event.acceptance_status is AcceptanceStatus.REJECTED:
                stats = replace(stats, rejected=stats.rejected + 1)
            if event.first_pass:
                stats = replace(stats, first_pass_accepted=stats.first_pass_accepted + 1)
            if event.outcome_category is OutcomeCategory.PLAN_INVALID:
                stats = replace(stats, invalid_plans=stats.invalid_plans + 1)
            if event.outcome_category is OutcomeCategory.STRUCTURED_OUTPUT_INVALID:
                stats = replace(
                    stats, structured_output_invalid=stats.structured_output_invalid + 1
                )
            if event.outcome_category is OutcomeCategory.DETERMINISTIC_VALIDATION_FAILURE:
                stats = replace(stats, deterministic_failures=stats.deterministic_failures + 1)
            if event.outcome_category is OutcomeCategory.CONCEPTUAL_FAILURE:
                stats = replace(stats, conceptual_failures=stats.conceptual_failures + 1)
            if event.authority_adherence is AuthorityAdherenceStatus.COMPLIANT:
                stats = replace(stats, authority_compliant=stats.authority_compliant + 1)
            elif event.authority_adherence is not None:
                stats = replace(stats, authority_violations=stats.authority_violations + 1)
            if event.repair_metadata is not None:
                stats = replace(stats, repairs_attempted=stats.repairs_attempted + 1)
                if event.repair_metadata.resolved:
                    stats = replace(stats, repairs_resolved=stats.repairs_resolved + 1)
                if event.repair_metadata.required_cross_model_escalation:
                    stats = replace(stats, cross_model_repairs=stats.cross_model_repairs + 1)
            if event.latency_seconds is not None:
                stats = replace(
                    stats, total_latency_seconds=stats.total_latency_seconds + event.latency_seconds
                )
            if event.provider_wait_seconds is not None:
                stats = replace(
                    stats,
                    total_provider_wait_seconds=stats.total_provider_wait_seconds
                    + event.provider_wait_seconds,
                )
            if event.usage.input_tokens is not None:
                stats = replace(
                    stats, total_input_tokens=stats.total_input_tokens + event.usage.input_tokens
                )
            if event.usage.output_tokens is not None:
                stats = replace(
                    stats, total_output_tokens=stats.total_output_tokens + event.usage.output_tokens
                )
            if event.usage.cached_tokens is not None:
                stats = replace(
                    stats, total_cached_tokens=stats.total_cached_tokens + event.usage.cached_tokens
                )
            if event.usage.reasoning_tokens is not None:
                stats = replace(
                    stats,
                    total_reasoning_tokens=stats.total_reasoning_tokens
                    + event.usage.reasoning_tokens,
                )
            stats = self._accumulate_cost(stats, event.direct_cost)

        if event.event_type is PerformanceEventType.REVIEWER_FINDING and event.model_id:
            finding_stats = dict(stats.reviewer_findings)
            for disposition in event.review_finding_dispositions.values():
                finding_stats[disposition] = finding_stats.get(disposition, 0) + 1
            stats = replace(stats, reviewer_findings=finding_stats)

        model_role = dict(bundle.model_role)
        model_role[key] = stats
        return replace(bundle, model_role=model_role)

    def _update_route(
        self, bundle: PerformanceStatisticsBundle, event: PerformanceEvent
    ) -> PerformanceStatisticsBundle:
        if not event.route_id:
            return bundle
        key = event.route_id
        stats = bundle.route.get(key) or RouteStatistics(
            route_id=event.route_id, provider_id=event.provider_id or ""
        )

        if event.event_type in {
            PerformanceEventType.TASK_OUTCOME,
            PerformanceEventType.RECOVERY_DECISION,
        }:
            stats = replace(stats, attempts=stats.attempts + 1)
            stats = replace(stats, request_count=stats.request_count + 1)
            if event.outcome_category in {
                OutcomeCategory.INFRASTRUCTURE_TRANSIENT,
                OutcomeCategory.ROUTE_FAILURE,
            }:
                stats = replace(stats, infrastructure_failures=stats.infrastructure_failures + 1)
                stats = replace(stats, error_count=stats.error_count + 1)
            if event.outcome_category is OutcomeCategory.QUOTA_EXHAUSTED:
                stats = replace(stats, quota_failures=stats.quota_failures + 1)
                stats = replace(stats, error_count=stats.error_count + 1)
            if event.outcome_category is OutcomeCategory.AUTH_FAILURE:
                stats = replace(stats, auth_failures=stats.auth_failures + 1)
                stats = replace(stats, error_count=stats.error_count + 1)
            if event.latency_seconds is not None:
                stats = replace(
                    stats, total_latency_seconds=stats.total_latency_seconds + event.latency_seconds
                )
            if event.provider_wait_seconds is not None:
                stats = replace(
                    stats,
                    total_provider_wait_seconds=stats.total_provider_wait_seconds
                    + event.provider_wait_seconds,
                )
            stats = self._accumulate_route_cost(stats, event.direct_cost)

        route = dict(bundle.route)
        route[key] = stats
        return replace(bundle, route=route)

    def _update_reviewer(
        self, bundle: PerformanceStatisticsBundle, event: PerformanceEvent
    ) -> PerformanceStatisticsBundle:
        if event.event_type is not PerformanceEventType.REVIEWER_FINDING or not event.model_id:
            return bundle
        key = event.model_id
        stats = bundle.reviewer.get(key) or ReviewerStatistics(model_id=event.model_id)
        stats = replace(stats, findings_created=stats.findings_created + 1)
        for disposition in event.review_finding_dispositions.values():
            if disposition is FindingDisposition.SUPPORTED:
                stats = replace(stats, supported=stats.supported + 1)
            elif disposition is FindingDisposition.UNSUPPORTED:
                stats = replace(stats, unsupported=stats.unsupported + 1)
            elif disposition is FindingDisposition.STALE:
                stats = replace(stats, stale=stats.stale + 1)
            elif disposition is FindingDisposition.DUPLICATE:
                stats = replace(stats, duplicate=stats.duplicate + 1)
            elif disposition is FindingDisposition.MIS_SEVERITY:
                stats = replace(stats, mis_severity=stats.mis_severity + 1)
            elif disposition is FindingDisposition.PENDING:
                stats = replace(stats, pending=stats.pending + 1)
        reviewer = dict(bundle.reviewer)
        reviewer[key] = stats
        return replace(bundle, reviewer=reviewer)

    def _update_reviewer_false_negative(
        self, bundle: PerformanceStatisticsBundle, event: PerformanceEvent
    ) -> PerformanceStatisticsBundle:
        if (
            event.event_type is not PerformanceEventType.REVIEWER_FALSE_NEGATIVE
            or not event.model_id
        ):
            return bundle
        key = event.model_id
        stats = bundle.reviewer.get(key) or ReviewerStatistics(model_id=event.model_id)
        stats = replace(stats, false_negatives=stats.false_negatives + 1)
        reviewer = dict(bundle.reviewer)
        reviewer[key] = stats
        return replace(bundle, reviewer=reviewer)

    def _update_context_strategy(
        self, bundle: PerformanceStatisticsBundle, event: PerformanceEvent
    ) -> PerformanceStatisticsBundle:
        if not event.context_strategy:
            return bundle
        key = event.context_strategy
        stats = bundle.context_strategy.get(key) or ContextStrategyStatistics(strategy=key)
        stats = replace(stats, attempts=stats.attempts + 1)
        if event.acceptance_status is AcceptanceStatus.ACCEPTED:
            stats = replace(stats, accepted=stats.accepted + 1)
        if event.first_pass:
            stats = replace(stats, first_pass_accepted=stats.first_pass_accepted + 1)
        if (
            event.authority_adherence is not None
            and event.authority_adherence is not AuthorityAdherenceStatus.COMPLIANT
        ):
            stats = replace(stats, authority_violations=stats.authority_violations + 1)
        if event.outcome_category is OutcomeCategory.CONTEXT_CAPACITY:
            stats = replace(stats, context_capacity_exceeded=stats.context_capacity_exceeded + 1)
        context_strategy = dict(bundle.context_strategy)
        context_strategy[key] = stats
        return replace(bundle, context_strategy=context_strategy)

    def _update_risk(
        self, bundle: PerformanceStatisticsBundle, event: PerformanceEvent
    ) -> PerformanceStatisticsBundle:
        if not event.risk:
            return bundle
        key = event.risk
        stats = bundle.risk.get(key) or RiskDifficultyStatistics(risk=key)
        stats = replace(stats, attempts=stats.attempts + 1)
        if event.acceptance_status is AcceptanceStatus.ACCEPTED:
            stats = replace(stats, accepted=stats.accepted + 1)
        if event.first_pass:
            stats = replace(stats, first_pass_accepted=stats.first_pass_accepted + 1)
        if (
            event.authority_adherence is not None
            and event.authority_adherence is not AuthorityAdherenceStatus.COMPLIANT
        ):
            stats = replace(stats, authority_violations=stats.authority_violations + 1)
        risk = dict(bundle.risk)
        risk[key] = stats
        return replace(bundle, risk=risk)

    def _update_language_framework(
        self, bundle: PerformanceStatisticsBundle, event: PerformanceEvent
    ) -> PerformanceStatisticsBundle:
        if not event.language_framework:
            return bundle
        key = event.language_framework
        stats = bundle.language_framework.get(key) or LanguageFrameworkStatistics(
            language_framework=key
        )
        stats = replace(stats, attempts=stats.attempts + 1)
        if event.acceptance_status is AcceptanceStatus.ACCEPTED:
            stats = replace(stats, accepted=stats.accepted + 1)
        if event.first_pass:
            stats = replace(stats, first_pass_accepted=stats.first_pass_accepted + 1)
        language_framework = dict(bundle.language_framework)
        language_framework[key] = stats
        return replace(bundle, language_framework=language_framework)

    def _update_project(
        self, bundle: PerformanceStatisticsBundle, event: PerformanceEvent
    ) -> PerformanceStatisticsBundle:
        if not event.project_id:
            return bundle
        key = event.project_id
        stats = bundle.project.get(key) or ProjectPerformanceStatistics(project_id=key)
        stats = replace(stats, attempts=stats.attempts + 1)
        if event.acceptance_status is AcceptanceStatus.ACCEPTED:
            stats = replace(stats, accepted=stats.accepted + 1)
        if event.first_pass:
            stats = replace(stats, first_pass_accepted=stats.first_pass_accepted + 1)
        if event.outcome_category is OutcomeCategory.DETERMINISTIC_VALIDATION_FAILURE:
            stats = replace(stats, deterministic_failures=stats.deterministic_failures + 1)
        if (
            event.authority_adherence is not None
            and event.authority_adherence is not AuthorityAdherenceStatus.COMPLIANT
        ):
            stats = replace(stats, authority_violations=stats.authority_violations + 1)
        if event.direct_cost.state.value == "actual" and event.direct_cost.amount is not None:
            stats = replace(
                stats, total_cost_actual=stats.total_cost_actual + event.direct_cost.amount
            )
        elif event.direct_cost.state.value == "estimated" and event.direct_cost.amount is not None:
            stats = replace(
                stats, total_cost_estimated=stats.total_cost_estimated + event.direct_cost.amount
            )
        project = dict(bundle.project)
        project[key] = stats
        return replace(bundle, project=project)

    def _accumulate_cost(self, stats: ModelRoleStatistics, cost: Cost) -> ModelRoleStatistics:
        if cost.state.value == "actual" and cost.amount is not None:
            return replace(stats, actual_cost_sum=stats.actual_cost_sum + cost.amount)
        if cost.state.value == "estimated" and cost.amount is not None:
            return replace(stats, estimated_cost_sum=stats.estimated_cost_sum + cost.amount)
        if cost.state.value == "unknown":
            return replace(stats, unknown_cost_count=stats.unknown_cost_count + 1)
        return stats

    def _accumulate_route_cost(self, stats: RouteStatistics, cost: Cost) -> RouteStatistics:
        if cost.state.value == "actual" and cost.amount is not None:
            return replace(stats, actual_cost_sum=stats.actual_cost_sum + cost.amount)
        if cost.state.value == "estimated" and cost.amount is not None:
            return replace(stats, estimated_cost_sum=stats.estimated_cost_sum + cost.amount)
        if cost.state.value == "unknown":
            return replace(stats, unknown_cost_count=stats.unknown_cost_count + 1)
        return stats
