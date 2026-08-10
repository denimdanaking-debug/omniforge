"""Persistence integration for Phase 11 performance intelligence."""

from __future__ import annotations

from typing import Any

from src.performance.builder import StatisticsBuilder
from src.performance.event import FindingDisposition
from src.performance.ledger import PerformanceLedger
from src.performance.statistics import PerformanceStatisticsBundle

CURRENT_PERFORMANCE_SCHEMA_VERSION = "1.0.0"


def performance_state_to_dict(
    ledger: PerformanceLedger,
    bundle: PerformanceStatisticsBundle | None = None,
) -> dict[str, Any]:
    """Serialize performance intelligence state for runtime-state persistence."""
    if bundle is None:
        bundle = StatisticsBuilder().build(ledger)
    return {
        "schema_version": CURRENT_PERFORMANCE_SCHEMA_VERSION,
        "ledger": ledger.to_dict(),
        "statistics": bundle.to_dict(),
    }


def performance_state_from_dict(
    data: dict[str, Any],
) -> tuple[PerformanceLedger, PerformanceStatisticsBundle]:
    """Deserialize performance intelligence state from runtime-state storage."""
    ledger_data = data.get("ledger", {"schema_version": "1.0.0", "events": []})
    ledger = PerformanceLedger.from_dict(ledger_data)
    statistics_data = data.get("statistics")
    if statistics_data is not None:
        bundle = _bundle_from_dict(statistics_data)
    else:
        bundle = StatisticsBuilder().build(ledger)
    return ledger, bundle


def _bundle_from_dict(data: dict[str, Any]) -> PerformanceStatisticsBundle:
    from src.performance.statistics import (
        ContextStrategyStatistics,
        LanguageFrameworkStatistics,
        ModelRoleStatistics,
        ProjectPerformanceStatistics,
        ReviewerStatistics,
        RiskDifficultyStatistics,
        RouteStatistics,
    )

    model_role: dict[tuple[str, str], ModelRoleStatistics] = {}
    for key, stats_data in (data.get("model_role") or {}).items():
        model_id, role = key.split(":", 1)
        model_role[(model_id, role)] = ModelRoleStatistics(
            model_id=model_id,
            role=role,
            attempts=int(stats_data.get("attempts", 0)),
            first_pass_accepted=int(stats_data.get("first_pass_accepted", 0)),
            accepted=int(stats_data.get("accepted", 0)),
            rejected=int(stats_data.get("rejected", 0)),
            invalid_plans=int(stats_data.get("invalid_plans", 0)),
            structured_output_invalid=int(stats_data.get("structured_output_invalid", 0)),
            deterministic_failures=int(stats_data.get("deterministic_failures", 0)),
            conceptual_failures=int(stats_data.get("conceptual_failures", 0)),
            authority_violations=int(stats_data.get("authority_violations", 0)),
            authority_compliant=int(stats_data.get("authority_compliant", 0)),
            repairs_attempted=int(stats_data.get("repairs_attempted", 0)),
            repairs_resolved=int(stats_data.get("repairs_resolved", 0)),
            cross_model_repairs=int(stats_data.get("cross_model_repairs", 0)),
            reviewer_findings={
                FindingDisposition(k): v
                for k, v in (stats_data.get("reviewer_findings") or {}).items()
            },
            total_latency_seconds=float(stats_data.get("total_latency_seconds", 0.0)),
            total_provider_wait_seconds=float(stats_data.get("total_provider_wait_seconds", 0.0)),
            total_input_tokens=int(stats_data.get("total_input_tokens", 0)),
            total_output_tokens=int(stats_data.get("total_output_tokens", 0)),
            total_cached_tokens=int(stats_data.get("total_cached_tokens", 0)),
            total_reasoning_tokens=int(stats_data.get("total_reasoning_tokens", 0)),
            actual_cost_sum=float(stats_data.get("actual_cost_sum", 0.0)),
            estimated_cost_sum=float(stats_data.get("estimated_cost_sum", 0.0)),
            unknown_cost_count=int(stats_data.get("unknown_cost_count", 0)),
        )

    route: dict[str, RouteStatistics] = {}
    for key, stats_data in (data.get("route") or {}).items():
        route[key] = RouteStatistics(
            route_id=key,
            provider_id=stats_data.get("provider_id", ""),
            attempts=int(stats_data.get("attempts", 0)),
            infrastructure_failures=int(stats_data.get("infrastructure_failures", 0)),
            quota_failures=int(stats_data.get("quota_failures", 0)),
            auth_failures=int(stats_data.get("auth_failures", 0)),
            route_failures=int(stats_data.get("route_failures", 0)),
            rate_limited_count=int(stats_data.get("rate_limited_count", 0)),
            total_latency_seconds=float(stats_data.get("total_latency_seconds", 0.0)),
            total_provider_wait_seconds=float(stats_data.get("total_provider_wait_seconds", 0.0)),
            request_count=int(stats_data.get("request_count", 0)),
            error_count=int(stats_data.get("error_count", 0)),
            actual_cost_sum=float(stats_data.get("actual_cost_sum", 0.0)),
            estimated_cost_sum=float(stats_data.get("estimated_cost_sum", 0.0)),
            unknown_cost_count=int(stats_data.get("unknown_cost_count", 0)),
        )

    reviewer: dict[str, ReviewerStatistics] = {}
    for key, stats_data in (data.get("reviewer") or {}).items():
        reviewer[key] = ReviewerStatistics(
            model_id=key,
            findings_created=int(stats_data.get("findings_created", 0)),
            supported=int(stats_data.get("supported", 0)),
            unsupported=int(stats_data.get("unsupported", 0)),
            stale=int(stats_data.get("stale", 0)),
            duplicate=int(stats_data.get("duplicate", 0)),
            mis_severity=int(stats_data.get("mis_severity", 0)),
            pending=int(stats_data.get("pending", 0)),
            false_negatives=int(stats_data.get("false_negatives", 0)),
        )

    context_strategy: dict[str, ContextStrategyStatistics] = {}
    for key, stats_data in (data.get("context_strategy") or {}).items():
        context_strategy[key] = ContextStrategyStatistics(
            strategy=key,
            attempts=int(stats_data.get("attempts", 0)),
            accepted=int(stats_data.get("accepted", 0)),
            first_pass_accepted=int(stats_data.get("first_pass_accepted", 0)),
            authority_violations=int(stats_data.get("authority_violations", 0)),
            context_capacity_exceeded=int(stats_data.get("context_capacity_exceeded", 0)),
        )

    risk: dict[str, RiskDifficultyStatistics] = {}
    for key, stats_data in (data.get("risk") or {}).items():
        risk[key] = RiskDifficultyStatistics(
            risk=key,
            attempts=int(stats_data.get("attempts", 0)),
            accepted=int(stats_data.get("accepted", 0)),
            first_pass_accepted=int(stats_data.get("first_pass_accepted", 0)),
            authority_violations=int(stats_data.get("authority_violations", 0)),
        )

    language_framework: dict[str, LanguageFrameworkStatistics] = {}
    for key, stats_data in (data.get("language_framework") or {}).items():
        language_framework[key] = LanguageFrameworkStatistics(
            language_framework=key,
            attempts=int(stats_data.get("attempts", 0)),
            accepted=int(stats_data.get("accepted", 0)),
            first_pass_accepted=int(stats_data.get("first_pass_accepted", 0)),
        )

    project: dict[str, ProjectPerformanceStatistics] = {}
    for key, stats_data in (data.get("project") or {}).items():
        project[key] = ProjectPerformanceStatistics(
            project_id=key,
            attempts=int(stats_data.get("attempts", 0)),
            accepted=int(stats_data.get("accepted", 0)),
            first_pass_accepted=int(stats_data.get("first_pass_accepted", 0)),
            deterministic_failures=int(stats_data.get("deterministic_failures", 0)),
            authority_violations=int(stats_data.get("authority_violations", 0)),
            total_cost_actual=float(stats_data.get("total_cost_actual", 0.0)),
            total_cost_estimated=float(stats_data.get("total_cost_estimated", 0.0)),
        )

    return PerformanceStatisticsBundle(
        model_role=model_role,
        route=route,
        reviewer=reviewer,
        context_strategy=context_strategy,
        risk=risk,
        language_framework=language_framework,
        project=project,
        total_events=int(data.get("total_events", 0)),
    )
