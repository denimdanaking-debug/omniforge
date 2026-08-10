"""Derived performance statistics rebuilt from the immutable event ledger."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

from src.performance.event import FindingDisposition


def safe_rate(numerator: float, denominator: float) -> float | None:
    """Return numerator/denominator or None when there is no data."""
    if denominator <= 0:
        return None
    return numerator / denominator


@dataclass(frozen=True)
class ModelRoleStatistics:
    """Marginal empirical statistics for a (model_id, execution_role) pair.

    Quality counters (attempts, accepted, rejected, first_pass_accepted, failure
    counters, authority, repairs) are updated only for events attributed to
    model quality.  Operational resource counters (calls, latency, tokens, cost)
    are updated for any actual model call so infrastructure failures do not
    contaminate quality denominators.
    """

    model_id: str
    role: str
    # Quality counters (MODEL_QUALITY attribution only).
    attempts: int = 0
    first_pass_accepted: int = 0
    accepted: int = 0
    rejected: int = 0
    invalid_plans: int = 0
    structured_output_invalid: int = 0
    deterministic_failures: int = 0
    conceptual_failures: int = 0
    authority_violations: int = 0
    authority_compliant: int = 0
    repairs_attempted: int = 0
    repairs_resolved: int = 0
    cross_model_repairs: int = 0
    reviewer_findings: dict[FindingDisposition, int] = field(default_factory=dict)
    # Operational resource counters (any model call).
    calls: int = 0
    total_latency_seconds: float = 0.0
    total_provider_wait_seconds: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_reasoning_tokens: int = 0
    actual_cost_sum: float = 0.0
    estimated_cost_sum: float = 0.0
    unknown_cost_count: int = 0

    def first_pass_rate(self) -> float | None:
        return safe_rate(self.first_pass_accepted, self.attempts)

    def acceptance_rate(self) -> float | None:
        return safe_rate(self.accepted, self.attempts)

    def repair_success_rate(self) -> float | None:
        return safe_rate(self.repairs_resolved, self.repairs_attempted)

    def average_latency_seconds(self) -> float | None:
        return safe_rate(self.total_latency_seconds, self.calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "role": self.role,
            "attempts": self.attempts,
            "first_pass_accepted": self.first_pass_accepted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "invalid_plans": self.invalid_plans,
            "structured_output_invalid": self.structured_output_invalid,
            "deterministic_failures": self.deterministic_failures,
            "conceptual_failures": self.conceptual_failures,
            "authority_violations": self.authority_violations,
            "authority_compliant": self.authority_compliant,
            "repairs_attempted": self.repairs_attempted,
            "repairs_resolved": self.repairs_resolved,
            "cross_model_repairs": self.cross_model_repairs,
            "reviewer_findings": {k.value: v for k, v in sorted(self.reviewer_findings.items())},
            "calls": self.calls,
            "total_latency_seconds": self.total_latency_seconds,
            "total_provider_wait_seconds": self.total_provider_wait_seconds,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "actual_cost_sum": self.actual_cost_sum,
            "estimated_cost_sum": self.estimated_cost_sum,
            "unknown_cost_count": self.unknown_cost_count,
            "first_pass_rate": self.first_pass_rate(),
            "acceptance_rate": self.acceptance_rate(),
            "repair_success_rate": self.repair_success_rate(),
            "average_latency_seconds": self.average_latency_seconds(),
        }


@dataclass(frozen=True)
class ModelRoleDimensionalStatistics:
    """Joint-dimensional model-quality statistics.

    Keyed by (model_id, role, risk, task_class, project_id, language_framework).
    """

    model_id: str
    role: str
    risk: str
    task_class: str
    project_id: str
    language_framework: str
    attempts: int = 0
    first_pass_accepted: int = 0
    accepted: int = 0
    rejected: int = 0
    invalid_plans: int = 0
    structured_output_invalid: int = 0
    deterministic_failures: int = 0
    conceptual_failures: int = 0
    authority_violations: int = 0
    authority_compliant: int = 0
    repairs_attempted: int = 0
    repairs_resolved: int = 0
    cross_model_repairs: int = 0

    def first_pass_rate(self) -> float | None:
        return safe_rate(self.first_pass_accepted, self.attempts)

    def acceptance_rate(self) -> float | None:
        return safe_rate(self.accepted, self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "role": self.role,
            "risk": self.risk,
            "task_class": self.task_class,
            "project_id": self.project_id,
            "language_framework": self.language_framework,
            "attempts": self.attempts,
            "first_pass_accepted": self.first_pass_accepted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "invalid_plans": self.invalid_plans,
            "structured_output_invalid": self.structured_output_invalid,
            "deterministic_failures": self.deterministic_failures,
            "conceptual_failures": self.conceptual_failures,
            "authority_violations": self.authority_violations,
            "authority_compliant": self.authority_compliant,
            "repairs_attempted": self.repairs_attempted,
            "repairs_resolved": self.repairs_resolved,
            "cross_model_repairs": self.cross_model_repairs,
            "first_pass_rate": self.first_pass_rate(),
            "acceptance_rate": self.acceptance_rate(),
        }


@dataclass(frozen=True)
class RouteStatistics:
    """Empirical statistics for a route/provider operational dimension."""

    route_id: str
    provider_id: str
    attempts: int = 0
    infrastructure_failures: int = 0
    quota_failures: int = 0
    auth_failures: int = 0
    route_failures: int = 0
    rate_limited_count: int = 0
    total_latency_seconds: float = 0.0
    total_provider_wait_seconds: float = 0.0
    request_count: int = 0
    error_count: int = 0
    actual_cost_sum: float = 0.0
    estimated_cost_sum: float = 0.0
    unknown_cost_count: int = 0

    def error_rate(self) -> float | None:
        return safe_rate(self.error_count, self.request_count)

    def average_latency_seconds(self) -> float | None:
        return safe_rate(self.total_latency_seconds, self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "provider_id": self.provider_id,
            "attempts": self.attempts,
            "infrastructure_failures": self.infrastructure_failures,
            "quota_failures": self.quota_failures,
            "auth_failures": self.auth_failures,
            "route_failures": self.route_failures,
            "rate_limited_count": self.rate_limited_count,
            "total_latency_seconds": self.total_latency_seconds,
            "total_provider_wait_seconds": self.total_provider_wait_seconds,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "actual_cost_sum": self.actual_cost_sum,
            "estimated_cost_sum": self.estimated_cost_sum,
            "unknown_cost_count": self.unknown_cost_count,
            "error_rate": self.error_rate(),
            "average_latency_seconds": self.average_latency_seconds(),
        }


@dataclass(frozen=True)
class ReviewerStatistics:
    """Empirical statistics for a reviewer model identity at full dimensionality.

    Keyed by (model_id, execution_role, risk, task_class, project_id).
    """

    model_id: str
    role: str
    risk: str
    task_class: str
    project_id: str
    findings_created: int = 0
    supported: int = 0
    unsupported: int = 0
    stale: int = 0
    duplicate: int = 0
    mis_severity: int = 0
    pending: int = 0
    false_negatives: int = 0

    def precision(self) -> float | None:
        denom = self.supported + self.unsupported + self.stale + self.duplicate + self.mis_severity
        return safe_rate(self.supported, denom)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "role": self.role,
            "risk": self.risk,
            "task_class": self.task_class,
            "project_id": self.project_id,
            "findings_created": self.findings_created,
            "supported": self.supported,
            "unsupported": self.unsupported,
            "stale": self.stale,
            "duplicate": self.duplicate,
            "mis_severity": self.mis_severity,
            "pending": self.pending,
            "false_negatives": self.false_negatives,
            "precision": self.precision(),
        }


@dataclass(frozen=True)
class ContextStrategyStatistics:
    """Empirical statistics for a Phase 7 context strategy at full dimensionality.

    Keyed by (strategy, model_id, role, risk, task_class, project_id).
    """

    strategy: str
    model_id: str
    role: str
    risk: str
    task_class: str
    project_id: str
    attempts: int = 0
    accepted: int = 0
    first_pass_accepted: int = 0
    authority_violations: int = 0
    context_capacity_exceeded: int = 0

    def acceptance_rate(self) -> float | None:
        return safe_rate(self.accepted, self.attempts)

    def first_pass_rate(self) -> float | None:
        return safe_rate(self.first_pass_accepted, self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "model_id": self.model_id,
            "role": self.role,
            "risk": self.risk,
            "task_class": self.task_class,
            "project_id": self.project_id,
            "attempts": self.attempts,
            "accepted": self.accepted,
            "first_pass_accepted": self.first_pass_accepted,
            "authority_violations": self.authority_violations,
            "context_capacity_exceeded": self.context_capacity_exceeded,
            "acceptance_rate": self.acceptance_rate(),
            "first_pass_rate": self.first_pass_rate(),
        }


@dataclass(frozen=True)
class RiskDifficultyStatistics:
    """Empirical statistics segmented by task risk."""

    risk: str
    attempts: int = 0
    accepted: int = 0
    first_pass_accepted: int = 0
    authority_violations: int = 0

    def acceptance_rate(self) -> float | None:
        return safe_rate(self.accepted, self.attempts)

    def first_pass_rate(self) -> float | None:
        return safe_rate(self.first_pass_accepted, self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk": self.risk,
            "attempts": self.attempts,
            "accepted": self.accepted,
            "first_pass_accepted": self.first_pass_accepted,
            "authority_violations": self.authority_violations,
            "acceptance_rate": self.acceptance_rate(),
            "first_pass_rate": self.first_pass_rate(),
        }


@dataclass(frozen=True)
class LanguageFrameworkStatistics:
    """Empirical statistics segmented by language/framework at full dimensionality.

    Keyed by (language_framework, model_id, role, risk, task_class, project_id).
    """

    language_framework: str
    model_id: str
    role: str
    risk: str
    task_class: str
    project_id: str
    attempts: int = 0
    accepted: int = 0
    first_pass_accepted: int = 0

    def acceptance_rate(self) -> float | None:
        return safe_rate(self.accepted, self.attempts)

    def first_pass_rate(self) -> float | None:
        return safe_rate(self.first_pass_accepted, self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language_framework": self.language_framework,
            "model_id": self.model_id,
            "role": self.role,
            "risk": self.risk,
            "task_class": self.task_class,
            "project_id": self.project_id,
            "attempts": self.attempts,
            "accepted": self.accepted,
            "first_pass_accepted": self.first_pass_accepted,
            "acceptance_rate": self.acceptance_rate(),
            "first_pass_rate": self.first_pass_rate(),
        }


@dataclass(frozen=True)
class ProjectPerformanceStatistics:
    """Project-specific empirical statistics."""

    project_id: str
    attempts: int = 0
    accepted: int = 0
    first_pass_accepted: int = 0
    deterministic_failures: int = 0
    authority_violations: int = 0
    total_cost_actual: float = 0.0
    total_cost_estimated: float = 0.0

    def acceptance_rate(self) -> float | None:
        return safe_rate(self.accepted, self.attempts)

    def first_pass_rate(self) -> float | None:
        return safe_rate(self.first_pass_accepted, self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "attempts": self.attempts,
            "accepted": self.accepted,
            "first_pass_accepted": self.first_pass_accepted,
            "deterministic_failures": self.deterministic_failures,
            "authority_violations": self.authority_violations,
            "total_cost_actual": self.total_cost_actual,
            "total_cost_estimated": self.total_cost_estimated,
            "acceptance_rate": self.acceptance_rate(),
            "first_pass_rate": self.first_pass_rate(),
        }


@dataclass(frozen=True)
class TaskLifecycleStatistics:
    """Derived task-level lifecycle cost and time aggregates.

    These are rebuilt from the raw event ledger rather than trusting a
    caller-supplied total.  Actual, estimated, and unknown cost states are
    preserved separately.
    """

    project_id: str
    task_id: str
    run_id: str
    start_time: datetime.datetime | None = None
    accepted_time: datetime.datetime | None = None
    accepted: bool = False
    abandoned: bool = False
    total_cost_actual: float = 0.0
    total_cost_estimated: float = 0.0
    unknown_cost_count: int = 0
    planning_cost_actual: float = 0.0
    planning_cost_estimated: float = 0.0
    implementation_cost_actual: float = 0.0
    implementation_cost_estimated: float = 0.0
    retry_cost_actual: float = 0.0
    retry_cost_estimated: float = 0.0
    review_cost_actual: float = 0.0
    review_cost_estimated: float = 0.0
    repair_cost_actual: float = 0.0
    repair_cost_estimated: float = 0.0
    arbitration_cost_actual: float = 0.0
    arbitration_cost_estimated: float = 0.0
    provider_wait_seconds: float = 0.0
    review_duration_seconds: float = 0.0
    repair_duration_seconds: float = 0.0
    arbitration_duration_seconds: float = 0.0

    def time_to_accepted_seconds(self) -> float | None:
        if self.start_time is None or self.accepted_time is None:
            return None
        return (self.accepted_time - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "accepted_time": self.accepted_time.isoformat() if self.accepted_time else None,
            "accepted": self.accepted,
            "abandoned": self.abandoned,
            "total_cost_actual": self.total_cost_actual,
            "total_cost_estimated": self.total_cost_estimated,
            "unknown_cost_count": self.unknown_cost_count,
            "planning_cost_actual": self.planning_cost_actual,
            "planning_cost_estimated": self.planning_cost_estimated,
            "implementation_cost_actual": self.implementation_cost_actual,
            "implementation_cost_estimated": self.implementation_cost_estimated,
            "retry_cost_actual": self.retry_cost_actual,
            "retry_cost_estimated": self.retry_cost_estimated,
            "review_cost_actual": self.review_cost_actual,
            "review_cost_estimated": self.review_cost_estimated,
            "repair_cost_actual": self.repair_cost_actual,
            "repair_cost_estimated": self.repair_cost_estimated,
            "arbitration_cost_actual": self.arbitration_cost_actual,
            "arbitration_cost_estimated": self.arbitration_cost_estimated,
            "provider_wait_seconds": self.provider_wait_seconds,
            "review_duration_seconds": self.review_duration_seconds,
            "repair_duration_seconds": self.repair_duration_seconds,
            "arbitration_duration_seconds": self.arbitration_duration_seconds,
            "time_to_accepted_seconds": self.time_to_accepted_seconds(),
        }


def _join_key(*parts: str) -> str:
    return "|".join(parts)


@dataclass(frozen=True)
class PerformanceStatisticsBundle:
    """All derived statistics rebuilt from a performance ledger."""

    model_role: dict[tuple[str, str], ModelRoleStatistics] = field(default_factory=dict)
    model_role_dimensional: dict[
        tuple[str, str, str, str, str, str], ModelRoleDimensionalStatistics
    ] = field(default_factory=dict)
    route: dict[str, RouteStatistics] = field(default_factory=dict)
    reviewer: dict[tuple[str, str, str, str, str], ReviewerStatistics] = field(default_factory=dict)
    context_strategy: dict[tuple[str, str, str, str, str, str], ContextStrategyStatistics] = field(
        default_factory=dict
    )
    risk: dict[str, RiskDifficultyStatistics] = field(default_factory=dict)
    language_framework: dict[tuple[str, str, str, str, str, str], LanguageFrameworkStatistics] = (
        field(default_factory=dict)
    )
    project: dict[str, ProjectPerformanceStatistics] = field(default_factory=dict)
    task_lifecycle: dict[tuple[str, str, str], TaskLifecycleStatistics] = field(
        default_factory=dict
    )
    total_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "model_role": {
                f"{m}:{r}": s.to_dict() for (m, r), s in sorted(self.model_role.items())
            },
            "model_role_dimensional": {
                _join_key(*k): s.to_dict() for k, s in sorted(self.model_role_dimensional.items())
            },
            "route": {k: v.to_dict() for k, v in sorted(self.route.items())},
            "reviewer": {_join_key(*k): s.to_dict() for k, s in sorted(self.reviewer.items())},
            "context_strategy": {
                _join_key(*k): s.to_dict() for k, s in sorted(self.context_strategy.items())
            },
            "risk": {k: v.to_dict() for k, v in sorted(self.risk.items())},
            "language_framework": {
                _join_key(*k): s.to_dict() for k, s in sorted(self.language_framework.items())
            },
            "project": {k: v.to_dict() for k, v in sorted(self.project.items())},
            "task_lifecycle": {
                _join_key(*k): s.to_dict() for k, s in sorted(self.task_lifecycle.items())
            },
        }
