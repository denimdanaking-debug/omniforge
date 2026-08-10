"""Derived performance statistics rebuilt from the immutable event ledger."""

from __future__ import annotations

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
    """Empirical statistics for a (model_id, execution_role) pair."""

    model_id: str
    role: str
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
        return safe_rate(self.total_latency_seconds, self.attempts)

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
    """Empirical statistics for a reviewer model identity."""

    model_id: str
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
    """Empirical statistics for a Phase 7 context strategy."""

    strategy: str
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
    """Empirical statistics segmented by language/framework tag."""

    language_framework: str
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
class PerformanceStatisticsBundle:
    """All derived statistics rebuilt from a performance ledger."""

    model_role: dict[tuple[str, str], ModelRoleStatistics] = field(default_factory=dict)
    route: dict[str, RouteStatistics] = field(default_factory=dict)
    reviewer: dict[str, ReviewerStatistics] = field(default_factory=dict)
    context_strategy: dict[str, ContextStrategyStatistics] = field(default_factory=dict)
    risk: dict[str, RiskDifficultyStatistics] = field(default_factory=dict)
    language_framework: dict[str, LanguageFrameworkStatistics] = field(default_factory=dict)
    project: dict[str, ProjectPerformanceStatistics] = field(default_factory=dict)
    total_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "model_role": {
                f"{m}:{r}": s.to_dict() for (m, r), s in sorted(self.model_role.items())
            },
            "route": {k: v.to_dict() for k, v in sorted(self.route.items())},
            "reviewer": {k: v.to_dict() for k, v in sorted(self.reviewer.items())},
            "context_strategy": {k: v.to_dict() for k, v in sorted(self.context_strategy.items())},
            "risk": {k: v.to_dict() for k, v in sorted(self.risk.items())},
            "language_framework": {
                k: v.to_dict() for k, v in sorted(self.language_framework.items())
            },
            "project": {k: v.to_dict() for k, v in sorted(self.project.items())},
        }
