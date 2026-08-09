"""Deterministic dynamic routing scorer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.policy.risk import lifecycle_eligible
from src.providers.identity import ProviderHealth
from src.recovery.quota_balance import QuotaPressure

from .candidate import RoutingCandidate
from .cost import CostToAcceptedEstimate, estimate_cost_to_accepted
from .priors import PriorBlender
from .request import DynamicRoutingRequest


class RoutingScoringError(ValueError):
    """Raised when scoring produces invalid or non-deterministic results."""


@dataclass(frozen=True)
class RoutingScoreFactors:
    """Normalized factor inputs to the routing scorer."""

    expected_success: float = 0.0
    role_fit: float = 0.0
    risk_fit: float = 0.0
    empirical_reliability: float = 0.0
    context_suitability: float = 0.0
    recent_performance: float = 0.0
    provider_health: float = 0.0
    quota_pressure: float = 0.0
    cost: float = 0.0
    latency: float = 0.0
    affinity: float = 0.0
    diversity_reserve: float = 0.0


@dataclass(frozen=True)
class WeightedFactor:
    """One weighted factor contribution with provenance."""

    name: str
    normalized_value: float
    weight: float
    contribution: float
    provenance: str


@dataclass(frozen=True)
class CandidateScore:
    """Immutable score result for one candidate."""

    total_score: float
    weighted_factors: tuple[WeightedFactor, ...]
    tie_break_key: str
    cost_estimate: CostToAcceptedEstimate | None = None


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for routing score factors."""

    expected_success: float = 1.0
    role_fit: float = 1.0
    risk_fit: float = 0.8
    empirical_reliability: float = 0.8
    context_suitability: float = 0.7
    recent_performance: float = 0.7
    provider_health: float = 0.6
    quota_pressure: float = 0.6
    cost: float = 0.5
    latency: float = 0.5
    affinity: float = 0.1
    diversity_reserve: float = 0.1

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(value):
                raise RoutingScoringError(f"weight {name} must be finite")
            if value < 0:
                raise RoutingScoringError(f"weight {name} cannot be negative")
        if all(value == 0 for value in self.__dict__.values()):
            raise RoutingScoringError("at least one weight must be non-zero")

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoringWeights:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ScoringState:
    """Shared scoring state (affinity hints, failure-domain concentration)."""

    last_selected_key: str | None = None
    failure_domain_counts: dict[str, int] = field(default_factory=dict)


class DeterministicRouterScorer:
    """Deterministic, weighted-factor scorer for routing candidates."""

    def __init__(
        self,
        weights: ScoringWeights | None = None,
        blender: PriorBlender | None = None,
        cost_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._weights = weights or ScoringWeights()
        self._blender = blender or PriorBlender()
        self._cost_metadata = cost_metadata or {}

    @property
    def weights(self) -> ScoringWeights:
        return self._weights

    @property
    def cost_metadata(self) -> dict[str, Any]:
        return self._cost_metadata

    _SIGNED_FACTORS: frozenset[str] = frozenset({"diversity_reserve"})

    def _finite(self, value: float, provenance: str) -> float:
        if not math.isfinite(value):
            raise RoutingScoringError(f"non-finite score value from {provenance}: {value}")
        return value

    def _normalize(self, value: float, factor_name: str) -> float:
        value = self._finite(value, factor_name)
        if factor_name in self._SIGNED_FACTORS:
            return max(-1.0, min(1.0, value))
        return max(0.0, min(1.0, value))

    def _role_fit(
        self, candidate: RoutingCandidate, request: DynamicRoutingRequest
    ) -> tuple[float, str]:
        supported = candidate.capabilities.supported_roles
        role_value = request.role.value
        if supported and role_value in supported:
            return 1.0, f"role {role_value} supported"
        return 0.0, f"role {role_value} not in supported roles"

    def _risk_fit(
        self, candidate: RoutingCandidate, request: DynamicRoutingRequest
    ) -> tuple[float, str]:
        lifecycle = candidate.model_identity.lifecycle
        if lifecycle_eligible(lifecycle, request.risk):
            return 1.0, f"lifecycle {lifecycle.value} eligible for {request.risk.name}"
        rank = {
            "SHADOW": 0,
            "DISABLED": 0,
            "LOW_RISK": 1,
            "NORMAL": 2,
            "HIGH_RISK": 3,
        }
        value = rank.get(lifecycle.value, 0) / 3.0
        return value, f"lifecycle {lifecycle.value} below requirement for {request.risk.name}"

    def _empirical_reliability(self, candidate: RoutingCandidate) -> tuple[float, str]:
        """Pure observed historical reliability.

        This factor intentionally does NOT blend priors. It reflects only the
        evidence recorded for this model. A cold-start candidate with no evidence
        receives a neutral conservative value so it does not dominate nor vanish.
        """
        evidence = candidate.performance_evidence
        if evidence is None or evidence.success_rate is None:
            return 0.5, "no empirical evidence"
        return (
            evidence.success_rate,
            f"observed_success_rate={evidence.success_rate:.3f}, attempts={evidence.attempts}",
        )

    def _context_suitability(
        self, candidate: RoutingCandidate, request: DynamicRoutingRequest
    ) -> tuple[float, str]:
        required = request.required_context_tokens
        if required is None:
            return 1.0, "no context requirement"
        capacity = candidate.capabilities.context_tokens
        if capacity < required:
            return 0.0, f"context {capacity} < required {required}"
        ratio = capacity / required
        if ratio >= 4.0:
            return 1.0, f"context ratio {ratio:.2f} comfortable"
        if ratio >= 2.0:
            return 0.8, f"context ratio {ratio:.2f} adequate"
        if ratio >= 1.5:
            return 0.6, f"context ratio {ratio:.2f} tight"
        return 0.4, f"context ratio {ratio:.2f} minimal"

    def _recent_performance(self, candidate: RoutingCandidate) -> tuple[float, str]:
        evidence = candidate.performance_evidence
        if evidence is None or evidence.recent_success_rate is None:
            return 0.5, "no recent performance data"
        return (
            evidence.recent_success_rate,
            f"recent_success_rate={evidence.recent_success_rate:.3f}",
        )

    def _provider_health(self, candidate: RoutingCandidate) -> tuple[float, str]:
        state = candidate.operational_state
        if state is None:
            return 0.5, "operational state unknown"
        health = state.health
        if health is ProviderHealth.HEALTHY:
            return 1.0, "provider healthy"
        if health is ProviderHealth.DEGRADED:
            return 0.6, "provider degraded"
        return 0.0, f"provider {health.value}"

    def _quota_pressure(self, candidate: RoutingCandidate) -> tuple[float, str]:
        quota = candidate.quota_state
        pressure = QuotaPressure.from_quota(quota)
        mapping = {
            QuotaPressure.EXHAUSTED: 0.0,
            QuotaPressure.CRITICAL: 0.2,
            QuotaPressure.HIGH: 0.4,
            QuotaPressure.MODERATE: 0.7,
            QuotaPressure.LOW: 1.0,
            QuotaPressure.UNKNOWN: 0.5,
        }
        return mapping[pressure], f"quota pressure {pressure.value}"

    def _cost(
        self, candidate: RoutingCandidate, request: DynamicRoutingRequest
    ) -> tuple[float, str, CostToAcceptedEstimate]:
        """Cost factor based on expected total cost to accepted integration.

        Uses ``estimate_cost_to_accepted`` so cheap but failure-prone models do
        not automatically win over more reliable alternatives.
        """
        estimate = estimate_cost_to_accepted(
            request,
            candidate,
            **self._cost_metadata,
        )
        if estimate.expected_total is None:
            return 0.5, "cost unknown", estimate
        # Normalize inverse: lower expected accepted-task cost is better.
        score = 1.0 / (1.0 + estimate.expected_total)
        return (
            self._normalize(score, "cost"),
            f"expected_total_cost={estimate.expected_total:.6f}",
            estimate,
        )

    def _latency(self, candidate: RoutingCandidate) -> tuple[float, str]:
        evidence = candidate.performance_evidence
        if evidence is not None and evidence.average_latency_ms is not None:
            ms = evidence.average_latency_ms
            score = 1.0 / (1.0 + ms / 1000.0)
            return self._normalize(score, "latency"), f"empirical_latency_ms={ms:.1f}"
        route_state = candidate.route_cost_state
        if route_state is not None and route_state.rolling_latency_ms is not None:
            ms = route_state.rolling_latency_ms
            score = 1.0 / (1.0 + ms / 1000.0)
            return self._normalize(score, "latency"), f"rolling_latency_ms={ms:.1f}"
        return 0.5, "latency unknown"

    def _affinity(self, candidate: RoutingCandidate, state: ScoringState) -> tuple[float, str]:
        if state.last_selected_key is None:
            return 0.0, "no affinity baseline"
        if candidate.identity_key == state.last_selected_key:
            return 0.1, "continuity bonus"
        return 0.0, "no continuity"

    def _diversity_reserve(
        self, candidate: RoutingCandidate, state: ScoringState
    ) -> tuple[float, str]:
        domain = candidate.route_identity.failure_domain
        count = state.failure_domain_counts.get(domain, 0)
        if count == 0:
            return 0.0, f"failure_domain {domain} unconcentrated"
        penalty = -min(0.1 * count, 0.3)
        return penalty, f"failure_domain {domain} count={count}"

    def _expected_success(
        self, candidate: RoutingCandidate, request: DynamicRoutingRequest
    ) -> tuple[float, str]:
        prior = self._blender.prior_for(
            model_id=candidate.model_id,
            role=request.role,
            task_class=request.task_class,
        )
        empirical = (
            candidate.performance_evidence.success_rate
            if candidate.performance_evidence is not None
            else None
        )
        count = (
            candidate.performance_evidence.attempts
            if candidate.performance_evidence is not None
            else 0
        )
        blended = self._blender.blend(prior, empirical, count)
        return blended, f"blended expected_success={blended:.3f}"

    def score(
        self,
        request: DynamicRoutingRequest,
        candidate: RoutingCandidate,
        state: ScoringState | None = None,
    ) -> CandidateScore:
        """Return a deterministic weighted score for a candidate."""
        state = state or ScoringState()

        from collections.abc import Callable

        cost_estimate: CostToAcceptedEstimate | None = None

        factor_methods: dict[str, Callable[[], tuple[float, str]]] = {
            "expected_success": lambda: self._expected_success(candidate, request),
            "role_fit": lambda: self._role_fit(candidate, request),
            "risk_fit": lambda: self._risk_fit(candidate, request),
            "empirical_reliability": lambda: self._empirical_reliability(candidate),
            "context_suitability": lambda: self._context_suitability(candidate, request),
            "recent_performance": lambda: self._recent_performance(candidate),
            "provider_health": lambda: self._provider_health(candidate),
            "quota_pressure": lambda: self._quota_pressure(candidate),
            "latency": lambda: self._latency(candidate),
            "affinity": lambda: self._affinity(candidate, state),
            "diversity_reserve": lambda: self._diversity_reserve(candidate, state),
        }

        weighted_factors: list[WeightedFactor] = []
        total = 0.0
        for name, method in factor_methods.items():
            raw_value, provenance = method()
            weight = getattr(self._weights, name)
            value = self._normalize(raw_value, name)
            contribution = value * weight
            weighted_factors.append(
                WeightedFactor(
                    name=name,
                    normalized_value=value,
                    weight=weight,
                    contribution=contribution,
                    provenance=provenance,
                )
            )
            total += contribution

        # Cost factor is computed separately so its estimate can be recorded.
        cost_value, cost_provenance, cost_estimate = self._cost(candidate, request)
        cost_weight = self._weights.cost
        cost_normalized = self._normalize(cost_value, "cost")
        cost_contribution = cost_normalized * cost_weight
        weighted_factors.append(
            WeightedFactor(
                name="cost",
                normalized_value=cost_normalized,
                weight=cost_weight,
                contribution=cost_contribution,
                provenance=cost_provenance,
            )
        )
        total += cost_contribution

        total = self._finite(total, "total")
        return CandidateScore(
            total_score=total,
            weighted_factors=tuple(weighted_factors),
            tie_break_key=candidate.identity_key,
            cost_estimate=cost_estimate,
        )
