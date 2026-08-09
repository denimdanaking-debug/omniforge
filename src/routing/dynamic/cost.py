"""Cost-to-accepted estimate for routing candidates."""

from __future__ import annotations

from dataclasses import dataclass

from .candidate import RoutingCandidate
from .request import DynamicRoutingRequest


@dataclass(frozen=True)
class CostToAcceptedEstimate:
    """Estimated cost to reach an accepted result for a candidate."""

    direct_attempt_cost: float | None
    expected_retry_cost: float | None
    expected_repair_cost: float | None
    expected_review_cost: float | None
    expected_total: float | None
    currency: str
    confidence: str


def estimate_cost_to_accepted(
    request: DynamicRoutingRequest,
    candidate: RoutingCandidate,
    *,
    review_cost_multiplier: float = 0.2,
    default_retry_rate: float = 0.1,
    default_repair_rate: float = 0.1,
    default_success_rate: float = 0.7,
) -> CostToAcceptedEstimate:
    """Compute expected total cost including retries, repairs, and reviews.

    Unknown pricing results in ``None`` expected_total, not zero.
    """
    route_state = candidate.route_cost_state
    caps = candidate.capabilities.cost
    input_cost = route_state.input_cost_per_million if route_state else None
    output_cost = route_state.output_cost_per_million if route_state else None
    if input_cost is None:
        input_cost = caps.input_per_million
    if output_cost is None:
        output_cost = caps.output_per_million

    input_tokens = request.expected_input_tokens or 1000
    output_tokens = request.expected_output_tokens or 500

    if input_cost is None or output_cost is None:
        return CostToAcceptedEstimate(
            direct_attempt_cost=None,
            expected_retry_cost=None,
            expected_repair_cost=None,
            expected_review_cost=None,
            expected_total=None,
            currency="USD",
            confidence="UNKNOWN",
        )

    direct = (input_tokens * input_cost + output_tokens * output_cost) / 1_000_000

    evidence = candidate.performance_evidence
    success_rate = (
        evidence.success_rate
        if evidence is not None and evidence.success_rate is not None
        else default_success_rate
    )
    retry_rate = (
        evidence.retry_rate
        if evidence is not None and evidence.retry_rate is not None
        else default_retry_rate
    )
    repair_rate = (
        evidence.repair_rate
        if evidence is not None and evidence.repair_rate is not None
        else default_repair_rate
    )

    expected_attempts = 1.0 / max(success_rate, 0.01)
    expected_retry_cost = direct * retry_rate * expected_attempts
    expected_repair_cost = direct * repair_rate * expected_attempts
    expected_review_cost = direct * review_cost_multiplier

    expected_total = (
        direct * expected_attempts
        + expected_retry_cost
        + expected_repair_cost
        + expected_review_cost
    )

    confidence = "HIGH" if (evidence is not None and evidence.attempts >= 10) else "LOW"

    return CostToAcceptedEstimate(
        direct_attempt_cost=direct,
        expected_retry_cost=expected_retry_cost,
        expected_repair_cost=expected_repair_cost,
        expected_review_cost=expected_review_cost,
        expected_total=expected_total,
        currency="USD",
        confidence=confidence,
    )
