"""Dynamic routing decision records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .candidate import RoutingCandidate
from .request import DynamicRoutingRequest
from .scoring import CandidateScore


@dataclass(frozen=True)
class ExclusionRecord:
    """One candidate exclusion with reason."""

    provider_id: str
    model_id: str
    route_id: str
    reason: str
    detail: str

    @property
    def identity_key(self) -> str:
        return f"{self.provider_id}:{self.model_id}:{self.route_id}"


@dataclass(frozen=True)
class RoutingDecisionRecord:
    """Immutable record of a dynamic routing decision."""

    decision_id: str
    request: DynamicRoutingRequest
    routing_mode: str
    exploration_enabled: bool
    candidates_considered: tuple[RoutingCandidate, ...]
    exclusions: tuple[ExclusionRecord, ...]
    eligible_candidates: tuple[RoutingCandidate, ...]
    scores: tuple[CandidateScore, ...]
    winner: RoutingCandidate | None
    runner_up: RoutingCandidate | None
    score_margin: float
    policy_effects: dict[str, Any]
    pin_effects: dict[str, Any]
    reserve_effects: dict[str, Any]
    quota_effects: dict[str, Any]
    fallback_used: bool
    fallback_reason: str | None
    context_metadata: dict[str, Any]
    input_fingerprint: str
    timestamp: datetime


@dataclass(frozen=True)
class RoutingDecision:
    """Final routing decision with ranked candidates and explanation."""

    selected_candidate: RoutingCandidate | None
    ranked_candidates: tuple[RoutingCandidate, ...]
    excluded: tuple[ExclusionRecord, ...]
    record: RoutingDecisionRecord
    explanation: str
    no_eligible_reason: str | None
    emergency_fallback_used: bool
