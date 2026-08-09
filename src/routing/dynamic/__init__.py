"""Dynamic deterministic router for OmniForge."""

from __future__ import annotations

from .candidate import PerformanceEvidence, RoutingCandidate
from .config import RouterConfig, RoutingCoordinatorState, load_router_config
from .cost import CostToAcceptedEstimate, estimate_cost_to_accepted
from .decision import ExclusionRecord, RoutingDecision, RoutingDecisionRecord
from .eligibility import CandidateEligibilityPipeline, EligibilityResult, ExclusionReason
from .explanation import ExplanationFormatter
from .fallback import EmergencyFallbackRouter
from .fingerprint import input_fingerprint, routing_input_fingerprint
from .priors import ModelRoutingPrior, PriorBlender
from .request import DynamicRoutingRequest
from .router import RoutingCoordinator
from .scoring import (
    CandidateScore,
    DeterministicRouterScorer,
    RoutingScoreFactors,
    RoutingScoringError,
    ScoringWeights,
    WeightedFactor,
)

__all__ = [
    "CandidateEligibilityPipeline",
    "CandidateScore",
    "CostToAcceptedEstimate",
    "DeterministicRouterScorer",
    "DynamicRoutingRequest",
    "EligibilityResult",
    "EmergencyFallbackRouter",
    "ExclusionReason",
    "ExclusionRecord",
    "ExplanationFormatter",
    "ModelRoutingPrior",
    "PerformanceEvidence",
    "PriorBlender",
    "RoutingCandidate",
    "RoutingCoordinator",
    "RoutingCoordinatorState",
    "RoutingDecision",
    "RoutingDecisionRecord",
    "RoutingScoreFactors",
    "RoutingScoringError",
    "RouterConfig",
    "ScoringWeights",
    "WeightedFactor",
    "estimate_cost_to_accepted",
    "input_fingerprint",
    "load_router_config",
    "routing_input_fingerprint",
]
