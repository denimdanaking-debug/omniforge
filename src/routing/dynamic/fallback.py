"""Emergency fallback router for dynamic routing."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.routing.roles import ExecutionRole

from .candidate import RoutingCandidate
from .decision import RoutingDecision, RoutingDecisionRecord
from .eligibility import CandidateEligibilityPipeline
from .explanation import ExplanationFormatter
from .request import DynamicRoutingRequest


class EmergencyFallbackRouter:
    """Emergency fallback router triggered when dynamic scoring fails."""

    def __init__(
        self,
        fallback_orders: dict[ExecutionRole, list[str]] | None = None,
    ) -> None:
        self._fallback_orders: dict[ExecutionRole, tuple[str, ...]] = {}
        defaults: dict[ExecutionRole, list[str]] = {
            ExecutionRole.CODING: [
                "openai:gpt-4o:openai-direct",
                "anthropic:claude-sonnet:anthropic-direct",
                "kimi:kimi-k3:kimi-direct",
                "qwen:qwen3.8-max:qwen-direct",
            ],
            ExecutionRole.REVIEW: [
                "anthropic:claude-sonnet:anthropic-direct",
                "openai:gpt-4o:openai-direct",
                "kimi:kimi-k3:kimi-direct",
                "qwen:qwen3.8-max:qwen-direct",
            ],
            ExecutionRole.HIGH_RISK_REVIEW: [
                "anthropic:claude-sonnet:anthropic-direct",
                "kimi:kimi-k3:kimi-direct",
                "qwen:qwen3.8-max:qwen-direct",
                "openai:gpt-4o:openai-direct",
            ],
            ExecutionRole.PLANNING: [
                "openai:gpt-4o:openai-direct",
                "anthropic:claude-sonnet:anthropic-direct",
                "kimi:kimi-k3:kimi-direct",
                "qwen:qwen3.8-max:qwen-direct",
            ],
            ExecutionRole.ARCHITECTURE: [
                "anthropic:claude-sonnet:anthropic-direct",
                "openai:gpt-4o:openai-direct",
                "kimi:kimi-k3:kimi-direct",
                "qwen:qwen3.8-max:qwen-direct",
            ],
            ExecutionRole.DEBUGGING: [
                "openai:gpt-4o:openai-direct",
                "kimi:kimi-k3:kimi-direct",
                "qwen:qwen3.8-max:qwen-direct",
                "anthropic:claude-sonnet:anthropic-direct",
            ],
            ExecutionRole.REPAIR: [
                "openai:gpt-4o:openai-direct",
                "kimi:kimi-k3:kimi-direct",
                "anthropic:claude-sonnet:anthropic-direct",
                "qwen:qwen3.8-max:qwen-direct",
            ],
        }
        orders = fallback_orders if fallback_orders is not None else defaults
        for role, keys in orders.items():
            self._fallback_orders[role] = tuple(keys)

    def route(
        self,
        request: DynamicRoutingRequest,
        candidates: tuple[RoutingCandidate, ...],
        pipeline: CandidateEligibilityPipeline,
        decision_id: str,
        context: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> RoutingDecision:
        """Return a fallback decision after filtering candidates through eligibility."""
        eligibility = pipeline.evaluate(request, candidates)
        ordered_keys = self._fallback_orders.get(request.role, ())
        winner: RoutingCandidate | None = None
        for key in ordered_keys:
            for candidate in eligibility.candidates:
                if candidate.identity_key == key:
                    winner = candidate
                    break
            if winner is not None:
                break
        if winner is None and eligibility.candidates:
            winner = eligibility.candidates[0]

        runner_up: RoutingCandidate | None = None
        if winner is not None and len(eligibility.candidates) > 1:
            for candidate in eligibility.candidates:
                if candidate.identity_key != winner.identity_key:
                    runner_up = candidate
                    break

        fallback_reason = "emergency_fallback_triggered"
        no_eligible_reason = None if winner is not None else "no eligible fallback candidates"
        record = RoutingDecisionRecord(
            decision_id=decision_id,
            request=request,
            routing_mode="dynamic",
            exploration_enabled=False,
            candidates_considered=candidates,
            exclusions=eligibility.exclusions,
            eligible_candidates=eligibility.candidates,
            scores=(),
            winner=winner,
            runner_up=runner_up,
            score_margin=0.0,
            policy_effects={},
            pin_effects={},
            reserve_effects={},
            quota_effects={},
            fallback_used=True,
            fallback_reason=fallback_reason,
            context_metadata=context or {},
            input_fingerprint="",
            timestamp=timestamp or request.timestamp or datetime.now(UTC),
        )
        decision = RoutingDecision(
            selected_candidate=winner,
            ranked_candidates=eligibility.candidates,
            excluded=eligibility.exclusions,
            record=record,
            explanation="",
            no_eligible_reason=no_eligible_reason,
            emergency_fallback_used=True,
        )
        explanation = ExplanationFormatter().format(decision)
        # Frozen dataclass: replace explanation via construction.
        decision = RoutingDecision(
            selected_candidate=decision.selected_candidate,
            ranked_candidates=decision.ranked_candidates,
            excluded=decision.excluded,
            record=decision.record,
            explanation=explanation,
            no_eligible_reason=decision.no_eligible_reason,
            emergency_fallback_used=decision.emergency_fallback_used,
        )
        return decision
