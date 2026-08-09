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

    def _deterministic_ranking(
        self,
        request: DynamicRoutingRequest,
        eligible_candidates: tuple[RoutingCandidate, ...],
    ) -> tuple[RoutingCandidate, ...]:
        """Return fallback candidates in deterministic order.

        Configured role-specific order takes priority. Any eligible candidates
        not explicitly listed are appended sorted by canonical identity key.
        """
        ordered_keys = self._fallback_orders.get(request.role, ())
        ranked: list[RoutingCandidate] = []
        seen: set[str] = set()
        key_to_candidate = {c.identity_key: c for c in eligible_candidates}

        for key in ordered_keys:
            candidate = key_to_candidate.get(key)
            if candidate is not None and candidate.identity_key not in seen:
                ranked.append(candidate)
                seen.add(candidate.identity_key)

        for candidate in sorted(eligible_candidates, key=lambda c: c.identity_key):
            if candidate.identity_key not in seen:
                ranked.append(candidate)
                seen.add(candidate.identity_key)

        return tuple(ranked)

    def route(
        self,
        request: DynamicRoutingRequest,
        candidates: tuple[RoutingCandidate, ...],
        pipeline: CandidateEligibilityPipeline,
        decision_id: str,
        *,
        input_fingerprint: str = "",
        exploration_enabled: bool = False,
        fallback_reason: str = "emergency_fallback_triggered",
        context: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> RoutingDecision:
        """Return a fallback decision after filtering candidates through eligibility."""
        eligibility = pipeline.evaluate(request, candidates)
        ranked = self._deterministic_ranking(request, eligibility.candidates)

        winner = ranked[0] if ranked else None
        runner_up = ranked[1] if len(ranked) > 1 else None

        no_eligible_reason = None if winner is not None else "no eligible fallback candidates"
        record = RoutingDecisionRecord(
            decision_id=decision_id,
            request=request,
            routing_mode="dynamic",
            exploration_enabled=exploration_enabled,
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
            input_fingerprint=input_fingerprint,
            timestamp=timestamp or request.timestamp or datetime.now(UTC),
        )
        decision = RoutingDecision(
            selected_candidate=winner,
            ranked_candidates=ranked,
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
