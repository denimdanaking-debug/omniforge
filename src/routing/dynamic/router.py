"""Routing coordinator: legacy and dynamic routing modes."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from src.context.budget import ContextBudget
from src.persistence.configuration import extract_administrative_state
from src.recovery.failure_domain import FailureDomainIndex
from src.recovery.reserve import ReserveCapacityPolicy
from src.routing.policy import ProjectRoutingPolicy, RoutingPolicyEngine

from .candidate import RoutingCandidate
from .config import RouterConfig, RoutingCoordinatorState, load_router_config
from .decision import RoutingDecision, RoutingDecisionRecord
from .eligibility import CandidateEligibilityPipeline
from .explanation import ExplanationFormatter
from .fingerprint import routing_input_fingerprint
from .request import DynamicRoutingRequest
from .scoring import CandidateScore, ScoringState


class RoutingCoordinator:
    """Coordinate legacy and dynamic routing decisions."""

    def __init__(self, config: RouterConfig | None = None) -> None:
        self._default_config = config or RouterConfig()
        self._state = RoutingCoordinatorState()

    @property
    def config(self) -> RouterConfig:
        return self._default_config

    @property
    def state(self) -> RoutingCoordinatorState:
        return self._state

    def _make_policy_engine(
        self,
        request: DynamicRoutingRequest,
        admin_state: dict[str, Any],
    ) -> RoutingPolicyEngine:
        provider_status = admin_state.get("provider_status", {})
        model_status = admin_state.get("model_status", {})
        route_status = admin_state.get("route_status", {})
        provider_enabled = {k: v.get("enabled", True) for k, v in provider_status.items()}
        model_enabled = {k: v.get("enabled", True) for k, v in model_status.items()}
        route_enabled = {k: v.get("enabled", True) for k, v in route_status.items()}

        policies = admin_state.get("project_policies", {})
        project_policy = ProjectRoutingPolicy.from_dict(policies.get(request.project_id, {}))
        pin = request.pin
        return RoutingPolicyEngine(
            provider_enabled=provider_enabled,
            model_enabled=model_enabled,
            route_enabled=route_enabled,
            project_policy=project_policy,
            pin=pin,
        )

    def _make_pipeline(
        self,
        policy_engine: RoutingPolicyEngine,
        admin_state: dict[str, Any],
        context_budget: ContextBudget | None,
    ) -> CandidateEligibilityPipeline:
        reserve_policy = None
        reserve_cfg = admin_state.get("reserve_policy")
        if reserve_cfg is not None:
            reserve_policy = ReserveCapacityPolicy.from_dict(reserve_cfg)
        failure_domain_index = None
        fdi_cfg = admin_state.get("failure_domain_index")
        if fdi_cfg is not None:
            failure_domain_index = FailureDomainIndex.from_dict(fdi_cfg)
        return CandidateEligibilityPipeline(
            policy_engine=policy_engine,
            reserve_policy=reserve_policy,
            failure_domain_index=failure_domain_index,
            context_requirement=context_budget,
        )

    def _route_legacy(
        self,
        request: DynamicRoutingRequest,
        candidates: tuple[RoutingCandidate, ...],
        admin_state: dict[str, Any],
        router_cfg: RouterConfig,
        context: dict[str, Any] | None,
        timestamp: datetime,
        exploration_enabled: bool,
    ) -> RoutingDecision:
        """Legacy mode: return the first candidate that survives policy checks."""
        policy_engine = self._make_policy_engine(request, admin_state)
        pipeline = self._make_pipeline(policy_engine, admin_state, None)
        eligibility = pipeline.evaluate(request, candidates)
        winner = eligibility.candidates[0] if eligibility.candidates else None
        runner_up = eligibility.candidates[1] if len(eligibility.candidates) > 1 else None
        margin = 0.0
        record = RoutingDecisionRecord(
            decision_id=f"decision-{uuid.uuid4()}",
            request=request,
            routing_mode="legacy",
            exploration_enabled=exploration_enabled,
            candidates_considered=candidates,
            exclusions=eligibility.exclusions,
            eligible_candidates=eligibility.candidates,
            scores=(),
            winner=winner,
            runner_up=runner_up,
            score_margin=margin,
            policy_effects={},
            pin_effects={},
            reserve_effects={},
            quota_effects={},
            fallback_used=False,
            fallback_reason=None,
            context_metadata=context or {},
            input_fingerprint=routing_input_fingerprint(
                request=request,
                candidates=candidates,
                policy_engine=policy_engine,
                router_config=router_cfg,
                scoring_state=self._state,
            ),
            timestamp=timestamp,
        )
        decision = RoutingDecision(
            selected_candidate=winner,
            ranked_candidates=eligibility.candidates,
            excluded=eligibility.exclusions,
            record=record,
            explanation="",
            no_eligible_reason="no eligible candidates" if winner is None else None,
            emergency_fallback_used=False,
        )
        explanation = ExplanationFormatter().format(decision)
        return RoutingDecision(
            selected_candidate=decision.selected_candidate,
            ranked_candidates=decision.ranked_candidates,
            excluded=decision.excluded,
            record=decision.record,
            explanation=explanation,
            no_eligible_reason=decision.no_eligible_reason,
            emergency_fallback_used=decision.emergency_fallback_used,
        )

    def _route_dynamic(
        self,
        request: DynamicRoutingRequest,
        candidates: tuple[RoutingCandidate, ...],
        admin_state: dict[str, Any],
        router_cfg: RouterConfig,
        context: dict[str, Any] | None,
        timestamp: datetime,
        exploration_enabled: bool,
    ) -> RoutingDecision:
        """Dynamic mode: eligibility -> score -> select winner -> record."""
        policy_engine = self._make_policy_engine(request, admin_state)
        context_budget = router_cfg.context_budget_for(request)
        pipeline = self._make_pipeline(policy_engine, admin_state, context_budget)
        eligibility = pipeline.evaluate(request, candidates)

        scorer = router_cfg.build_scorer()
        fallback_router = router_cfg.build_fallback_router()

        if not eligibility.candidates:
            return fallback_router.route(
                request=request,
                candidates=candidates,
                pipeline=pipeline,
                decision_id=f"decision-{uuid.uuid4()}",
                context=context,
                timestamp=timestamp,
            )

        scoring_state = ScoringState(
            last_selected_key=self._state.last_selected_key,
            failure_domain_counts=dict(self._state.failure_domain_counts),
        )

        try:
            scored: list[tuple[RoutingCandidate, CandidateScore]] = []
            for candidate in eligibility.candidates:
                score = scorer.score(request, candidate, scoring_state)
                scored.append((candidate, score))
        except Exception as exc:
            return fallback_router.route(
                request=request,
                candidates=candidates,
                pipeline=pipeline,
                decision_id=f"decision-{uuid.uuid4()}",
                context={**(context or {}), "fallback_reason": f"scoring_failed:{exc}"},
                timestamp=timestamp,
            )

        # Deterministic ranking: higher score first, then canonical tie-break key.
        scored.sort(key=lambda item: (-item[1].total_score, item[1].tie_break_key))
        winner, winner_score = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else None
        margin = 0.0
        if runner_up is not None:
            margin = winner_score.total_score - scored[1][1].total_score

        scores = tuple(score for _, score in scored)
        ranked = tuple(candidate for candidate, _ in scored)

        # Update local scoring state for future continuity/diversity.
        domain = winner.route_identity.failure_domain
        self._state.failure_domain_counts[domain] = (
            self._state.failure_domain_counts.get(domain, 0) + 1
        )
        self._state.last_selected_key = winner.identity_key

        record = RoutingDecisionRecord(
            decision_id=f"decision-{uuid.uuid4()}",
            request=request,
            routing_mode="dynamic",
            exploration_enabled=exploration_enabled,
            candidates_considered=candidates,
            exclusions=eligibility.exclusions,
            eligible_candidates=eligibility.candidates,
            scores=scores,
            winner=winner,
            runner_up=runner_up,
            score_margin=margin,
            policy_effects={},
            pin_effects={},
            reserve_effects={},
            quota_effects={},
            fallback_used=False,
            fallback_reason=None,
            context_metadata=context or {},
            input_fingerprint=routing_input_fingerprint(
                request=request,
                candidates=candidates,
                policy_engine=policy_engine,
                router_config=router_cfg,
                scoring_state=self._state,
            ),
            timestamp=timestamp,
        )
        decision = RoutingDecision(
            selected_candidate=winner,
            ranked_candidates=ranked,
            excluded=eligibility.exclusions,
            record=record,
            explanation="",
            no_eligible_reason=None,
            emergency_fallback_used=False,
        )
        explanation = ExplanationFormatter().format(decision)
        return RoutingDecision(
            selected_candidate=decision.selected_candidate,
            ranked_candidates=decision.ranked_candidates,
            excluded=decision.excluded,
            record=decision.record,
            explanation=explanation,
            no_eligible_reason=decision.no_eligible_reason,
            emergency_fallback_used=decision.emergency_fallback_used,
        )

    def route(
        self,
        request: DynamicRoutingRequest,
        candidates: tuple[RoutingCandidate, ...],
        config: dict[str, Any],
        state: dict[str, Any] | None = None,
    ) -> RoutingDecision:
        """Route a request using the configured routing mode."""
        admin_state = extract_administrative_state(config)
        router_cfg = load_router_config(config.get("router_config", {}))
        routing_mode = admin_state.get("routing_mode", "legacy")
        project_override = None
        policies = admin_state.get("project_policies", {})
        if request.project_id in policies:
            policy = ProjectRoutingPolicy.from_dict(policies[request.project_id])
            project_override = policy.routing_mode_override
        if project_override is not None:
            routing_mode = project_override

        # Exploration is recorded in metadata but selection remains deterministic.
        exploration_enabled = router_cfg.exploration_enabled

        timestamp = datetime.now(UTC)
        context: dict[str, Any] = {
            "exploration_enabled": exploration_enabled,
            "router_config_loaded": True,
        }
        if state is not None:
            context["state_snapshot_ref"] = state.get("run_id")

        if routing_mode == "legacy":
            return self._route_legacy(
                request,
                candidates,
                admin_state,
                router_cfg,
                context,
                timestamp,
                exploration_enabled,
            )
        return self._route_dynamic(
            request, candidates, admin_state, router_cfg, context, timestamp, exploration_enabled
        )
