from __future__ import annotations

from datetime import UTC, datetime

from src.policy.risk import RiskLevel
from src.routing.dynamic.candidate import RoutingCandidate
from src.routing.dynamic.decision import RoutingDecision, RoutingDecisionRecord
from src.routing.dynamic.explanation import ExplanationFormatter
from src.routing.dynamic.request import DynamicRoutingRequest
from src.routing.roles import ExecutionRole


def test_explanation_mentions_winner(healthy_candidate: RoutingCandidate) -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )
    record = RoutingDecisionRecord(
        decision_id="dec-1",
        request=request,
        routing_mode="dynamic",
        exploration_enabled=False,
        candidates_considered=(healthy_candidate,),
        exclusions=(),
        eligible_candidates=(healthy_candidate,),
        scores=(),
        winner=healthy_candidate,
        runner_up=None,
        score_margin=0.0,
        policy_effects={},
        pin_effects={},
        reserve_effects={},
        quota_effects={},
        fallback_used=False,
        fallback_reason=None,
        context_metadata={},
        input_fingerprint="abc",
        timestamp=datetime.now(UTC),
    )
    decision = RoutingDecision(
        selected_candidate=healthy_candidate,
        ranked_candidates=(healthy_candidate,),
        excluded=(),
        record=record,
        explanation="",
        no_eligible_reason=None,
        emergency_fallback_used=False,
    )
    formatter = ExplanationFormatter()
    text = formatter.format(decision)
    assert "Winner" in text
    assert "openai:gpt-4o:openai-direct" in text


def test_explanation_no_hidden_reasoning(healthy_candidate: RoutingCandidate) -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )
    record = RoutingDecisionRecord(
        decision_id="dec-1",
        request=request,
        routing_mode="dynamic",
        exploration_enabled=False,
        candidates_considered=(healthy_candidate,),
        exclusions=(),
        eligible_candidates=(healthy_candidate,),
        scores=(),
        winner=healthy_candidate,
        runner_up=None,
        score_margin=0.0,
        policy_effects={},
        pin_effects={},
        reserve_effects={},
        quota_effects={},
        fallback_used=False,
        fallback_reason=None,
        context_metadata={"secret": "hidden"},
        input_fingerprint="abc",
        timestamp=datetime.now(UTC),
    )
    decision = RoutingDecision(
        selected_candidate=healthy_candidate,
        ranked_candidates=(healthy_candidate,),
        excluded=(),
        record=record,
        explanation="",
        no_eligible_reason=None,
        emergency_fallback_used=False,
    )
    formatter = ExplanationFormatter()
    text = formatter.format(decision)
    # Explanation is derived from record only; should not leak hidden context metadata.
    assert "hidden" not in text
