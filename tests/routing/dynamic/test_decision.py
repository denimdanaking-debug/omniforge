from __future__ import annotations

from datetime import UTC, datetime

from src.policy.risk import RiskLevel
from src.routing.dynamic.decision import ExclusionRecord, RoutingDecision, RoutingDecisionRecord
from src.routing.dynamic.request import DynamicRoutingRequest
from src.routing.roles import ExecutionRole


def test_decision_record_fields() -> None:
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
        candidates_considered=(),
        exclusions=(),
        eligible_candidates=(),
        scores=(),
        winner=None,
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
        selected_candidate=None,
        ranked_candidates=(),
        excluded=(),
        record=record,
        explanation="no eligible",
        no_eligible_reason="none",
        emergency_fallback_used=False,
    )
    assert decision.record.decision_id == "dec-1"
    assert decision.record.input_fingerprint == "abc"
    assert decision.no_eligible_reason == "none"


def test_exclusion_record_identity_key() -> None:
    record = ExclusionRecord(
        provider_id="openai",
        model_id="gpt-4o",
        route_id="openai-direct",
        reason="test",
        detail="detail",
    )
    assert record.identity_key == "openai:gpt-4o:openai-direct"
