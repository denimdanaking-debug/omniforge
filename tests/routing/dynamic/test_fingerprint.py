from __future__ import annotations

from datetime import UTC, datetime

from src.policy.risk import RiskLevel
from src.routing.dynamic.fingerprint import input_fingerprint
from src.routing.dynamic.request import DynamicRoutingRequest
from src.routing.roles import ExecutionRole


def test_fingerprint_deterministic() -> None:
    request = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )
    f1 = input_fingerprint(request)
    f2 = input_fingerprint(request)
    assert f1 == f2
    assert len(f1) == 64


def test_fingerprint_excludes_timestamp() -> None:
    request1 = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
        timestamp=datetime.now(UTC),
    )
    request2 = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
        timestamp=datetime(2020, 1, 1, tzinfo=UTC),
    )
    assert input_fingerprint(request1) == input_fingerprint(request2)


def test_fingerprint_changes_with_role() -> None:
    request1 = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )
    request2 = DynamicRoutingRequest(
        task_id="t1",
        project_id="p1",
        role=ExecutionRole.REVIEW,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )
    assert input_fingerprint(request1) != input_fingerprint(request2)
