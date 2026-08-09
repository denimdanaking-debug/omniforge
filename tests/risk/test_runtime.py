from __future__ import annotations

import pytest

from src.policy.risk import RiskLevel
from src.risk import (
    RiskRuntimeEvent,
    RiskRuntimeEventType,
    RuntimeRiskEscalator,
)


def _event(
    event_type: RiskRuntimeEventType,
    material: bool = True,
    evidence: str = "evidence",
    count: int = 1,
    threshold: int = 1,
) -> RiskRuntimeEvent:
    return RiskRuntimeEvent(
        event_type=event_type,
        material=material,
        evidence=evidence,
        count=count,
        threshold=threshold,
    )


def test_single_test_failure_below_threshold_does_not_escalate() -> None:
    escalator = RuntimeRiskEscalator.default()
    event = _event(RiskRuntimeEventType.TEST_FAILURE, count=1, threshold=3)
    new_risk, record = escalator.escalate(RiskLevel.R2_NORMAL, event)
    assert new_risk == RiskLevel.R2_NORMAL
    assert record is not None
    assert record.new_risk == RiskLevel.R2_NORMAL


def test_repeated_test_failure_escalates_at_threshold() -> None:
    escalator = RuntimeRiskEscalator.default()
    event = _event(RiskRuntimeEventType.TEST_FAILURE, count=3, threshold=3)
    new_risk, record = escalator.escalate(RiskLevel.R2_NORMAL, event)
    assert new_risk == RiskLevel.R3_HIGH
    assert record is not None
    assert record.new_risk == RiskLevel.R3_HIGH


def test_material_model_disagreement_escalates() -> None:
    escalator = RuntimeRiskEscalator.default()
    event = _event(RiskRuntimeEventType.MODEL_DISAGREEMENT, material=True)
    new_risk, record = escalator.escalate(RiskLevel.R2_NORMAL, event)
    assert new_risk == RiskLevel.R3_HIGH


def test_non_material_model_disagreement_does_not_escalate() -> None:
    escalator = RuntimeRiskEscalator.default()
    event = _event(RiskRuntimeEventType.MODEL_DISAGREEMENT, material=False)
    new_risk, record = escalator.escalate(RiskLevel.R2_NORMAL, event)
    assert new_risk == RiskLevel.R2_NORMAL


def test_authority_violation_escalates_to_r4() -> None:
    escalator = RuntimeRiskEscalator.default()
    event = _event(RiskRuntimeEventType.AUTHORITY_VIOLATION)
    new_risk, record = escalator.escalate(RiskLevel.R1_LOW, event)
    assert new_risk == RiskLevel.R4_CRITICAL_AUTHORITY


def test_unexpected_file_touch_escalates_to_r3() -> None:
    escalator = RuntimeRiskEscalator.default()
    event = _event(RiskRuntimeEventType.UNEXPECTED_FILE_TOUCH)
    new_risk, record = escalator.escalate(RiskLevel.R1_LOW, event)
    assert new_risk == RiskLevel.R3_HIGH


def test_merge_conflict_escalates_to_r3() -> None:
    escalator = RuntimeRiskEscalator.default()
    event = _event(RiskRuntimeEventType.MERGE_CONFLICT)
    new_risk, record = escalator.escalate(RiskLevel.R1_LOW, event)
    assert new_risk == RiskLevel.R3_HIGH


def test_integration_anomaly_escalates_to_r3() -> None:
    escalator = RuntimeRiskEscalator.default()
    event = _event(RiskRuntimeEventType.INTEGRATION_ANOMALY)
    new_risk, record = escalator.escalate(RiskLevel.R1_LOW, event)
    assert new_risk == RiskLevel.R3_HIGH


def test_repair_loop_escalates_at_threshold() -> None:
    escalator = RuntimeRiskEscalator.default()
    event = _event(RiskRuntimeEventType.REPAIR_LOOP, count=3, threshold=3)
    new_risk, record = escalator.escalate(RiskLevel.R2_NORMAL, event)
    assert new_risk == RiskLevel.R3_HIGH


def test_repair_loop_below_threshold_stays_lower() -> None:
    escalator = RuntimeRiskEscalator.default()
    event = _event(RiskRuntimeEventType.REPAIR_LOOP, count=1, threshold=3)
    new_risk, record = escalator.escalate(RiskLevel.R2_NORMAL, event)
    assert new_risk == RiskLevel.R2_NORMAL


def test_escalation_never_lowers_risk() -> None:
    escalator = RuntimeRiskEscalator.default()
    event = _event(RiskRuntimeEventType.TEST_FAILURE, count=1, threshold=5)
    new_risk, record = escalator.escalate(RiskLevel.R4_CRITICAL_AUTHORITY, event)
    assert new_risk == RiskLevel.R4_CRITICAL_AUTHORITY


def test_event_ordering_produces_same_final_risk() -> None:
    escalator = RuntimeRiskEscalator.default()
    e1 = _event(RiskRuntimeEventType.TEST_FAILURE, count=5, threshold=3)
    e2 = _event(RiskRuntimeEventType.AUTHORITY_VIOLATION)
    r1, _ = escalator.apply_all(RiskLevel.R2_NORMAL, (e1, e2))
    r2, _ = escalator.apply_all(RiskLevel.R2_NORMAL, (e2, e1))
    assert r1 == r2 == RiskLevel.R4_CRITICAL_AUTHORITY


def test_event_count_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RiskRuntimeEvent(
            event_type=RiskRuntimeEventType.TEST_FAILURE,
            material=True,
            evidence="x",
            count=-1,
        )


def test_event_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        RiskRuntimeEvent(
            event_type=RiskRuntimeEventType.TEST_FAILURE,
            material=True,
            evidence="x",
            threshold=0,
        )


def test_event_evidence_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        RiskRuntimeEvent(
            event_type=RiskRuntimeEventType.TEST_FAILURE,
            material=True,
            evidence="",
        )


def test_apply_all_records_sequence() -> None:
    escalator = RuntimeRiskEscalator.default()
    events = (
        _event(RiskRuntimeEventType.TEST_FAILURE, count=3, threshold=3),
        _event(RiskRuntimeEventType.REPAIR_LOOP, count=3, threshold=3),
    )
    _, records = escalator.apply_all(RiskLevel.R2_NORMAL, events, start_sequence=5)
    assert len(records) == 2
    assert records[0].sequence == 5
    assert records[1].sequence == 6


def test_unexpected_file_touch_without_paths_is_r3() -> None:
    escalator = RuntimeRiskEscalator.default()
    event = _event(RiskRuntimeEventType.UNEXPECTED_FILE_TOUCH)
    new_risk, _ = escalator.escalate(RiskLevel.R1_LOW, event)
    assert new_risk == RiskLevel.R3_HIGH


def test_unexpected_authority_file_touch_is_r4() -> None:
    from src.risk import AuthoritySensitivePolicy

    escalator = RuntimeRiskEscalator.default(
        authority_policy=AuthoritySensitivePolicy.default(),
    )
    event = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.UNEXPECTED_FILE_TOUCH,
        material=True,
        evidence="touched authority file",
        affected_paths=("docs/PROJECT_STATE.json",),
    )
    new_risk, _ = escalator.escalate(RiskLevel.R1_LOW, event)
    assert new_risk == RiskLevel.R4_CRITICAL_AUTHORITY


def test_unexpected_security_file_touch_is_at_least_r3() -> None:
    from src.risk import SecuritySensitivePolicy

    escalator = RuntimeRiskEscalator.default(
        security_policy=SecuritySensitivePolicy.default(),
    )
    event = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.UNEXPECTED_FILE_TOUCH,
        material=True,
        evidence="touched security file",
        affected_paths=("src/security/secrets.py",),
    )
    new_risk, _ = escalator.escalate(RiskLevel.R1_LOW, event)
    assert new_risk >= RiskLevel.R3_HIGH


def test_unexpected_authority_path_normalized_bypass_still_r4() -> None:
    from src.risk import AuthoritySensitivePolicy

    escalator = RuntimeRiskEscalator.default(
        authority_policy=AuthoritySensitivePolicy.default(),
    )
    event = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.UNEXPECTED_FILE_TOUCH,
        material=True,
        evidence="touched authority file",
        affected_paths=("./docs/../docs/PROJECT_STATE.json",),
    )
    new_risk, _ = escalator.escalate(RiskLevel.R1_LOW, event)
    assert new_risk == RiskLevel.R4_CRITICAL_AUTHORITY


def test_evidence_text_alone_does_not_drive_path_decisions() -> None:
    escalator = RuntimeRiskEscalator.default()
    event = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.UNEXPECTED_FILE_TOUCH,
        material=True,
        evidence="docs/PROJECT_STATE.json was touched",
    )
    new_risk, _ = escalator.escalate(RiskLevel.R1_LOW, event)
    assert new_risk == RiskLevel.R3_HIGH


def test_configured_test_failure_threshold_is_authoritative() -> None:
    escalator = RuntimeRiskEscalator(test_failure_threshold=3)
    event = _event(RiskRuntimeEventType.TEST_FAILURE, count=1, threshold=1)
    new_risk, record = escalator.escalate(RiskLevel.R2_NORMAL, event)
    assert new_risk == RiskLevel.R2_NORMAL
    assert record is not None
    assert record.threshold == 3


def test_repeated_test_failure_at_configured_threshold_escalates() -> None:
    escalator = RuntimeRiskEscalator(test_failure_threshold=3)
    event = _event(RiskRuntimeEventType.TEST_FAILURE, count=3, threshold=1)
    new_risk, record = escalator.escalate(RiskLevel.R2_NORMAL, event)
    assert new_risk == RiskLevel.R3_HIGH
    assert record is not None
    assert record.threshold == 3


def test_configured_repair_loop_threshold_is_authoritative() -> None:
    escalator = RuntimeRiskEscalator(repair_loop_threshold=4)
    event = _event(RiskRuntimeEventType.REPAIR_LOOP, count=2, threshold=1)
    new_risk, record = escalator.escalate(RiskLevel.R2_NORMAL, event)
    assert new_risk == RiskLevel.R2_NORMAL
    assert record is not None
    assert record.threshold == 4


def test_repair_loop_at_configured_threshold_escalates() -> None:
    escalator = RuntimeRiskEscalator(repair_loop_threshold=4)
    event = _event(RiskRuntimeEventType.REPAIR_LOOP, count=4, threshold=1)
    new_risk, record = escalator.escalate(RiskLevel.R2_NORMAL, event)
    assert new_risk == RiskLevel.R3_HIGH
    assert record is not None
    assert record.threshold == 4
