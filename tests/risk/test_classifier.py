from __future__ import annotations

from dataclasses import replace

from src.policy.risk import RiskLevel
from src.risk import (
    InitialRiskClassifier,
    RiskAssessmentRequest,
    RiskFactorCode,
    RuntimeRiskEscalator,
)
from src.routing.roles import ExecutionRole


def _request(
    changed_files: tuple[str, ...] = (),
    changed_lines: int = 0,
    operation: str = "modify",
    task_class: str = "default",
    runtime_events: tuple = (),
    project_policy: dict | None = None,
    generated_files: tuple[str, ...] = (),
    dependency_changes: tuple[str, ...] = (),
) -> RiskAssessmentRequest:
    return RiskAssessmentRequest(
        project_id="project-a",
        task_id="task-1",
        role=ExecutionRole.CODING,
        task_class=task_class,
        operation=operation,
        changed_files=changed_files,
        changed_lines_estimate=changed_lines,
        runtime_events=runtime_events,
        project_policy=project_policy or {},
        generated_files=generated_files,
        dependency_changes=dependency_changes,
    )


def test_trivial_docs_change_is_r0() -> None:
    classifier = InitialRiskClassifier.default()
    request = _request(changed_files=("docs/README.md",), changed_lines=10)
    result = classifier.classify(request)
    assert result.final_risk == RiskLevel.R0_TRIVIAL


def test_narrow_isolated_change_is_r1() -> None:
    classifier = InitialRiskClassifier.default()
    request = _request(changed_files=("src/providers/openai/adapter.py",), changed_lines=50)
    result = classifier.classify(request)
    assert result.final_risk == RiskLevel.R1_LOW


def test_ordinary_multiple_files_is_r2() -> None:
    classifier = InitialRiskClassifier.default()
    request = _request(
        changed_files=(
            "src/providers/openai/adapter.py",
            "src/providers/openai/request.py",
            "src/providers/openai/response.py",
        ),
        changed_lines=150,
    )
    result = classifier.classify(request)
    assert result.final_risk == RiskLevel.R2_NORMAL


def test_project_state_modification_is_r4() -> None:
    classifier = InitialRiskClassifier.default()
    request = _request(changed_files=("docs/PROJECT_STATE.json",), changed_lines=5)
    result = classifier.classify(request)
    assert result.final_risk == RiskLevel.R4_CRITICAL_AUTHORITY
    assert any(f.code == RiskFactorCode.AUTHORITY_SENSITIVE for f in result.factors)


def test_roadmap_modification_is_r4() -> None:
    classifier = InitialRiskClassifier.default()
    request = _request(changed_files=("docs/OMNIFORGE_FULL_ROADMAP_v1.0.md",), changed_lines=5)
    result = classifier.classify(request)
    assert result.final_risk == RiskLevel.R4_CRITICAL_AUTHORITY


def test_authority_path_normalization_no_bypass() -> None:
    classifier = InitialRiskClassifier.default()
    request = _request(changed_files=("./docs/../docs/PROJECT_STATE.json",), changed_lines=5)
    result = classifier.classify(request)
    assert result.final_risk == RiskLevel.R4_CRITICAL_AUTHORITY


def test_read_authority_is_not_modify() -> None:
    classifier = InitialRiskClassifier.default()
    request = _request(
        changed_files=("docs/PROJECT_STATE.json",),
        changed_lines=0,
        operation="read",
    )
    result = classifier.classify(request)
    assert result.final_risk == RiskLevel.R1_LOW


def test_security_credential_change_is_at_least_r3() -> None:
    classifier = InitialRiskClassifier.default()
    request = _request(changed_files=("src/security/secrets.py",), changed_lines=20)
    result = classifier.classify(request)
    assert result.final_risk >= RiskLevel.R3_HIGH
    assert any(f.code == RiskFactorCode.SECURITY_SENSITIVE for f in result.factors)


def test_ordinary_text_password_does_not_trigger_security() -> None:
    classifier = InitialRiskClassifier.default()
    request = _request(
        changed_files=("src/providers/openai/password_policy_comment.py",), changed_lines=10
    )
    result = classifier.classify(request)
    assert not any(f.code == RiskFactorCode.SECURITY_SENSITIVE for f in result.factors)


def test_broad_central_interface_change_is_architectural() -> None:
    classifier = InitialRiskClassifier.default()
    request = _request(changed_files=("src/routing/capabilities.py",), changed_lines=10)
    result = classifier.classify(request)
    assert any(f.code == RiskFactorCode.ARCHITECTURAL_CHANGE for f in result.factors)


def test_cross_subsystem_architectural_signal() -> None:
    classifier = InitialRiskClassifier.default()
    request = _request(
        changed_files=(
            "src/providers/openai/adapter.py",
            "src/routing/capabilities.py",
            "src/context/schema.py",
            "src/recovery/state_machine.py",
        ),
        changed_lines=100,
    )
    result = classifier.classify(request)
    assert result.final_risk >= RiskLevel.R3_HIGH


def test_generated_only_change_avoids_architectural_escalation() -> None:
    classifier = InitialRiskClassifier.default()
    request = _request(
        changed_files=("src/generated/big_lock.py",),
        generated_files=("src/generated/big_lock.py",),
        changed_lines=10000,
    )
    result = classifier.classify(request)
    assert not any(f.code == RiskFactorCode.ARCHITECTURAL_CHANGE for f in result.factors)


def test_classification_deterministic_across_orderings() -> None:
    classifier = InitialRiskClassifier.default()
    files_a = ("src/a.py", "src/b.py", "src/c.py")
    files_b = ("src/c.py", "src/a.py", "src/b.py")
    r1 = classifier.classify(_request(changed_files=files_a, changed_lines=100))
    r2 = classifier.classify(_request(changed_files=files_b, changed_lines=100))
    assert r1.final_risk == r2.final_risk
    assert r1.fingerprint == r2.fingerprint


def test_repeated_classification_stable() -> None:
    classifier = InitialRiskClassifier.default()
    request = _request(changed_files=("src/foo.py",), changed_lines=50)
    results = [classifier.classify(request) for _ in range(100)]
    assert all(r.final_risk == results[0].final_risk for r in results)
    assert all(r.fingerprint == results[0].fingerprint for r in results)


def test_explanation_has_no_hidden_reasoning() -> None:
    classifier = InitialRiskClassifier.default()
    request = _request(changed_files=("docs/PROJECT_STATE.json",), changed_lines=5)
    result = classifier.classify(request)
    assert result.final_risk.name in result.explanation
    assert "deterministic" in result.explanation.lower() or "factor" in result.explanation.lower()


def test_runtime_authority_violation_escalates_to_r4() -> None:
    from src.risk import RiskRuntimeEvent, RiskRuntimeEventType

    classifier = InitialRiskClassifier.default()
    event = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.AUTHORITY_VIOLATION,
        material=True,
        evidence="attempted roadmap mutation detected",
    )
    request = _request(changed_files=("src/foo.py",), runtime_events=(event,))
    result = classifier.classify(request)
    assert result.final_risk == RiskLevel.R4_CRITICAL_AUTHORITY


def test_runtime_escalation_never_lowers_risk() -> None:
    from src.risk import RiskRuntimeEvent, RiskRuntimeEventType

    classifier = InitialRiskClassifier.default()
    event = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.TEST_FAILURE,
        material=True,
        evidence="test passed",
        count=1,
        threshold=5,
    )
    request = _request(
        changed_files=("docs/PROJECT_STATE.json",),
        runtime_events=(event,),
    )
    result = classifier.classify(request)
    assert result.final_risk == RiskLevel.R4_CRITICAL_AUTHORITY


def test_event_ordering_produces_same_maximum_risk() -> None:
    from src.risk import RiskRuntimeEvent, RiskRuntimeEventType

    classifier = InitialRiskClassifier.default()
    e1 = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.TEST_FAILURE,
        material=True,
        evidence="repeated failures",
        count=5,
        threshold=3,
    )
    e2 = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.AUTHORITY_VIOLATION,
        material=True,
        evidence="roadmap mutation",
    )
    r1 = classifier.classify(_request(changed_files=("src/foo.py",), runtime_events=(e1, e2)))
    r2 = classifier.classify(_request(changed_files=("src/foo.py",), runtime_events=(e2, e1)))
    assert r1.final_risk == r2.final_risk == RiskLevel.R4_CRITICAL_AUTHORITY


def test_runtime_event_changes_assessment_fingerprint() -> None:
    from src.risk import RiskRuntimeEvent, RiskRuntimeEventType

    classifier = InitialRiskClassifier.default()
    base = classifier.classify(_request(changed_files=("src/foo.py",)))
    event = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.MODEL_DISAGREEMENT,
        material=True,
        evidence="material disagreement on approach",
    )
    with_event = classifier.classify(
        _request(changed_files=("src/foo.py",), runtime_events=(event,))
    )
    assert with_event.final_risk == RiskLevel.R3_HIGH
    assert base.fingerprint != with_event.fingerprint


def test_runtime_events_included_in_fingerprint_cover_decision_inputs() -> None:
    from src.risk import RiskRuntimeEvent, RiskRuntimeEventType

    classifier = InitialRiskClassifier.default()
    event1 = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.MODEL_DISAGREEMENT,
        material=False,
        evidence="not material",
    )
    event2 = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.MODEL_DISAGREEMENT,
        material=True,
        evidence="material",
    )
    r1 = classifier.classify(_request(changed_files=("src/foo.py",), runtime_events=(event1,)))
    r2 = classifier.classify(_request(changed_files=("src/foo.py",), runtime_events=(event2,)))
    assert r1.fingerprint != r2.fingerprint
    assert r2.final_risk == RiskLevel.R3_HIGH


def test_unexpected_authority_touch_at_runtime_reaches_r4() -> None:
    from src.risk import RiskRuntimeEvent, RiskRuntimeEventType

    classifier = InitialRiskClassifier.default()
    event = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.UNEXPECTED_FILE_TOUCH,
        material=True,
        evidence="runtime touch",
        affected_paths=("docs/PROJECT_STATE.json",),
    )
    result = classifier.classify(_request(changed_files=("src/foo.py",), runtime_events=(event,)))
    assert result.final_risk == RiskLevel.R4_CRITICAL_AUTHORITY


def test_unexpected_security_touch_at_runtime_is_at_least_r3() -> None:
    from src.risk import RiskRuntimeEvent, RiskRuntimeEventType

    classifier = InitialRiskClassifier.default()
    event = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.UNEXPECTED_FILE_TOUCH,
        material=True,
        evidence="runtime touch",
        affected_paths=("src/security/secrets.py",),
    )
    result = classifier.classify(_request(changed_files=("src/foo.py",), runtime_events=(event,)))
    assert result.final_risk >= RiskLevel.R3_HIGH


def test_dict_runtime_event_caller_threshold_ignored_for_fingerprint() -> None:
    """Dict-form TEST_FAILURE caller threshold must not affect the fingerprint
    when the engine threshold is authoritative."""
    classifier = InitialRiskClassifier.default()
    low = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            {
                "event_type": "test_failure",
                "material": True,
                "evidence": "failures",
                "count": 3,
                "threshold": 1,
            },
        ),
    )
    high = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            {
                "event_type": "test_failure",
                "material": True,
                "evidence": "failures",
                "count": 3,
                "threshold": 99,
            },
        ),
    )
    r_low = classifier.classify(low)
    r_high = classifier.classify(high)
    assert r_low.final_risk == r_high.final_risk == RiskLevel.R3_HIGH
    assert r_low.fingerprint == r_high.fingerprint


def test_dict_repair_loop_caller_threshold_ignored_for_fingerprint() -> None:
    classifier = InitialRiskClassifier.default()
    low = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            {
                "event_type": "repair_loop",
                "material": True,
                "evidence": "repairs",
                "count": 3,
                "threshold": 1,
            },
        ),
    )
    high = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            {
                "event_type": "repair_loop",
                "material": True,
                "evidence": "repairs",
                "count": 3,
                "threshold": 99,
            },
        ),
    )
    r_low = classifier.classify(low)
    r_high = classifier.classify(high)
    assert r_low.final_risk == r_high.final_risk == RiskLevel.R3_HIGH
    assert r_low.fingerprint == r_high.fingerprint


def test_dict_test_failure_engine_threshold_change_changes_fingerprint() -> None:
    """Changing the authoritative engine threshold must change the fingerprint
    even when caller dict thresholds differ."""
    base = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            {
                "event_type": "test_failure",
                "material": True,
                "evidence": "failures",
                "count": 3,
                "threshold": 1,
            },
        ),
    )
    classifier_t3 = InitialRiskClassifier.default()
    classifier_t5 = replace(
        classifier_t3,
        runtime_escalator=RuntimeRiskEscalator(
            test_failure_threshold=5,
            authority_policy=classifier_t3.authority_policy,
            security_policy=classifier_t3.security_policy,
        ),
    )
    r3 = classifier_t3.classify(base)
    r5 = classifier_t5.classify(base)
    assert r3.fingerprint != r5.fingerprint
    assert r3.final_risk == RiskLevel.R3_HIGH
    assert r5.final_risk == RiskLevel.R1_LOW


def test_typed_and_dict_runtime_events_equivalent() -> None:
    """A typed RiskRuntimeEvent and an equivalent dict event must produce the same
    final risk and fingerprint."""
    from src.risk import RiskRuntimeEvent, RiskRuntimeEventType

    classifier = InitialRiskClassifier.default()
    typed = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            RiskRuntimeEvent(
                event_type=RiskRuntimeEventType.TEST_FAILURE,
                material=True,
                evidence="failures",
                count=3,
                threshold=99,
            ),
        ),
    )
    dict_form = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            {
                "event_type": "test_failure",
                "material": True,
                "evidence": "failures",
                "count": 3,
                "threshold": 99,
            },
        ),
    )
    r_typed = classifier.classify(typed)
    r_dict = classifier.classify(dict_form)
    assert r_typed.final_risk == r_dict.final_risk == RiskLevel.R3_HIGH
    assert r_typed.fingerprint == r_dict.fingerprint


def test_dict_repair_loop_engine_threshold_change_changes_fingerprint() -> None:
    base = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            {
                "event_type": "repair_loop",
                "material": True,
                "evidence": "repairs",
                "count": 3,
                "threshold": 1,
            },
        ),
    )
    classifier_t3 = InitialRiskClassifier.default()
    classifier_t5 = replace(
        classifier_t3,
        runtime_escalator=RuntimeRiskEscalator(
            repair_loop_threshold=5,
            authority_policy=classifier_t3.authority_policy,
            security_policy=classifier_t3.security_policy,
        ),
    )
    r3 = classifier_t3.classify(base)
    r5 = classifier_t5.classify(base)
    assert r3.fingerprint != r5.fingerprint
    assert r3.final_risk == RiskLevel.R3_HIGH
    assert r5.final_risk == RiskLevel.R1_LOW


def test_dict_test_failure_count_threshold_crossing_changes_fingerprint() -> None:
    """A count below the engine threshold versus at/above it must produce different
    fingerprints and different escalation results."""
    classifier = InitialRiskClassifier.default()
    below = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            {
                "event_type": "test_failure",
                "material": True,
                "evidence": "failures",
                "count": 2,
                "threshold": 1,
            },
        ),
    )
    at_threshold = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            {
                "event_type": "test_failure",
                "material": True,
                "evidence": "failures",
                "count": 3,
                "threshold": 1,
            },
        ),
    )
    r_below = classifier.classify(below)
    r_at = classifier.classify(at_threshold)
    assert r_below.fingerprint != r_at.fingerprint
    assert r_below.final_risk == RiskLevel.R1_LOW
    assert r_at.final_risk == RiskLevel.R3_HIGH
