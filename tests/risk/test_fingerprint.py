from __future__ import annotations

from src.policy.risk import RiskLevel
from src.risk import (
    RiskAssessmentRequest,
    RiskRuntimeEvent,
    RiskRuntimeEventType,
    risk_assessment_fingerprint,
)
from src.routing.roles import ExecutionRole


def _request(
    changed_files: tuple[str, ...] = (),
    project_policy: dict | None = None,
) -> RiskAssessmentRequest:
    return RiskAssessmentRequest(
        project_id="project-a",
        task_id="task-1",
        role=ExecutionRole.CODING,
        task_class="default",
        changed_files=changed_files,
        project_policy=project_policy or {},
    )


def test_same_input_same_fingerprint() -> None:
    req = _request(changed_files=("src/a.py", "src/b.py"))
    assert risk_assessment_fingerprint(req) == risk_assessment_fingerprint(req)


def test_fingerprint_invariant_to_file_order() -> None:
    req1 = _request(changed_files=("src/a.py", "src/b.py", "src/c.py"))
    req2 = _request(changed_files=("src/c.py", "src/a.py", "src/b.py"))
    assert risk_assessment_fingerprint(req1) == risk_assessment_fingerprint(req2)


def test_material_path_change_alters_fingerprint() -> None:
    req1 = _request(changed_files=("src/a.py",))
    req2 = _request(changed_files=("src/b.py",))
    assert risk_assessment_fingerprint(req1) != risk_assessment_fingerprint(req2)


def test_baseline_risk_change_alters_fingerprint() -> None:
    req1 = RiskAssessmentRequest(
        project_id="p",
        task_id="t",
        role=ExecutionRole.CODING,
        task_class="default",
        baseline_risk=RiskLevel.R1_LOW,
    )
    req2 = RiskAssessmentRequest(
        project_id="p",
        task_id="t",
        role=ExecutionRole.CODING,
        task_class="default",
        baseline_risk=RiskLevel.R2_NORMAL,
    )
    assert risk_assessment_fingerprint(req1) != risk_assessment_fingerprint(req2)


def test_project_policy_change_alters_fingerprint() -> None:
    req1 = _request(project_policy={"minimum_risk": 1})
    req2 = _request(project_policy={"minimum_risk": 3})
    assert risk_assessment_fingerprint(req1) != risk_assessment_fingerprint(req2)


def test_role_change_alters_fingerprint() -> None:
    req1 = RiskAssessmentRequest(
        project_id="p",
        task_id="t",
        role=ExecutionRole.CODING,
        task_class="default",
    )
    req2 = RiskAssessmentRequest(
        project_id="p",
        task_id="t",
        role=ExecutionRole.REVIEW,
        task_class="default",
    )
    assert risk_assessment_fingerprint(req1) != risk_assessment_fingerprint(req2)


def test_operation_change_alters_fingerprint() -> None:
    req1 = RiskAssessmentRequest(
        project_id="p",
        task_id="t",
        role=ExecutionRole.CODING,
        task_class="default",
        operation="modify",
    )
    req2 = RiskAssessmentRequest(
        project_id="p",
        task_id="t",
        role=ExecutionRole.CODING,
        task_class="default",
        operation="delete",
    )
    assert risk_assessment_fingerprint(req1) != risk_assessment_fingerprint(req2)


def test_changed_lines_change_alters_fingerprint() -> None:
    req1 = RiskAssessmentRequest(
        project_id="p",
        task_id="t",
        role=ExecutionRole.CODING,
        task_class="default",
        changed_files=("src/a.py",),
        changed_lines_estimate=10,
    )
    req2 = RiskAssessmentRequest(
        project_id="p",
        task_id="t",
        role=ExecutionRole.CODING,
        task_class="default",
        changed_files=("src/a.py",),
        changed_lines_estimate=100,
    )
    assert risk_assessment_fingerprint(req1) != risk_assessment_fingerprint(req2)


def test_repeated_fingerprint_stable() -> None:
    req = _request(changed_files=("src/a.py",))
    fingerprints = [risk_assessment_fingerprint(req) for _ in range(100)]
    assert all(f == fingerprints[0] for f in fingerprints)


def test_normalization_does_not_change_fingerprint() -> None:
    req1 = _request(changed_files=("./src/a.py",))
    req2 = _request(changed_files=("src/a.py",))
    assert risk_assessment_fingerprint(req1) == risk_assessment_fingerprint(req2)


def test_runtime_event_changes_fingerprint() -> None:
    base = _request(changed_files=("src/foo.py",))
    with_event = RiskAssessmentRequest(
        project_id="project-a",
        task_id="task-1",
        role=ExecutionRole.CODING,
        task_class="default",
        changed_files=("src/foo.py",),
        runtime_events=(
            RiskRuntimeEvent(
                event_type=RiskRuntimeEventType.MODEL_DISAGREEMENT,
                material=True,
                evidence="material disagreement",
            ),
        ),
    )
    assert risk_assessment_fingerprint(base) != risk_assessment_fingerprint(with_event)


def test_authority_violation_changes_fingerprint() -> None:
    base = _request(changed_files=("src/foo.py",))
    with_violation = RiskAssessmentRequest(
        project_id="project-a",
        task_id="task-1",
        role=ExecutionRole.CODING,
        task_class="default",
        changed_files=("src/foo.py",),
        runtime_events=(
            RiskRuntimeEvent(
                event_type=RiskRuntimeEventType.AUTHORITY_VIOLATION,
                material=True,
                evidence="authority violation",
            ),
        ),
    )
    assert risk_assessment_fingerprint(base) != risk_assessment_fingerprint(with_violation)


def test_runtime_event_order_does_not_change_fingerprint() -> None:
    e1 = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.TEST_FAILURE,
        material=True,
        evidence="failures",
        count=2,
    )
    e2 = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.REPAIR_LOOP,
        material=True,
        evidence="repairs",
        count=2,
    )
    req1 = RiskAssessmentRequest(
        project_id="project-a",
        task_id="task-1",
        role=ExecutionRole.CODING,
        task_class="default",
        changed_files=("src/foo.py",),
        runtime_events=(e1, e2),
    )
    req2 = RiskAssessmentRequest(
        project_id="project-a",
        task_id="task-1",
        role=ExecutionRole.CODING,
        task_class="default",
        changed_files=("src/foo.py",),
        runtime_events=(e2, e1),
    )
    assert risk_assessment_fingerprint(req1) == risk_assessment_fingerprint(req2)


def test_fingerprint_excludes_event_evidence_text() -> None:
    req1 = RiskAssessmentRequest(
        project_id="project-a",
        task_id="task-1",
        role=ExecutionRole.CODING,
        task_class="default",
        changed_files=("src/foo.py",),
        runtime_events=(
            RiskRuntimeEvent(
                event_type=RiskRuntimeEventType.MODEL_DISAGREEMENT,
                material=True,
                evidence="description one",
            ),
        ),
    )
    req2 = RiskAssessmentRequest(
        project_id="project-a",
        task_id="task-1",
        role=ExecutionRole.CODING,
        task_class="default",
        changed_files=("src/foo.py",),
        runtime_events=(
            RiskRuntimeEvent(
                event_type=RiskRuntimeEventType.MODEL_DISAGREEMENT,
                material=True,
                evidence="description two",
            ),
        ),
    )
    assert risk_assessment_fingerprint(req1) == risk_assessment_fingerprint(req2)


def test_affected_paths_in_event_change_fingerprint() -> None:
    req1 = RiskAssessmentRequest(
        project_id="project-a",
        task_id="task-1",
        role=ExecutionRole.CODING,
        task_class="default",
        changed_files=("src/foo.py",),
        runtime_events=(
            RiskRuntimeEvent(
                event_type=RiskRuntimeEventType.UNEXPECTED_FILE_TOUCH,
                material=True,
                evidence="touch",
                affected_paths=("src/a.py",),
            ),
        ),
    )
    req2 = RiskAssessmentRequest(
        project_id="project-a",
        task_id="task-1",
        role=ExecutionRole.CODING,
        task_class="default",
        changed_files=("src/foo.py",),
        runtime_events=(
            RiskRuntimeEvent(
                event_type=RiskRuntimeEventType.UNEXPECTED_FILE_TOUCH,
                material=True,
                evidence="touch",
                affected_paths=("src/b.py",),
            ),
        ),
    )
    assert risk_assessment_fingerprint(req1) != risk_assessment_fingerprint(req2)
