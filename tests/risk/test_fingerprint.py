from __future__ import annotations

from typing import Any

from src.policy.risk import RiskLevel
from src.risk import (
    ArchitectureThresholds,
    AuthoritySensitivePolicy,
    ProjectRiskPolicy,
    RiskAssessmentRequest,
    RiskDecisionInputs,
    RiskRuntimeEvent,
    RiskRuntimeEventType,
    RuntimeRiskEscalator,
    SecuritySensitivePolicy,
    risk_assessment_fingerprint,
)
from src.routing.roles import ExecutionRole


def _request(
    changed_files: tuple[str, ...] = (),
    project_policy: dict | None = None,
    **kwargs: Any,
) -> RiskAssessmentRequest:
    return RiskAssessmentRequest(
        project_id="project-a",
        task_id="task-1",
        role=kwargs.pop("role", ExecutionRole.CODING),
        task_class="default",
        changed_files=changed_files,
        project_policy=project_policy or {},
        **kwargs,
    )


def _inputs(
    request: RiskAssessmentRequest,
    project_policy_data: dict | None = None,
) -> RiskDecisionInputs:
    """Build effective decision inputs from the classifier's effective policy."""
    project_policy = ProjectRiskPolicy.from_dict(project_policy_data or {})
    authority_policy = project_policy.authority_policy
    security_policy = project_policy.security_policy
    architecture_thresholds = ArchitectureThresholds.from_dict(
        project_policy.architecture_thresholds
    )
    runtime_escalator = RuntimeRiskEscalator.default(
        authority_policy=authority_policy,
        security_policy=security_policy,
    )
    return RiskDecisionInputs(
        request=request,
        project_policy=project_policy,
        authority_policy=authority_policy,
        security_policy=security_policy,
        architecture_thresholds=architecture_thresholds,
        runtime_escalator=runtime_escalator,
        normalized_runtime_events=tuple(request.runtime_events),
    )


def test_same_input_same_fingerprint() -> None:
    req = _request(changed_files=("src/a.py", "src/b.py"))
    inputs = _inputs(req)
    assert risk_assessment_fingerprint(inputs) == risk_assessment_fingerprint(inputs)


def test_fingerprint_invariant_to_file_order() -> None:
    req1 = _request(changed_files=("src/a.py", "src/b.py", "src/c.py"))
    req2 = _request(changed_files=("src/c.py", "src/a.py", "src/b.py"))
    assert risk_assessment_fingerprint(_inputs(req1)) == risk_assessment_fingerprint(_inputs(req2))


def test_material_path_change_alters_fingerprint() -> None:
    req1 = _request(changed_files=("src/a.py",))
    req2 = _request(changed_files=("src/b.py",))
    assert risk_assessment_fingerprint(_inputs(req1)) != risk_assessment_fingerprint(_inputs(req2))


def test_baseline_risk_change_alters_fingerprint() -> None:
    req1 = _request(baseline_risk=RiskLevel.R1_LOW)
    req2 = _request(baseline_risk=RiskLevel.R2_NORMAL)
    assert risk_assessment_fingerprint(_inputs(req1)) != risk_assessment_fingerprint(_inputs(req2))


def test_project_policy_change_alters_fingerprint() -> None:
    req = _request()
    inputs1 = _inputs(req, project_policy_data={"minimum_risk": 1})
    inputs2 = _inputs(req, project_policy_data={"minimum_risk": 3})
    assert risk_assessment_fingerprint(inputs1) != risk_assessment_fingerprint(inputs2)


def test_role_change_alters_fingerprint() -> None:
    req1 = _request(role=ExecutionRole.CODING)
    req2 = _request(role=ExecutionRole.REVIEW)
    assert risk_assessment_fingerprint(_inputs(req1)) != risk_assessment_fingerprint(_inputs(req2))


def test_operation_change_alters_fingerprint() -> None:
    req1 = _request(operation="modify")
    req2 = _request(operation="delete")
    assert risk_assessment_fingerprint(_inputs(req1)) != risk_assessment_fingerprint(_inputs(req2))


def test_changed_lines_change_alters_fingerprint() -> None:
    req1 = _request(changed_files=("src/a.py",), changed_lines_estimate=10)
    req2 = _request(changed_files=("src/a.py",), changed_lines_estimate=100)
    assert risk_assessment_fingerprint(_inputs(req1)) != risk_assessment_fingerprint(_inputs(req2))


def test_repeated_fingerprint_stable() -> None:
    inputs = _inputs(_request(changed_files=("src/a.py",)))
    fingerprints = [risk_assessment_fingerprint(inputs) for _ in range(100)]
    assert all(f == fingerprints[0] for f in fingerprints)


def test_normalization_does_not_change_fingerprint() -> None:
    req1 = _request(changed_files=("./src/a.py",))
    req2 = _request(changed_files=("src/a.py",))
    assert risk_assessment_fingerprint(_inputs(req1)) == risk_assessment_fingerprint(_inputs(req2))


def test_runtime_event_changes_fingerprint() -> None:
    base = _request(changed_files=("src/foo.py",))
    with_event = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            RiskRuntimeEvent(
                event_type=RiskRuntimeEventType.MODEL_DISAGREEMENT,
                material=True,
                evidence="material disagreement",
            ),
        ),
    )
    assert risk_assessment_fingerprint(_inputs(base)) != risk_assessment_fingerprint(
        _inputs(with_event)
    )


def test_authority_violation_changes_fingerprint() -> None:
    base = _request(changed_files=("src/foo.py",))
    with_violation = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            RiskRuntimeEvent(
                event_type=RiskRuntimeEventType.AUTHORITY_VIOLATION,
                material=True,
                evidence="authority violation",
            ),
        ),
    )
    assert risk_assessment_fingerprint(_inputs(base)) != risk_assessment_fingerprint(
        _inputs(with_violation)
    )


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
    req1 = _request(changed_files=("src/foo.py",), runtime_events=(e1, e2))
    req2 = _request(changed_files=("src/foo.py",), runtime_events=(e2, e1))
    assert risk_assessment_fingerprint(_inputs(req1)) == risk_assessment_fingerprint(_inputs(req2))


def test_fingerprint_excludes_event_evidence_text() -> None:
    req1 = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            RiskRuntimeEvent(
                event_type=RiskRuntimeEventType.MODEL_DISAGREEMENT,
                material=True,
                evidence="description one",
            ),
        ),
    )
    req2 = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            RiskRuntimeEvent(
                event_type=RiskRuntimeEventType.MODEL_DISAGREEMENT,
                material=True,
                evidence="description two",
            ),
        ),
    )
    assert risk_assessment_fingerprint(_inputs(req1)) == risk_assessment_fingerprint(_inputs(req2))


def test_affected_paths_in_event_change_fingerprint() -> None:
    req1 = _request(
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
    req2 = _request(
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
    assert risk_assessment_fingerprint(_inputs(req1)) != risk_assessment_fingerprint(_inputs(req2))


def test_effective_policy_differs_in_fingerprint() -> None:
    """The fingerprint must reflect the policy that actually drives the decision."""
    req = _request(changed_files=("src/foo.py",))
    inputs_default = _inputs(req)
    inputs_project = _inputs(req, project_policy_data={"minimum_risk": 4})
    assert risk_assessment_fingerprint(inputs_default) != risk_assessment_fingerprint(
        inputs_project
    )


def test_empty_request_policy_with_custom_effective_policy() -> None:
    """A request with no project_policy but a classifier built from a custom policy
    must still reflect that effective policy in the fingerprint."""
    req = _request(changed_files=("src/foo.py",), project_policy={})
    custom_policy = ProjectRiskPolicy(minimum_risk=RiskLevel.R3_HIGH)
    inputs_custom = RiskDecisionInputs(
        request=req,
        project_policy=custom_policy,
        authority_policy=custom_policy.authority_policy,
        security_policy=custom_policy.security_policy,
        architecture_thresholds=ArchitectureThresholds.from_dict(
            custom_policy.architecture_thresholds
        ),
        runtime_escalator=RuntimeRiskEscalator.default(
            authority_policy=custom_policy.authority_policy,
            security_policy=custom_policy.security_policy,
        ),
        normalized_runtime_events=(),
    )
    inputs_default = _inputs(req)
    assert risk_assessment_fingerprint(inputs_custom) != risk_assessment_fingerprint(inputs_default)


def test_runtime_threshold_differs_in_fingerprint() -> None:
    """Different effective test-failure thresholds change the decision fingerprint."""
    req = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            RiskRuntimeEvent(
                event_type=RiskRuntimeEventType.TEST_FAILURE,
                material=True,
                evidence="failures",
                count=3,
            ),
        ),
    )
    inputs_threshold_3 = RiskDecisionInputs(
        request=req,
        project_policy=ProjectRiskPolicy.default(),
        authority_policy=AuthoritySensitivePolicy.default(),
        security_policy=SecuritySensitivePolicy.default(),
        architecture_thresholds=ArchitectureThresholds.default(),
        runtime_escalator=RuntimeRiskEscalator(test_failure_threshold=3),
        normalized_runtime_events=req.runtime_events,
    )
    inputs_threshold_5 = RiskDecisionInputs(
        request=req,
        project_policy=ProjectRiskPolicy.default(),
        authority_policy=AuthoritySensitivePolicy.default(),
        security_policy=SecuritySensitivePolicy.default(),
        architecture_thresholds=ArchitectureThresholds.default(),
        runtime_escalator=RuntimeRiskEscalator(test_failure_threshold=5),
        normalized_runtime_events=req.runtime_events,
    )
    assert risk_assessment_fingerprint(inputs_threshold_3) != risk_assessment_fingerprint(
        inputs_threshold_5
    )


def test_caller_event_threshold_ignored_in_fingerprint() -> None:
    """TEST_FAILURE uses the engine threshold, so caller-supplied thresholds must not
    change the fingerprint or the decision.
    """
    req_low = _request(
        changed_files=("src/foo.py",),
        runtime_events=(
            RiskRuntimeEvent(
                event_type=RiskRuntimeEventType.TEST_FAILURE,
                material=True,
                evidence="failures",
                count=3,
                threshold=1,
            ),
        ),
    )
    req_high = _request(
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
    inputs_low = RiskDecisionInputs(
        request=req_low,
        project_policy=ProjectRiskPolicy.default(),
        authority_policy=AuthoritySensitivePolicy.default(),
        security_policy=SecuritySensitivePolicy.default(),
        architecture_thresholds=ArchitectureThresholds.default(),
        runtime_escalator=RuntimeRiskEscalator(test_failure_threshold=3),
        normalized_runtime_events=req_low.runtime_events,
    )
    inputs_high = RiskDecisionInputs(
        request=req_high,
        project_policy=ProjectRiskPolicy.default(),
        authority_policy=AuthoritySensitivePolicy.default(),
        security_policy=SecuritySensitivePolicy.default(),
        architecture_thresholds=ArchitectureThresholds.default(),
        runtime_escalator=RuntimeRiskEscalator(test_failure_threshold=3),
        normalized_runtime_events=req_high.runtime_events,
    )
    assert risk_assessment_fingerprint(inputs_low) == risk_assessment_fingerprint(inputs_high)


def test_fingerprint_changes_with_event_count_threshold_crossing() -> None:
    """Below-threshold and at-threshold counts produce different fingerprints."""

    def base_inputs(count: int) -> RiskDecisionInputs:
        req = _request(
            changed_files=("src/foo.py",),
            runtime_events=(
                RiskRuntimeEvent(
                    event_type=RiskRuntimeEventType.TEST_FAILURE,
                    material=True,
                    evidence="failures",
                    count=count,
                ),
            ),
        )
        return RiskDecisionInputs(
            request=req,
            project_policy=ProjectRiskPolicy.default(),
            authority_policy=AuthoritySensitivePolicy.default(),
            security_policy=SecuritySensitivePolicy.default(),
            architecture_thresholds=ArchitectureThresholds.default(),
            runtime_escalator=RuntimeRiskEscalator(test_failure_threshold=3),
            normalized_runtime_events=req.runtime_events,
        )

    assert risk_assessment_fingerprint(base_inputs(2)) != risk_assessment_fingerprint(
        base_inputs(3)
    )
