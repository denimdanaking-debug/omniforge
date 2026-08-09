from __future__ import annotations

import inspect

from src.policy.risk import RiskLevel
from src.risk import (
    AuthoritySensitivePolicy,
    InitialRiskClassifier,
    ProjectRiskPolicy,
    RiskAssessmentRequest,
    RiskRuntimeEvent,
    RiskRuntimeEventType,
    RuntimeRiskEscalator,
    SecuritySensitivePolicy,
)
from src.routing.roles import ExecutionRole


def test_only_one_risk_level_enum() -> None:
    # RiskLevel must remain canonical; no competing enum introduced.
    assert RiskLevel.R0_TRIVIAL.value == 0
    assert RiskLevel.R4_CRITICAL_AUTHORITY.value == 4


def test_classifier_does_not_call_provider_or_llm() -> None:
    source = inspect.getsource(InitialRiskClassifier.classify)
    # No provider imports or network calls inside classification.
    assert "requests." not in source
    assert "openai." not in source
    assert "anthropic." not in source


def test_runtime_escalator_does_not_import_provider_logic() -> None:
    source = inspect.getsource(RuntimeRiskEscalator)
    assert "ProviderHealth" not in source
    assert "quota" not in source.lower()


def test_no_brand_based_risk_logic() -> None:
    classifier_source = inspect.getsource(InitialRiskClassifier)
    authority_source = inspect.getsource(AuthoritySensitivePolicy)
    security_source = inspect.getsource(SecuritySensitivePolicy)
    combined = classifier_source + authority_source + security_source
    # No provider/model brand names should influence risk directly.
    for brand in ("openai", "anthropic", "kimi", "qwen", "deepseek", "gemini", "xai", "zai"):
        assert brand not in combined.lower(), brand


def test_classification_is_deterministic_under_100_repetitions() -> None:
    classifier = InitialRiskClassifier.default()
    request = RiskAssessmentRequest(
        project_id="p",
        task_id="t",
        role=ExecutionRole.CODING,
        task_class="feature",
        changed_files=("src/a.py", "src/b.py"),
        changed_lines_estimate=100,
    )
    results = [classifier.classify(request) for _ in range(100)]
    assert all(r.final_risk == results[0].final_risk for r in results)
    assert all(r.fingerprint == results[0].fingerprint for r in results)
    reasons = [tuple(f.evidence for f in r.factors) for r in results]
    assert all(reason == reasons[0] for reason in reasons)


def test_authority_violation_reaches_r4() -> None:
    classifier = InitialRiskClassifier.default()
    event = RiskRuntimeEvent(
        event_type=RiskRuntimeEventType.AUTHORITY_VIOLATION,
        material=True,
        evidence="roadmap mutation",
    )
    request = RiskAssessmentRequest(
        project_id="p",
        task_id="t",
        role=ExecutionRole.CODING,
        task_class="feature",
        changed_files=("src/foo.py",),
        runtime_events=(event,),
    )
    result = classifier.classify(request)
    assert result.final_risk == RiskLevel.R4_CRITICAL_AUTHORITY


def test_project_override_cannot_lower_core_authority_floor() -> None:
    policy = ProjectRiskPolicy(minimum_risk=RiskLevel.R1_LOW)
    classifier = InitialRiskClassifier.from_project_policy(policy)
    request = RiskAssessmentRequest(
        project_id="p",
        task_id="t",
        role=ExecutionRole.CODING,
        task_class="feature",
        changed_files=("docs/PROJECT_STATE.json",),
    )
    result = classifier.classify(request)
    assert result.final_risk == RiskLevel.R4_CRITICAL_AUTHORITY


def test_no_phase10_retry_engine_present() -> None:
    classifier_source = inspect.getsource(InitialRiskClassifier)
    escalator_source = inspect.getsource(RuntimeRiskEscalator)
    combined = classifier_source + escalator_source
    assert "retry" not in combined.lower()
    assert "transient" not in combined.lower()


def test_risk_explanation_has_no_hidden_reasoning() -> None:
    from src.risk import format_explanation

    classifier = InitialRiskClassifier.default()
    request = RiskAssessmentRequest(
        project_id="p",
        task_id="t",
        role=ExecutionRole.CODING,
        task_class="feature",
        changed_files=("docs/PROJECT_STATE.json",),
    )
    result = classifier.classify(request)
    explanation = format_explanation(result, project_id="p")
    assert explanation.risk == RiskLevel.R4_CRITICAL_AUTHORITY
    # Explanation must derive from explicit factor codes and evidence only.
    assert all(code in (f.code.value for f in result.factors) for code in explanation.factor_codes)
    assert all(reason in (f.evidence for f in result.factors) for reason in explanation.reasons)


def test_runtime_events_do_not_affect_input_fingerprint() -> None:
    from src.risk import risk_assessment_fingerprint

    base = RiskAssessmentRequest(
        project_id="p",
        task_id="t",
        role=ExecutionRole.CODING,
        task_class="feature",
        changed_files=("src/foo.py",),
    )
    with_event = RiskAssessmentRequest(
        project_id="p",
        task_id="t",
        role=ExecutionRole.CODING,
        task_class="feature",
        changed_files=("src/foo.py",),
        runtime_events=(
            RiskRuntimeEvent(
                event_type=RiskRuntimeEventType.AUTHORITY_VIOLATION,
                material=True,
                evidence="x",
            ),
        ),
    )
    assert risk_assessment_fingerprint(base) == risk_assessment_fingerprint(with_event)
