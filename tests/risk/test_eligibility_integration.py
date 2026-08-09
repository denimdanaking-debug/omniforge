from __future__ import annotations

from src.policy.risk import RiskLevel, lifecycle_eligible
from src.risk import (
    InitialRiskClassifier,
    RiskAssessmentRequest,
    RiskAssessmentResult,
    RiskEligibilityConnector,
    RiskRoutingIntegration,
)
from src.routing.capabilities import CapabilityRequirement
from src.routing.model_identity import ModelLifecycle
from src.routing.roles import ExecutionRole


def test_connector_builds_dynamic_routing_request() -> None:
    classifier = InitialRiskClassifier.default()
    request = RiskAssessmentRequest(
        project_id="project-a",
        task_id="task-1",
        role=ExecutionRole.CODING,
        task_class="feature",
        changed_files=("src/foo.py",),
    )
    result = classifier.classify(request)
    connector = RiskEligibilityConnector.default()
    routing_request = connector.build_routing_request(
        result,
        task_id="task-1",
        project_id="project-a",
        role=ExecutionRole.CODING,
        task_class="feature",
        capability_requirement=CapabilityRequirement(min_context_tokens=1000),
        required_context_tokens=2000,
    )
    assert routing_request.risk == result.final_risk
    assert routing_request.task_id == "task-1"
    assert routing_request.project_id == "project-a"
    assert routing_request.role == ExecutionRole.CODING
    assert routing_request.required_context_tokens == 2000


def test_integrate_produces_all_risk_derived_outputs() -> None:
    classifier = InitialRiskClassifier.default()
    request = RiskAssessmentRequest(
        project_id="project-a",
        task_id="task-1",
        role=ExecutionRole.CODING,
        task_class="feature",
        changed_files=("src/foo.py",),
    )
    result = classifier.classify(request)
    connector = RiskEligibilityConnector.default()
    integration = connector.integrate(
        result,
        task_id="task-1",
        project_id="project-a",
        role=ExecutionRole.CODING,
        task_class="feature",
        global_exploration_enabled=False,
    )
    assert isinstance(integration, RiskRoutingIntegration)
    assert integration.routing_request.risk == result.final_risk
    assert integration.review_requirement.reviewer_count >= 0
    assert not integration.experimentation.allowed


def test_r2_allows_normal_lifecycle_candidate() -> None:
    assert lifecycle_eligible(ModelLifecycle.NORMAL, RiskLevel.R2_NORMAL)
    assert lifecycle_eligible(ModelLifecycle.HIGH_RISK, RiskLevel.R2_NORMAL)
    assert not lifecycle_eligible(ModelLifecycle.LOW_RISK, RiskLevel.R2_NORMAL)


def test_r3_excludes_normal_lifecycle_candidate() -> None:
    assert not lifecycle_eligible(ModelLifecycle.NORMAL, RiskLevel.R3_HIGH)
    assert lifecycle_eligible(ModelLifecycle.HIGH_RISK, RiskLevel.R3_HIGH)
    assert not lifecycle_eligible(ModelLifecycle.LOW_RISK, RiskLevel.R3_HIGH)


def test_r4_requires_high_risk_lifecycle() -> None:
    assert lifecycle_eligible(ModelLifecycle.HIGH_RISK, RiskLevel.R4_CRITICAL_AUTHORITY)
    assert not lifecycle_eligible(ModelLifecycle.NORMAL, RiskLevel.R4_CRITICAL_AUTHORITY)
    assert not lifecycle_eligible(ModelLifecycle.LOW_RISK, RiskLevel.R4_CRITICAL_AUTHORITY)


def test_disabled_and_shadow_never_eligible() -> None:
    for level in RiskLevel:
        assert not lifecycle_eligible(ModelLifecycle.DISABLED, level)
        assert not lifecycle_eligible(ModelLifecycle.SHADOW, level)


def test_risk_classification_does_not_mutate_model_reputation() -> None:
    classifier = InitialRiskClassifier.default()
    request = RiskAssessmentRequest(
        project_id="project-a",
        task_id="task-1",
        role=ExecutionRole.CODING,
        task_class="feature",
        changed_files=("docs/PROJECT_STATE.json",),
    )
    result = classifier.classify(request)
    # Risk assessment is a pure function of task metadata; no model state changes.
    assert result.factors
    assert all(f.provenance for f in result.factors)
    assert result.final_risk == RiskLevel.R4_CRITICAL_AUTHORITY


def test_project_review_minimum_applies_through_connector() -> None:
    from src.risk import ProjectRiskPolicy

    policy = ProjectRiskPolicy(review_minimum=3)
    connector = RiskEligibilityConnector(project_risk_policy=policy)
    result = RiskAssessmentResult(
        baseline_risk=RiskLevel.R2_NORMAL,
        final_risk=RiskLevel.R2_NORMAL,
        factors=(),
        policy_effects={},
        fingerprint="fp",
        explanation="",
    )
    integration = connector.integrate(
        result,
        task_id="task-1",
        project_id="project-a",
        role=ExecutionRole.CODING,
        task_class="feature",
    )
    assert integration.review_requirement.reviewer_count == 3


def test_project_review_minimum_cannot_weaken_r3_requirement() -> None:
    from src.risk import ProjectRiskPolicy

    policy = ProjectRiskPolicy(review_minimum=1)
    connector = RiskEligibilityConnector(project_risk_policy=policy)
    result = RiskAssessmentResult(
        baseline_risk=RiskLevel.R3_HIGH,
        final_risk=RiskLevel.R3_HIGH,
        factors=(),
        policy_effects={},
        fingerprint="fp",
        explanation="",
    )
    integration = connector.integrate(
        result,
        task_id="task-1",
        project_id="project-a",
        role=ExecutionRole.CODING,
        task_class="feature",
    )
    assert integration.review_requirement.reviewer_count >= 2
    assert integration.review_requirement.distinct_failure_domains_required


def test_project_exploration_ceiling_tightens() -> None:
    from src.risk import ProjectRiskPolicy

    policy = ProjectRiskPolicy(exploration_max_risk=RiskLevel.R0_TRIVIAL)
    connector = RiskEligibilityConnector(project_risk_policy=policy)
    result = RiskAssessmentResult(
        baseline_risk=RiskLevel.R1_LOW,
        final_risk=RiskLevel.R1_LOW,
        factors=(),
        policy_effects={},
        fingerprint="fp",
        explanation="",
    )
    integration = connector.integrate(
        result,
        task_id="task-1",
        project_id="project-a",
        role=ExecutionRole.CODING,
        task_class="feature",
        global_exploration_enabled=True,
        project_exploration_allowed=True,
    )
    assert not integration.experimentation.allowed


def test_project_exploration_ceiling_cannot_loosen_core() -> None:
    from src.risk import ProjectRiskPolicy

    policy = ProjectRiskPolicy(exploration_max_risk=RiskLevel.R3_HIGH)
    connector = RiskEligibilityConnector(project_risk_policy=policy)
    result = RiskAssessmentResult(
        baseline_risk=RiskLevel.R2_NORMAL,
        final_risk=RiskLevel.R2_NORMAL,
        factors=(),
        policy_effects={},
        fingerprint="fp",
        explanation="",
    )
    integration = connector.integrate(
        result,
        task_id="task-1",
        project_id="project-a",
        role=ExecutionRole.CODING,
        task_class="feature",
        global_exploration_enabled=True,
        project_exploration_allowed=True,
    )
    assert not integration.experimentation.allowed


def test_project_context_minimum_deepens() -> None:
    from src.risk import ProjectRiskPolicy

    policy = ProjectRiskPolicy(context_depth_minimum="large_context")
    connector = RiskEligibilityConnector(project_risk_policy=policy)
    result = RiskAssessmentResult(
        baseline_risk=RiskLevel.R1_LOW,
        final_risk=RiskLevel.R1_LOW,
        factors=(),
        policy_effects={},
        fingerprint="fp",
        explanation="",
    )
    integration = connector.integrate(
        result,
        task_id="task-1",
        project_id="project-a",
        role=ExecutionRole.CODING,
        task_class="feature",
    )
    assert integration.context_requirements.strategy_preference == "large_context"


def test_project_context_minimum_cannot_weaken_r4_raw_authority() -> None:
    from src.risk import ProjectRiskPolicy

    policy = ProjectRiskPolicy(context_depth_minimum="targeted")
    connector = RiskEligibilityConnector(project_risk_policy=policy)
    result = RiskAssessmentResult(
        baseline_risk=RiskLevel.R4_CRITICAL_AUTHORITY,
        final_risk=RiskLevel.R4_CRITICAL_AUTHORITY,
        factors=(),
        policy_effects={},
        fingerprint="fp",
        explanation="",
    )
    integration = connector.integrate(
        result,
        task_id="task-1",
        project_id="project-a",
        role=ExecutionRole.CODING,
        task_class="feature",
    )
    assert integration.context_requirements.strategy_preference == "large_context"
    assert integration.context_requirements.require_raw_authority
