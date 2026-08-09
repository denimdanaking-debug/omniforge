"""Integration seam between Phase 9 risk assessment and Phase 8 routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.routing.capabilities import CapabilityRequirement
from src.routing.dynamic.request import DynamicRoutingRequest
from src.routing.roles import ExecutionRole

from .assessment import RiskAssessmentResult
from .context_policy import RiskContextPolicy, RiskContextRequirements
from .experimentation import ExperimentationEligibility, ExperimentationEligibilityPolicy
from .review_policy import RiskReviewPolicy, RiskReviewRequirement


@dataclass(frozen=True)
class RiskRoutingIntegration:
    """Risk-derived inputs for dynamic routing and context construction."""

    routing_request: DynamicRoutingRequest
    review_requirement: RiskReviewRequirement
    experimentation: ExperimentationEligibility
    context_requirements: RiskContextRequirements


class RiskEligibilityConnector:
    """Wire a RiskAssessmentResult into downstream routing decisions."""

    def __init__(
        self,
        review_policy: RiskReviewPolicy | None = None,
        experimentation_policy: ExperimentationEligibilityPolicy | None = None,
        context_policy: RiskContextPolicy | None = None,
    ) -> None:
        self._review_policy = review_policy or RiskReviewPolicy.default()
        self._experimentation_policy = (
            experimentation_policy or ExperimentationEligibilityPolicy.default()
        )
        self._context_policy = context_policy or RiskContextPolicy.default()

    @classmethod
    def default(cls) -> RiskEligibilityConnector:
        return cls()

    def build_routing_request(
        self,
        result: RiskAssessmentResult,
        *,
        task_id: str,
        project_id: str,
        role: ExecutionRole,
        task_class: str,
        capability_requirement: CapabilityRequirement | None = None,
        required_context_tokens: int | None = None,
        expected_input_tokens: int | None = None,
        expected_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> DynamicRoutingRequest:
        """Create a DynamicRoutingRequest using the assessed risk."""
        return DynamicRoutingRequest(
            task_id=task_id,
            project_id=project_id,
            role=role,
            risk=result.final_risk,
            task_class=task_class,
            capability_requirement=capability_requirement,
            required_context_tokens=required_context_tokens,
            expected_input_tokens=expected_input_tokens,
            expected_output_tokens=expected_output_tokens,
            **kwargs,
        )

    def integrate(
        self,
        result: RiskAssessmentResult,
        *,
        task_id: str,
        project_id: str,
        role: ExecutionRole,
        task_class: str,
        global_exploration_enabled: bool = False,
        project_exploration_allowed: bool | None = None,
        project_review_minimum: int | None = None,
        **routing_kwargs: Any,
    ) -> RiskRoutingIntegration:
        """Return risk-derived routing, review, experimentation, and context inputs."""
        routing_request = self.build_routing_request(
            result,
            task_id=task_id,
            project_id=project_id,
            role=role,
            task_class=task_class,
            **routing_kwargs,
        )
        review_requirement = self._review_policy.requirement_for(
            result.final_risk,
            project_minimum=project_review_minimum,
        )
        experimentation = self._experimentation_policy.check(
            result.final_risk,
            global_exploration_enabled=global_exploration_enabled,
            project_exploration_allowed=project_exploration_allowed,
        )
        context_requirements = self._context_policy.requirements_for(result.final_risk)
        return RiskRoutingIntegration(
            routing_request=routing_request,
            review_requirement=review_requirement,
            experimentation=experimentation,
            context_requirements=context_requirements,
        )
