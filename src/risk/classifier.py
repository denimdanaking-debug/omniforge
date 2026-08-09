"""Initial deterministic risk classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.policy.risk import RiskLevel

from .architecture import ArchitectureImpactDetector, ArchitectureThresholds
from .assessment import (
    OperationType,
    RiskAssessmentRequest,
    RiskAssessmentResult,
    RiskFactor,
)
from .authority import AuthoritySensitivePolicy
from .explanation import format_explanation
from .fingerprint import RiskDecisionInputs, risk_assessment_fingerprint
from .project_policy import ProjectRiskPolicy
from .runtime import RiskRuntimeEvent, RuntimeRiskEscalator
from .security import SecuritySensitivePolicy


class RiskClassificationError(ValueError):
    """Raised when a risk classification request is invalid."""


@dataclass(frozen=True)
class InitialRiskClassifier:
    """Deterministic initial risk classifier."""

    authority_policy: AuthoritySensitivePolicy
    security_policy: SecuritySensitivePolicy
    architecture_detector: ArchitectureImpactDetector
    runtime_escalator: RuntimeRiskEscalator
    project_policy: ProjectRiskPolicy

    @classmethod
    def default(cls) -> InitialRiskClassifier:
        authority_policy = AuthoritySensitivePolicy.default()
        security_policy = SecuritySensitivePolicy.default()
        return cls(
            authority_policy=authority_policy,
            security_policy=security_policy,
            architecture_detector=ArchitectureImpactDetector.default(),
            runtime_escalator=RuntimeRiskEscalator.default(
                authority_policy=authority_policy,
                security_policy=security_policy,
            ),
            project_policy=ProjectRiskPolicy.default(),
        )

    @classmethod
    def from_project_policy(cls, policy: ProjectRiskPolicy | None) -> InitialRiskClassifier:
        policy = policy or ProjectRiskPolicy.default()
        return cls(
            authority_policy=policy.authority_policy,
            security_policy=policy.security_policy,
            architecture_detector=ArchitectureImpactDetector(
                ArchitectureThresholds.from_dict(policy.architecture_thresholds)
            ),
            runtime_escalator=RuntimeRiskEscalator.default(
                authority_policy=policy.authority_policy,
                security_policy=policy.security_policy,
            ),
            project_policy=policy,
        )

    def classify(self, request: RiskAssessmentRequest) -> RiskAssessmentResult:
        """Return a deterministic initial risk assessment."""
        paths = request.changed_files + request.explicit_paths
        operation = (
            request.operation
            if isinstance(request.operation, OperationType)
            else OperationType(request.operation)
        )

        factors: list[RiskFactor] = []
        risk = self._baseline_risk(request, paths, operation)

        # Authority-sensitive surfaces.
        authority_factor = self.authority_policy.assess(paths, operation)
        if authority_factor is not None:
            factors.append(authority_factor)
            risk = max(risk, authority_factor.risk_level)

        # Security-sensitive surfaces.
        security_factor = self.security_policy.assess(paths)
        if security_factor is not None:
            factors.append(security_factor)
            risk = max(risk, security_factor.risk_level)

        # Broad architectural impact.
        architecture_factor = self.architecture_detector.detect(
            paths,
            request.changed_lines_estimate,
            request.generated_files,
        )
        if architecture_factor is not None:
            factors.append(architecture_factor)
            risk = max(risk, architecture_factor.risk_level)

        # Project-specific path floors.
        risk, floor_factor = self.project_policy.apply_floor(paths, risk)
        if floor_factor is not None:
            factors.append(floor_factor)

        # Project minimum risk.
        risk, min_factor = self.project_policy.apply_minimum(risk)
        if min_factor is not None:
            factors.append(min_factor)

        # Runtime escalation events. Coerce once and use the canonical tuple for
        # both escalation and fingerprinting so the audit input matches the
        # decision inputs exactly.
        events = self._coerce_events(request.runtime_events)
        if events:
            risk, _records = self.runtime_escalator.apply_all(risk, events)

        # Final determinism: sort factors by code and evidence for stable output.
        factors = sorted(factors, key=lambda f: (f.code.value, f.evidence))

        decision_inputs = RiskDecisionInputs(
            request=request,
            project_policy=self.project_policy,
            authority_policy=self.authority_policy,
            security_policy=self.security_policy,
            architecture_thresholds=self.architecture_detector._thresholds,
            runtime_escalator=self.runtime_escalator,
            normalized_runtime_events=events,
        )
        fingerprint = risk_assessment_fingerprint(decision_inputs)
        explanation = format_explanation(
            RiskAssessmentResult(
                baseline_risk=self._baseline_risk(request, paths, operation),
                final_risk=risk,
                factors=tuple(factors),
                policy_effects={},
                fingerprint=fingerprint,
                explanation="",
            ),
            request.project_id,
        )

        return RiskAssessmentResult(
            baseline_risk=self._baseline_risk(request, paths, operation),
            final_risk=risk,
            factors=tuple(factors),
            policy_effects={"operation": operation.value},
            fingerprint=fingerprint,
            explanation=explanation.summary,
        )

    def _baseline_risk(
        self,
        request: RiskAssessmentRequest,
        paths: tuple[str, ...],
        operation: OperationType,
    ) -> RiskLevel:
        """Determine baseline risk from task metadata and scope."""
        if request.baseline_risk is not None:
            return request.baseline_risk

        # Formatting/docs-only trivial changes.
        docs_only = paths and all(p.endswith((".md", ".rst", ".txt")) for p in paths)
        if (
            docs_only
            and operation
            in {
                OperationType.READ,
                OperationType.REFERENCE,
                OperationType.MODIFY,
            }
            and request.changed_lines_estimate < 50
            and len(paths) <= 2
        ):
            return RiskLevel.R0_TRIVIAL

        # Generated-only large changes stay lower.
        if request.generated_files and all(
            p.replace("\\", "/") in request.generated_files for p in paths
        ):
            return RiskLevel.R1_LOW

        file_count = len(paths)
        if file_count == 1 and request.changed_lines_estimate < 100:
            return RiskLevel.R1_LOW
        if file_count <= 3 and request.changed_lines_estimate < 300:
            return RiskLevel.R2_NORMAL
        return RiskLevel.R2_NORMAL

    def _coerce_events(self, raw_events: tuple[Any, ...]) -> tuple[RiskRuntimeEvent, ...]:
        """Normalize runtime events into canonical RiskRuntimeEvent objects."""
        events: list[RiskRuntimeEvent] = []
        for item in raw_events:
            if isinstance(item, RiskRuntimeEvent):
                events.append(item)
            elif isinstance(item, dict):
                from .runtime import RiskRuntimeEventType

                events.append(
                    RiskRuntimeEvent(
                        event_type=RiskRuntimeEventType(item["event_type"]),
                        material=bool(item.get("material", True)),
                        evidence=str(item["evidence"]),
                        count=int(item.get("count", 1)),
                        threshold=int(item.get("threshold", 1)),
                        affected_paths=tuple(item.get("affected_paths", ())),
                        operation=item.get("operation"),
                    )
                )
            else:
                raise RiskClassificationError(f"unsupported runtime event type: {type(item)}")
        return tuple(events)
