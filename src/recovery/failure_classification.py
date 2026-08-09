"""Canonical failure classification for Phase 10 retry/escalation decisions.

Classification is deterministic and rule-based. No LLM or hidden reasoning is
used. The classifier consumes normalized provider errors, validation results,
and task context, then emits a typed ``FailureClassification`` with explicit
metadata for downstream recovery coordination.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from src.policy.risk import RiskLevel
from src.providers.errors import ProviderError, ProviderErrorCode
from src.routing.roles import ExecutionRole
from src.security.redaction import redact


class FailureCategory(StrEnum):
    """Typed failure categories used by the recovery coordinator."""

    INFRASTRUCTURE_TRANSIENT = "INFRASTRUCTURE_TRANSIENT"
    INFRASTRUCTURE_QUOTA = "INFRASTRUCTURE_QUOTA"
    INFRASTRUCTURE_AUTH = "INFRASTRUCTURE_AUTH"
    INFRASTRUCTURE_UNAVAILABLE = "INFRASTRUCTURE_UNAVAILABLE"
    CAPABILITY_MISMATCH = "CAPABILITY_MISMATCH"
    CONTEXT_CAPACITY = "CONTEXT_CAPACITY"
    STRUCTURED_OUTPUT_INVALID = "STRUCTURED_OUTPUT_INVALID"
    PLANNING_OUTPUT_INVALID = "PLANNING_OUTPUT_INVALID"
    IMPLEMENTATION_DETERMINISTIC = "IMPLEMENTATION_DETERMINISTIC"
    IMPLEMENTATION_CONCEPTUAL = "IMPLEMENTATION_CONCEPTUAL"
    AUTHORITY_VIOLATION = "AUTHORITY_VIOLATION"
    INTEGRATION_FAILURE = "INTEGRATION_FAILURE"
    CANCELLED = "CANCELLED"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


class FailureSubtype(StrEnum):
    """More specific failure subtypes for evidence and telemetry."""

    TRANSIENT_TRANSPORT = "TRANSIENT_TRANSPORT"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    AUTH_FAILURE = "AUTH_FAILURE"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    MALFORMED_JSON = "MALFORMED_JSON"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    MISSING_REQUIRED_FIELDS = "MISSING_REQUIRED_FIELDS"
    INVALID_ENUM = "INVALID_ENUM"
    PARSE_FAILURE = "PARSE_FAILURE"
    INCOMPLETE_RESPONSE = "INCOMPLETE_RESPONSE"
    MALFORMED_PLAN = "MALFORMED_PLAN"
    MISSING_PLAN_STEPS = "MISSING_PLAN_STEPS"
    PLAN_AUTHORITY_VIOLATION = "PLAN_AUTHORITY_VIOLATION"
    PLAN_SCHEMA_INVALID = "PLAN_SCHEMA_INVALID"
    COMPILE_FAILURE = "COMPILE_FAILURE"
    LINT_FAILURE = "LINT_FAILURE"
    TYPE_CHECK_FAILURE = "TYPE_CHECK_FAILURE"
    TEST_FAILURE = "TEST_FAILURE"
    BUILD_FAILURE = "BUILD_FAILURE"
    INVARIANT_FAILURE = "INVARIANT_FAILURE"
    REPEATED_SIGNATURE = "REPEATED_SIGNATURE"
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
    INTEGRATION_ANOMALY = "INTEGRATION_ANOMALY"
    USER_CANCELLED = "USER_CANCELLED"
    UNKNOWN = "UNKNOWN"


class Retryability(StrEnum):
    """Canonical retryability classification."""

    YES = "yes"
    BOUNDED = "bounded"
    NO = "no"
    WAIT = "wait"


@dataclass(frozen=True)
class FailureDomain:
    """Failure-domain attribution: which layer owns the failure."""

    provider_id: str | None = None
    route_id: str | None = None
    model_id: str | None = None
    failure_domain: str | None = None
    stage: str = ""


@dataclass(frozen=True)
class ValidationResultSummary:
    """Bounded summary of a deterministic validation result."""

    validator: str
    passed: bool
    failing_check_names: tuple[str, ...] = ()
    affected_files: tuple[str, ...] = ()
    error_excerpts: tuple[str, ...] = ()
    exit_status: int | None = None


@dataclass(frozen=True)
class StructuredOutputValidationResult:
    """Bounded summary of structured-output validation."""

    schema_name: str | None = None
    missing_required_fields: tuple[str, ...] = ()
    invalid_enum_values: tuple[str, ...] = ()
    parse_error: str | None = None
    schema_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanningValidationResult:
    """Bounded summary of planning-output validation."""

    plan_schema_name: str | None = None
    missing_steps: tuple[str, ...] = ()
    authority_violations: tuple[str, ...] = ()
    schema_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextOverflowMetadata:
    """Context-overflow-specific metadata for classification."""

    estimated_input_chars: int | None = None
    model_context_tokens: int | None = None
    authority_required: bool = False
    authority_items_present: int = 0
    authority_items_raw: int = 0
    rebuild_attempts: int = 0


@dataclass(frozen=True)
class AuthorityViolationData:
    """Authority-violation-specific metadata for classification."""

    touched_authority_paths: tuple[str, ...] = ()
    attempted_state_advancement: bool = False
    ignored_immutable_authority: bool = False
    summary_substituted_for_raw: bool = False
    integration_state_mismatch: bool = False


@dataclass(frozen=True)
class FailureClassifierInput:
    """Normalized inputs for deterministic failure classification."""

    task_id: str
    role: ExecutionRole
    task_risk: RiskLevel = RiskLevel.R2_NORMAL
    stage: str = ""
    provider_error: ProviderError | None = None
    deterministic_validation: ValidationResultSummary | None = None
    structured_output_validation: StructuredOutputValidationResult | None = None
    planning_validation: PlanningValidationResult | None = None
    review_findings: tuple[str, ...] = ()
    retry_history: tuple[Any, ...] = ()
    provider_id: str | None = None
    model_id: str | None = None
    route_id: str | None = None
    failure_domain: str | None = None
    context_overflow: ContextOverflowMetadata | None = None
    authority_violation: AuthorityViolationData | None = None
    integration_evidence: tuple[str, ...] = ()
    cancelled: bool = False

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")

    def with_provider_error(self, provider_error: ProviderError) -> FailureClassifierInput:
        return replace(self, provider_error=provider_error)

    def with_deterministic_validation(
        self, deterministic_validation: ValidationResultSummary
    ) -> FailureClassifierInput:
        return replace(self, deterministic_validation=deterministic_validation)

    def with_structured_output_validation(
        self, structured_output_validation: StructuredOutputValidationResult
    ) -> FailureClassifierInput:
        return replace(self, structured_output_validation=structured_output_validation)

    def with_planning_validation(
        self, planning_validation: PlanningValidationResult
    ) -> FailureClassifierInput:
        return replace(self, planning_validation=planning_validation)

    def with_context_overflow(
        self, context_overflow: ContextOverflowMetadata
    ) -> FailureClassifierInput:
        return replace(self, context_overflow=context_overflow)

    def with_authority_violation(
        self, authority_violation: AuthorityViolationData
    ) -> FailureClassifierInput:
        return replace(self, authority_violation=authority_violation)

    def with_cancelled(self, cancelled: bool) -> FailureClassifierInput:
        return replace(self, cancelled=cancelled)


@dataclass(frozen=True)
class FailureClassification:
    """Immutable output of the failure classifier."""

    category: FailureCategory
    subtype: FailureSubtype
    retryability: Retryability
    failure_domain: FailureDomain
    model_quality_effect: bool
    provider_health_effect: bool
    route_health_effect: bool
    recommended_action_class: str
    evidence_refs: tuple[str, ...]
    deterministic_fingerprint: str
    confidence: str = "deterministic_rules"
    explanation: str = ""


class FailureClassifier:
    """Deterministic rule-based failure classifier."""

    def classify(self, inputs: FailureClassifierInput) -> FailureClassification:
        """Return a deterministic classification for the given inputs."""
        if inputs.cancelled:
            return self._build(
                inputs,
                FailureCategory.CANCELLED,
                FailureSubtype.USER_CANCELLED,
                Retryability.NO,
                model_quality_effect=False,
                provider_health_effect=False,
                route_health_effect=False,
                recommended_action_class="CANCEL",
                explanation="Cancellation is terminal unless explicitly resumed.",
            )

        if inputs.authority_violation is not None and (
            inputs.authority_violation.touched_authority_paths
            or inputs.authority_violation.attempted_state_advancement
            or inputs.authority_violation.ignored_immutable_authority
            or inputs.authority_violation.summary_substituted_for_raw
            or inputs.authority_violation.integration_state_mismatch
        ):
            return self._build(
                inputs,
                FailureCategory.AUTHORITY_VIOLATION,
                FailureSubtype.PLAN_AUTHORITY_VIOLATION,
                Retryability.NO,
                model_quality_effect=False,
                provider_health_effect=False,
                route_health_effect=False,
                recommended_action_class="BLOCK",
                explanation="Authority violation is a severe task-integrity failure.",
            )

        if inputs.provider_error is not None:
            return self._classify_provider_error(inputs)

        if inputs.context_overflow is not None and (
            inputs.context_overflow.estimated_input_chars is not None
            and inputs.context_overflow.model_context_tokens is not None
            and inputs.context_overflow.estimated_input_chars
            > inputs.context_overflow.model_context_tokens
        ):
            return self._build(
                inputs,
                FailureCategory.CONTEXT_CAPACITY,
                FailureSubtype.CONTEXT_OVERFLOW,
                Retryability.BOUNDED,
                model_quality_effect=False,
                provider_health_effect=False,
                route_health_effect=False,
                recommended_action_class="REBUILD_CONTEXT",
                explanation="Context overflow requires context-aware recovery.",
            )

        if inputs.planning_validation is not None and (
            inputs.planning_validation.missing_steps
            or inputs.planning_validation.authority_violations
            or inputs.planning_validation.schema_errors
        ):
            return self._build(
                inputs,
                FailureCategory.PLANNING_OUTPUT_INVALID,
                self._planning_subtype(inputs.planning_validation),
                Retryability.BOUNDED,
                model_quality_effect=True,
                provider_health_effect=False,
                route_health_effect=False,
                recommended_action_class="REPLAN",
                explanation="Planning output failed deterministic validation.",
            )

        if inputs.structured_output_validation is not None and (
            inputs.structured_output_validation.missing_required_fields
            or inputs.structured_output_validation.invalid_enum_values
            or inputs.structured_output_validation.parse_error
            or inputs.structured_output_validation.schema_errors
        ):
            return self._build(
                inputs,
                FailureCategory.STRUCTURED_OUTPUT_INVALID,
                self._structured_output_subtype(inputs.structured_output_validation),
                Retryability.BOUNDED,
                model_quality_effect=True,
                provider_health_effect=False,
                route_health_effect=False,
                recommended_action_class="CONSTRAINED_OUTPUT_RETRY",
                explanation="Structured output failed schema/format validation.",
            )

        validation = inputs.deterministic_validation
        if validation is not None and not validation.passed:
            return self._build(
                inputs,
                FailureCategory.IMPLEMENTATION_DETERMINISTIC,
                self._implementation_subtype(validation),
                Retryability.BOUNDED,
                model_quality_effect=True,
                provider_health_effect=False,
                route_health_effect=False,
                recommended_action_class="REPAIR_WITH_EVIDENCE",
                explanation="Deterministic validation produced failing evidence.",
            )

        if inputs.review_findings:
            return self._build(
                inputs,
                FailureCategory.IMPLEMENTATION_CONCEPTUAL,
                FailureSubtype.MODEL_DISAGREEMENT,
                Retryability.BOUNDED,
                model_quality_effect=True,
                provider_health_effect=False,
                route_health_effect=False,
                recommended_action_class="CROSS_MODEL_REPAIR",
                explanation="Conceptual reviewer finding persists across attempts.",
            )

        if inputs.integration_evidence:
            return self._build(
                inputs,
                FailureCategory.INTEGRATION_FAILURE,
                FailureSubtype.INTEGRATION_ANOMALY,
                Retryability.BOUNDED,
                model_quality_effect=False,
                provider_health_effect=False,
                route_health_effect=False,
                recommended_action_class="BLOCK",
                explanation="Integration anomaly requires review before proceeding.",
            )

        return self._build(
            inputs,
            FailureCategory.UNKNOWN_FAILURE,
            FailureSubtype.UNKNOWN,
            Retryability.BOUNDED,
            model_quality_effect=False,
            provider_health_effect=False,
            route_health_effect=False,
            recommended_action_class="BLOCK",
            explanation="Unknown failure: conservative bounded recovery then block.",
        )

    def _classify_provider_error(self, inputs: FailureClassifierInput) -> FailureClassification:
        error = inputs.provider_error
        assert error is not None
        code = error.code

        if code is ProviderErrorCode.TRANSIENT_TRANSPORT:
            return self._build(
                inputs,
                FailureCategory.INFRASTRUCTURE_TRANSIENT,
                FailureSubtype.TRANSIENT_TRANSPORT,
                Retryability.YES,
                model_quality_effect=False,
                provider_health_effect=True,
                route_health_effect=True,
                recommended_action_class="RETRY_ALTERNATE_ROUTE",
                explanation="Transient transport failure is infrastructure-scoped.",
            )

        if code is ProviderErrorCode.PROVIDER_UNAVAILABLE:
            return self._build(
                inputs,
                FailureCategory.INFRASTRUCTURE_UNAVAILABLE,
                FailureSubtype.PROVIDER_UNAVAILABLE,
                Retryability.YES,
                model_quality_effect=False,
                provider_health_effect=True,
                route_health_effect=True,
                recommended_action_class="REROUTE_PROVIDER",
                explanation="Provider unavailable is infrastructure-scoped.",
            )

        if code is ProviderErrorCode.RATE_LIMITED:
            return self._build(
                inputs,
                FailureCategory.INFRASTRUCTURE_TRANSIENT,
                FailureSubtype.RATE_LIMITED,
                Retryability.YES,
                model_quality_effect=False,
                provider_health_effect=True,
                route_health_effect=True,
                recommended_action_class="RETRY_ALTERNATE_ROUTE",
                explanation="Rate limit is temporary infrastructure capacity pressure.",
            )

        if code is ProviderErrorCode.QUOTA_EXHAUSTED:
            return self._build(
                inputs,
                FailureCategory.INFRASTRUCTURE_QUOTA,
                FailureSubtype.QUOTA_EXHAUSTED,
                Retryability.WAIT,
                model_quality_effect=False,
                provider_health_effect=True,
                route_health_effect=True,
                recommended_action_class="WAIT_FOR_PROVIDER",
                explanation="Quota exhaustion is capacity state, not model quality.",
            )

        if code is ProviderErrorCode.AUTH_FAILURE:
            return self._build(
                inputs,
                FailureCategory.INFRASTRUCTURE_AUTH,
                FailureSubtype.AUTH_FAILURE,
                Retryability.NO,
                model_quality_effect=False,
                provider_health_effect=True,
                route_health_effect=True,
                recommended_action_class="BLOCK",
                explanation="Auth failure requires configuration change; no hot-loop retry.",
            )

        if code is ProviderErrorCode.UNSUPPORTED_CAPABILITY:
            return self._build(
                inputs,
                FailureCategory.CAPABILITY_MISMATCH,
                FailureSubtype.UNSUPPORTED_CAPABILITY,
                Retryability.NO,
                model_quality_effect=False,
                provider_health_effect=False,
                route_health_effect=False,
                recommended_action_class="REROUTE_MODEL",
                explanation="Capability mismatch requires different candidate selection.",
            )

        if code is ProviderErrorCode.CONTEXT_OVERFLOW:
            return self._build(
                inputs,
                FailureCategory.CONTEXT_CAPACITY,
                FailureSubtype.CONTEXT_OVERFLOW,
                Retryability.BOUNDED,
                model_quality_effect=False,
                provider_health_effect=False,
                route_health_effect=False,
                recommended_action_class="REBUILD_CONTEXT",
                explanation="Context overflow requires context-aware recovery.",
            )

        if code is ProviderErrorCode.INVALID_MODEL_OUTPUT:
            return self._build(
                inputs,
                FailureCategory.STRUCTURED_OUTPUT_INVALID,
                FailureSubtype.SCHEMA_MISMATCH,
                Retryability.BOUNDED,
                model_quality_effect=True,
                provider_health_effect=False,
                route_health_effect=False,
                recommended_action_class="CONSTRAINED_OUTPUT_RETRY",
                explanation="Invalid model output is a model/task-output failure.",
            )

        if code is ProviderErrorCode.TASK_FAILURE:
            return self._build(
                inputs,
                FailureCategory.IMPLEMENTATION_DETERMINISTIC,
                FailureSubtype.UNKNOWN,
                Retryability.BOUNDED,
                model_quality_effect=True,
                provider_health_effect=False,
                route_health_effect=False,
                recommended_action_class="REPAIR_WITH_EVIDENCE",
                explanation="Task failure requires deterministic evidence for repair.",
            )

        if code is ProviderErrorCode.CANCELLED:
            return self._build(
                inputs,
                FailureCategory.CANCELLED,
                FailureSubtype.USER_CANCELLED,
                Retryability.NO,
                model_quality_effect=False,
                provider_health_effect=False,
                route_health_effect=False,
                recommended_action_class="CANCEL",
                explanation="Cancellation is terminal unless explicitly resumed.",
            )

        return self._build(
            inputs,
            FailureCategory.UNKNOWN_FAILURE,
            FailureSubtype.UNKNOWN,
            Retryability.BOUNDED,
            model_quality_effect=False,
            provider_health_effect=True,
            route_health_effect=True,
            recommended_action_class="BLOCK",
            explanation="Unknown provider error: conservative bounded recovery.",
        )

    def _structured_output_subtype(
        self, result: StructuredOutputValidationResult
    ) -> FailureSubtype:
        if result.parse_error:
            return FailureSubtype.PARSE_FAILURE
        if result.missing_required_fields:
            return FailureSubtype.MISSING_REQUIRED_FIELDS
        if result.invalid_enum_values:
            return FailureSubtype.INVALID_ENUM
        if result.schema_errors:
            return FailureSubtype.SCHEMA_MISMATCH
        return FailureSubtype.INCOMPLETE_RESPONSE

    def _planning_subtype(self, result: PlanningValidationResult) -> FailureSubtype:
        if result.authority_violations:
            return FailureSubtype.PLAN_AUTHORITY_VIOLATION
        if result.missing_steps:
            return FailureSubtype.MISSING_PLAN_STEPS
        if result.schema_errors:
            return FailureSubtype.PLAN_SCHEMA_INVALID
        return FailureSubtype.MALFORMED_PLAN

    def _implementation_subtype(self, result: ValidationResultSummary) -> FailureSubtype:
        validator = result.validator.lower()
        if "compile" in validator or "build" in validator:
            return FailureSubtype.BUILD_FAILURE
        if "mypy" in validator or "type" in validator:
            return FailureSubtype.TYPE_CHECK_FAILURE
        if "lint" in validator or "ruff" in validator:
            return FailureSubtype.LINT_FAILURE
        if "test" in validator:
            return FailureSubtype.TEST_FAILURE
        if "invariant" in validator or "architecture" in validator:
            return FailureSubtype.INVARIANT_FAILURE
        return FailureSubtype.BUILD_FAILURE

    def _build(
        self,
        inputs: FailureClassifierInput,
        category: FailureCategory,
        subtype: FailureSubtype,
        retryability: Retryability,
        *,
        model_quality_effect: bool,
        provider_health_effect: bool,
        route_health_effect: bool,
        recommended_action_class: str,
        explanation: str,
    ) -> FailureClassification:
        domain = FailureDomain(
            provider_id=inputs.provider_id,
            route_id=inputs.route_id,
            model_id=inputs.model_id,
            failure_domain=inputs.failure_domain,
            stage=inputs.stage,
        )
        refs = self._evidence_refs(inputs)
        fingerprint = failure_classification_fingerprint(inputs, category, subtype)
        return FailureClassification(
            category=category,
            subtype=subtype,
            retryability=retryability,
            failure_domain=domain,
            model_quality_effect=model_quality_effect,
            provider_health_effect=provider_health_effect,
            route_health_effect=route_health_effect,
            recommended_action_class=recommended_action_class,
            evidence_refs=refs,
            deterministic_fingerprint=fingerprint,
            explanation=explanation,
        )

    def _evidence_refs(self, inputs: FailureClassifierInput) -> tuple[str, ...]:
        refs: list[str] = []
        if inputs.provider_error is not None:
            refs.append(f"provider_error:{inputs.provider_error.code.value}")
        if inputs.deterministic_validation is not None:
            refs.append(f"validation:{inputs.deterministic_validation.validator}")
        if inputs.structured_output_validation is not None:
            refs.append("structured_output_validation")
        if inputs.planning_validation is not None:
            refs.append("planning_validation")
        if inputs.review_findings:
            refs.append(f"review_findings:{len(inputs.review_findings)}")
        if inputs.authority_violation is not None:
            refs.append("authority_violation")
        if inputs.integration_evidence:
            refs.append(f"integration_evidence:{len(inputs.integration_evidence)}")
        return tuple(refs)


def _normalize_classification_input(inputs: FailureClassifierInput) -> dict[str, Any]:
    """Return a deterministic, redacted dict of decision-driving inputs."""

    def _provider_error_dict(error: ProviderError | None) -> dict[str, Any] | None:
        if error is None:
            return None
        return {
            "code": error.code.value,
            "category": error.category.name,
            "retryable": error.retryable,
            "http_status": error.http_status,
            "provider_error_code": error.provider_error_code,
            "retry_after_seconds": error.retry_after_seconds,
        }

    def _validation_dict(v: ValidationResultSummary | None) -> dict[str, Any] | None:
        if v is None:
            return None
        return {
            "validator": v.validator,
            "passed": v.passed,
            "failing_check_names": sorted(v.failing_check_names),
            "affected_files": sorted(v.affected_files),
            "exit_status": v.exit_status,
        }

    def _structured_dict(v: StructuredOutputValidationResult | None) -> dict[str, Any] | None:
        if v is None:
            return None
        return {
            "schema_name": v.schema_name,
            "missing_required_fields": sorted(v.missing_required_fields),
            "invalid_enum_values": sorted(v.invalid_enum_values),
            "schema_errors": sorted(v.schema_errors),
            "has_parse_error": v.parse_error is not None,
        }

    def _planning_dict(v: PlanningValidationResult | None) -> dict[str, Any] | None:
        if v is None:
            return None
        return {
            "plan_schema_name": v.plan_schema_name,
            "missing_steps": sorted(v.missing_steps),
            "authority_violations": sorted(v.authority_violations),
            "schema_errors": sorted(v.schema_errors),
        }

    def _context_dict(v: ContextOverflowMetadata | None) -> dict[str, Any] | None:
        if v is None:
            return None
        return {
            "estimated_input_chars": v.estimated_input_chars,
            "model_context_tokens": v.model_context_tokens,
            "authority_required": v.authority_required,
            "authority_items_present": v.authority_items_present,
            "authority_items_raw": v.authority_items_raw,
            "rebuild_attempts": v.rebuild_attempts,
        }

    def _authority_dict(v: AuthorityViolationData | None) -> dict[str, Any] | None:
        if v is None:
            return None
        return {
            "touched_authority_paths": sorted(v.touched_authority_paths),
            "attempted_state_advancement": v.attempted_state_advancement,
            "ignored_immutable_authority": v.ignored_immutable_authority,
            "summary_substituted_for_raw": v.summary_substituted_for_raw,
            "integration_state_mismatch": v.integration_state_mismatch,
        }

    data: dict[str, Any] = {
        "task_id": inputs.task_id,
        "role": inputs.role.value,
        "task_risk": inputs.task_risk.value,
        "stage": inputs.stage,
        "provider_error": _provider_error_dict(inputs.provider_error),
        "deterministic_validation": _validation_dict(inputs.deterministic_validation),
        "structured_output_validation": _structured_dict(inputs.structured_output_validation),
        "planning_validation": _planning_dict(inputs.planning_validation),
        "review_findings": sorted(inputs.review_findings),
        "provider_id": inputs.provider_id,
        "model_id": inputs.model_id,
        "route_id": inputs.route_id,
        "failure_domain": inputs.failure_domain,
        "context_overflow": _context_dict(inputs.context_overflow),
        "authority_violation": _authority_dict(inputs.authority_violation),
        "integration_evidence": sorted(inputs.integration_evidence),
        "cancelled": inputs.cancelled,
    }
    return redact(data) or {}


def failure_classification_fingerprint(
    inputs: FailureClassifierInput,
    category: FailureCategory | None = None,
    subtype: FailureSubtype | None = None,
) -> str:
    """Deterministic fingerprint of all decision-driving inputs.

    Same failure state -> same fingerprint. Material failure change -> different
    fingerprint. Excludes arbitrary secret-bearing logs and volatile IDs.
    """
    data = _normalize_classification_input(inputs)
    if category is not None:
        data["category"] = category.value
    if subtype is not None:
        data["subtype"] = subtype.value
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
