"""Tests for Phase 10 failure classification."""

from __future__ import annotations

import pytest

from src.policy.risk import RiskLevel
from src.providers.errors import ProviderError, ProviderErrorCode
from src.recovery.failure_classification import (
    AuthorityViolationData,
    ContextOverflowMetadata,
    FailureCategory,
    FailureClassifier,
    FailureClassifierInput,
    FailureSubtype,
    PlanningValidationResult,
    Retryability,
    StructuredOutputValidationResult,
    ValidationResultSummary,
    failure_classification_fingerprint,
)
from src.routing.roles import ExecutionRole
from src.security.redaction import contains_secret

SENTINEL = "OMNIFORGE_TEST_SECRET_SENTINEL_PHASE10_777"


@pytest.fixture
def classifier() -> FailureClassifier:
    return FailureClassifier()


@pytest.fixture
def base_input() -> FailureClassifierInput:
    return FailureClassifierInput(
        task_id="task-1",
        role=ExecutionRole.CODING,
        task_risk=RiskLevel.R2_NORMAL,
        stage="execution",
    )


class TestInfrastructureClassification:
    def test_transient_transport_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_provider_error(
            ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT,
                message="connection reset",
            )
        )
        result = classifier.classify(inputs)
        assert result.category is FailureCategory.INFRASTRUCTURE_TRANSIENT
        assert result.subtype is FailureSubtype.TRANSIENT_TRANSPORT
        assert result.retryability is Retryability.YES
        assert not result.model_quality_effect
        assert result.provider_health_effect
        assert result.route_health_effect

    def test_provider_unavailable_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_provider_error(
            ProviderError(
                code=ProviderErrorCode.PROVIDER_UNAVAILABLE,
                message="provider down",
            )
        )
        result = classifier.classify(inputs)
        assert result.category is FailureCategory.INFRASTRUCTURE_UNAVAILABLE
        assert result.subtype is FailureSubtype.PROVIDER_UNAVAILABLE
        assert not result.model_quality_effect

    def test_rate_limited_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_provider_error(
            ProviderError(
                code=ProviderErrorCode.RATE_LIMITED,
                message="rate limited",
                retry_after_seconds=30,
            )
        )
        result = classifier.classify(inputs)
        assert result.category is FailureCategory.INFRASTRUCTURE_TRANSIENT
        assert result.subtype is FailureSubtype.RATE_LIMITED
        assert not result.model_quality_effect

    def test_quota_exhausted_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_provider_error(
            ProviderError(
                code=ProviderErrorCode.QUOTA_EXHAUSTED,
                message="quota exhausted",
            )
        )
        result = classifier.classify(inputs)
        assert result.category is FailureCategory.INFRASTRUCTURE_QUOTA
        assert result.subtype is FailureSubtype.QUOTA_EXHAUSTED
        assert result.retryability is Retryability.WAIT
        assert not result.model_quality_effect

    def test_auth_failure_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_provider_error(
            ProviderError(
                code=ProviderErrorCode.AUTH_FAILURE,
                message="auth failed",
            )
        )
        result = classifier.classify(inputs)
        assert result.category is FailureCategory.INFRASTRUCTURE_AUTH
        assert result.subtype is FailureSubtype.AUTH_FAILURE
        assert result.retryability is Retryability.NO
        assert not result.model_quality_effect

    def test_unsupported_capability_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_provider_error(
            ProviderError(
                code=ProviderErrorCode.UNSUPPORTED_CAPABILITY,
                message="unsupported",
            )
        )
        result = classifier.classify(inputs)
        assert result.category is FailureCategory.CAPABILITY_MISMATCH
        assert result.subtype is FailureSubtype.UNSUPPORTED_CAPABILITY
        assert result.retryability is Retryability.NO
        assert not result.model_quality_effect


class TestOutputClassification:
    def test_malformed_json_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_structured_output_validation(
            StructuredOutputValidationResult(parse_error="unexpected token")
        )
        result = classifier.classify(inputs)
        assert result.category is FailureCategory.STRUCTURED_OUTPUT_INVALID
        assert result.subtype is FailureSubtype.PARSE_FAILURE
        assert result.retryability is Retryability.BOUNDED
        assert result.model_quality_effect

    def test_schema_mismatch_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_structured_output_validation(
            StructuredOutputValidationResult(schema_errors=("field 'risk' required",))
        )
        result = classifier.classify(inputs)
        assert result.category is FailureCategory.STRUCTURED_OUTPUT_INVALID
        assert result.subtype is FailureSubtype.SCHEMA_MISMATCH
        assert result.model_quality_effect

    def test_missing_required_fields_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_structured_output_validation(
            StructuredOutputValidationResult(missing_required_fields=("risk",))
        )
        result = classifier.classify(inputs)
        assert result.subtype is FailureSubtype.MISSING_REQUIRED_FIELDS

    def test_invalid_plan_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_planning_validation(
            PlanningValidationResult(missing_steps=("validate",))
        )
        result = classifier.classify(inputs)
        assert result.category is FailureCategory.PLANNING_OUTPUT_INVALID
        assert result.subtype is FailureSubtype.MISSING_PLAN_STEPS
        assert result.model_quality_effect

    def test_plan_authority_violation_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_planning_validation(
            PlanningValidationResult(authority_violations=("docs/PROJECT_STATE.json",))
        )
        result = classifier.classify(inputs)
        assert result.subtype is FailureSubtype.PLAN_AUTHORITY_VIOLATION


class TestImplementationClassification:
    def test_compile_failure_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_deterministic_validation(
            ValidationResultSummary(
                validator="python -m compileall",
                passed=False,
                failing_check_names=("src/a.py",),
                exit_status=1,
            )
        )
        result = classifier.classify(inputs)
        assert result.category is FailureCategory.IMPLEMENTATION_DETERMINISTIC
        assert result.subtype is FailureSubtype.BUILD_FAILURE
        assert result.model_quality_effect

    def test_lint_failure_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_deterministic_validation(
            ValidationResultSummary(
                validator="ruff check",
                passed=False,
                failing_check_names=("E501",),
            )
        )
        result = classifier.classify(inputs)
        assert result.subtype is FailureSubtype.LINT_FAILURE

    def test_mypy_failure_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_deterministic_validation(
            ValidationResultSummary(
                validator="mypy",
                passed=False,
                failing_check_names=("arg-type",),
            )
        )
        result = classifier.classify(inputs)
        assert result.subtype is FailureSubtype.TYPE_CHECK_FAILURE

    def test_unit_test_failure_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_deterministic_validation(
            ValidationResultSummary(
                validator="pytest",
                passed=False,
                failing_check_names=("test_foo",),
            )
        )
        result = classifier.classify(inputs)
        assert result.subtype is FailureSubtype.TEST_FAILURE


class TestContextAndAuthority:
    def test_context_overflow_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_context_overflow(
            ContextOverflowMetadata(
                estimated_input_chars=10000,
                model_context_tokens=1000,
                authority_required=True,
            )
        )
        result = classifier.classify(inputs)
        assert result.category is FailureCategory.CONTEXT_CAPACITY
        assert result.subtype is FailureSubtype.CONTEXT_OVERFLOW
        assert result.retryability is Retryability.BOUNDED
        assert not result.model_quality_effect

    def test_authority_violation_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_authority_violation(
            AuthorityViolationData(
                touched_authority_paths=("docs/PROJECT_STATE.json",),
                attempted_state_advancement=True,
            )
        )
        result = classifier.classify(inputs)
        assert result.category is FailureCategory.AUTHORITY_VIOLATION
        assert result.retryability is Retryability.NO

    def test_cancelled_classified(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        inputs = base_input.with_cancelled(True)
        result = classifier.classify(inputs)
        assert result.category is FailureCategory.CANCELLED
        assert result.retryability is Retryability.NO


class TestFingerprint:
    def test_same_inputs_same_fingerprint(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        result1 = classifier.classify(base_input)
        result2 = classifier.classify(base_input)
        assert result1.deterministic_fingerprint == result2.deterministic_fingerprint

    def test_material_change_changes_fingerprint(
        self, classifier: FailureClassifier, base_input: FailureClassifierInput
    ) -> None:
        result1 = classifier.classify(base_input)
        inputs2 = base_input.with_provider_error(
            ProviderError(code=ProviderErrorCode.QUOTA_EXHAUSTED, message="quota")
        )
        result2 = classifier.classify(inputs2)
        assert result1.deterministic_fingerprint != result2.deterministic_fingerprint

    def test_fingerprint_excludes_secrets(self, base_input: FailureClassifierInput) -> None:
        inputs = base_input.with_provider_error(
            ProviderError(
                code=ProviderErrorCode.AUTH_FAILURE,
                message=f"bearer {SENTINEL}",
            )
        )
        fingerprint = failure_classification_fingerprint(inputs)
        assert not contains_secret(fingerprint, SENTINEL)

    def test_fingerprint_invariant_to_irrelevant_timestamps(
        self, base_input: FailureClassifierInput
    ) -> None:
        inputs1 = base_input.with_provider_error(
            ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT,
                message="error at 2026-01-01T00:00:00Z",
            )
        )
        inputs2 = base_input.with_provider_error(
            ProviderError(
                code=ProviderErrorCode.TRANSIENT_TRANSPORT,
                message="error at 2026-01-02T00:00:00Z",
            )
        )
        fp1 = failure_classification_fingerprint(inputs1)
        fp2 = failure_classification_fingerprint(inputs2)
        assert fp1 == fp2
