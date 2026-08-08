from __future__ import annotations

import unittest

from src.telemetry.outcomes import (
    ModelLearningSignal,
    OutcomeAttribution,
    OutcomeKind,
    TaskOutcome,
    model_failure,
    provider_failure,
    successful_outcome,
)


class TaskOutcomeTests(unittest.TestCase):
    def test_success_is_positive_model_learning_signal(self) -> None:
        outcome = successful_outcome()
        self.assertEqual(ModelLearningSignal.POSITIVE, outcome.model_learning_signal)
        self.assertTrue(outcome.affects_model_quality)

    def test_quota_exhaustion_never_penalizes_model_quality(self) -> None:
        outcome = provider_failure("QUOTA_EXHAUSTED", "provider quota depleted")
        self.assertEqual(OutcomeKind.PROVIDER_FAILURE, outcome.kind)
        self.assertEqual(OutcomeAttribution.PROVIDER, outcome.attribution)
        self.assertEqual(ModelLearningSignal.NONE, outcome.model_learning_signal)
        self.assertFalse(outcome.affects_model_quality)

    def test_provider_unavailable_never_penalizes_model_quality(self) -> None:
        outcome = provider_failure("PROVIDER_UNAVAILABLE")
        self.assertFalse(outcome.affects_model_quality)

    def test_invalid_model_output_is_negative_model_signal(self) -> None:
        outcome = model_failure(OutcomeKind.INVALID_MODEL_OUTPUT, "INVALID_JSON")
        self.assertEqual(ModelLearningSignal.NEGATIVE, outcome.model_learning_signal)

    def test_authority_violation_is_negative_model_signal(self) -> None:
        outcome = model_failure(OutcomeKind.AUTHORITY_VIOLATION, "FUTURE_STEP_LEAKAGE")
        self.assertEqual(ModelLearningSignal.NEGATIVE, outcome.model_learning_signal)

    def test_model_validation_failure_is_negative_model_signal(self) -> None:
        outcome = model_failure(
            OutcomeKind.DETERMINISTIC_VALIDATION_FAILURE,
            "TEST_FAILURE",
        )
        self.assertTrue(outcome.affects_model_quality)

    def test_system_failure_does_not_penalize_model(self) -> None:
        outcome = TaskOutcome(
            OutcomeKind.SYSTEM_FAILURE,
            OutcomeAttribution.SYSTEM,
            reason_code="WORKTREE_IO_ERROR",
        )
        self.assertEqual(ModelLearningSignal.NONE, outcome.model_learning_signal)

    def test_invalid_attribution_combinations_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TaskOutcome(OutcomeKind.PROVIDER_FAILURE, OutcomeAttribution.MODEL)
        with self.assertRaises(ValueError):
            TaskOutcome(OutcomeKind.SYSTEM_FAILURE, OutcomeAttribution.PROVIDER)
        with self.assertRaises(ValueError):
            TaskOutcome(OutcomeKind.SUCCESS, OutcomeAttribution.MODEL)

    def test_model_failure_helper_refuses_provider_failure_kind(self) -> None:
        with self.assertRaises(ValueError):
            model_failure(OutcomeKind.PROVIDER_FAILURE)


if __name__ == "__main__":
    unittest.main()
