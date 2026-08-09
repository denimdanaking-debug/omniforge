"""Tests for Phase 10 failure recovery policy."""

from __future__ import annotations

import math

import pytest

from src.recovery.retry_policy import FailureRecoveryPolicy


class TestFailureRecoveryPolicy:
    def test_default_policy_is_finite_and_non_negative(self) -> None:
        policy = FailureRecoveryPolicy()
        for name, value in policy.to_dict().items():
            assert isinstance(value, int), name
            assert value >= 0, name
            assert math.isfinite(value), name

    def test_max_total_attempts_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_total_attempts"):
            FailureRecoveryPolicy(max_total_attempts=0)

    def test_negative_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_transient_retries"):
            FailureRecoveryPolicy(max_transient_retries=-1)

    def test_same_signature_threshold_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_same_signature_attempts"):
            FailureRecoveryPolicy(max_same_signature_attempts=0)

    def test_policy_round_trips_through_dict(self) -> None:
        policy = FailureRecoveryPolicy(
            max_total_attempts=5,
            max_structured_output_retries=1,
        )
        restored = FailureRecoveryPolicy.from_dict(policy.to_dict())
        assert restored == policy

    def test_partial_dict_uses_defaults(self) -> None:
        restored = FailureRecoveryPolicy.from_dict({"max_total_attempts": 7})
        assert restored.max_total_attempts == 7
        assert restored.max_transient_retries == 3
