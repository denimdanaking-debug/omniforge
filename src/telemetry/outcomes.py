"""Normalized task outcomes and learning attribution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class OutcomeKind(StrEnum):
    SUCCESS = "success"
    DETERMINISTIC_VALIDATION_FAILURE = "deterministic_validation_failure"
    AUTHORITY_VIOLATION = "authority_violation"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    TASK_FAILURE = "task_failure"
    PROVIDER_FAILURE = "provider_failure"
    SYSTEM_FAILURE = "system_failure"


class OutcomeAttribution(StrEnum):
    NONE = "none"
    MODEL = "model"
    PROVIDER = "provider"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ModelLearningSignal(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NONE = "none"


@dataclass(frozen=True)
class TaskOutcome:
    kind: OutcomeKind
    attribution: OutcomeAttribution
    reason_code: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.reason_code is not None and not self.reason_code.strip():
            raise ValueError("reason_code must be non-empty when provided")
        if self.detail is not None and not self.detail.strip():
            raise ValueError("detail must be non-empty when provided")

        if self.kind == OutcomeKind.SUCCESS and self.attribution != OutcomeAttribution.NONE:
            raise ValueError("successful outcomes must use NONE attribution")
        if self.kind == OutcomeKind.PROVIDER_FAILURE and self.attribution != OutcomeAttribution.PROVIDER:
            raise ValueError("provider failures must use PROVIDER attribution")
        if self.kind == OutcomeKind.SYSTEM_FAILURE and self.attribution != OutcomeAttribution.SYSTEM:
            raise ValueError("system failures must use SYSTEM attribution")

    @property
    def model_learning_signal(self) -> ModelLearningSignal:
        if self.kind == OutcomeKind.SUCCESS:
            return ModelLearningSignal.POSITIVE
        if self.attribution != OutcomeAttribution.MODEL:
            return ModelLearningSignal.NONE
        if self.kind in {
            OutcomeKind.DETERMINISTIC_VALIDATION_FAILURE,
            OutcomeKind.AUTHORITY_VIOLATION,
            OutcomeKind.INVALID_MODEL_OUTPUT,
            OutcomeKind.TASK_FAILURE,
        }:
            return ModelLearningSignal.NEGATIVE
        return ModelLearningSignal.NONE

    @property
    def affects_model_quality(self) -> bool:
        return self.model_learning_signal != ModelLearningSignal.NONE


def successful_outcome() -> TaskOutcome:
    return TaskOutcome(OutcomeKind.SUCCESS, OutcomeAttribution.NONE)


def provider_failure(reason_code: str, detail: str | None = None) -> TaskOutcome:
    return TaskOutcome(
        OutcomeKind.PROVIDER_FAILURE,
        OutcomeAttribution.PROVIDER,
        reason_code=reason_code,
        detail=detail,
    )


def model_failure(
    kind: OutcomeKind, reason_code: str | None = None, detail: str | None = None
) -> TaskOutcome:
    if kind not in {
        OutcomeKind.DETERMINISTIC_VALIDATION_FAILURE,
        OutcomeKind.AUTHORITY_VIOLATION,
        OutcomeKind.INVALID_MODEL_OUTPUT,
        OutcomeKind.TASK_FAILURE,
    }:
        raise ValueError(f"{kind.value} is not a model-failure outcome kind")
    return TaskOutcome(
        kind,
        OutcomeAttribution.MODEL,
        reason_code=reason_code,
        detail=detail,
    )
