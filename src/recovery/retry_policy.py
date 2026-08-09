"""Canonical failure recovery policy with bounded, deterministic limits.

Policy values are validated to be finite and non-negative. No unlimited sentinel
is accidentally allowed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FailureRecoveryPolicy:
    """Configurable limits for failure recovery behavior."""

    max_total_attempts: int = 10
    max_same_signature_attempts: int = 3
    max_transient_retries: int = 3
    max_structured_output_retries: int = 2
    max_planning_retries: int = 2
    max_same_model_repairs: int = 2
    max_context_rebuilds: int = 2
    max_provider_switches: int = 5
    max_model_switches: int = 5
    max_consecutive_infrastructure_retries: int = 3
    require_cross_provider_after_same_signature: int = 2

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.max_total_attempts == 0:
            raise ValueError("max_total_attempts must be positive")
        if self.max_same_signature_attempts == 0:
            raise ValueError("max_same_signature_attempts must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_total_attempts": self.max_total_attempts,
            "max_same_signature_attempts": self.max_same_signature_attempts,
            "max_transient_retries": self.max_transient_retries,
            "max_structured_output_retries": self.max_structured_output_retries,
            "max_planning_retries": self.max_planning_retries,
            "max_same_model_repairs": self.max_same_model_repairs,
            "max_context_rebuilds": self.max_context_rebuilds,
            "max_provider_switches": self.max_provider_switches,
            "max_model_switches": self.max_model_switches,
            "max_consecutive_infrastructure_retries": (self.max_consecutive_infrastructure_retries),
            "require_cross_provider_after_same_signature": (
                self.require_cross_provider_after_same_signature
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureRecoveryPolicy:
        expected = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs: dict[str, Any] = {}
        for name in expected:
            value = data.get(name)
            if value is None:
                value = cls.__dataclass_fields__[name].default
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
            kwargs[name] = value
        return cls(**kwargs)
