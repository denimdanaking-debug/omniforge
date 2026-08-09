"""Bounded retry/backoff policy to prevent hot-loop retries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Jitter(Protocol):
    """Optional jitter strategy. Default is no jitter for determinism."""

    def apply(self, delay_seconds: float, attempt: int) -> float: ...


class NoJitter:
    """Deterministic: no jitter."""

    def apply(self, delay_seconds: float, attempt: int) -> float:
        return delay_seconds


@dataclass(frozen=True)
class BackoffPolicy:
    """Stepped exponential backoff with a hard ceiling."""

    steps: tuple[float, ...] = (5.0, 15.0, 30.0, 60.0, 120.0, 300.0)
    max_seconds: float = 600.0
    jitter: Jitter | None = None

    def delay_for_attempt(self, attempt: int) -> float:
        """Return the backoff delay for the given 0-based attempt index."""
        if attempt < 0:
            raise ValueError("attempt must be non-negative")
        if attempt < len(self.steps):
            base = self.steps[attempt]
        else:
            base = self.steps[-1] if self.steps else 0.0
        base = min(base, self.max_seconds)
        jitter = self.jitter or NoJitter()
        return jitter.apply(base, attempt)


@dataclass(frozen=True)
class RetryBudget:
    """Per-route retry budget."""

    max_consecutive_failures_before_unavailable: int = 3
    max_attempts_per_window: int | None = None
    window_seconds: float | None = None

    def exceeds_consecutive(self, consecutive_failures: int) -> bool:
        return consecutive_failures >= self.max_consecutive_failures_before_unavailable


@dataclass(frozen=True)
class HotLoopPolicy:
    """Determines which normalized errors are never retried automatically."""

    never_retry: frozenset[str] = frozenset(
        {
            "QUOTA_EXHAUSTED",
            "AUTH_FAILURE",
            "UNSUPPORTED_CAPABILITY",
            "CONTEXT_OVERFLOW",
            "INVALID_MODEL_OUTPUT",
            "TASK_FAILURE",
            "CANCELLED",
        }
    )

    def can_retry(self, error_code: str) -> bool:
        return error_code not in self.never_retry
