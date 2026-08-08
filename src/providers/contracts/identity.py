"""Stable identities for providers, models, and inference routes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderId:
    """Stable provider identity (e.g. 'anthropic', 'openai', 'moonshot')."""

    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("ProviderId must be a non-empty string")
        object.__setattr__(self, "value", self.value.lower().strip())


@dataclass(frozen=True, slots=True)
class ModelId:
    """Stable model identity independent of provider or route."""

    family: str
    version: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        if not self.family or not self.family.strip():
            raise ValueError("ModelId.family must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RouteId:
    """Inference route identity (e.g. direct API, OpenRouter, local endpoint)."""

    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("RouteId must be a non-empty string")
