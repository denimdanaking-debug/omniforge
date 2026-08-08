"""First-class model identity and model-specific reputation state."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

_MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._:/-]{1,127}$")


class ModelIdentityError(ValueError):
    pass


class ModelLifecycle(StrEnum):
    SHADOW = "SHADOW"
    LOW_RISK = "LOW_RISK"
    NORMAL = "NORMAL"
    HIGH_RISK = "HIGH_RISK"
    DISABLED = "DISABLED"


@dataclass(frozen=True)
class ModelIdentity:
    """Stable model identity independent of provider operational health."""

    model_id: str
    family: str
    version: str | None = None
    revision: str | None = None
    capability_metadata: Mapping[str, Any] = field(default_factory=dict)
    lifecycle: ModelLifecycle = ModelLifecycle.NORMAL

    def __post_init__(self) -> None:
        if not _MODEL_ID_PATTERN.fullmatch(self.model_id):
            raise ModelIdentityError(
                "model_id must be 2-128 lowercase letters/digits plus '.', '_', ':', '/' or '-'"
            )
        if not self.family.strip():
            raise ModelIdentityError("family must be non-empty")
        if self.version is not None and not self.version.strip():
            raise ModelIdentityError("version must be non-empty when provided")
        if self.revision is not None and not self.revision.strip():
            raise ModelIdentityError("revision must be non-empty when provided")
        if not isinstance(self.capability_metadata, Mapping):
            raise ModelIdentityError("capability_metadata must be a mapping")


@dataclass(frozen=True)
class ModelReputation:
    attempts: int = 0
    accepted: int = 0
    authority_violations: int = 0
    score_hint: float | None = None

    def __post_init__(self) -> None:
        if self.attempts < 0 or self.accepted < 0 or self.authority_violations < 0:
            raise ModelIdentityError("model reputation counters cannot be negative")
        if self.accepted > self.attempts:
            raise ModelIdentityError("accepted cannot exceed attempts")


@dataclass(frozen=True)
class ModelRegistration:
    identity: ModelIdentity
    reputation: ModelReputation = field(default_factory=ModelReputation)


class ModelRegistry:
    """Registry separating immutable identity from mutable model reputation."""

    def __init__(self) -> None:
        self._models: dict[str, ModelRegistration] = {}

    def register(self, identity: ModelIdentity) -> None:
        existing = self._models.get(identity.model_id)
        if existing is not None and existing.identity != identity:
            raise ModelIdentityError(
                f"model_id {identity.model_id!r} is already bound to another identity"
            )
        if existing is None:
            self._models[identity.model_id] = ModelRegistration(identity=identity)

    def get(self, model_id: str) -> ModelRegistration:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise ModelIdentityError(f"unknown model_id {model_id!r}") from exc

    def set_reputation(self, model_id: str, reputation: ModelReputation) -> None:
        registration = self.get(model_id)
        self._models[model_id] = replace(registration, reputation=reputation)

    def set_lifecycle(self, model_id: str, lifecycle: ModelLifecycle) -> None:
        registration = self.get(model_id)
        self._models[model_id] = replace(
            registration,
            identity=replace(registration.identity, lifecycle=lifecycle),
        )

    def registrations(self) -> tuple[ModelRegistration, ...]:
        return tuple(self._models[key] for key in sorted(self._models))
