"""Model routing priors and prior/empirical blending."""

from __future__ import annotations

from dataclasses import dataclass

from src.routing.roles import ExecutionRole


@dataclass(frozen=True)
class ModelRoutingPrior:
    """A prior belief about a model's performance for a role/task class."""

    model_id: str
    role: ExecutionRole | None
    task_class: str | None
    factor_name: str
    prior_value: float
    confidence: int = 0

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not 0.0 <= self.prior_value <= 1.0:
            raise ValueError("prior_value must be between 0.0 and 1.0")
        if self.confidence < 0:
            raise ValueError("confidence must be non-negative")


def _default_priors() -> list[ModelRoutingPrior]:
    """Return conservative cold-start priors for known model families.

    Loaded as data, not hardcoded branches, to avoid brand preference.
    """
    priors: list[ModelRoutingPrior] = []
    base_models = {
        "kimi": {
            "coding": 0.78,
            "review": 0.75,
            "planning": 0.76,
            "debugging": 0.74,
            "repair": 0.73,
            "architecture": 0.75,
        },
        "qwen": {
            "coding": 0.77,
            "review": 0.74,
            "planning": 0.75,
            "debugging": 0.73,
            "repair": 0.72,
            "architecture": 0.74,
        },
        "claude": {
            "coding": 0.76,
            "review": 0.78,
            "planning": 0.77,
            "debugging": 0.75,
            "repair": 0.76,
            "architecture": 0.77,
        },
        "openai": {
            "coding": 0.77,
            "review": 0.76,
            "planning": 0.76,
            "debugging": 0.75,
            "repair": 0.75,
            "architecture": 0.76,
        },
        "cursor": {
            "coding": 0.75,
            "review": 0.73,
            "planning": 0.73,
            "debugging": 0.74,
            "repair": 0.73,
            "architecture": 0.73,
        },
    }
    default = 0.70
    for family, roles in base_models.items():
        for role_name, value in roles.items():
            try:
                role = ExecutionRole(role_name)
            except ValueError:
                continue
            priors.append(
                ModelRoutingPrior(
                    model_id=family,
                    role=role,
                    task_class=None,
                    factor_name="expected_success",
                    prior_value=value,
                    confidence=1,
                )
            )
        # Generic task_class default for the family.
        priors.append(
            ModelRoutingPrior(
                model_id=family,
                role=None,
                task_class="default",
                factor_name="expected_success",
                prior_value=default,
                confidence=1,
            )
        )
    return priors


class PriorBlender:
    """Blend prior beliefs with empirical evidence deterministically."""

    def __init__(self, priors: list[ModelRoutingPrior] | None = None) -> None:
        self._priors = priors if priors is not None else _default_priors()
        self._index: dict[tuple[str, str | None, str | None, str], ModelRoutingPrior] = {}
        for prior in self._priors:
            key = (
                prior.model_id,
                prior.role.value if prior.role else None,
                prior.task_class,
                prior.factor_name,
            )
            self._index[key] = prior

    @property
    def priors(self) -> tuple[ModelRoutingPrior, ...]:
        return tuple(self._priors)

    def prior_for(
        self,
        *,
        model_id: str,
        role: ExecutionRole,
        task_class: str,
        factor_name: str = "expected_success",
        default: float = 0.5,
    ) -> float:
        """Return the best-matching prior for a model/role/task."""
        # Collect known model_id prefixes from priors, longest first.
        model_prefixes = sorted(
            {key[0] for key in self._index if key[0]},
            key=len,
            reverse=True,
        )
        family = model_id
        for prefix in model_prefixes:
            if model_id.startswith(prefix):
                family = prefix
                break
        candidates = [
            (model_id, role.value, None, factor_name),
            (model_id, None, task_class, factor_name),
            (model_id, None, "default", factor_name),
            (family, role.value, None, factor_name),
            (family, None, task_class, factor_name),
            (family, None, "default", factor_name),
        ]
        for key in candidates:
            prior = self._index.get(key)
            if prior is not None:
                return prior.prior_value
        return default

    def blend(
        self,
        prior_value: float,
        empirical_value: float | None,
        evidence_count: int,
        *,
        evidence_growth_rate: float = 0.1,
    ) -> float:
        """Blend prior and empirical values using weighted average."""
        if empirical_value is None or evidence_count <= 0:
            return prior_value
        prior_weight = 1.0
        evidence_weight = evidence_count * evidence_growth_rate
        total_weight = prior_weight + evidence_weight
        blended = (prior_value * prior_weight + empirical_value * evidence_weight) / total_weight
        return max(0.0, min(1.0, blended))
