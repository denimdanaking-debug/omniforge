"""Router configuration loading and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.context.budget import BudgetType, ContextBudget
from src.routing.roles import ExecutionRole

from .priors import PriorBlender
from .request import DynamicRoutingRequest
from .scoring import ScoringWeights


@dataclass
class RoutingCoordinatorState:
    """Mutable routing state container."""

    last_selected_key: str | None = None
    failure_domain_counts: dict[str, int] = field(default_factory=dict)


class RouterConfigError(ValueError):
    """Raised when router configuration is invalid."""


@dataclass(frozen=True)
class RouterConfig:
    """Validated dynamic router configuration."""

    factor_weights: ScoringWeights = field(default_factory=ScoringWeights)
    priors: list[dict[str, Any]] = field(default_factory=list)
    emergency_fallback_orders: dict[ExecutionRole, list[str]] = field(default_factory=dict)
    cost_metadata: dict[str, Any] = field(default_factory=dict)
    default_safety_margin_fraction: float = 0.1
    exploration_enabled: bool = False
    state: RoutingCoordinatorState = field(default_factory=RoutingCoordinatorState)

    def __post_init__(self) -> None:
        if not 0.0 <= self.default_safety_margin_fraction <= 1.0:
            raise RouterConfigError("default_safety_margin_fraction must be between 0.0 and 1.0")

    @property
    def prior_blender(self) -> PriorBlender:
        priors = [self._prior_from_dict(p) for p in self.priors]
        return PriorBlender(priors if priors else None)

    def context_budget_for(self, request: DynamicRoutingRequest) -> ContextBudget | None:
        """Return context budget for a request, or None if no requirement."""
        if request.required_context_tokens is None:
            return None
        return ContextBudget(
            primary_budget=request.required_context_tokens,
            budget_type=BudgetType.TOKENS_ESTIMATE,
            safety_margin_fraction=self.default_safety_margin_fraction,
        )

    def _prior_from_dict(self, data: dict[str, Any]) -> Any:
        from .priors import ModelRoutingPrior

        role = data.get("role")
        return ModelRoutingPrior(
            model_id=data["model_id"],
            role=ExecutionRole(role) if role else None,
            task_class=data.get("task_class"),
            factor_name=data.get("factor_name", "expected_success"),
            prior_value=float(data["prior_value"]),
            confidence=int(data.get("confidence", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_weights": self.factor_weights.to_dict(),
            "priors": self.priors,
            "emergency_fallback_orders": {
                role.value: keys for role, keys in self.emergency_fallback_orders.items()
            },
            "cost_metadata": self.cost_metadata,
            "default_safety_margin_fraction": self.default_safety_margin_fraction,
            "exploration_enabled": self.exploration_enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RouterConfig:
        return load_router_config(data)


def _validate_weights(data: dict[str, Any]) -> None:
    """Validate factor weights are finite, non-negative, and known."""
    known = set(ScoringWeights.__dataclass_fields__.keys())
    for name, value in data.items():
        if name not in known:
            raise RouterConfigError(f"unknown factor weight: {name}")
        if not isinstance(value, (int, float)):
            raise RouterConfigError(f"weight {name} must be numeric")
        if not math.isfinite(value):
            raise RouterConfigError(f"weight {name} must be finite")
        if value < 0:
            raise RouterConfigError(f"weight {name} cannot be negative")
    if all(value == 0 for value in data.values()):
        raise RouterConfigError("at least one weight must be non-zero")


def _fallback_orders_from_dict(data: dict[str, list[str]]) -> dict[ExecutionRole, list[str]]:
    result: dict[ExecutionRole, list[str]] = {}
    for role_value, keys in data.items():
        role = ExecutionRole(role_value)
        if not isinstance(keys, list):
            raise RouterConfigError(f"fallback order for {role_value} must be a list")
        result[role] = [str(k) for k in keys]
    return result


def _default_fallback_orders() -> dict[ExecutionRole, list[str]]:
    return {
        ExecutionRole.CODING: [
            "openai:gpt-4o:openai-direct",
            "anthropic:claude-sonnet:anthropic-direct",
            "kimi:kimi-k3:kimi-direct",
            "qwen:qwen3.8-max:qwen-direct",
        ],
        ExecutionRole.REVIEW: [
            "anthropic:claude-sonnet:anthropic-direct",
            "openai:gpt-4o:openai-direct",
            "kimi:kimi-k3:kimi-direct",
            "qwen:qwen3.8-max:qwen-direct",
        ],
        ExecutionRole.HIGH_RISK_REVIEW: [
            "anthropic:claude-sonnet:anthropic-direct",
            "kimi:kimi-k3:kimi-direct",
            "qwen:qwen3.8-max:qwen-direct",
            "openai:gpt-4o:openai-direct",
        ],
        ExecutionRole.PLANNING: [
            "openai:gpt-4o:openai-direct",
            "anthropic:claude-sonnet:anthropic-direct",
            "kimi:kimi-k3:kimi-direct",
            "qwen:qwen3.8-max:qwen-direct",
        ],
        ExecutionRole.ARCHITECTURE: [
            "anthropic:claude-sonnet:anthropic-direct",
            "openai:gpt-4o:openai-direct",
            "kimi:kimi-k3:kimi-direct",
            "qwen:qwen3.8-max:qwen-direct",
        ],
        ExecutionRole.DEBUGGING: [
            "openai:gpt-4o:openai-direct",
            "kimi:kimi-k3:kimi-direct",
            "qwen:qwen3.8-max:qwen-direct",
            "anthropic:claude-sonnet:anthropic-direct",
        ],
        ExecutionRole.REPAIR: [
            "openai:gpt-4o:openai-direct",
            "kimi:kimi-k3:kimi-direct",
            "anthropic:claude-sonnet:anthropic-direct",
            "qwen:qwen3.8-max:qwen-direct",
        ],
    }


def load_router_config(config_dict: dict[str, Any]) -> RouterConfig:
    """Load and validate a router configuration dict."""
    factor_weights_data = config_dict.get("factor_weights", {})
    if factor_weights_data:
        _validate_weights(factor_weights_data)
    weights = (
        ScoringWeights.from_dict(factor_weights_data) if factor_weights_data else ScoringWeights()
    )

    priors = config_dict.get("priors", [])
    if not isinstance(priors, list):
        raise RouterConfigError("priors must be a list")

    fallback_orders_data = config_dict.get("emergency_fallback_orders", {})
    if fallback_orders_data:
        fallback_orders = _fallback_orders_from_dict(fallback_orders_data)
    else:
        fallback_orders = _default_fallback_orders()

    cost_metadata = config_dict.get("cost_metadata", {})
    if not isinstance(cost_metadata, dict):
        raise RouterConfigError("cost_metadata must be a dict")

    margin = config_dict.get("default_safety_margin_fraction", 0.1)
    if not isinstance(margin, (int, float)):
        raise RouterConfigError("default_safety_margin_fraction must be numeric")

    exploration = config_dict.get("exploration_enabled", False)
    if not isinstance(exploration, bool):
        raise RouterConfigError("exploration_enabled must be a boolean")

    return RouterConfig(
        factor_weights=weights,
        priors=priors,
        emergency_fallback_orders=fallback_orders,
        cost_metadata=cost_metadata,
        default_safety_margin_fraction=float(margin),
        exploration_enabled=exploration,
    )
