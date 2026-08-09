"""Administrative routing policy: enable/disable, pins, and project overrides.

This module implements a single eligibility seam that routing can call before
preference/scoring. It keeps policy checks out of provider adapters and
orchestration branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.routing.capabilities import (
    CapabilityRequirement,
    DeploymentMode,
    ModelCapabilities,
    match_capabilities,
)
from src.routing.inference_route import InferenceRouteIdentity
from src.routing.model_identity import ModelIdentity
from src.routing.roles import ExecutionRole


@dataclass(frozen=True)
class RoutingPin:
    """Manual administrative narrowing of routing choices.

    A pin selects from eligible candidates; it does not override project
    prohibitions or administrative disablement.
    """

    provider_id: str | None = None
    model_id: str | None = None
    route_id: str | None = None

    def __post_init__(self) -> None:
        if all(field is None for field in (self.provider_id, self.model_id, self.route_id)):
            raise ValueError(
                "RoutingPin must specify at least one of provider_id, model_id, route_id"
            )


@dataclass(frozen=True)
class ProjectRoutingPolicy:
    """Project-specific overrides that can tighten but not weaken global policy."""

    prohibited_provider_ids: frozenset[str] = field(default_factory=frozenset)
    prohibited_model_ids: frozenset[str] = field(default_factory=frozenset)
    prohibited_route_ids: frozenset[str] = field(default_factory=frozenset)
    allowed_deployment_modes: frozenset[DeploymentMode] | None = None
    minimum_review_independence: str | None = None
    allow_exploration: bool | None = None
    routing_mode_override: str | None = None

    def __post_init__(self) -> None:
        valid_independence = {None, "same_provider", "same_model", "independent"}
        if self.minimum_review_independence not in valid_independence:
            raise ValueError(
                "minimum_review_independence must be one of "
                "same_provider, same_model, independent, or None"
            )
        if self.routing_mode_override is not None and self.routing_mode_override not in {
            "legacy",
            "dynamic",
        }:
            raise ValueError("routing_mode_override must be 'legacy' or 'dynamic'")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ProjectRoutingPolicy:
        modes = raw.get("allowed_deployment_modes")
        return cls(
            prohibited_provider_ids=frozenset(raw.get("prohibited_provider_ids", [])),
            prohibited_model_ids=frozenset(raw.get("prohibited_model_ids", [])),
            prohibited_route_ids=frozenset(raw.get("prohibited_route_ids", [])),
            allowed_deployment_modes=frozenset(DeploymentMode(m) for m in modes) if modes else None,
            minimum_review_independence=raw.get("minimum_review_independence"),
            allow_exploration=raw.get("allow_exploration"),
            routing_mode_override=raw.get("routing_mode_override"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "prohibited_provider_ids": sorted(self.prohibited_provider_ids),
            "prohibited_model_ids": sorted(self.prohibited_model_ids),
            "prohibited_route_ids": sorted(self.prohibited_route_ids),
        }
        if self.allowed_deployment_modes is not None:
            result["allowed_deployment_modes"] = sorted(
                m.value for m in self.allowed_deployment_modes
            )
        if self.minimum_review_independence is not None:
            result["minimum_review_independence"] = self.minimum_review_independence
        if self.allow_exploration is not None:
            result["allow_exploration"] = self.allow_exploration
        if self.routing_mode_override is not None:
            result["routing_mode_override"] = self.routing_mode_override
        return result


@dataclass(frozen=True)
class EligibilityResult:
    """Outcome of a policy eligibility check."""

    eligible: bool
    reason: str


class RoutingPolicyEngine:
    """Evaluate administrative eligibility for provider/model/route candidates.

    Precedence:
    1. Hard global schema/validation (enforced before this engine is called).
    2. Project prohibitions and stricter overrides.
    3. Administrative enable/disable.
    4. Manual pin narrowing.
    5. Capability/risk requirements (delegated to existing matcher).
    """

    def __init__(
        self,
        *,
        provider_enabled: dict[str, bool] | None = None,
        model_enabled: dict[str, bool] | None = None,
        route_enabled: dict[str, bool] | None = None,
        project_policy: ProjectRoutingPolicy | None = None,
        pin: RoutingPin | None = None,
    ) -> None:
        self._provider_enabled = provider_enabled or {}
        self._model_enabled = model_enabled or {}
        self._route_enabled = route_enabled or {}
        self._project_policy = project_policy or ProjectRoutingPolicy()
        self._pin = pin

    def evaluate(
        self,
        provider_id: str,
        model: ModelIdentity,
        route: InferenceRouteIdentity,
        model_capabilities: ModelCapabilities | None = None,
        role: ExecutionRole | None = None,
    ) -> EligibilityResult:
        # Project prohibitions.
        if provider_id in self._project_policy.prohibited_provider_ids:
            return EligibilityResult(False, "project_prohibited:provider")
        if model.model_id in self._project_policy.prohibited_model_ids:
            return EligibilityResult(False, "project_prohibited:model")
        if route.route_id in self._project_policy.prohibited_route_ids:
            return EligibilityResult(False, "project_prohibited:route")

        if self._project_policy.allowed_deployment_modes is not None:
            mode = model.capability_metadata.get("deployment_mode")
            if mode is not None:
                deployment = DeploymentMode(mode)
                if deployment not in self._project_policy.allowed_deployment_modes:
                    return EligibilityResult(False, "project_prohibited:deployment_mode")

        # Administrative enable/disable.
        if not self._provider_enabled.get(provider_id, True):
            return EligibilityResult(False, "provider_disabled")
        if not self._model_enabled.get(model.model_id, True):
            return EligibilityResult(False, "model_disabled")
        if not self._route_enabled.get(route.route_id, True):
            return EligibilityResult(False, "route_disabled")

        # Manual pin narrowing.
        if self._pin is not None:
            if self._pin.provider_id is not None and self._pin.provider_id != provider_id:
                return EligibilityResult(False, "pin_identity_mismatch:provider")
            if self._pin.model_id is not None and self._pin.model_id != model.model_id:
                return EligibilityResult(False, "pin_identity_mismatch:model")
            if self._pin.route_id is not None and self._pin.route_id != route.route_id:
                return EligibilityResult(False, "pin_identity_mismatch:route")

        # Capability/risk requirement.
        if model_capabilities is not None:
            capability_req = CapabilityRequirement(
                min_context_tokens=1,
                required_roles=frozenset({role.value}) if role else frozenset(),
            )
            match = match_capabilities(model_capabilities, capability_req)
            if not match.eligible:
                return EligibilityResult(False, f"capability_mismatch:{':'.join(match.missing)}")

        return EligibilityResult(True, "eligible")

    def filter_candidates(
        self,
        candidates: list[tuple[str, ModelIdentity, InferenceRouteIdentity]],
        model_capabilities: dict[str, ModelCapabilities] | None = None,
        role: ExecutionRole | None = None,
    ) -> list[tuple[str, ModelIdentity, InferenceRouteIdentity]]:
        """Return candidates that survive policy checks.

        ``model_capabilities`` maps model_id -> capabilities for the capability check.
        """
        caps = model_capabilities or {}
        return [
            (provider_id, model, route)
            for provider_id, model, route in candidates
            if self.evaluate(provider_id, model, route, caps.get(model.model_id), role).eligible
        ]

    def evaluate_pin(self) -> EligibilityResult | None:
        """Return a diagnostic if the pin itself refers to disabled/prohibited targets."""
        if self._pin is None:
            return None

        if self._pin.provider_id is not None and not self._provider_enabled.get(
            self._pin.provider_id, True
        ):
            return EligibilityResult(False, "pin_target_disabled:provider")
        if self._pin.model_id is not None and not self._model_enabled.get(self._pin.model_id, True):
            return EligibilityResult(False, "pin_target_disabled:model")
        if self._pin.route_id is not None and not self._route_enabled.get(self._pin.route_id, True):
            return EligibilityResult(False, "pin_target_disabled:route")

        if self._pin.provider_id in self._project_policy.prohibited_provider_ids:
            return EligibilityResult(False, "pin_target_prohibited:provider")
        if self._pin.model_id in self._project_policy.prohibited_model_ids:
            return EligibilityResult(False, "pin_target_prohibited:model")
        if self._pin.route_id in self._project_policy.prohibited_route_ids:
            return EligibilityResult(False, "pin_target_prohibited:route")

        return EligibilityResult(True, "pin_eligible")
