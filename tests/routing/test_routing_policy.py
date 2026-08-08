"""Tests for administrative routing policy: enable/disable, pins, project overrides."""

from __future__ import annotations

import pytest

from src.routing.capabilities import DeploymentMode, ModelCapabilities
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity
from src.routing.policy import (
    EligibilityResult,
    ProjectRoutingPolicy,
    RoutingPin,
    RoutingPolicyEngine,
)
from src.routing.roles import ExecutionRole


def _provider_id() -> str:
    return "openai"


def _model(model_id: str = "gpt-4o") -> ModelIdentity:
    return ModelIdentity(model_id=model_id, family="gpt")


def _route(route_id: str = "openai-direct") -> InferenceRouteIdentity:
    return InferenceRouteIdentity(
        route_id=route_id,
        provider_id="openai",
        route_type=RouteType.DIRECT,
        endpoint_key="https://api.openai.com/v1",
        failure_domain="openai.com",
    )


def test_enabled_provider_is_eligible() -> None:
    engine = RoutingPolicyEngine(provider_enabled={"openai": True})
    result = engine.evaluate("openai", _model(), _route())
    assert result == EligibilityResult(True, "eligible")


def test_disabled_provider_excluded() -> None:
    engine = RoutingPolicyEngine(provider_enabled={"openai": False})
    result = engine.evaluate("openai", _model(), _route())
    assert result == EligibilityResult(False, "provider_disabled")


def test_disabled_model_excluded() -> None:
    engine = RoutingPolicyEngine(model_enabled={"gpt-4o": False})
    result = engine.evaluate("openai", _model("gpt-4o"), _route())
    assert result == EligibilityResult(False, "model_disabled")


def test_other_models_from_same_provider_remain_eligible() -> None:
    engine = RoutingPolicyEngine(model_enabled={"gpt-4o": False})
    result = engine.evaluate("openai", _model("gpt-4o-mini"), _route())
    assert result.eligible is True


def test_disabled_route_excluded() -> None:
    engine = RoutingPolicyEngine(route_enabled={"openai-direct": False})
    result = engine.evaluate("openai", _model(), _route("openai-direct"))
    assert result == EligibilityResult(False, "route_disabled")


def test_same_model_via_alternate_route_eligible() -> None:
    engine = RoutingPolicyEngine(route_enabled={"openai-direct": False})
    result = engine.evaluate("openai", _model(), _route("openrouter-openai"))
    assert result.eligible is True


def test_project_prohibits_provider() -> None:
    policy = ProjectRoutingPolicy(prohibited_provider_ids=frozenset({"openai"}))
    engine = RoutingPolicyEngine(project_policy=policy)
    result = engine.evaluate("openai", _model(), _route())
    assert result == EligibilityResult(False, "project_prohibited:provider")


def test_project_prohibits_model() -> None:
    policy = ProjectRoutingPolicy(prohibited_model_ids=frozenset({"gpt-4o"}))
    engine = RoutingPolicyEngine(project_policy=policy)
    result = engine.evaluate("openai", _model("gpt-4o"), _route())
    assert result == EligibilityResult(False, "project_prohibited:model")


def test_project_prohibits_route() -> None:
    policy = ProjectRoutingPolicy(prohibited_route_ids=frozenset({"openai-direct"}))
    engine = RoutingPolicyEngine(project_policy=policy)
    result = engine.evaluate("openai", _model(), _route("openai-direct"))
    assert result == EligibilityResult(False, "project_prohibited:route")


def test_project_deployment_mode_restriction() -> None:
    policy = ProjectRoutingPolicy(allowed_deployment_modes=frozenset({DeploymentMode.LOCAL}))
    engine = RoutingPolicyEngine(project_policy=policy)
    model = ModelIdentity(
        model_id="cloud-model",
        family="cloud",
        capability_metadata={"deployment_mode": DeploymentMode.CLOUD.value},
    )
    result = engine.evaluate("openai", model, _route())
    assert result == EligibilityResult(False, "project_prohibited:deployment_mode")


def test_manual_pin_succeeds_when_eligible() -> None:
    pin = RoutingPin(provider_id="openai", model_id="gpt-4o", route_id="openai-direct")
    engine = RoutingPolicyEngine(pin=pin)
    result = engine.evaluate("openai", _model("gpt-4o"), _route("openai-direct"))
    assert result.eligible is True


def test_pin_fails_when_provider_mismatches() -> None:
    pin = RoutingPin(provider_id="anthropic")
    engine = RoutingPolicyEngine(pin=pin)
    result = engine.evaluate("openai", _model(), _route())
    assert result == EligibilityResult(False, "pin_identity_mismatch:provider")


def test_pin_fails_when_model_mismatches() -> None:
    pin = RoutingPin(model_id="claude-sonnet")
    engine = RoutingPolicyEngine(pin=pin)
    result = engine.evaluate("openai", _model("gpt-4o"), _route())
    assert result == EligibilityResult(False, "pin_identity_mismatch:model")


def test_pin_fails_when_route_mismatches() -> None:
    pin = RoutingPin(route_id="openrouter-openai")
    engine = RoutingPolicyEngine(pin=pin)
    result = engine.evaluate("openai", _model(), _route("openai-direct"))
    assert result == EligibilityResult(False, "pin_identity_mismatch:route")


def test_pin_does_not_override_disabled_target() -> None:
    pin = RoutingPin(provider_id="openai")
    engine = RoutingPolicyEngine(
        provider_enabled={"openai": False},
        pin=pin,
    )
    result = engine.evaluate("openai", _model(), _route())
    assert result == EligibilityResult(False, "provider_disabled")


def test_pin_does_not_override_project_prohibition() -> None:
    pin = RoutingPin(provider_id="openai")
    policy = ProjectRoutingPolicy(prohibited_provider_ids=frozenset({"openai"}))
    engine = RoutingPolicyEngine(project_policy=policy, pin=pin)
    result = engine.evaluate("openai", _model(), _route())
    assert result == EligibilityResult(False, "project_prohibited:provider")


def test_project_prohibition_beats_pin_evaluation() -> None:
    pin = RoutingPin(provider_id="openai")
    policy = ProjectRoutingPolicy(prohibited_provider_ids=frozenset({"openai"}))
    engine = RoutingPolicyEngine(project_policy=policy, pin=pin)
    pin_result = engine.evaluate_pin()
    assert pin_result is not None
    assert pin_result.eligible is False


def test_capability_mismatch_reported() -> None:
    engine = RoutingPolicyEngine()
    model = ModelIdentity(
        model_id="tiny",
        family="tiny",
        capability_metadata={},
    )
    model_capabilities = ModelCapabilities(context_tokens=1000, streaming=True)
    result = engine.evaluate(
        "openai",
        model,
        _route(),
        model_capabilities=model_capabilities,
        role=ExecutionRole.CODING,
    )
    assert result.eligible is False
    assert "capability_mismatch" in result.reason


def test_filter_candidates_returns_eligible_only() -> None:
    engine = RoutingPolicyEngine(
        provider_enabled={"openai": True, "anthropic": True},
        model_enabled={"gpt-4o": True, "claude": False},
    )
    candidates = [
        ("openai", _model("gpt-4o"), _route("openai-direct")),
        ("anthropic", _model("claude"), _route("anthropic-direct")),
    ]
    result = engine.filter_candidates(candidates)
    assert len(result) == 1
    assert result[0][0] == "openai"


def test_project_policy_round_trip() -> None:
    policy = ProjectRoutingPolicy(
        prohibited_provider_ids=frozenset({"xai"}),
        allowed_deployment_modes=frozenset({DeploymentMode.CLOUD}),
        minimum_review_independence="independent",
        allow_exploration=False,
        routing_mode_override="legacy",
    )
    restored = ProjectRoutingPolicy.from_dict(policy.to_dict())
    assert restored == policy


def test_project_policy_rejects_invalid_independence() -> None:
    with pytest.raises(ValueError):
        ProjectRoutingPolicy(minimum_review_independence="invalid")


def test_project_policy_rejects_invalid_routing_mode_override() -> None:
    with pytest.raises(ValueError):
        ProjectRoutingPolicy(routing_mode_override="magic")


def test_pin_requires_at_least_one_field() -> None:
    with pytest.raises(ValueError):
        RoutingPin()
