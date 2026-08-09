from __future__ import annotations

import pytest

from src.persistence import configuration
from src.routing.dynamic.config import RouterConfig, RouterConfigError, load_router_config
from src.routing.roles import ExecutionRole


def test_default_config_loads() -> None:
    config = load_router_config({})
    assert config.factor_weights == RouterConfig().factor_weights
    assert config.exploration_enabled is False


def test_custom_weights_load() -> None:
    config = load_router_config(
        {
            "factor_weights": {"cost": 2.0, "latency": 0.0},
        }
    )
    assert config.factor_weights.cost == 2.0
    assert config.factor_weights.latency == 0.0


def test_negative_weight_rejected() -> None:
    with pytest.raises(RouterConfigError):
        load_router_config({"factor_weights": {"cost": -1.0}})


def test_nan_weight_rejected() -> None:
    with pytest.raises(RouterConfigError):
        load_router_config({"factor_weights": {"cost": float("nan")}})


def test_unknown_factor_rejected() -> None:
    with pytest.raises(RouterConfigError):
        load_router_config({"factor_weights": {"unknown_factor": 1.0}})


def test_all_zero_weights_rejected() -> None:
    zeros = dict.fromkeys(RouterConfig().factor_weights.__dict__, 0.0)
    with pytest.raises(RouterConfigError):
        load_router_config({"factor_weights": zeros})


def test_fallback_orders_load() -> None:
    config = load_router_config(
        {
            "emergency_fallback_orders": {
                "coding": ["openai:gpt-4o:openai-direct"],
            }
        }
    )
    assert ExecutionRole.CODING in config.emergency_fallback_orders


def test_invalid_fallback_role_rejected() -> None:
    with pytest.raises((RouterConfigError, ValueError)):
        load_router_config(
            {"emergency_fallback_orders": {"invalid_role": ["openai:gpt-4o:openai-direct"]}}
        )


def test_invalid_safety_margin_rejected() -> None:
    with pytest.raises(RouterConfigError):
        load_router_config({"default_safety_margin_fraction": 1.5})


def test_invalid_exploration_type_rejected() -> None:
    with pytest.raises(RouterConfigError):
        load_router_config({"exploration_enabled": "yes"})


def test_config_migration_adds_router_config() -> None:
    config = {
        "schema_version": "1.1.0",
        "routing_mode": "legacy",
        "exploration_enabled": False,
        "providers": {"openai": {"enabled": True, "models": {}, "routes": {}}},
        "pins": {},
        "project_policies": {},
    }
    result = configuration.validate_config(config)
    assert result["schema_version"] == "1.3.0"
    assert "router_config" in result
    assert "factor_weights" in result["router_config"]
    assert "priors" in result["router_config"]
    assert "risk_policy" in result


def test_extract_router_config_from_admin_state() -> None:
    config = {
        "schema_version": "1.3.0",
        "routing_mode": "dynamic",
        "exploration_enabled": False,
        "providers": {"openai": {"enabled": True, "models": {}, "routes": {}}},
        "pins": {},
        "project_policies": {},
        "router_config": {"default_safety_margin_fraction": 0.2},
        "risk_policy": {},
    }
    admin = configuration.extract_administrative_state(config)
    router_cfg = configuration.extract_router_config(admin)
    assert router_cfg.default_safety_margin_fraction == 0.2


def test_invalid_router_config_in_config_rejected() -> None:
    config = {
        "schema_version": "1.3.0",
        "routing_mode": "legacy",
        "exploration_enabled": False,
        "providers": {"openai": {"enabled": True, "models": {}, "routes": {}}},
        "pins": {},
        "project_policies": {},
        "router_config": {"factor_weights": {"cost": -1.0}},
        "risk_policy": {},
    }
    with pytest.raises(configuration.InvalidConfiguration):
        configuration.validate_config(config)
