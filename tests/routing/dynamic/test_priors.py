from __future__ import annotations

import math

import pytest

from src.routing.dynamic.priors import ModelRoutingPrior, PriorBlender
from src.routing.roles import ExecutionRole


def test_default_priors_loaded_as_data() -> None:
    blender = PriorBlender()
    assert len(blender.priors) > 0
    families = {p.model_id for p in blender.priors}
    assert "kimi" in families
    assert "qwen" in families
    assert "claude" in families
    assert "openai" in families
    assert "cursor" in families


def test_cold_start_prior_for_known_family() -> None:
    blender = PriorBlender()
    prior = blender.prior_for(model_id="kimi-k3", role=ExecutionRole.CODING, task_class="default")
    assert 0.0 < prior <= 1.0


def test_cold_start_default_for_unknown_family() -> None:
    blender = PriorBlender()
    prior = blender.prior_for(
        model_id="unknown-model", role=ExecutionRole.CODING, task_class="default"
    )
    assert prior == 0.5


def test_empirical_override_increases_with_evidence() -> None:
    blender = PriorBlender()
    prior = 0.5
    empirical = 0.9
    blended_1 = blender.blend(prior, empirical, 1)
    blended_10 = blender.blend(prior, empirical, 10)
    blended_100 = blender.blend(prior, empirical, 100)
    assert blended_1 < blended_10 < blended_100
    assert math.isclose(blended_100, empirical, rel_tol=0.05)


def test_no_empirical_uses_prior() -> None:
    blender = PriorBlender()
    result = blender.blend(0.7, None, 0)
    assert result == 0.7


def test_custom_prior_loading() -> None:
    prior = ModelRoutingPrior(
        model_id="custom",
        role=ExecutionRole.CODING,
        task_class=None,
        factor_name="expected_success",
        prior_value=0.95,
        confidence=5,
    )
    blender = PriorBlender([prior])
    value = blender.prior_for(model_id="custom", role=ExecutionRole.CODING, task_class="default")
    assert value == 0.95


def test_invalid_prior_value_rejected() -> None:
    with pytest.raises(ValueError):
        ModelRoutingPrior(
            model_id="x",
            role=None,
            task_class=None,
            factor_name="expected_success",
            prior_value=1.5,
        )


def test_invalid_confidence_rejected() -> None:
    with pytest.raises(ValueError):
        ModelRoutingPrior(
            model_id="x",
            role=None,
            task_class=None,
            factor_name="expected_success",
            prior_value=0.5,
            confidence=-1,
        )
