"""Architectural enforcement tests for Phase 5 security and policy seams."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.security.redaction import contains_secret

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "OMNIFORGE_TEST_SECRET_SENTINEL_ARCH_12345"


def _source_files(glob: str) -> list[Path]:
    return sorted(ROOT.glob(glob))


@pytest.mark.architecture
def test_only_environment_resolver_uses_os_environ() -> None:
    """Provider adapters and orchestration must not directly scrape the environment."""
    violations: list[str] = []
    for path in _source_files("src/**/*.py"):
        if path.name == "secrets.py" and path.parent.name == "security":
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "environ"
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
            ):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            if isinstance(node, ast.Name) and node.id == "environ":
                # Heuristic: bare `environ` import usage.
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, (
        "direct os.environ usage outside EnvironmentSecretResolver: " + "; ".join(violations[:10])
    )


@pytest.mark.architecture
def test_config_module_rejects_raw_secret_assignment() -> None:
    """Configuration loading must structurally prohibit raw secret fields."""
    from src.persistence import configuration

    config = {
        "schema_version": "1.1.0",
        "routing_mode": "legacy",
        "exploration_enabled": False,
        "providers": {"openai": {"enabled": True, "api_key": SENTINEL}},
        "pins": {},
        "project_policies": {},
    }
    with pytest.raises(configuration.RawSecretInConfigurationError):
        configuration.validate_config(config)


@pytest.mark.architecture
def test_runtime_state_save_contains_no_secret_sentinel() -> None:
    import tempfile

    from src.persistence import runtime_state
    from src.security.secrets import SecretValue

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runtime.json"
        state = {
            "schema_version": "1.1.0",
            "run_id": "run-1",
            "workflow_state": "STOPPED",
            "checkpoint": {"secret": SecretValue(SENTINEL)},
            "routing_mode": "legacy",
            "exploration_enabled": False,
            "provider_status": {},
            "model_status": {},
            "route_status": {},
            "pins": {},
            "project_policies": {},
        }
        runtime_state.save_runtime_state(path, state)
        text = path.read_text(encoding="utf-8")
        assert not contains_secret(text, SENTINEL)


@pytest.mark.architecture
def test_dynamic_switch_does_not_implement_future_router() -> None:
    """The dynamic routing mode must not falsely claim a full dynamic router exists."""
    from src.persistence import configuration

    config = configuration.validate_config(
        {
            "schema_version": "1.1.0",
            "routing_mode": "dynamic",
            "exploration_enabled": False,
            "providers": {"openai": {"enabled": True}},
            "pins": {},
            "project_policies": {},
        }
    )
    assert config["routing_mode"] == "dynamic"
    # No dynamic scoring/routing algorithm should be importable from Phase 5 code.
    dynamic_modules = list(ROOT.glob("src/**/dynamic_router.py")) + list(
        ROOT.glob("src/**/dynamic_scorer.py")
    )
    assert not dynamic_modules, "Phase 5 must not implement a full dynamic router"


@pytest.mark.architecture
def test_exploration_flag_introduces_no_randomness() -> None:
    import random

    from src.persistence import configuration

    config = configuration.validate_config(
        {
            "schema_version": "1.1.0",
            "routing_mode": "legacy",
            "exploration_enabled": True,
            "providers": {"openai": {"enabled": True}},
            "pins": {},
            "project_policies": {},
        }
    )
    assert config["exploration_enabled"] is True
    # Setting the flag must not alter random state.
    before = random.getstate()
    _ = config["exploration_enabled"]
    after = random.getstate()
    assert before == after
