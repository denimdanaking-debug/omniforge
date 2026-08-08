"""Architecture tests for the enterprise-route abstraction.

These tests prove that enterprise routes are configurable data rather than
hardcoded orchestration branches, and that the abstraction contains no
credentials or SDK-specific objects.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.routing.enterprise import EnterprisePlatform, EnterpriseRouteConfig
from src.routing.inference_route import InferenceRouteIdentity, RouteType


def _route_identity() -> InferenceRouteIdentity:
    return InferenceRouteIdentity(
        route_id="bedrock-us-east-1-claude",
        provider_id="aws",
        route_type=RouteType.ENTERPRISE,
        endpoint_key="bedrock://us-east-1/anthropic.claude-3-sonnet",
        failure_domain="us-east-1.bedrock.amazonaws.com",
    )


def _direct_route() -> InferenceRouteIdentity:
    return InferenceRouteIdentity(
        route_id="anthropic-direct-claude",
        provider_id="anthropic",
        route_type=RouteType.DIRECT,
        endpoint_key="https://api.anthropic.com/v1",
        failure_domain="anthropic.com",
    )


@pytest.mark.architecture
async def test_all_platforms_are_representable_as_data() -> None:
    for platform in EnterprisePlatform:
        config = EnterpriseRouteConfig(
            route_identity=_route_identity(),
            platform=platform,
            region="us-east-1",
            endpoint_or_deployment_id="deployment-1",
            failure_domain=f"{platform.value}.example",
            underlying_provider_id="anthropic",
            underlying_model_id="claude-sonnet",
            capability_metadata={"streaming": True},
            administrative_metadata={"owner": "platform-team"},
        )
        assert config.platform == platform
        assert config.route_identity is not None


@pytest.mark.architecture
async def test_config_contains_no_credentials() -> None:
    config = EnterpriseRouteConfig(
        route_identity=_route_identity(),
        platform=EnterprisePlatform.AWS_BEDROCK,
    )
    data = str(config)
    forbidden = {"api_key", "secret", "password", "token", "credential"}
    assert not any(term in data.lower() for term in forbidden)


@pytest.mark.architecture
async def test_config_has_no_sdk_objects() -> None:
    config = EnterpriseRouteConfig(
        route_identity=_route_identity(),
        platform=EnterprisePlatform.AZURE_AI,
        capability_metadata={"supports_streaming": True},
    )
    # The dataclass should contain only primitive/routing types.
    assert isinstance(config.route_identity, InferenceRouteIdentity)
    assert isinstance(config.platform, EnterprisePlatform)
    assert isinstance(config.capability_metadata, dict)


@pytest.mark.architecture
async def test_failure_domain_is_resolved() -> None:
    config = EnterpriseRouteConfig(
        route_identity=_route_identity(),
        platform=EnterprisePlatform.GOOGLE_VERTEX,
        region="us-central1",
    )
    assert "google_vertex" in config.resolved_failure_domain
    assert "us-central1" in config.resolved_failure_domain


@pytest.mark.architecture
async def test_route_identity_is_required() -> None:
    with pytest.raises(ValueError):
        EnterpriseRouteConfig(route_identity=None, platform=EnterprisePlatform.AWS_BEDROCK)  # type: ignore[arg-type]


@pytest.mark.architecture
async def test_platform_must_be_non_empty() -> None:
    with pytest.raises(ValueError):
        EnterpriseRouteConfig(route_identity=_route_identity(), platform=EnterprisePlatform(""))


@pytest.mark.architecture
def test_core_orchestration_has_no_enterprise_branching() -> None:
    """Core orchestration must not contain provider-specific enterprise branches.

    Enterprise routes should be handled as data through the enterprise-route
    abstraction, not via ``if bedrock / if azure / if vertex`` conditionals.
    """
    orchestration_dir = Path("src/orchestration")
    enterprise_platforms = {
        "bedrock",
        "azure",
        "vertex",
        "aws_bedrock",
        "azure_ai",
        "google_vertex",
    }
    violations: list[str] = []
    for path in orchestration_dir.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.lower()
                if any(platform in lowered for platform in enterprise_platforms):
                    violations.append(f"{path}:{node.lineno}: string constant {node.value!r}")
            if isinstance(node, ast.Name):
                lowered = node.id.lower()
                if any(platform in lowered for platform in enterprise_platforms):
                    violations.append(f"{path}:{node.lineno}: name {node.id!r}")
    assert not violations, (
        "core orchestration contains enterprise-specific branching: " + "; ".join(violations[:10])
    )


@pytest.mark.architecture
def test_enterprise_module_has_no_sdk_imports() -> None:
    """The enterprise route module must not depend on cloud SDKs."""
    source = Path("src/routing/enterprise.py").read_text(encoding="utf-8")
    forbidden_sdk_prefixes = ("boto3", "azure", "google.cloud", "vertexai")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            for prefix in forbidden_sdk_prefixes:
                assert prefix not in stripped, f"enterprise module imports SDK: {stripped}"


@pytest.mark.architecture
def test_enterprise_config_requires_enterprise_route() -> None:
    with pytest.raises(ValueError, match="EnterpriseRouteConfig requires RouteType.ENTERPRISE"):
        EnterpriseRouteConfig(
            route_identity=_direct_route(),
            platform=EnterprisePlatform.AWS_BEDROCK,
        )


@pytest.mark.architecture
def test_enterprise_config_rejects_gateway_route() -> None:
    gateway_route = InferenceRouteIdentity(
        route_id="openrouter-claude",
        provider_id="openrouter",
        route_type=RouteType.GATEWAY,
        endpoint_key="openrouter://anthropic/claude-sonnet",
        failure_domain="openrouter.ai",
    )
    with pytest.raises(ValueError, match="EnterpriseRouteConfig requires RouteType.ENTERPRISE"):
        EnterpriseRouteConfig(
            route_identity=gateway_route,
            platform=EnterprisePlatform.AWS_BEDROCK,
        )


@pytest.mark.architecture
def test_enterprise_config_rejects_local_route() -> None:
    local_route = InferenceRouteIdentity(
        route_id="ollama-qwen",
        provider_id="local-ollama",
        route_type=RouteType.LOCAL,
        endpoint_key="http://localhost:11434/v1",
        failure_domain="localhost:11434",
    )
    with pytest.raises(ValueError, match="EnterpriseRouteConfig requires RouteType.ENTERPRISE"):
        EnterpriseRouteConfig(
            route_identity=local_route,
            platform=EnterprisePlatform.AWS_BEDROCK,
        )
