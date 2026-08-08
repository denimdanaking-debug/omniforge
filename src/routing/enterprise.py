"""Enterprise-route abstraction for future AWS Bedrock, Azure AI, and Vertex AI.

This module provides provider-neutral enterprise-route metadata. It does NOT
implement full provider integrations, does NOT depend on cloud SDKs, and does
NOT contain credentials. Core orchestration can represent an enterprise route as
plain configuration data rather than hardcoded ``if bedrock / if azure`` branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.routing.inference_route import InferenceRouteIdentity, RouteType


class EnterprisePlatform(StrEnum):
    """Supported enterprise inference platforms."""

    AWS_BEDROCK = "aws_bedrock"
    AZURE_AI = "azure_ai"
    GOOGLE_VERTEX = "google_vertex"


@dataclass(frozen=True)
class EnterpriseRouteConfig:
    """Provider-neutral configuration for an enterprise inference route.

    No credentials and no SDK-specific objects are stored here. The route carries
    enough metadata for core routing to treat enterprise routes as data rather
    than hardcoded branches.
    """

    route_identity: InferenceRouteIdentity
    platform: EnterprisePlatform
    region: str | None = None
    endpoint_or_deployment_id: str | None = None
    failure_domain: str | None = None
    underlying_provider_id: str | None = None
    underlying_model_id: str | None = None
    capability_metadata: dict[str, Any] = field(default_factory=dict)
    administrative_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.route_identity is None:
            raise ValueError("route_identity is required")
        if self.route_identity.route_type is not RouteType.ENTERPRISE:
            raise ValueError(
                "EnterpriseRouteConfig requires RouteType.ENTERPRISE, got "
                f"{self.route_identity.route_type.value}"
            )
        if not self.platform.value.strip():
            raise ValueError("platform must be non-empty")

    @property
    def resolved_failure_domain(self) -> str:
        """Return the effective failure domain for this enterprise route."""
        if self.failure_domain:
            return self.failure_domain
        platform_domain = {
            EnterprisePlatform.AWS_BEDROCK: "amazonaws.com",
            EnterprisePlatform.AZURE_AI: "azure.com",
            EnterprisePlatform.GOOGLE_VERTEX: "googleapis.com",
        }
        region = self.region or "unknown"
        return f"{self.platform.value}.{region}.{platform_domain.get(self.platform, 'unknown')}"
