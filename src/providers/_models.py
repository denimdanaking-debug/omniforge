"""Shared helpers for provider model descriptor construction.

These helpers wrap the canonical ModelIdentity and ModelCapabilities types so
that provider adapter modules can declare their supported models without
repeating boilerplate or accidentally mutating canonical identity objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.routing.capabilities import CostMetadata, DeploymentMode, ModelCapabilities, RateMetadata
from src.routing.model_identity import ModelIdentity, ModelLifecycle


@dataclass(frozen=True)
class ModelDescriptor:
    """Provider-side model descriptor used to build canonical identities."""

    model_id: str
    family: str
    version: str | None = None
    revision: str | None = None
    lifecycle: ModelLifecycle = ModelLifecycle.NORMAL
    context_tokens: int = 128_000
    structured_output: bool = True
    tool_use: bool = True
    streaming: bool = True
    reasoning: bool = False
    code_generation: bool = True
    multimodal: bool = False
    deployment_mode: DeploymentMode = DeploymentMode.CLOUD
    cost_input_per_million: float | None = None
    cost_output_per_million: float | None = None
    cost_cached_input_per_million: float | None = None
    max_concurrency: int | None = None
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    capability_metadata: dict[str, object] | None = None

    def to_identity(self) -> ModelIdentity:
        """Build a canonical ModelIdentity from this descriptor."""
        return ModelIdentity(
            model_id=self.model_id,
            family=self.family,
            version=self.version,
            revision=self.revision,
            lifecycle=self.lifecycle,
            capability_metadata=self.capability_metadata or {},
        )

    def to_capabilities(self) -> ModelCapabilities:
        """Build canonical ModelCapabilities from this descriptor."""
        return ModelCapabilities(
            context_tokens=self.context_tokens,
            structured_output=self.structured_output,
            tool_use=self.tool_use,
            streaming=self.streaming,
            reasoning=self.reasoning,
            code_generation=self.code_generation,
            multimodal=self.multimodal,
            deployment_mode=self.deployment_mode,
            cost=CostMetadata(
                input_per_million=self.cost_input_per_million,
                output_per_million=self.cost_output_per_million,
                cached_input_per_million=self.cost_cached_input_per_million,
            ),
            rate=RateMetadata(
                max_concurrency=self.max_concurrency,
                requests_per_minute=self.requests_per_minute,
                tokens_per_minute=self.tokens_per_minute,
            ),
        )
