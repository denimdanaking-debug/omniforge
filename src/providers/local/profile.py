"""Local endpoint profile and route configuration.

A local endpoint profile captures the runtime-specific route information for a
local model without conflating the runtime name with the model identity. The same
underlying model can be served by Ollama, vLLM, llama.cpp, etc., each represented
by a distinct ``LocalEndpointProfile`` and therefore a distinct route identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.routing.capabilities import DeploymentMode
from src.routing.inference_route import InferenceRouteIdentity


class LocalRuntimeKind(StrEnum):
    """Known local inference runtimes."""

    OLLAMA = "ollama"
    LLAMA_CPP = "llama.cpp"
    VLLM = "vllm"
    LM_STUDIO = "lm_studio"
    SGLANG = "sglang"
    GENERIC = "generic"


@dataclass(frozen=True)
class LocalEndpointProfile:
    """Configuration for a local OpenAI-compatible inference endpoint.

    Contains no credentials and no SDK-specific objects. The runtime kind is a
    label, not a model family.
    """

    runtime_kind: LocalRuntimeKind
    base_url: str
    route_identity: InferenceRouteIdentity
    failure_domain: str
    capability_metadata: dict[str, Any] = field(default_factory=dict)
    health_endpoint: str | None = None
    models_endpoint: str | None = None

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url must be non-empty")
        if not self.failure_domain.strip():
            raise ValueError("failure_domain must be non-empty")

    @property
    def deployment_mode(self) -> DeploymentMode:
        return DeploymentMode.LOCAL


@dataclass(frozen=True)
class LocalModelConfig:
    """A locally-served model resolved to a specific local endpoint profile."""

    model_id: str
    family: str
    profile: LocalEndpointProfile
    explicit_capabilities: dict[str, Any] = field(default_factory=dict)
