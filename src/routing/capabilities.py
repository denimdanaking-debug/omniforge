"""Normalized model capabilities and deterministic eligibility checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class CapabilityError(ValueError):
    pass


class DeploymentMode(StrEnum):
    CLOUD = "cloud"
    LOCAL = "local"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class CostMetadata:
    input_per_million: float | None = None
    output_per_million: float | None = None
    cached_input_per_million: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("input_per_million", self.input_per_million),
            ("output_per_million", self.output_per_million),
            ("cached_input_per_million", self.cached_input_per_million),
        ):
            if value is not None and value < 0:
                raise CapabilityError(f"{name} cannot be negative")


@dataclass(frozen=True)
class RateMetadata:
    max_concurrency: int | None = None
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_concurrency", self.max_concurrency),
            ("requests_per_minute", self.requests_per_minute),
            ("tokens_per_minute", self.tokens_per_minute),
        ):
            if value is not None and value <= 0:
                raise CapabilityError(f"{name} must be positive when provided")


@dataclass(frozen=True)
class ModelCapabilities:
    context_tokens: int
    structured_output: bool = False
    tool_use: bool = False
    streaming: bool = False
    reasoning: bool = False
    code_generation: bool = False
    multimodal: bool = False
    deployment_mode: DeploymentMode = DeploymentMode.CLOUD
    cost: CostMetadata = field(default_factory=CostMetadata)
    rate: RateMetadata = field(default_factory=RateMetadata)
    supported_roles: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.context_tokens <= 0:
            raise CapabilityError("context_tokens must be positive")
        if any(not isinstance(role, str) or not role.strip() for role in self.supported_roles):
            raise CapabilityError("supported_roles must contain non-empty strings")


@dataclass(frozen=True)
class CapabilityRequirement:
    min_context_tokens: int = 1
    structured_output: bool = False
    tool_use: bool = False
    streaming: bool = False
    reasoning: bool = False
    code_generation: bool = False
    multimodal: bool = False
    allowed_deployment_modes: frozenset[DeploymentMode] = frozenset()
    required_roles: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.min_context_tokens <= 0:
            raise CapabilityError("min_context_tokens must be positive")
        if any(not isinstance(role, str) or not role.strip() for role in self.required_roles):
            raise CapabilityError("required_roles must contain non-empty strings")


@dataclass(frozen=True)
class CapabilityMatch:
    eligible: bool
    missing: tuple[str, ...] = ()


def match_capabilities(
    capabilities: ModelCapabilities, requirement: CapabilityRequirement
) -> CapabilityMatch:
    """Return deterministic eligibility; unsupported features never become prompt guesses."""

    missing: list[str] = []
    if capabilities.context_tokens < requirement.min_context_tokens:
        missing.append("context_tokens")

    boolean_requirements = (
        ("structured_output", requirement.structured_output, capabilities.structured_output),
        ("tool_use", requirement.tool_use, capabilities.tool_use),
        ("streaming", requirement.streaming, capabilities.streaming),
        ("reasoning", requirement.reasoning, capabilities.reasoning),
        ("code_generation", requirement.code_generation, capabilities.code_generation),
        ("multimodal", requirement.multimodal, capabilities.multimodal),
    )
    for name, required, supported in boolean_requirements:
        if required and not supported:
            missing.append(name)

    if (
        requirement.allowed_deployment_modes
        and capabilities.deployment_mode not in requirement.allowed_deployment_modes
    ):
        missing.append("deployment_mode")

    unsupported_roles = sorted(requirement.required_roles - capabilities.supported_roles)
    missing.extend(f"role:{role}" for role in unsupported_roles)
    return CapabilityMatch(eligible=not missing, missing=tuple(missing))


def filter_capable_models(
    candidates: dict[str, ModelCapabilities], requirement: CapabilityRequirement
) -> tuple[str, ...]:
    """Filter model IDs by hard capability requirements in deterministic order."""

    return tuple(
        model_id
        for model_id in sorted(candidates)
        if match_capabilities(candidates[model_id], requirement).eligible
    )
