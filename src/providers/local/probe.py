"""Local-model capability discovery and probing.

Probing discovers or confirms model capabilities without assuming that an
OpenAI-compatible endpoint supports every cloud feature. Explicit administrator
configuration always takes precedence over probe results, and absent metadata
remains unknown/conservative rather than being fabricated as support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.providers.errors import ProviderError
from src.providers.local.profile import LocalEndpointProfile, LocalModelConfig
from src.routing.capabilities import (
    CostMetadata,
    DeploymentMode,
    ModelCapabilities,
    RateMetadata,
)
from src.routing.model_identity import ModelIdentity


class CapabilitySource(StrEnum):
    """Provenance of a capability value."""

    EXPLICIT_CONFIG = "explicit_config"
    PROBE_RESULT = "probe_result"
    PROVIDER_METADATA = "provider_metadata"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CapabilityEvidence:
    """A single capability assertion with provenance."""

    capability: str
    supported: bool | None
    source: CapabilitySource
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityProbeResult:
    """Result of probing a local endpoint for model capabilities."""

    model_identity: ModelIdentity
    capabilities: ModelCapabilities
    evidence: tuple[CapabilityEvidence, ...] = ()
    errors: tuple[ProviderError, ...] = ()


class LocalCapabilityProber:
    """Probe a local OpenAI-compatible endpoint for model capabilities.

    Probing is explicit and never runs at import time. Probe failures do not
    permanently mark a model as incapable; missing evidence remains unknown.
    """

    def __init__(self, http_client: Any | None = None) -> None:
        self._http_client = http_client

    async def probe(
        self,
        profile: LocalEndpointProfile,
        model_id: str,
        family: str,
        explicit_capabilities: dict[str, Any] | None = None,
    ) -> CapabilityProbeResult:
        """Probe the endpoint and merge explicit config > probe > metadata > unknown."""
        explicit = explicit_capabilities or {}
        evidence: list[CapabilityEvidence] = []
        errors: list[ProviderError] = []
        metadata: dict[str, Any] = {}

        if explicit:
            for key, value in explicit.items():
                if isinstance(value, bool):
                    evidence.append(
                        CapabilityEvidence(
                            capability=key,
                            supported=value,
                            source=CapabilitySource.EXPLICIT_CONFIG,
                        )
                    )

        probe_metadata = await self._fetch_models_metadata(profile, model_id)
        if probe_metadata is not None:
            metadata.update(probe_metadata)
        else:
            evidence.append(
                CapabilityEvidence(
                    capability="probe",
                    supported=None,
                    source=CapabilitySource.UNKNOWN,
                    metadata={"reason": "models endpoint unavailable or probe failed"},
                )
            )

        context_tokens = self._resolve_context_tokens(explicit, metadata)
        structured_output = self._resolve_boolean("structured_output", explicit, metadata, evidence)
        tool_use = self._resolve_boolean("tool_use", explicit, metadata, evidence)
        streaming = self._resolve_boolean("streaming", explicit, metadata, evidence)
        reasoning = self._resolve_boolean("reasoning", explicit, metadata, evidence)
        code_generation = self._resolve_boolean(
            "code_generation", explicit, metadata, evidence, default=True
        )
        multimodal = self._resolve_boolean("multimodal", explicit, metadata, evidence)

        model_identity = ModelIdentity(
            model_id=model_id,
            family=family,
            capability_metadata={
                "deployment_mode": DeploymentMode.LOCAL.value,
                **metadata,
            },
        )
        capabilities = ModelCapabilities(
            context_tokens=context_tokens,
            structured_output=structured_output or False,
            tool_use=tool_use or False,
            streaming=streaming or False,
            reasoning=reasoning or False,
            code_generation=code_generation or False,
            multimodal=multimodal or False,
            deployment_mode=DeploymentMode.LOCAL,
            cost=CostMetadata(),
            rate=RateMetadata(),
        )
        return CapabilityProbeResult(
            model_identity=model_identity,
            capabilities=capabilities,
            evidence=tuple(evidence),
            errors=tuple(errors),
        )

    async def probe_model_config(self, model_config: LocalModelConfig) -> CapabilityProbeResult:
        """Convenience probe using a ``LocalModelConfig``."""
        return await self.probe(
            profile=model_config.profile,
            model_id=model_config.model_id,
            family=model_config.family,
            explicit_capabilities=model_config.explicit_capabilities,
        )

    async def _fetch_models_metadata(
        self, profile: LocalEndpointProfile, model_id: str
    ) -> dict[str, Any] | None:
        """Fetch model metadata from a local endpoint without destructive prompts.

        Returns None if the endpoint cannot be reached or does not expose metadata.
        """
        import json

        endpoint = profile.models_endpoint or "/v1/models"
        if not endpoint.startswith("/") and not endpoint.startswith("http"):
            endpoint = "/" + endpoint
        url = profile.base_url.rstrip("/") + endpoint
        try:
            if self._http_client is not None:
                body = await self._http_client.get(url)
            else:
                return None
            data = json.loads(body) if isinstance(body, str) else body
            models = data.get("data", []) if isinstance(data, dict) else []
            for entry in models:
                if not isinstance(entry, dict):
                    continue
                if entry.get("id") == model_id:
                    return dict(entry)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _resolve_context_tokens(self, explicit: dict[str, Any], metadata: dict[str, Any]) -> int:
        if "context_tokens" in explicit:
            return int(explicit["context_tokens"])
        for key in ("context_length", "max_context_length", "n_ctx", "context_window"):
            value = metadata.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return 4096

    def _resolve_boolean(
        self,
        capability: str,
        explicit: dict[str, Any],
        metadata: dict[str, Any],
        evidence: list[CapabilityEvidence],
        *,
        default: bool | None = None,
    ) -> bool | None:
        if capability in explicit:
            return bool(explicit[capability])
        metadata_keys = {
            "structured_output": {"supports_structured_output", "structured_output"},
            "tool_use": {"supports_tool_calls", "tool_calls", "function_calling"},
            "streaming": {"supports_streaming", "streaming"},
            "reasoning": {"supports_reasoning", "reasoning"},
            "code_generation": {"supports_code_generation", "code_generation"},
            "multimodal": {"supports_vision", "multimodal", "vision"},
        }
        for key in metadata_keys.get(capability, {capability}):
            value = metadata.get(key)
            if isinstance(value, bool):
                evidence.append(
                    CapabilityEvidence(
                        capability=capability,
                        supported=value,
                        source=CapabilitySource.PROVIDER_METADATA,
                        metadata={"key": key},
                    )
                )
                return value
        if default is not None:
            evidence.append(
                CapabilityEvidence(
                    capability=capability,
                    supported=default,
                    source=CapabilitySource.UNKNOWN,
                    metadata={"default": True},
                )
            )
            return default
        evidence.append(
            CapabilityEvidence(
                capability=capability,
                supported=None,
                source=CapabilitySource.UNKNOWN,
            )
        )
        return None
