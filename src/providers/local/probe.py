"""Local-model capability discovery and probing.

Probing discovers or confirms model capabilities without assuming that an
OpenAI-compatible endpoint supports every cloud feature. Explicit administrator
configuration always takes precedence over probe results; provider metadata is
explicitly tagged as metadata (not a verified probe); and absent evidence remains
unknown/conservative rather than being fabricated as support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.providers._common import translate_exception
from src.providers.errors import ProviderError, ProviderErrorCode
from src.providers.identity import ProviderIdentity
from src.providers.local.profile import LocalEndpointProfile, LocalModelConfig
from src.routing.capabilities import (
    CostMetadata,
    DeploymentMode,
    ModelCapabilities,
    RateMetadata,
)
from src.routing.model_identity import ModelIdentity


class CapabilitySource(StrEnum):
    """Provenance of a capability assertion."""

    EXPLICIT_CONFIG = "explicit_config"
    PROBE_RESULT = "probe_result"
    PROVIDER_METADATA = "provider_metadata"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CapabilityEvidence:
    """A single capability assertion with provenance.

    ``supported`` may be ``None`` when evidence is absent; the final canonical
    ``ModelCapabilities`` boolean will then fall back to ``False`` for safety,
    but the evidence retains the distinction between "verified unsupported" and
    "unknown/conservative".
    """

    capability: str
    supported: bool | None
    source: CapabilitySource
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityProbeResult:
    """Result of probing a local endpoint for model capabilities.

    Probe failures are represented as normalized ``ProviderError`` values with
    route attribution. A failure does NOT permanently mark the model as
    incapable; capabilities with no evidence remain unknown/conservative.
    """

    model_identity: ModelIdentity
    capabilities: ModelCapabilities
    evidence: tuple[CapabilityEvidence, ...] = ()
    errors: tuple[ProviderError, ...] = ()


@dataclass(frozen=True)
class _MetadataFetchResult:
    metadata: dict[str, Any] | None = None
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

        for key, value in explicit.items():
            if isinstance(value, bool):
                evidence.append(
                    CapabilityEvidence(
                        capability=key,
                        supported=value,
                        source=CapabilitySource.EXPLICIT_CONFIG,
                    )
                )

        fetch_result = await self._fetch_models_metadata(profile, model_id)
        metadata = fetch_result.metadata or {}
        errors.extend(fetch_result.errors)

        # The only deterministic probe performed in Phase 4 is verifying that the
        # endpoint reports this model in its model list. It is cheap, safe, and
        # produces genuine PROBE_RESULT evidence without sending chat prompts.
        model_listed = fetch_result.metadata is not None
        evidence.append(
            CapabilityEvidence(
                capability="model_listed",
                supported=model_listed,
                source=CapabilitySource.PROBE_RESULT,
                metadata={"endpoint": profile.models_endpoint or "/v1/models"},
            )
        )

        context_tokens, context_source, context_metadata = self._resolve_context_tokens(
            explicit, metadata
        )
        evidence.append(
            CapabilityEvidence(
                capability="context_tokens",
                supported=context_source is not CapabilitySource.UNKNOWN,
                source=context_source,
                metadata=context_metadata,
            )
        )

        structured_output = self._resolve_boolean("structured_output", explicit, metadata, evidence)
        tool_use = self._resolve_boolean("tool_use", explicit, metadata, evidence)
        streaming = self._resolve_boolean("streaming", explicit, metadata, evidence)
        reasoning = self._resolve_boolean("reasoning", explicit, metadata, evidence)
        code_generation = self._resolve_boolean("code_generation", explicit, metadata, evidence)
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
    ) -> _MetadataFetchResult:
        """Fetch model metadata from a local endpoint without destructive prompts.

        Returns the model entry if the endpoint exposes a model list and the
        model is present. Returns normalized ``ProviderError`` values on failure.
        """
        if self._http_client is None:
            return _MetadataFetchResult(
                errors=(
                    ProviderError(
                        code=ProviderErrorCode.UNKNOWN,
                        message="No HTTP client configured for capability probe",
                        provider_id=self._local_provider_identity(profile),
                        route_id=profile.route_identity,
                        retryable=False,
                    ),
                )
            )

        endpoint = profile.models_endpoint or "/v1/models"
        if not endpoint.startswith("/") and not endpoint.startswith("http"):
            endpoint = "/" + endpoint
        url = profile.base_url.rstrip("/") + endpoint
        provider_id = self._local_provider_identity(profile)

        try:
            body = await self._http_client.get(url)
        except Exception as exc:  # noqa: BLE001
            return _MetadataFetchResult(
                errors=(translate_exception(exc, provider_id, profile.route_identity),)
            )

        try:
            data = json.loads(body) if isinstance(body, str) else body
        except Exception as exc:  # noqa: BLE001
            return _MetadataFetchResult(
                errors=(
                    ProviderError(
                        code=ProviderErrorCode.UNKNOWN,
                        message=f"Invalid metadata response from {url}: {exc}",
                        provider_id=provider_id,
                        route_id=profile.route_identity,
                        retryable=False,
                    ),
                )
            )

        if not isinstance(data, dict):
            return _MetadataFetchResult(
                errors=(
                    ProviderError(
                        code=ProviderErrorCode.UNKNOWN,
                        message=f"Unexpected metadata payload shape from {url}",
                        provider_id=provider_id,
                        route_id=profile.route_identity,
                        retryable=False,
                    ),
                )
            )

        models = data.get("data", [])
        if not isinstance(models, list):
            return _MetadataFetchResult(
                errors=(
                    ProviderError(
                        code=ProviderErrorCode.UNKNOWN,
                        message=f"Metadata endpoint at {url} did not return a model list",
                        provider_id=provider_id,
                        route_id=profile.route_identity,
                        retryable=False,
                    ),
                )
            )

        for entry in models:
            if isinstance(entry, dict) and entry.get("id") == model_id:
                return _MetadataFetchResult(metadata=dict(entry))

        # Model not listed: not a hard error, just no metadata for this model.
        return _MetadataFetchResult()

    def _local_provider_identity(self, profile: LocalEndpointProfile) -> ProviderIdentity:
        return ProviderIdentity(
            provider_id=f"local-{profile.runtime_kind.value}",
            display_name=f"Local ({profile.runtime_kind.value})",
            failure_domain=profile.failure_domain,
            metadata={"runtime_kind": profile.runtime_kind.value},
        )

    def _resolve_context_tokens(
        self,
        explicit: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[int, CapabilitySource, dict[str, Any]]:
        if "context_tokens" in explicit:
            return int(explicit["context_tokens"]), CapabilitySource.EXPLICIT_CONFIG, {}
        for key in ("context_length", "max_context_length", "n_ctx", "context_window"):
            value = metadata.get(key)
            if value is not None:
                try:
                    return int(value), CapabilitySource.PROVIDER_METADATA, {"key": key}
                except (TypeError, ValueError):
                    continue
        return (
            4096,
            CapabilitySource.UNKNOWN,
            {"reason": "context capacity not verified; using conservative executable floor"},
        )

    def _resolve_boolean(
        self,
        capability: str,
        explicit: dict[str, Any],
        metadata: dict[str, Any],
        evidence: list[CapabilityEvidence],
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
        evidence.append(
            CapabilityEvidence(
                capability=capability,
                supported=None,
                source=CapabilitySource.UNKNOWN,
            )
        )
        return None
