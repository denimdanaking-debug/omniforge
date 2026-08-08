"""Normalized capability metadata used for provider/model eligibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class CapabilitySupport(Enum):
    """How a capability is supported by a model/route."""

    REQUIRED = auto()
    PREFERRED = auto()
    SUPPORTED = auto()
    UNSUPPORTED = auto()


class ExecutionRole(Enum):
    """Normalized execution roles."""

    PLANNING = "planning"
    ARCHITECTURE = "architecture"
    CODING = "coding"
    DEBUGGING = "debugging"
    REPAIR = "repair"
    REVIEW = "review"
    HIGH_RISK_REVIEW = "high_risk_review"
    ARBITRATION = "arbitration"
    CONTEXT_ANALYSIS = "context_analysis"
    INTEGRATION_ANALYSIS = "integration_analysis"


class RiskLevel(Enum):
    """Normalized risk taxonomy."""

    R0_TRIVIAL = "R0_TRIVIAL"
    R1_LOW = "R1_LOW"
    R2_NORMAL = "R2_NORMAL"
    R3_HIGH = "R3_HIGH"
    R4_CRITICAL_AUTHORITY = "R4_CRITICAL_AUTHORITY"


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Normalized capability metadata for a model/route combination.

    Provider-neutral does not mean pretending all providers support every feature.
    Unsupported required capabilities must fail eligibility before dispatch.
    """

    context_size: int
    supports_structured_output: CapabilitySupport = CapabilitySupport.SUPPORTED
    supports_tool_use: CapabilitySupport = CapabilitySupport.SUPPORTED
    supports_streaming: CapabilitySupport = CapabilitySupport.SUPPORTED
    supports_reasoning: CapabilitySupport = CapabilitySupport.UNSUPPORTED
    supports_multimodal: CapabilitySupport = CapabilitySupport.UNSUPPORTED
    supports_temperature: CapabilitySupport = CapabilitySupport.SUPPORTED
    supports_max_tokens: CapabilitySupport = CapabilitySupport.SUPPORTED
    supports_stop_sequences: CapabilitySupport = CapabilitySupport.SUPPORTED
    supported_roles: frozenset[ExecutionRole] = field(default_factory=frozenset)
    local_or_cloud: str = "cloud"
    cost_metadata: dict[str, Any] = field(default_factory=dict)
    concurrency_metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def supports_role(self, role: ExecutionRole) -> bool:
        """Return True if the model declares support for the execution role."""
        return role in self.supported_roles

    def is_eligible(self, required: list[CapabilitySupport]) -> bool:
        """Return True if no required capability is unsupported."""
        return all(cap is not CapabilitySupport.UNSUPPORTED for cap in required)
