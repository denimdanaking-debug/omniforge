"""OmniForge Phase 7 context construction engine.

Public API surface for context packets, provenance, budgeting, strategies,
summarization, arbitration evidence, validation, telemetry, and outcomes.
"""

from src.context.arbitration import (
    ArbitrationEvidencePacket,
    DisputedFinding,
    ReviewerPosition,
)
from src.context.budget import (
    BudgetResult,
    BudgetType,
    ContextBudget,
    compute_usable_budget,
    estimate_tokens,
)
from src.context.hierarchical import (
    DeterministicTestSummarizer,
    HierarchicalContextStrategy,
    Summarizer,
    SummaryResult,
)
from src.context.hybrid import HybridContextStrategy
from src.context.large_context import LargeContextStrategy
from src.context.outcomes import ContextOutcomeRecord
from src.context.provenance import ProvenanceIndex
from src.context.schema import (
    CONTEXT_PACKET_SCHEMA_VERSION,
    AcceptanceCriterion,
    AuthorityPresence,
    ContextPacket,
    ContextSummary,
    DiffInfo,
    Exclusion,
    HistoricalFinding,
    ProvenanceRef,
    RelevantFile,
    TaskMetadata,
    TestEvidence,
)
from src.context.strategy import (
    ContextBuildRequest,
    ContextStrategy,
    ContextStrategyResult,
)
from src.context.targeted import TargetedContextStrategy
from src.context.telemetry import ContextStrategyTelemetry
from src.context.validation import ContextPacketValidator, ValidationIssue

__all__ = [
    "CONTEXT_PACKET_SCHEMA_VERSION",
    "AcceptanceCriterion",
    "ArbitrationEvidencePacket",
    "AuthorityPresence",
    "BudgetResult",
    "BudgetType",
    "ContextBudget",
    "ContextBuildRequest",
    "ContextOutcomeRecord",
    "ContextPacket",
    "ContextPacketValidator",
    "ContextStrategy",
    "ContextStrategyResult",
    "ContextStrategyTelemetry",
    "ContextSummary",
    "DeterministicTestSummarizer",
    "DiffInfo",
    "DisputedFinding",
    "Exclusion",
    "HistoricalFinding",
    "HierarchicalContextStrategy",
    "HybridContextStrategy",
    "LargeContextStrategy",
    "ProvenanceIndex",
    "ProvenanceRef",
    "RelevantFile",
    "ReviewerPosition",
    "Summarizer",
    "SummaryResult",
    "TargetedContextStrategy",
    "TaskMetadata",
    "TestEvidence",
    "ValidationIssue",
    "compute_usable_budget",
    "estimate_tokens",
]
