"""Persistence integration for Phase 11 performance intelligence."""

from __future__ import annotations

from typing import Any

from src.performance.builder import StatisticsBuilder
from src.performance.ledger import PerformanceLedger
from src.performance.statistics import PerformanceStatisticsBundle

CURRENT_PERFORMANCE_SCHEMA_VERSION = "1.0.0"


def performance_state_to_dict(
    ledger: PerformanceLedger,
    bundle: PerformanceStatisticsBundle | None = None,
) -> dict[str, Any]:
    """Serialize performance intelligence state for runtime-state persistence."""
    if bundle is None:
        bundle = StatisticsBuilder().build(ledger)
    return {
        "schema_version": CURRENT_PERFORMANCE_SCHEMA_VERSION,
        "ledger": ledger.to_dict(),
        "statistics": bundle.to_dict(),
    }


def performance_state_from_dict(
    data: dict[str, Any],
) -> tuple[PerformanceLedger, PerformanceStatisticsBundle]:
    """Deserialize performance intelligence state from runtime-state storage.

    The raw immutable event ledger is the source of truth.  Persisted derived
    statistics, if present, are treated as an ignored cache; aggregates are
    always rebuilt from the ledger so stale or corrupted counters cannot win.
    """
    ledger_data = data.get("ledger", {"schema_version": "1.0.0", "events": []})
    ledger = PerformanceLedger.from_dict(ledger_data)
    bundle = StatisticsBuilder().build(ledger)
    return ledger, bundle
