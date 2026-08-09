"""Tests for context outcome records."""

from __future__ import annotations

import pytest

from src.context.outcomes import ContextOutcomeRecord


def test_outcome_record_construction() -> None:
    record = ContextOutcomeRecord(
        packet_id="pkt-1",
        strategy="targeted",
        model_id="model-a",
        role="coding",
        risk="R2_NORMAL",
        task_class="feature",
        context_size=1234,
        accepted=True,
        validation_result={"issues": []},
        review_result="approved",
        repair_required=False,
        failure_category=None,
    )
    assert record.packet_id == "pkt-1"
    assert record.accepted is True


def test_outcome_record_rejects_negative_context_size() -> None:
    with pytest.raises(ValueError):
        ContextOutcomeRecord(
            packet_id="pkt-1",
            strategy="targeted",
            model_id="model-a",
            role="coding",
            risk="R2_NORMAL",
            task_class="feature",
            context_size=-1,
            accepted=True,
            validation_result={},
            review_result=None,
            repair_required=False,
            failure_category=None,
        )
