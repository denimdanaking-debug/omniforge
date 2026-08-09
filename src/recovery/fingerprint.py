"""Deterministic recovery-decision fingerprint.

The failure signature identifies the normalized failure. The recovery input
fingerprint represents the complete logical state that drives the recovery
action, including history, policy, candidates, and task context. The same full
input must always produce the same action and fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.recovery.failure_classification import FailureClassification
from src.security.redaction import redact


def _sort_candidates(candidates: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Return a deterministically sorted list of candidate descriptions."""
    items: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda c: c.key):
        items.append(
            {
                "key": candidate.key,
                "lifecycle": candidate.model_identity.lifecycle.value,
                "context_tokens": candidate.capabilities.context_tokens,
                "supported_roles": sorted(candidate.capabilities.supported_roles),
                "recovery_health": candidate.recovery_state.health.value,
                "quota_exhausted": bool(
                    candidate.quota is not None and candidate.quota.is_exhausted()
                ),
                "failure_domain": candidate.failure_domain,
            }
        )
    return items


def _ledger_state(ledger: Any) -> dict[str, Any]:
    """Return deterministic retry-ledger state."""
    wait: dict[str, Any] | None = None
    if ledger.current_wait is not None:
        wait = {
            "reason": ledger.current_wait.reason,
            "next_recheck_at": ledger.current_wait.next_recheck_at.isoformat(),
            "affected_failure_domains": sorted(ledger.current_wait.affected_failure_domains),
        }
    return {
        "attempt_count": ledger.attempt_count,
        "transient_retry_count": ledger.transient_retry_count(),
        "constrained_output_retry_count": ledger.constrained_output_retry_count(),
        "planning_retry_count": ledger.planning_retry_count(),
        "repair_count": ledger.repair_count(),
        "context_rebuild_count": ledger.context_rebuild_count(),
        "provider_switch_count": ledger.provider_switch_count(),
        "model_switch_count": ledger.model_switch_count(),
        "exhausted_paths": sorted(
            {"signature": s, "provider_id": p, "model_id": m} for s, p, m in ledger.exhausted_paths
        ),
        "current_wait": wait,
        "current_context_rebuild": dict(ledger.current_context_rebuild),
    }


def recovery_input_fingerprint(
    inputs: Any,
    classification: FailureClassification,
    eligible_candidates: tuple[Any, ...] | None = None,
) -> str:
    """Return a deterministic fingerprint of all recovery-decision inputs."""
    classifier_input = inputs.classifier_input
    role_value = inputs.role.value if inputs.role is not None else classifier_input.role.value

    policy = inputs.policy.to_dict()

    candidate_list = _sort_candidates(
        eligible_candidates if eligible_candidates is not None else inputs.candidates
    )

    pin: dict[str, Any] | None = None
    if inputs.pin is not None:
        pin = {
            "provider_id": inputs.pin.provider_id,
            "model_id": inputs.pin.model_id,
            "route_id": inputs.pin.route_id,
        }

    capability: dict[str, Any] | None = None
    if inputs.capability_requirement is not None:
        req = inputs.capability_requirement
        capability = {
            "min_context_tokens": req.min_context_tokens,
            "required_roles": sorted(req.required_roles),
            "structured_output": req.structured_output,
            "tool_use": req.tool_use,
            "streaming": req.streaming,
            "reasoning": req.reasoning,
            "code_generation": req.code_generation,
            "multimodal": req.multimodal,
        }

    data: dict[str, Any] = {
        "failure": {
            "category": classification.category.value,
            "subtype": classification.subtype.value,
            "failure_signature": classification.deterministic_fingerprint,
        },
        "history": _ledger_state(inputs.ledger),
        "policy": policy,
        "candidates": candidate_list,
        "task": {
            "current_risk": inputs.current_risk.value,
            "role": role_value,
            "required_context_tokens": inputs.required_context_tokens,
            "capability_requirement": capability,
            "pin": pin,
            "reviewer_identities": sorted(inputs.reviewer_identities),
            "coder_identities": sorted(inputs.coder_identities),
        },
    }

    canonical = json.dumps(redact(data), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
