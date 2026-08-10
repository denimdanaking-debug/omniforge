"""Deterministic recovery-decision fingerprint.

The failure signature identifies the normalized failure. The recovery input
fingerprint represents the complete logical state that drives the recovery
action, including history, policy, eligibility-producing configuration,
candidates, and task context. The same full input must always produce the same
action and fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.context.schema import AuthorityPresence, ContextPacket
from src.context.validation import ContextPacketValidator
from src.providers.identity import ProviderQuotaState
from src.recovery.eligibility import _effective_quota
from src.recovery.failure_classification import FailureClassification
from src.risk.context_policy import RiskContextPolicy, RiskContextRequirements
from src.security.redaction import redact


def _effective_quota_dict(
    candidate: Any,
    quota_domain_states: dict[str, ProviderQuotaState] | None,
) -> dict[str, Any] | None:
    """Return a normalized dict of the candidate's effective quota state."""
    effective: ProviderQuotaState | None = _effective_quota(candidate, quota_domain_states)
    if effective is None:
        return None
    return {
        "provider_signal": effective.provider_signal.value,
        "is_exhausted": effective.is_exhausted(),
        "remaining_fraction": effective.remaining_fraction,
        "remaining_requests": effective.remaining_requests,
        "remaining_tokens": effective.remaining_tokens,
    }


def _sort_candidates(
    candidates: tuple[Any, ...],
    quota_domain_states: dict[str, ProviderQuotaState] | None,
) -> list[dict[str, Any]]:
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
                "quota": _effective_quota_dict(candidate, quota_domain_states),
                "quota_domain": candidate.quota_domain,
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

    # Canonicalize exhausted paths with an explicitly sortable representation so
    # optional provider_id/model_id values do not raise TypeError during sorting.
    def _exhausted_sort_key(
        item: tuple[str, str | None, str | None],
    ) -> tuple[str, tuple[bool, str], tuple[bool, str]]:
        signature, provider_id, model_id = item
        return (
            signature,
            (provider_id is None, provider_id or ""),
            (model_id is None, model_id or ""),
        )

    exhausted = sorted(ledger.exhausted_paths, key=_exhausted_sort_key)
    return {
        "attempt_count": ledger.attempt_count,
        "transient_retry_count": ledger.transient_retry_count(),
        "constrained_output_retry_count": ledger.constrained_output_retry_count(),
        "planning_retry_count": ledger.planning_retry_count(),
        "repair_count": ledger.repair_count(),
        "context_rebuild_count": ledger.context_rebuild_count(),
        "provider_switch_count": ledger.provider_switch_count(),
        "model_switch_count": ledger.model_switch_count(),
        "exhausted_paths": [
            {"signature": s, "provider_id": p, "model_id": m} for s, p, m in exhausted
        ],
        "current_wait": wait,
        "current_context_rebuild": dict(ledger.current_context_rebuild),
    }


def _project_policy_dict(project_policy: Any) -> dict[str, Any] | None:
    """Return a deterministic, redacted representation of the project policy."""
    if project_policy is None:
        return None
    return {
        "prohibited_provider_ids": sorted(project_policy.prohibited_provider_ids),
        "prohibited_model_ids": sorted(project_policy.prohibited_model_ids),
        "prohibited_route_ids": sorted(project_policy.prohibited_route_ids),
        "minimum_review_independence": project_policy.minimum_review_independence,
    }


def _reserve_policy_dict(reserve_policy: Any) -> dict[str, Any] | None:
    """Return a deterministic representation of the reserve policy."""
    if reserve_policy is None:
        return None
    result: dict[str, Any] = reserve_policy.to_dict()
    return result


def _quota_domain_states_dict(
    quota_domain_states: dict[str, ProviderQuotaState] | None,
) -> dict[str, dict[str, Any]] | None:
    """Return a deterministic representation of shared quota-domain states."""
    if quota_domain_states is None:
        return None
    return {
        domain: {
            "provider_signal": state.provider_signal.value,
            "is_exhausted": state.is_exhausted(),
            "remaining_fraction": state.remaining_fraction,
        }
        for domain, state in sorted(quota_domain_states.items())
    }


def _exclusions_dict(exclusions: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Return a deterministic list of eligibility exclusion records."""
    return sorted(
        [
            {
                "provider_id": ex.provider_id,
                "model_id": ex.model_id,
                "route_id": ex.route_id,
                "reason": ex.reason,
            }
            for ex in exclusions
        ],
        key=lambda d: (d["provider_id"], d["model_id"], d["route_id"], d["reason"]),
    )


def effective_risk_context_requirements(
    inputs: Any,
) -> RiskContextRequirements:
    """Return the effective risk context requirements using coordinator precedence.

    Precedence:
    1. Explicitly supplied risk_context_requirements.
    2. Supplied risk_context_policy.requirements_for(current_risk).
    3. RiskContextPolicy.default().requirements_for(current_risk).
    """
    explicit: RiskContextRequirements | None = inputs.risk_context_requirements
    if explicit is not None:
        return explicit
    policy: RiskContextPolicy | None = inputs.risk_context_policy
    if policy is not None:
        result: RiskContextRequirements = policy.requirements_for(inputs.current_risk)
        return result
    default: RiskContextRequirements = RiskContextPolicy.default().requirements_for(
        inputs.current_risk
    )
    return default


def _risk_context_requirements_dict(requirements: RiskContextRequirements) -> dict[str, Any]:
    """Return deterministic decision-driving requirement fields."""
    return {
        "strategy_preference": requirements.strategy_preference,
        "authority_required": requirements.authority_required,
        "require_raw_authority": requirements.require_raw_authority,
        "include_test_evidence": requirements.include_test_evidence,
        "include_historical_findings": requirements.include_historical_findings,
        "budget_multiplier": requirements.budget_multiplier,
    }


def _context_authority_fingerprint(inputs: Any) -> dict[str, Any] | None:
    """Return deterministic context-authority decision inputs.

    Uses the SAME effective RiskContextRequirements as the coordinator for
    validation. Material authority differences therefore change the recovery
    input fingerprint.
    """
    packet: ContextPacket | None = inputs.context_packet
    requirements = effective_risk_context_requirements(inputs)

    req_dict = _risk_context_requirements_dict(requirements)

    if packet is None:
        return {
            "packet_present": False,
            "requirements": req_dict,
            "validation": None,
        }

    issues = list(ContextPacketValidator().validate(packet))
    issue_dicts = [{"severity": issue.severity, "code": issue.code} for issue in issues]
    has_error = any(issue["severity"] == "error" for issue in issue_dicts)
    authority_safe = not has_error
    if (
        requirements.require_raw_authority
        and packet.authority_presence is not AuthorityPresence.RAW_INCLUDED
    ):
        authority_safe = False
    if (
        requirements.authority_required
        and packet.authority_presence is AuthorityPresence.NOT_REQUIRED
    ):
        authority_safe = False

    return {
        "packet_present": True,
        "packet_content_hash": packet.content_hash(),
        "authority_presence": packet.authority_presence.name,
        "raw_item_count": packet.raw_item_count,
        "summary_count": packet.summary_count,
        "requirements": req_dict,
        "validation": {
            "issues": sorted(issue_dicts, key=lambda d: (d["code"], d["severity"])),
            "safe": authority_safe,
        },
    }


def recovery_input_fingerprint(
    inputs: Any,
    classification: FailureClassification,
    eligible_candidates: tuple[Any, ...] | None = None,
    exclusions: tuple[Any, ...] | None = None,
) -> str:
    """Return a deterministic fingerprint of all recovery-decision inputs."""
    classifier_input = inputs.classifier_input
    role_value = inputs.role.value if inputs.role is not None else classifier_input.role.value

    policy = inputs.policy.to_dict()

    candidate_list = _sort_candidates(
        eligible_candidates if eligible_candidates is not None else inputs.candidates,
        inputs.quota_domain_states,
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
        "eligibility": {
            "provider_enabled": dict(sorted((inputs.provider_enabled or {}).items())),
            "model_enabled": dict(sorted((inputs.model_enabled or {}).items())),
            "route_enabled": dict(sorted((inputs.route_enabled or {}).items())),
            "project_policy": _project_policy_dict(inputs.project_policy),
            "reserve_policy": _reserve_policy_dict(inputs.reserve_policy),
            "failure_domain_index": (
                inputs.failure_domain_index.to_dict()
                if inputs.failure_domain_index is not None
                else {}
            ),
            "quota_domain_states": _quota_domain_states_dict(inputs.quota_domain_states),
            "exclusions": _exclusions_dict(exclusions if exclusions is not None else ()),
        },
        "task": {
            "current_risk": inputs.current_risk.value,
            "role": role_value,
            "required_context_tokens": inputs.required_context_tokens,
            "capability_requirement": capability,
            "pin": pin,
            "reviewer_identities": sorted(inputs.reviewer_identities),
            "coder_identities": sorted(inputs.coder_identities),
        },
        "context_authority": _context_authority_fingerprint(inputs),
    }

    canonical = json.dumps(redact(data), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
