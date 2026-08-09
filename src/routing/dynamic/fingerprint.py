"""Deterministic input fingerprint for routing decisions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from .request import DynamicRoutingRequest

# Fields excluded because they are volatile or not part of routing intent.
_VOLATILE_FIELDS = frozenset({"timestamp", "state_snapshot_ref"})


def _canonical_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable canonical form."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, frozenset | set):
        return sorted(str(v) for v in value)
    if isinstance(value, tuple | list):
        return [_canonical_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical_value(v) for k, v in value.items()}
    return str(value)


def _request_to_dict(request: DynamicRoutingRequest) -> dict[str, Any]:
    """Convert request to canonical dict excluding volatile fields."""
    raw: dict[str, Any] = {
        "task_id": request.task_id,
        "project_id": request.project_id,
        "role": request.role.value,
        "risk": request.risk.name,
        "task_class": request.task_class,
        "capability_requirement": None,
        "required_context_tokens": request.required_context_tokens,
        "pin": None,
        "reviewer_identities": request.reviewer_identities,
        "coder_identities": request.coder_identities,
        "expected_input_tokens": request.expected_input_tokens,
        "expected_output_tokens": request.expected_output_tokens,
    }
    if request.capability_requirement is not None:
        req = request.capability_requirement
        raw["capability_requirement"] = {
            "min_context_tokens": req.min_context_tokens,
            "structured_output": req.structured_output,
            "tool_use": req.tool_use,
            "streaming": req.streaming,
            "reasoning": req.reasoning,
            "code_generation": req.code_generation,
            "multimodal": req.multimodal,
            "allowed_deployment_modes": sorted(m.value for m in req.allowed_deployment_modes),
            "required_roles": sorted(req.required_roles),
        }
    if request.pin is not None:
        raw["pin"] = {
            "provider_id": request.pin.provider_id,
            "model_id": request.pin.model_id,
            "route_id": request.pin.route_id,
        }
    return raw


def input_fingerprint(request: DynamicRoutingRequest) -> str:
    """Return a deterministic SHA-256 fingerprint of the routing request only."""
    canonical = _canonical_value(_request_to_dict(request))
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_to_dict(candidate: Any) -> dict[str, Any]:
    """Canonical representation of one routing candidate."""
    from .candidate import RoutingCandidate

    if not isinstance(candidate, RoutingCandidate):
        return {"non_candidate_value": _canonical_value(candidate)}

    caps = candidate.capabilities
    cap_dict: dict[str, Any] = {
        "context_tokens": caps.context_tokens,
        "structured_output": caps.structured_output,
        "tool_use": caps.tool_use,
        "streaming": caps.streaming,
        "reasoning": caps.reasoning,
        "code_generation": caps.code_generation,
        "multimodal": caps.multimodal,
        "supported_roles": sorted(caps.supported_roles),
    }

    recovery = candidate.recovery_state
    recovery_dict: dict[str, Any] | None = None
    if recovery is not None:
        recovery_dict = {
            "health": recovery.health.value,
            "failure_domain": recovery.failure_domain,
            "consecutive_failures": recovery.consecutive_failures,
        }

    operational = candidate.operational_state
    operational_dict: dict[str, Any] | None = None
    if operational is not None:
        operational_dict = {"health": operational.health.value}

    quota = candidate.quota_state
    quota_dict: dict[str, Any] | None = None
    if quota is not None:
        quota_dict = {
            "provider_signal": quota.provider_signal.value,
            "remaining_fraction": quota.remaining_fraction,
            "is_exhausted": quota.is_exhausted(),
        }

    route_state = candidate.route_cost_state
    route_state_dict: dict[str, Any] | None = None
    if route_state is not None:
        route_state_dict = {
            "health": route_state.health.value,
            "input_cost_per_million": route_state.input_cost_per_million,
            "output_cost_per_million": route_state.output_cost_per_million,
            "rolling_latency_ms": route_state.rolling_latency_ms,
        }

    evidence = candidate.performance_evidence
    evidence_dict: dict[str, Any] | None = None
    if evidence is not None:
        evidence_dict = {
            "attempts": evidence.attempts,
            "success_rate": evidence.success_rate,
            "recent_success_rate": evidence.recent_success_rate,
            "repair_rate": evidence.repair_rate,
            "retry_rate": evidence.retry_rate,
            "average_latency_ms": evidence.average_latency_ms,
        }

    return {
        "identity_key": candidate.identity_key,
        "provider_id": candidate.provider_id,
        "model_id": candidate.model_id,
        "route_id": candidate.route_id,
        "model_lifecycle": candidate.model_identity.lifecycle.value,
        "capabilities": cap_dict,
        "route_failure_domain": candidate.route_identity.failure_domain,
        "recovery_state": recovery_dict,
        "operational_state": operational_dict,
        "quota_state": quota_dict,
        "route_cost_state": route_state_dict,
        "performance_evidence": evidence_dict,
    }


def _policy_engine_to_dict(policy_engine: Any) -> dict[str, Any]:
    """Canonical representation of policy inputs affecting the decision."""
    policy = policy_engine._project_policy
    return {
        "provider_enabled": dict(sorted(policy_engine._provider_enabled.items())),
        "model_enabled": dict(sorted(policy_engine._model_enabled.items())),
        "route_enabled": dict(sorted(policy_engine._route_enabled.items())),
        "project_policy": {
            "prohibited_provider_ids": sorted(policy.prohibited_provider_ids),
            "prohibited_model_ids": sorted(policy.prohibited_model_ids),
            "prohibited_route_ids": sorted(policy.prohibited_route_ids),
            "allowed_deployment_modes": (
                sorted(m.value for m in policy.allowed_deployment_modes)
                if policy.allowed_deployment_modes is not None
                else None
            ),
            "minimum_review_independence": policy.minimum_review_independence,
            "allow_exploration": policy.allow_exploration,
            "routing_mode_override": policy.routing_mode_override,
        },
        "pin": {
            "provider_id": policy_engine._pin.provider_id if policy_engine._pin else None,
            "model_id": policy_engine._pin.model_id if policy_engine._pin else None,
            "route_id": policy_engine._pin.route_id if policy_engine._pin else None,
        },
    }


def _router_config_to_dict(router_config: Any) -> dict[str, Any]:
    """Canonical representation of decision-driving router configuration."""
    if router_config is None:
        return {}
    return {
        "factor_weights": dict(sorted(router_config.factor_weights.to_dict().items())),
        "priors": router_config.priors,
        "emergency_fallback_orders": {
            role.value: keys
            for role, keys in sorted(
                router_config.emergency_fallback_orders.items(), key=lambda item: item[0].value
            )
        },
        "cost_metadata": dict(sorted(router_config.cost_metadata.items())),
        "default_safety_margin_fraction": router_config.default_safety_margin_fraction,
        "exploration_enabled": router_config.exploration_enabled,
    }


def _scoring_state_to_dict(state: Any) -> dict[str, Any]:
    """Canonical representation of continuity/diversity state."""
    if state is None:
        return {}
    return {
        "last_selected_key": state.last_selected_key,
        "failure_domain_counts": dict(sorted(state.failure_domain_counts.items())),
    }


def routing_input_fingerprint(
    *,
    request: DynamicRoutingRequest,
    candidates: tuple[Any, ...],
    policy_engine: Any,
    router_config: Any,
    scoring_state: Any | None = None,
) -> str:
    """Return a deterministic SHA-256 fingerprint of the full routing input.

    Includes the request, all candidates (in canonical order), policy settings,
    router configuration, and continuity/diversity state. Excludes volatile
    values such as timestamps and decision UUIDs.
    """
    candidate_dicts = [
        _candidate_to_dict(c)
        for c in sorted(candidates, key=lambda c: getattr(c, "identity_key", str(c)))
    ]
    payload: dict[str, Any] = {
        "request": _request_to_dict(request),
        "candidates": candidate_dicts,
        "policy": _policy_engine_to_dict(policy_engine),
        "router_config": _router_config_to_dict(router_config),
        "scoring_state": _scoring_state_to_dict(scoring_state),
    }
    canonical = _canonical_value(payload)
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
