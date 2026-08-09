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
    result: Any
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
    """Return a deterministic SHA-256 fingerprint of the routing request."""
    canonical = _canonical_value(_request_to_dict(request))
    # Sort keys recursively via json.dumps.
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
