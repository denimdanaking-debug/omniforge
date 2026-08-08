"""Recursive secret redaction for diagnostics and serialized output.

This is defense in depth: the primary protection is never persisting secrets.
Redaction catches accidental leakage in exceptions, metadata, and diagnostics.

Redaction is both key-aware and value-pattern-aware. Structured fields whose
keys are known to carry secret material are redacted regardless of value shape,
while ordinary string content is scanned for common credential patterns.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Patterns are intentionally specific to common credential shapes.
_REDACTION_PATTERNS = [
    (re.compile(r"sk-[a-zA-Z0-9_-]{10,}"), "sk-***"),
    (re.compile(r"Bearer\s+[a-zA-Z0-9_.\-+/=]{10,}", re.IGNORECASE), "Bearer ***"),
    (re.compile(r"Basic\s+[a-zA-Z0-9_.\-+/=]{10,}", re.IGNORECASE), "Basic ***"),
    (
        re.compile(r"api[_\-]?key['\"\\s]*[:=]['\"\\s]*[a-zA-Z0-9_.\-+/=]{4,}", re.IGNORECASE),
        "api_key=***",
    ),
    (
        re.compile(r"secret[_\-]?key['\"\\s]*[:=]['\"\\s]*[a-zA-Z0-9_.\-+/=]{4,}", re.IGNORECASE),
        "secret_key=***",
    ),
    (
        re.compile(r"access[_\-]?token['\"\\s]*[:=]['\"\\s]*[a-zA-Z0-9_.\-+/=]{4,}", re.IGNORECASE),
        "access_token=***",
    ),
    (
        re.compile(r"bearer[_\-]?token['\"\\s]*[:=]['\"\\s]*[a-zA-Z0-9_.\-+/=]{4,}", re.IGNORECASE),
        "bearer_token=***",
    ),
    (
        re.compile(
            r"client[_\-]?secret['\"\\s]*[:=]['\"\\s]*[a-zA-Z0-9_.\-+/=]{4,}", re.IGNORECASE
        ),
        "client_secret=***",
    ),
    (re.compile(r"password['\"\\s]*[:=]['\"\\s]*[^\s&]{4,}", re.IGNORECASE), "password=***"),
    (re.compile(r"Authorization['\"\\s]*:\s*[^\r\n]+", re.IGNORECASE), "Authorization: ***"),
]

# Canonical sensitive structured-field names. Variants are matched after
# normalizing separators and case, so api_key, api-key, apikey, and "API Key"
# are all treated as sensitive.
SENSITIVE_STRUCTURED_KEYS = frozenset(
    {
        "apikey",
        "secretkey",
        "accesstoken",
        "bearertoken",
        "clientsecret",
        "password",
        "passwd",
        "authorization",
        "proxyauthorization",
        "credential",
        "credentials",
    }
)


def _normalize_key(key: str) -> str:
    """Lowercase and remove common separators for key comparison."""
    lowered = key.lower()
    for separator in ("_", "-", " ", "."):
        lowered = lowered.replace(separator, "")
    return lowered


def _is_sensitive_key(key: Any) -> bool:
    """Return True if ``key`` is a known secret-bearing structured field."""
    if not isinstance(key, str):
        return False
    return _normalize_key(key) in SENSITIVE_STRUCTURED_KEYS


def _redact_string(text: str) -> str:
    for pattern, replacement in _REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact(value: Any) -> Any:
    """Recursively redact likely secret material while preserving structure.

    Dictionary keys recognized as secret-bearing cause their entire value to be
    replaced with ``<redacted>``. Other values receive recursive value-pattern
    redaction. Innocent prose is preserved.
    """
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(key):
                result[key] = "<redacted>"
            else:
                result[key] = redact(item)
        return result
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


def contains_secret(value: Any, sentinel: str) -> bool:
    """Return True if ``sentinel`` appears anywhere in a serialized form of ``value``."""
    try:
        serialized = json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        serialized = str(value)
    return sentinel in serialized


def redact_json_text(text: str) -> str:
    """Parse JSON text, redact recursively, and re-serialize deterministically."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _redact_string(text)
    return json.dumps(redact(data), indent=2, sort_keys=True)
