"""Tests for recursive secret redaction."""

from __future__ import annotations

import json

import pytest

from src.security.redaction import contains_secret, redact, redact_json_text

SENTINEL = "OMNIFORGE_TEST_SECRET_SENTINEL_7a8b9c"
OPENAI_KEY = "sk-openai-test-key-1234567890abcdef"
BEARER = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"


def test_redacts_openai_key_in_string() -> None:
    assert "sk-***" in redact(f"my key is {OPENAI_KEY}")
    assert OPENAI_KEY not in redact(f"my key is {OPENAI_KEY}")


def test_redacts_bearer_token_in_string() -> None:
    result = redact(f"Authorization: {BEARER}")
    assert "Bearer ***" in result or "Authorization: ***" in result
    assert BEARER not in result


def test_redacts_nested_dicts() -> None:
    payload = {
        "error": "auth failed",
        "headers": {"Authorization": f"Bearer {OPENAI_KEY}"},
        "body": {"api_key": OPENAI_KEY},
    }
    result = redact(payload)
    assert OPENAI_KEY not in json.dumps(result)
    assert result["headers"]["Authorization"] != f"Bearer {OPENAI_KEY}"


def test_redacts_lists() -> None:
    payload = ["ok", f"api_key={OPENAI_KEY}", {"token": BEARER}]
    result = redact(payload)
    assert OPENAI_KEY not in json.dumps(result)
    assert BEARER not in json.dumps(result)


def test_preserves_innocent_strings() -> None:
    result = redact("This is a token of appreciation")
    assert "token of appreciation" in result


def test_contains_secret_detects_sentinel() -> None:
    payload = {"nested": {"list": ["ok", SENTINEL]}}
    assert contains_secret(payload, SENTINEL) is True


def test_contains_secret_false_when_absent() -> None:
    payload = {"nested": {"list": ["ok", "other"]}}
    assert contains_secret(payload, SENTINEL) is False


def test_redact_json_text_parses_and_redacts() -> None:
    text = json.dumps({"api_key": OPENAI_KEY, "message": "hello"})
    result = redact_json_text(text)
    assert OPENAI_KEY not in result
    assert "message" in result


def test_redact_json_text_falls_back_on_invalid_json() -> None:
    text = f"error: api_key={OPENAI_KEY}"
    result = redact_json_text(text)
    assert OPENAI_KEY not in result


def test_secret_value_wrapper_is_redacted() -> None:
    from src.security.secrets import SecretValue

    value = SecretValue(SENTINEL)
    result = redact({"secret": value})
    assert SENTINEL not in json.dumps(result, default=str)


@pytest.mark.parametrize(
    "key",
    [
        "password",
        "PASSWORD",
        "api_key",
        "api-key",
        "apiKey",
        "API Key",
        "secret_key",
        "access_token",
        "bearer_token",
        "client_secret",
        "authorization",
        "Authorization",
        "credentials",
    ],
)
def test_redacts_sensitive_keys_regardless_of_value_shape(key: str) -> None:
    payload = {key: SENTINEL}
    result = redact(payload)
    assert result[key] == "<redacted>"
    assert SENTINEL not in json.dumps(result)


def test_redacts_nested_sensitive_keys_in_lists() -> None:
    payload = {"outer": [{"client_secret": SENTINEL}, {"password": SENTINEL}]}
    result = redact(payload)
    assert SENTINEL not in json.dumps(result)
    assert result["outer"][0]["client_secret"] == "<redacted>"
    assert result["outer"][1]["password"] == "<redacted>"


def test_preserves_innocent_text_with_sensitive_words() -> None:
    result = redact("This token represents a parser node")
    assert "token represents a parser node" in result
    result = redact("password policy requires 16 characters")
    assert "password policy requires 16 characters" in result


def test_normal_key_retains_non_secret_sentinel() -> None:
    payload = {"normal": SENTINEL}
    result = redact(payload)
    assert result["normal"] == SENTINEL


@pytest.mark.architecture
def test_runtime_state_save_does_not_persist_resolved_secret() -> None:
    import tempfile
    from pathlib import Path

    from src.persistence import runtime_state
    from src.security.redaction import contains_secret
    from src.security.secrets import SecretValue

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runtime.json"
        state = {
            "schema_version": "1.1.0",
            "run_id": "run-1",
            "workflow_state": "STOPPED",
            "checkpoint": {"resolved": SecretValue(SENTINEL)},
            "routing_mode": "legacy",
            "exploration_enabled": False,
            "provider_status": {},
            "model_status": {},
            "route_status": {},
            "pins": {},
            "project_policies": {},
        }
        runtime_state.save_runtime_state(path, state)
        text = path.read_text(encoding="utf-8")
        assert not contains_secret(text, SENTINEL)
        assert "<redacted>" in text
