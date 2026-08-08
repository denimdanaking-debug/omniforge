"""Optional live integration tests for Mistral / Devstral.

These tests are skipped unless credentials are present and are never run in CI.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.providers.mistral.adapter import MistralAdapter

pytestmark = [pytest.mark.live]


@pytest.mark.live
async def test_mistral_basic_completion(skip_without_api_key: Callable[[str], None]) -> None:
    skip_without_api_key("MISTRAL_API_KEY")
    adapter = MistralAdapter(api_key="resolved-from-env")
    assert adapter.identity.provider_id == "mistral"
