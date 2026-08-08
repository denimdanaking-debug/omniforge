"""Optional live integration tests for MiniMax.

These tests are skipped unless credentials are present and are never run in CI.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.providers.minimax.adapter import MiniMaxAdapter

pytestmark = [pytest.mark.live]


@pytest.mark.live
async def test_minimax_basic_completion(skip_without_api_key: Callable[[str], None]) -> None:
    skip_without_api_key("MINIMAX_API_KEY")
    adapter = MiniMaxAdapter(api_key="resolved-from-env")
    assert adapter.identity.provider_id == "minimax"
