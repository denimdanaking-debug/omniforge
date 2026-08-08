"""Shared live-test helpers."""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest


@pytest.fixture
def skip_without_api_key() -> Callable[[str], None]:
    """Marker fixture to skip a live test when no API key is available."""

    def _skip(provider_env_var: str) -> None:
        if not os.environ.get(provider_env_var):
            pytest.skip(f"Live test skipped: {provider_env_var} not set")

    return _skip
