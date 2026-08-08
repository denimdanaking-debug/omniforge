"""Cursor route adapter contract tests using the shared suite."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.providers.adapter import ProviderAdapter
from src.providers.cursor.adapter import CursorRouteAdapter
from tests.providers.adapter_contract_suite import AdapterContractSuite


class TestCursorRouteAdapterContract(AdapterContractSuite):
    """Run the provider-neutral contract suite against CursorRouteAdapter.

    Cursor is intentionally a non-executable route in Phase 3, so tests that
    assume a healthy, capable execution backend are skipped.
    """

    @pytest.fixture
    def adapter_factory(self) -> Callable[[], ProviderAdapter]:
        return CursorRouteAdapter

    @pytest.fixture
    def limited_adapter_factory(self) -> Callable[[], ProviderAdapter]:
        return CursorRouteAdapter

    @pytest.fixture
    def failing_adapter_factory(self) -> Callable[[], ProviderAdapter]:
        return CursorRouteAdapter

    @pytest.mark.contract
    async def test_normalized_response(self, *args: object, **kwargs: object) -> None:
        pytest.skip("Cursor is a non-executable route in Phase 3")

    @pytest.mark.contract
    async def test_streaming_semantics(self, *args: object, **kwargs: object) -> None:
        pytest.skip("Cursor is a non-executable route in Phase 3")

    @pytest.mark.contract
    async def test_normalized_errors(self, *args: object, **kwargs: object) -> None:
        pytest.skip("Cursor always returns UNSUPPORTED_CAPABILITY; no RATE_LIMITED path")

    @pytest.mark.contract
    async def test_clean_lifecycle(self, *args: object, **kwargs: object) -> None:
        pytest.skip("Cursor reports UNAVAILABLE by design in Phase 3")
