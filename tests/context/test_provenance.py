"""Tests for the provenance index."""

from __future__ import annotations

import pytest

from src.context.provenance import ProvenanceIndex
from src.context.schema import ProvenanceRef


def test_register_and_lookup() -> None:
    index = ProvenanceIndex()
    ref = ProvenanceRef(source_type="git", path="src/a.py", revision="abc123")
    index.register("item-1", ref)
    assert index.sources_for("item-1") == (ref,)
    assert index.sources_for("missing") == ()


def test_items_from_source_key() -> None:
    index = ProvenanceIndex()
    index.register("item-1", ProvenanceRef(source_type="git", path="src/a.py", revision="main"))
    index.register("item-2", ProvenanceRef(source_type="git", path="src/a.py", revision="main"))
    index.register("item-3", ProvenanceRef(source_type="git", path="src/b.py", revision="main"))
    assert index.items_from("git::src/a.py:main") == ("item-1", "item-2")


def test_authority_sources() -> None:
    index = ProvenanceIndex()
    index.register(
        "auth", ProvenanceRef(source_type="roadmap", path="roadmap.md", authority_level="primary")
    )
    index.register("other", ProvenanceRef(source_type="git", path="src/a.py"))
    assert index.authority_sources() == (
        ProvenanceRef(source_type="roadmap", path="roadmap.md", authority_level="primary"),
    )


def test_validate_no_dangling() -> None:
    index = ProvenanceIndex()
    index.register("item-1", ProvenanceRef(source_type="git", path="src/a.py"))
    dangling = index.validate_no_dangling(["item-1", "item-2"])
    assert dangling == ["item-2"]


def test_register_rejects_empty_item_id() -> None:
    index = ProvenanceIndex()
    with pytest.raises(ValueError):
        index.register("", ProvenanceRef(source_type="git"))
