"""Deterministic helpers for context construction."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def hash_text(text: str) -> str:
    """Return a stable SHA-256 hex digest over normalized UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_path(path: str | Path) -> str:
    """Return a forward-slash, relative path with no trailing separator."""
    normalized = Path(path).as_posix()
    return normalized.rstrip("/")


def item_provenance_id(item: Any) -> str | None:
    """Return the provenance_id attribute/key of an item, if present."""
    if isinstance(item, dict):
        return item.get("provenance_id")
    return getattr(item, "provenance_id", None)


def item_content_hash(item: Any) -> str | None:
    """Return the content_hash attribute/key of an item, if present."""
    if isinstance(item, dict):
        return item.get("content_hash")
    return getattr(item, "content_hash", None)


def deterministic_sort(
    items: Iterable[T],
    key: Callable[[T], Any] | None = None,
) -> list[T]:
    """Return a stably sorted list with a predictable ordering."""
    return sorted(items, key=key)  # type: ignore[arg-type,type-var]
