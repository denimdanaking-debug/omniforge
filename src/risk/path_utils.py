"""Canonical repository-relative path normalization for risk classification."""

from __future__ import annotations

from pathlib import PurePosixPath


class PathNormalizationError(ValueError):
    """Raised when a path cannot be normalized safely within the repository."""


def normalize_repo_path(path: str) -> str:
    """Return a deterministic, repository-root-relative normalized path.

    Treats both ``/`` and ``\\`` as path separators, removes ``./`` components,
    and resolves ``../`` components that stay within the repository root.
    Raises ``PathNormalizationError`` if the path attempts to escape the
    repository root, because such a path must not silently bypass path-based
    safety rules.
    """
    normalized = path.replace("\\", "/").lstrip("/")
    p = PurePosixPath(normalized)
    parts: list[str] = []
    for part in p.parts:
        if part == "..":
            if not parts:
                raise PathNormalizationError(f"path escapes repository root: {path!r}")
            parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)
