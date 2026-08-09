"""Provenance index for context packets."""

from __future__ import annotations

from src.context.schema import ProvenanceRef


class ProvenanceIndex:
    """Maps item IDs to their provenance references and supports reverse lookups."""

    def __init__(self) -> None:
        self._refs: dict[str, ProvenanceRef] = {}

    def register(self, item_id: str, ref: ProvenanceRef) -> None:
        """Register a provenance reference for an item ID."""
        if not item_id or not item_id.strip():
            raise ValueError("item_id must be non-empty")
        self._refs[item_id] = ref

    def sources_for(self, item_id: str) -> tuple[ProvenanceRef, ...]:
        """Return the provenance reference(s) for an item ID."""
        ref = self._refs.get(item_id)
        if ref is None:
            return ()
        return (ref,)

    def items_from(self, source_key: str) -> tuple[str, ...]:
        """Return item IDs whose provenance matches the given source key.

        A source key has the form ``source_type:repository:path:revision``.
        """
        parts = source_key.split(":")
        source_type = parts[0] if len(parts) > 0 else ""
        repository = parts[1] if len(parts) > 1 and parts[1] != "" else None
        path = parts[2] if len(parts) > 2 and parts[2] != "" else None
        revision = parts[3] if len(parts) > 3 and parts[3] != "" else None

        results: list[str] = []
        for item_id, ref in sorted(self._refs.items()):
            if ref.source_type != source_type:
                continue
            if repository is not None and ref.repository != repository:
                continue
            if path is not None and ref.path != path:
                continue
            if revision is not None and ref.revision != revision:
                continue
            results.append(item_id)
        return tuple(results)

    def authority_sources(self) -> tuple[ProvenanceRef, ...]:
        """Return all authority-level provenance references."""
        return tuple(
            ref
            for ref in sorted(self._refs.values(), key=lambda r: (r.source_type, r.path or ""))
            if ref.authority_level is not None
        )

    def summary_sources(self) -> tuple[ProvenanceRef, ...]:
        """Return all summary provenance references."""
        return tuple(
            ref
            for ref in sorted(self._refs.values(), key=lambda r: (r.source_type, r.path or ""))
            if ref.source_type == "summary"
        )

    def by_source_type(self, source_type: str) -> tuple[ProvenanceRef, ...]:
        """Return all provenance references of the given source type."""
        return tuple(
            ref
            for ref in sorted(self._refs.values(), key=lambda r: (r.source_type, r.path or ""))
            if ref.source_type == source_type
        )

    def validate_no_dangling(self, items: list[str]) -> list[str]:
        """Return item IDs that lack a registered provenance reference."""
        return sorted(item_id for item_id in items if item_id not in self._refs)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._refs
