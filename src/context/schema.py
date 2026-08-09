"""Canonical context packet schema and deterministic serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any

from src.context._utils import deterministic_sort, hash_text
from src.security.redaction import redact

CONTEXT_PACKET_SCHEMA_VERSION = "1.0.0"


class AuthorityPresence(Enum):
    """How raw authority material is present in the packet."""

    RAW_INCLUDED = auto()
    RAW_REFERENCED = auto()
    NOT_REQUIRED = auto()


@dataclass(frozen=True)
class ProvenanceRef:
    """Reference to an authoritative or raw source."""

    source_type: str
    repository: str | None = None
    path: str | None = None
    revision: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    commit_sha: str | None = None
    pr_or_issue_id: str | None = None
    content_hash: str | None = None
    authority_level: str | None = None


@dataclass(frozen=True)
class AcceptanceCriterion:
    """One acceptance criterion with authority linkage."""

    criterion_id: str
    text: str
    authority_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelevantFile:
    """A file referenced as relevant context."""

    file_id: str
    path: str
    reason: str | None = None
    provenance_id: str | None = None
    content_hash: str | None = None


@dataclass(frozen=True)
class DiffInfo:
    """Current diff excerpt for one path."""

    diff_id: str
    path: str
    before_ref: str | None = None
    after_ref: str | None = None
    content: str = ""
    provenance_id: str | None = None


@dataclass(frozen=True)
class TestEvidence:
    """Evidence from a test run."""

    evidence_id: str
    test_name: str
    outcome: str
    log_excerpt: str = ""
    provenance_id: str | None = None


TestEvidence.__test__ = False  # type: ignore[attr-defined]


@dataclass(frozen=True)
class HistoricalFinding:
    """A prior finding relevant to the current task."""

    finding_id: str
    summary: str
    provenance_id: str | None = None
    related_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskMetadata:
    """Metadata describing the task that requested context."""

    task_id: str
    role: str | None = None
    risk: str | None = None
    requested_objective: str = ""
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextSummary:
    """A lossy summary of supporting context."""

    text: str
    source_provenance_ids: tuple[str, ...]
    source_hashes: tuple[str, ...]
    level: str
    lossy: bool = True
    generated_by: str = ""
    source_revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_provenance_ids": list(self.source_provenance_ids),
            "source_hashes": list(self.source_hashes),
            "level": self.level,
            "lossy": self.lossy,
            "generated_by": self.generated_by,
            "source_revision": self.source_revision,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextSummary:
        return cls(
            text=data["text"],
            source_provenance_ids=tuple(data.get("source_provenance_ids") or ()),
            source_hashes=tuple(data.get("source_hashes") or ()),
            level=data.get("level", ""),
            lossy=bool(data.get("lossy", True)),
            generated_by=data.get("generated_by", ""),
            source_revision=data.get("source_revision"),
        )


@dataclass(frozen=True)
class AuthorityContextItem:
    """One authority item with immutable provenance metadata."""

    authority_id: str
    provenance_id: str
    full_source_ref: str
    revision: str
    content_hash: str
    content: str | None = None
    raw_included: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_id": self.authority_id,
            "provenance_id": self.provenance_id,
            "full_source_ref": self.full_source_ref,
            "revision": self.revision,
            "content_hash": self.content_hash,
            "content": self.content,
            "raw_included": self.raw_included,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthorityContextItem:
        return cls(
            authority_id=data["authority_id"],
            provenance_id=data["provenance_id"],
            full_source_ref=data["full_source_ref"],
            revision=data["revision"],
            content_hash=data["content_hash"],
            content=data.get("content"),
            raw_included=bool(data.get("raw_included", True)),
        )


@dataclass(frozen=True)
class ReviewerPosition:
    """One reviewer's stance on a disputed finding."""

    reviewer_id: str
    position: str
    reasoning: str
    evidence_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "position": self.position,
            "reasoning": self.reasoning,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewerPosition:
        return cls(
            reviewer_id=data["reviewer_id"],
            position=data["position"],
            reasoning=data["reasoning"],
            evidence_refs=tuple(data.get("evidence_refs") or ()),
        )


@dataclass(frozen=True)
class DisputedFinding:
    """A finding under arbitration with independent reviewer positions."""

    finding_id: str
    exact_text: str
    positions: tuple[ReviewerPosition, ...]
    evidence_refs: tuple[str, ...]
    authority_refs: tuple[str, ...]
    unresolved_question: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "exact_text": self.exact_text,
            "positions": [p.to_dict() for p in self.positions],
            "evidence_refs": list(self.evidence_refs),
            "authority_refs": list(self.authority_refs),
            "unresolved_question": self.unresolved_question,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DisputedFinding:
        return cls(
            finding_id=data["finding_id"],
            exact_text=data["exact_text"],
            positions=tuple(ReviewerPosition.from_dict(p) for p in data.get("positions") or ()),
            evidence_refs=tuple(data.get("evidence_refs") or ()),
            authority_refs=tuple(data.get("authority_refs") or ()),
            unresolved_question=data.get("unresolved_question", ""),
        )


@dataclass(frozen=True)
class ArbitrationContext:
    """Arbitration evidence kept separate from authority."""

    dispute_id: str
    disputed_findings: tuple[DisputedFinding, ...]
    reviewer_positions: tuple[ReviewerPosition, ...]
    evidence_refs: tuple[str, ...]
    authority_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dispute_id": self.dispute_id,
            "disputed_findings": [f.to_dict() for f in self.disputed_findings],
            "reviewer_positions": [p.to_dict() for p in self.reviewer_positions],
            "evidence_refs": list(self.evidence_refs),
            "authority_refs": list(self.authority_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArbitrationContext:
        return cls(
            dispute_id=data["dispute_id"],
            disputed_findings=tuple(
                DisputedFinding.from_dict(f) for f in data.get("disputed_findings") or ()
            ),
            reviewer_positions=tuple(
                ReviewerPosition.from_dict(p) for p in data.get("reviewer_positions") or ()
            ),
            evidence_refs=tuple(data.get("evidence_refs") or ()),
            authority_refs=tuple(data.get("authority_refs") or ()),
        )


@dataclass(frozen=True)
class Exclusion:
    """Record of an item dropped from the context packet."""

    exclusion_id: str
    reason: str
    source_item_id: str | None = None
    estimated_chars: int = 0


@dataclass(frozen=True)
class ContextPacket:
    """Canonical context packet with provenance and budget metadata.

    The ``kind``/``source``/``payload`` fields are backward-compatibility shims
    for the minimal packet used by early provider-request tests. New code should
    use the canonical typed fields.
    """

    schema_version: str = CONTEXT_PACKET_SCHEMA_VERSION
    packet_id: str = ""
    authority: tuple[AuthorityContextItem, ...] = ()
    acceptance_criteria: tuple[AcceptanceCriterion, ...] = ()
    relevant_files: tuple[RelevantFile, ...] = ()
    current_diff: tuple[DiffInfo, ...] = ()
    test_evidence: tuple[TestEvidence, ...] = ()
    historical_findings: tuple[HistoricalFinding, ...] = ()
    task_metadata: TaskMetadata | None = None
    exclusions: tuple[Exclusion, ...] = ()
    summaries: tuple[ContextSummary, ...] = ()
    arbitration: ArbitrationContext | None = None
    provenance_index: dict[str, ProvenanceRef] = field(default_factory=dict)
    authority_presence: AuthorityPresence = AuthorityPresence.NOT_REQUIRED
    raw_item_count: int = 0
    summary_count: int = 0
    estimated_input_chars: int = 0
    budget: dict[str, Any] = field(default_factory=dict)
    kind: str | None = None
    source: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != CONTEXT_PACKET_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported context packet schema version {self.schema_version!r}; "
                f"expected {CONTEXT_PACKET_SCHEMA_VERSION!r}"
            )
        if self.raw_item_count < 0:
            raise ValueError("raw_item_count must be non-negative")
        if self.summary_count < 0:
            raise ValueError("summary_count must be non-negative")
        if len(self.summaries) != self.summary_count:
            raise ValueError(
                f"summary_count {self.summary_count} does not match len(summaries) "
                f"{len(self.summaries)}"
            )
        if self.estimated_input_chars < 0:
            raise ValueError("estimated_input_chars must be non-negative")

    def content_hash(self) -> str:
        """Deterministic content hash for this packet."""
        return hash_text(self._canonical_text())

    def _canonical_text(self) -> str:
        data = self.to_dict()
        return _json_dumps_sorted(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a deterministic JSON-compatible dict."""
        return _packet_to_dict(self)

    def to_safe_dict(self) -> dict[str, Any]:
        """Serialize with secret redaction for diagnostics and telemetry."""
        return dict(redact(self.to_dict()))

    def to_safe_json(self) -> str:
        """Serialize to a redacted JSON string."""
        return _json_dumps_sorted(self.to_safe_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextPacket:
        """Deserialize from a dict produced by ``to_dict``."""
        return _packet_from_dict(data)


def _json_dumps_sorted(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _packet_to_dict(packet: ContextPacket) -> dict[str, Any]:
    def _provenance(ref: ProvenanceRef) -> dict[str, Any]:
        return {
            "source_type": ref.source_type,
            "repository": ref.repository,
            "path": ref.path,
            "revision": ref.revision,
            "line_start": ref.line_start,
            "line_end": ref.line_end,
            "commit_sha": ref.commit_sha,
            "pr_or_issue_id": ref.pr_or_issue_id,
            "content_hash": ref.content_hash,
            "authority_level": ref.authority_level,
        }

    def _dicts(items: tuple[Any, ...]) -> list[dict[str, Any]]:
        return [asdict(item) for item in items]

    provenance_index: dict[str, Any] = {
        key: _provenance(value)
        for key, value in deterministic_sort(packet.provenance_index.items())
    }

    return {
        "schema_version": packet.schema_version,
        "packet_id": packet.packet_id,
        "authority": [item.to_dict() for item in packet.authority],
        "acceptance_criteria": _dicts(packet.acceptance_criteria),
        "relevant_files": _dicts(packet.relevant_files),
        "current_diff": _dicts(packet.current_diff),
        "test_evidence": _dicts(packet.test_evidence),
        "historical_findings": _dicts(packet.historical_findings),
        "task_metadata": asdict(packet.task_metadata) if packet.task_metadata else None,
        "exclusions": _dicts(packet.exclusions),
        "summaries": [s.to_dict() for s in packet.summaries],
        "arbitration": packet.arbitration.to_dict() if packet.arbitration else None,
        "provenance_index": provenance_index,
        "authority_presence": packet.authority_presence.name,
        "raw_item_count": packet.raw_item_count,
        "summary_count": packet.summary_count,
        "estimated_input_chars": packet.estimated_input_chars,
        "budget": dict(packet.budget),
        "kind": packet.kind,
        "source": packet.source,
        "payload": dict(packet.payload),
    }


def _packet_from_dict(data: dict[str, Any]) -> ContextPacket:
    def _provenance(raw: dict[str, Any]) -> ProvenanceRef:
        return ProvenanceRef(
            source_type=raw.get("source_type", ""),
            repository=raw.get("repository"),
            path=raw.get("path"),
            revision=raw.get("revision"),
            line_start=raw.get("line_start"),
            line_end=raw.get("line_end"),
            commit_sha=raw.get("commit_sha"),
            pr_or_issue_id=raw.get("pr_or_issue_id"),
            content_hash=raw.get("content_hash"),
            authority_level=raw.get("authority_level"),
        )

    def _items(raw: dict[str, Any], key: str) -> tuple[Any, ...]:
        return tuple(raw.get(key) or ())

    raw_index = data.get("provenance_index") or {}
    provenance_index = {key: _provenance(value) for key, value in raw_index.items()}

    def _criterion(raw: dict[str, Any]) -> AcceptanceCriterion:
        return AcceptanceCriterion(
            criterion_id=raw["criterion_id"],
            text=raw["text"],
            authority_refs=tuple(raw.get("authority_refs") or ()),
        )

    def _file(raw: dict[str, Any]) -> RelevantFile:
        return RelevantFile(
            file_id=raw["file_id"],
            path=raw["path"],
            reason=raw.get("reason"),
            provenance_id=raw.get("provenance_id"),
            content_hash=raw.get("content_hash"),
        )

    def _diff(raw: dict[str, Any]) -> DiffInfo:
        return DiffInfo(
            diff_id=raw["diff_id"],
            path=raw["path"],
            before_ref=raw.get("before_ref"),
            after_ref=raw.get("after_ref"),
            content=raw.get("content", ""),
            provenance_id=raw.get("provenance_id"),
        )

    def _test(raw: dict[str, Any]) -> TestEvidence:
        return TestEvidence(
            evidence_id=raw["evidence_id"],
            test_name=raw["test_name"],
            outcome=raw["outcome"],
            log_excerpt=raw.get("log_excerpt", ""),
            provenance_id=raw.get("provenance_id"),
        )

    def _finding(raw: dict[str, Any]) -> HistoricalFinding:
        return HistoricalFinding(
            finding_id=raw["finding_id"],
            summary=raw["summary"],
            provenance_id=raw.get("provenance_id"),
            related_symbols=tuple(raw.get("related_symbols") or ()),
        )

    def _task(raw: dict[str, Any] | None) -> TaskMetadata | None:
        if raw is None:
            return None
        return TaskMetadata(
            task_id=raw["task_id"],
            role=raw.get("role"),
            risk=raw.get("risk"),
            requested_objective=raw.get("requested_objective", ""),
            constraints=tuple(raw.get("constraints") or ()),
        )

    def _exclusion(raw: dict[str, Any]) -> Exclusion:
        return Exclusion(
            exclusion_id=raw["exclusion_id"],
            reason=raw["reason"],
            source_item_id=raw.get("source_item_id"),
            estimated_chars=raw.get("estimated_chars", 0),
        )

    def _authority(raw: dict[str, Any]) -> AuthorityContextItem:
        return AuthorityContextItem.from_dict(raw)

    authority_data = data.get("authority") or ()
    authority: tuple[AuthorityContextItem, ...]
    if authority_data and isinstance(authority_data[0], str):
        # Backward compatibility for legacy string-only authority tuples.
        authority = tuple(
            AuthorityContextItem(
                authority_id=f"legacy-{i}",
                provenance_id="",
                full_source_ref=text,
                revision="",
                content_hash="",
                content=text,
                raw_included=True,
            )
            for i, text in enumerate(authority_data)
        )
    else:
        authority = tuple(_authority(item) for item in authority_data)

    arbitration_raw = data.get("arbitration")
    arbitration = ArbitrationContext.from_dict(arbitration_raw) if arbitration_raw else None

    authority_presence_value = data.get("authority_presence", AuthorityPresence.NOT_REQUIRED.name)
    try:
        authority_presence = AuthorityPresence[authority_presence_value]
    except KeyError:
        authority_presence = AuthorityPresence.NOT_REQUIRED

    return ContextPacket(
        schema_version=data.get("schema_version", CONTEXT_PACKET_SCHEMA_VERSION),
        packet_id=data.get("packet_id", ""),
        authority=authority,
        acceptance_criteria=tuple(_criterion(item) for item in _items(data, "acceptance_criteria")),
        relevant_files=tuple(_file(item) for item in _items(data, "relevant_files")),
        current_diff=tuple(_diff(item) for item in _items(data, "current_diff")),
        test_evidence=tuple(_test(item) for item in _items(data, "test_evidence")),
        historical_findings=tuple(_finding(item) for item in _items(data, "historical_findings")),
        task_metadata=_task(data.get("task_metadata")),
        exclusions=tuple(_exclusion(item) for item in _items(data, "exclusions")),
        summaries=tuple(ContextSummary.from_dict(s) for s in data.get("summaries") or ()),
        arbitration=arbitration,
        provenance_index=provenance_index,
        authority_presence=authority_presence,
        raw_item_count=data.get("raw_item_count", 0),
        summary_count=data.get("summary_count", 0),
        estimated_input_chars=data.get("estimated_input_chars", 0),
        budget=dict(data.get("budget") or {}),
        kind=data.get("kind"),
        source=data.get("source"),
        payload=dict(data.get("payload") or {}),
    )
