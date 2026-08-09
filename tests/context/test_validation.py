"""Tests for context packet validation."""

from __future__ import annotations

from src.context.schema import (
    AuthorityContextItem,
    AuthorityPresence,
    ContextPacket,
    ContextSummary,
    ProvenanceRef,
    RelevantFile,
)
from src.context.validation import ContextPacketValidator, ValidationIssue

_FAKE_COMMIT = "a" * 40
_FAKE_HASH = "b" * 64


def test_authority_protection() -> None:
    packet = ContextPacket(
        authority_presence=AuthorityPresence.RAW_INCLUDED,
        authority=(),
    )
    issues = ContextPacketValidator().validate(packet)
    assert any(issue.code == "MISSING_AUTHORITY" for issue in issues)


def test_summary_claiming_authority_rejected() -> None:
    summary = ContextSummary(
        text="summary of roadmap",
        source_provenance_ids=("s1",),
        source_hashes=(_FAKE_HASH,),
        level="section",
    )
    packet = ContextPacket(
        authority=(
            AuthorityContextItem(
                authority_id="auth-1",
                provenance_id="auth-prov-1",
                full_source_ref="summary of roadmap",
                revision=_FAKE_COMMIT,
                content_hash=_FAKE_HASH,
                content="summary of roadmap",
                raw_included=True,
            ),
        ),
        summaries=(summary,),
        summary_count=1,
        provenance_index={
            "s1": ProvenanceRef("summary", path="summary", content_hash=_FAKE_HASH),
            "auth-prov-1": ProvenanceRef(
                "authority",
                path="summary of roadmap",
                revision=_FAKE_COMMIT,
                content_hash=_FAKE_HASH,
                authority_level="primary",
            ),
        },
        authority_presence=AuthorityPresence.NOT_REQUIRED,
    )
    issues = ContextPacketValidator().validate(packet)
    assert any(issue.code == "SUMMARY_CLAIMS_AUTHORITY" for issue in issues)


def test_dangling_provenance_detected() -> None:
    packet = ContextPacket(
        relevant_files=(RelevantFile("f1", "a.py", provenance_id="missing"),),
    )
    issues = ContextPacketValidator().validate(packet)
    assert any(issue.code == "DANGLING_PROVENANCE" for issue in issues)


def test_valid_packet_has_no_issues() -> None:
    packet = ContextPacket(
        packet_id="p1",
        authority=(
            AuthorityContextItem(
                authority_id="auth-1",
                provenance_id="p1",
                full_source_ref="roadmap.md",
                revision=_FAKE_COMMIT,
                content_hash=_FAKE_HASH,
                content="roadmap.md",
                raw_included=True,
            ),
        ),
        authority_presence=AuthorityPresence.RAW_INCLUDED,
        provenance_index={
            "p1": ProvenanceRef(
                "authority",
                path="roadmap.md",
                revision=_FAKE_COMMIT,
                content_hash=_FAKE_HASH,
                authority_level="primary",
            )
        },
    )
    issues = ContextPacketValidator().validate(packet)
    assert not issues


def test_validation_issue_is_frozen() -> None:
    issue = ValidationIssue("error", "CODE", "message")
    assert issue.severity == "error"


def test_raw_included_missing_revision_and_hash_rejected() -> None:
    packet = ContextPacket(
        authority=(
            AuthorityContextItem(
                authority_id="auth-1",
                provenance_id="auth-prov-1",
                full_source_ref="roadmap.md",
                revision="",
                content_hash="",
                content="roadmap content",
                raw_included=True,
            ),
        ),
        authority_presence=AuthorityPresence.RAW_INCLUDED,
        provenance_index={
            "auth-prov-1": ProvenanceRef(
                "authority",
                path="roadmap.md",
                authority_level="roadmap",
            ),
        },
    )
    issues = ContextPacketValidator().validate(packet)
    assert any(issue.code == "AUTHORITY_ITEM_MISSING_IMMUTABLE_IDENTITY" for issue in issues)


def test_raw_referenced_missing_revision_and_hash_rejected() -> None:
    packet = ContextPacket(
        authority=(
            AuthorityContextItem(
                authority_id="auth-1",
                provenance_id="auth-prov-1",
                full_source_ref="roadmap.md",
                revision="",
                content_hash="",
                content=None,
                raw_included=False,
            ),
        ),
        authority_presence=AuthorityPresence.RAW_REFERENCED,
        provenance_index={
            "auth-prov-1": ProvenanceRef(
                "authority",
                path="roadmap.md",
                authority_level="roadmap",
            ),
        },
    )
    issues = ContextPacketValidator().validate(packet)
    assert any(issue.code == "AUTHORITY_ITEM_MISSING_IMMUTABLE_IDENTITY" for issue in issues)


def test_authority_revision_mismatch_rejected() -> None:
    packet = ContextPacket(
        authority=(
            AuthorityContextItem(
                authority_id="auth-1",
                provenance_id="auth-prov-1",
                full_source_ref="roadmap.md",
                revision="a" * 40,
                content_hash="b" * 64,
                content="roadmap content",
                raw_included=True,
            ),
        ),
        authority_presence=AuthorityPresence.RAW_INCLUDED,
        provenance_index={
            "auth-prov-1": ProvenanceRef(
                "authority",
                path="roadmap.md",
                revision="c" * 40,
                content_hash="b" * 64,
                authority_level="roadmap",
            ),
        },
    )
    issues = ContextPacketValidator().validate(packet)
    assert any(issue.code == "AUTHORITY_REVISION_MISMATCH" for issue in issues)


def test_authority_hash_mismatch_rejected() -> None:
    packet = ContextPacket(
        authority=(
            AuthorityContextItem(
                authority_id="auth-1",
                provenance_id="auth-prov-1",
                full_source_ref="roadmap.md",
                revision="a" * 40,
                content_hash="b" * 64,
                content="roadmap content",
                raw_included=True,
            ),
        ),
        authority_presence=AuthorityPresence.RAW_INCLUDED,
        provenance_index={
            "auth-prov-1": ProvenanceRef(
                "authority",
                path="roadmap.md",
                revision="a" * 40,
                content_hash="d" * 64,
                authority_level="roadmap",
            ),
        },
    )
    issues = ContextPacketValidator().validate(packet)
    assert any(issue.code == "AUTHORITY_HASH_MISMATCH" for issue in issues)


def test_legacy_string_authority_does_not_satisfy_strict_validation() -> None:
    """A bare legacy authority string has no pinned revision/hash and must fail."""
    data = {
        "packet_id": "legacy",
        "authority": ("roadmap.md",),
        "authority_presence": "RAW_INCLUDED",
        "provenance_index": {},
    }
    packet = ContextPacket.from_dict(data)
    issues = ContextPacketValidator().validate(packet)
    assert any(
        issue.code
        in {
            "AUTHORITY_ITEM_MISSING_IMMUTABLE_IDENTITY",
            "AUTHORITY_DANGLING_PROVENANCE",
        }
        for issue in issues
    )


def test_valid_rich_authority_has_no_issues() -> None:
    packet = ContextPacket(
        authority=(
            AuthorityContextItem(
                authority_id="auth-1",
                provenance_id="auth-prov-1",
                full_source_ref="roadmap.md",
                revision="a" * 40,
                content_hash="b" * 64,
                content="roadmap content",
                raw_included=True,
            ),
        ),
        authority_presence=AuthorityPresence.RAW_INCLUDED,
        provenance_index={
            "auth-prov-1": ProvenanceRef(
                "authority",
                path="roadmap.md",
                revision="a" * 40,
                content_hash="b" * 64,
                authority_level="roadmap",
            ),
        },
    )
    issues = ContextPacketValidator().validate(packet)
    assert not issues
