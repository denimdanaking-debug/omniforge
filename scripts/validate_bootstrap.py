#!/usr/bin/env python3
"""Dependency-free bootstrap validator for OmniForge."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ROADMAP = DOCS / "OMNIFORGE_FULL_ROADMAP_v1.0.md"
STATE = DOCS / "PROJECT_STATE.json"
AUTHORITY = DOCS / "ROADMAP_AUTHORITY.json"

TEXT_SUFFIXES = {".md", ".json", ".py", ".yml", ".yaml", ".toml", ".txt"}


def iter_text_files():
    ignored = {".git", ".venv", "venv", "node_modules", "bin", "obj"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in ignored for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore", ".editorconfig"}:
            yield path


def lint() -> list[str]:
    errors: list[str] = []
    for path in iter_text_files():
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"{path.relative_to(ROOT)}: not UTF-8")
            continue
        if b"\r\n" in data:
            errors.append(f"{path.relative_to(ROOT)}: CRLF line endings detected")
        if text and not text.endswith("\n"):
            errors.append(f"{path.relative_to(ROOT)}: missing final newline")
        for number, line in enumerate(text.splitlines(), start=1):
            if line.rstrip(" \t") != line:
                errors.append(f"{path.relative_to(ROOT)}:{number}: trailing whitespace")
    for path in ROOT.rglob("*.json"):
        if ".git" in path.parts:
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path.relative_to(ROOT)}: invalid JSON: {exc}")
    return errors


def build() -> list[str]:
    errors: list[str] = []
    required = [ROADMAP, STATE, AUTHORITY]
    for path in required:
        if not path.exists():
            errors.append(f"missing required authority file: {path.relative_to(ROOT)}")
    if errors:
        return errors

    state = json.loads(STATE.read_text(encoding="utf-8"))
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    roadmap_bytes = ROADMAP.read_bytes()
    roadmap_text = roadmap_bytes.decode("utf-8")

    step_count = len(re.findall(r"^##\s+\d+\.\d+\s+", roadmap_text, flags=re.MULTILINE))
    digest = hashlib.sha256(roadmap_bytes).hexdigest()

    if state.get("project_id") != "omniforge":
        errors.append("PROJECT_STATE project_id must be omniforge")
    if state.get("roadmap_file") != authority.get("roadmap_path"):
        errors.append("roadmap path mismatch between state and authority")
    if state.get("total_verified_steps") != step_count:
        errors.append(f"PROJECT_STATE step count {state.get('total_verified_steps')} != roadmap {step_count}")
    if authority.get("verified_step_count") != step_count:
        errors.append(f"ROADMAP_AUTHORITY step count {authority.get('verified_step_count')} != roadmap {step_count}")
    if authority.get("sha256") != digest:
        errors.append(f"roadmap SHA-256 mismatch: authority={authority.get('sha256')} actual={digest}")
    completed = state.get("completed_steps", [])
    if state.get("completed_verified_steps") != len(completed):
        errors.append("completed_verified_steps does not equal completed_steps length")
    if not authority.get("workspace_manager_included"):
        errors.append("workspace manager must be included in authority record")
    for model_key in ("kimi_k3_initial_status", "qwen_3_8_max_initial_status"):
        if authority.get(model_key) != "HIGH_RISK":
            errors.append(f"{model_key} must remain HIGH_RISK in v1.0 authority")
    return errors


def test() -> list[str]:
    errors: list[str] = []
    roadmap_text = ROADMAP.read_text(encoding="utf-8") if ROADMAP.exists() else ""
    required_phrases = [
        "Only work that has been:",
        "## 21.14 Implement first-class Workspace Manager",
        "## 21.20 Implement Workspace Janitor",
        "Kimi K3 and Qwen3.8-Max are first-class production models",
    ]
    for phrase in required_phrases:
        if phrase not in roadmap_text:
            errors.append(f"roadmap invariant missing: {phrase}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("lint", "build", "test", "all"), default="all")
    args = parser.parse_args()

    checks = {"lint": lint, "build": build, "test": test}
    modes = checks if args.mode == "all" else {args.mode: checks[args.mode]}
    failures = 0
    for name, check in modes.items():
        errors = check()
        if errors:
            failures += len(errors)
            print(f"[{name}] FAIL")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[{name}] PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
