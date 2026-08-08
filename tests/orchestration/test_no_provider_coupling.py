"""Architectural enforcement: core orchestration must not branch on provider names.

This test implements Step 2.8. It uses AST inspection to detect provider-specific
conditionals, string comparisons, and forbidden imports in core orchestration
modules. The check is structural rather than a simplistic string search.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import pkgutil
import re
from pathlib import Path
from typing import Any

import pytest

from providers.contracts.adapter import ProviderAdapter

_ORCHESTRATION_ROOT = Path(__file__).parents[2] / "src" / "orchestration"

_PROVIDER_NAME_LITERALS = {
    "openai",
    "anthropic",
    "claude",
    "kimi",
    "qwen",
    "deepseek",
    "google",
    "gemini",
    "xai",
    "grok",
    "z.ai",
    "glm",
    "minimax",
    "mistral",
    "openrouter",
}

_FORBIDDEN_ADAPTER_IMPORTS = {
    "providers.adapters",
    "providers.adapters.stub",
}


def _iter_orchestration_source_paths() -> list[Path]:
    if not _ORCHESTRATION_ROOT.exists():
        return []
    paths: list[Path] = []
    for path in _ORCHESTRATION_ROOT.rglob("*.py"):
        if path.name == "__init__.py" and path.read_text(encoding="utf-8").strip() == "":
            continue
        paths.append(path)
    return paths


def _read_module_ast(path: Path) -> ast.AST:
    source = path.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(path))


def _provider_name_in_literal(value: str) -> str | None:
    """Return the matched provider name if ``value`` contains one as a whole word."""
    lowered = value.lower()
    for name in _PROVIDER_NAME_LITERALS:
        pattern = rf"\b{re.escape(name)}\b"
        if re.search(pattern, lowered):
            return name
    return None


def _collect_provider_name_violations(tree: ast.AST) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            matched = _provider_name_in_literal(node.value)
            if matched:
                violations.append(
                    f"provider name literal ({matched}) at line {node.lineno}: {node.value!r}"
                )
        elif isinstance(node, ast.Compare):
            # Detect comparisons like provider == "openai" or "anthropic" in <expression>
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    matched = _provider_name_in_literal(comparator.value)
                    if matched:
                        violations.append(
                            f"provider comparison ({matched}) at line {comparator.lineno}: "
                            f"{comparator.value!r}"
                        )
    return violations


def _collect_forbidden_imports(tree: ast.AST, path: Path) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(
                    alias.name.startswith(forbidden) for forbidden in _FORBIDDEN_ADAPTER_IMPORTS
                ):
                    violations.append(f"forbidden import at line {node.lineno}: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if any(module.startswith(forbidden) for forbidden in _FORBIDDEN_ADAPTER_IMPORTS):
                violations.append(f"forbidden import at line {node.lineno}: {module}")
    return violations


@pytest.mark.architecture
def test_no_provider_name_literals_in_orchestration() -> None:
    paths = _iter_orchestration_source_paths()
    all_violations: list[str] = []
    for path in paths:
        tree = _read_module_ast(path)
        for violation in _collect_provider_name_violations(tree):
            all_violations.append(f"{path}: {violation}")
    assert not all_violations, "\n".join(all_violations)


@pytest.mark.architecture
def test_no_forbidden_adapter_imports_in_orchestration() -> None:
    paths = _iter_orchestration_source_paths()
    all_violations: list[str] = []
    for path in paths:
        tree = _read_module_ast(path)
        for violation in _collect_forbidden_imports(tree, path):
            all_violations.append(f"{path}: {violation}")
    assert not all_violations, "\n".join(all_violations)


@pytest.mark.architecture
def test_orchestration_depends_only_on_normalized_adapter_contract() -> None:
    """Core orchestration modules may import from providers.contracts but not adapters."""
    paths = _iter_orchestration_source_paths()
    for path in paths:
        tree = _read_module_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("providers.") and not module.startswith("providers.contracts"):
                    pytest.fail(
                        f"{path}: orchestration imports from {module}; "
                        "only providers.contracts is allowed"
                    )


@pytest.mark.architecture
def test_provider_adapter_is_abstract_protocol() -> None:
    """The normalized adapter contract is abstract and provider-name agnostic."""
    assert inspect.isabstract(ProviderAdapter)
    source = inspect.getsource(ProviderAdapter)
    lowered = source.lower()
    for name in _PROVIDER_NAME_LITERALS:
        assert name not in lowered, f"ProviderAdapter mentions provider name {name!r}"


@pytest.mark.architecture
def test_no_provider_specific_attributes_on_adapter() -> None:
    """The normalized adapter contract has no provider-specific attributes."""
    attrs = [name for name in dir(ProviderAdapter) if not name.startswith("_")]
    for attr in attrs:
        for provider_name in _PROVIDER_NAME_LITERALS:
            assert provider_name not in attr.lower(), (
                f"ProviderAdapter attribute {attr!r} contains provider name {provider_name!r}"
            )


def _import_all_orchestration_modules() -> list[Any]:
    imported: list[Any] = []
    if not _ORCHESTRATION_ROOT.exists():
        return imported
    for _, module_name, _ in pkgutil.iter_modules([str(_ORCHESTRATION_ROOT)], "orchestration."):
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        imported.append(module)
    return imported


@pytest.mark.architecture
def test_orchestration_modules_import_cleanly() -> None:
    """All orchestration modules import without provider-specific runtime errors."""
    modules = _import_all_orchestration_modules()
    assert all(module is not None for module in modules)
