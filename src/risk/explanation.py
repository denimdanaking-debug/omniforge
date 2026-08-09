"""Structured, audit-safe risk explanations."""

from __future__ import annotations

from dataclasses import dataclass

from src.policy.risk import RiskLevel

from .assessment import RiskAssessmentResult


@dataclass(frozen=True)
class RiskExplanation:
    """User-visible structured explanation of a risk assessment."""

    risk: RiskLevel
    summary: str
    factor_codes: tuple[str, ...]
    reasons: tuple[str, ...]


def format_explanation(result: RiskAssessmentResult, project_id: str = "") -> RiskExplanation:
    """Format a risk assessment result into an explicit explanation."""
    factors = result.factors
    reasons = tuple(f.evidence for f in factors)
    codes = tuple(f.code.value for f in factors)
    prefix = f"project {project_id}: " if project_id else ""
    summary = (
        f"{prefix}Risk {result.final_risk.name} based on {len(factors)} deterministic factor(s)"
    )
    return RiskExplanation(
        risk=result.final_risk,
        summary=summary,
        factor_codes=codes,
        reasons=reasons,
    )
