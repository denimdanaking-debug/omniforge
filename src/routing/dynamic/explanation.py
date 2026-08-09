"""Explanation formatting derived only from decision records."""

from __future__ import annotations

from .decision import RoutingDecision


class ExplanationFormatter:
    """Format a human-readable explanation from a routing decision record."""

    def format(self, decision: RoutingDecision) -> str:
        """Return a structured multi-line summary of the decision."""
        lines: list[str] = []
        record = decision.record

        lines.append(f"Decision: {record.decision_id}")
        lines.append(f"Mode: {record.routing_mode}")
        lines.append(f"Exploration enabled: {record.exploration_enabled}")
        lines.append(f"Fallback used: {record.fallback_used}")
        if record.fallback_reason:
            lines.append(f"Fallback reason: {record.fallback_reason}")

        if decision.selected_candidate is not None:
            winner = decision.selected_candidate
            lines.append(f"Winner: {winner.provider_id}:{winner.model_id}:{winner.route_id}")
        else:
            lines.append("Winner: none")
            if decision.no_eligible_reason:
                lines.append(f"Reason: {decision.no_eligible_reason}")

        if record.runner_up is not None:
            runner = record.runner_up
            lines.append(f"Runner-up: {runner.provider_id}:{runner.model_id}:{runner.route_id}")
            lines.append(f"Score margin: {record.score_margin:.6f}")

        lines.append(f"Candidates considered: {len(record.candidates_considered)}")
        lines.append(f"Eligible candidates: {len(record.eligible_candidates)}")
        lines.append(f"Excluded candidates: {len(record.exclusions)}")

        if record.exclusions:
            lines.append("Excluded key candidates:")
            for exclusion in record.exclusions[:10]:
                lines.append(
                    f"  - {exclusion.identity_key}: {exclusion.reason} ({exclusion.detail})"
                )
            if len(record.exclusions) > 10:
                lines.append(f"  ... and {len(record.exclusions) - 10} more")

        if record.scores:
            lines.append("Top scores:")
            for score in record.scores[:5]:
                lines.append(f"  - {score.tie_break_key}: {score.total_score:.6f}")

        return "\n".join(lines)
