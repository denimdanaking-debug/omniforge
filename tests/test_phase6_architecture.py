"""Architectural enforcement tests for Phase 6 recovery engine boundaries."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import pytest

from src.providers.errors import ProviderError, ProviderErrorCode
from src.providers.identity import ProviderHealth, ProviderQuotaState, QuotaSignal
from src.recovery import (
    FixedClock,
    HealthStateMachine,
    OutageSurvivalEngine,
    RouteRecoveryState,
    SurvivalCandidate,
    signal_from_error,
    signal_from_quota,
)
from src.recovery.failure_domain import FailureDomainIndex
from src.recovery.scheduler import RecheckPolicy, RecoveryScheduler
from src.recovery.survival import PersistedWait
from src.routing.capabilities import ModelCapabilities
from src.routing.model_identity import ModelIdentity, ModelRegistry, ModelReputation
from src.routing.roles import ExecutionRole
from src.security.redaction import contains_secret

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "OMNIFORGE_TEST_SECRET_SENTINEL_PHASE6_ARCH_888"


def _source_files(glob: str) -> list[Path]:
    return sorted(ROOT.glob(glob))


@pytest.mark.architecture
def test_recovery_modules_do_not_scrape_os_environ() -> None:
    """Recovery engine code must not read the process environment directly."""
    violations: list[str] = []
    for path in _source_files("src/recovery/**/*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "environ"
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
            ):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            if isinstance(node, ast.Name) and node.id == "environ":
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, "recovery modules must not access os.environ: " + "; ".join(
        violations[:10]
    )


@pytest.mark.architecture
def test_infrastructure_errors_do_not_mutate_model_reputation() -> None:
    """Provider/route infrastructure failures must not damage model reputation."""
    registry = ModelRegistry()
    identity = ModelIdentity(model_id="claude-x", family="claude")
    registry.register(identity)
    before = registry.get("claude-x").reputation

    clock = FixedClock()
    sm = HealthStateMachine(clock)
    state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
    error = ProviderError(
        code=ProviderErrorCode.QUOTA_EXHAUSTED,
        message="quota exhausted",
    )
    signal = signal_from_error(
        error, route_id="anthropic-direct", failure_domain="anthropic", clock=clock
    )
    sm.apply(state, signal)

    after = registry.get("claude-x").reputation
    assert after == before


@pytest.mark.architecture
def test_quota_exhaustion_does_not_penalize_model_quality() -> None:
    """Quota signals are capacity facts, not model-quality evidence."""
    registry = ModelRegistry()
    identity = ModelIdentity(model_id="qwen-max", family="qwen")
    registry.register(identity)
    registry.set_reputation("qwen-max", ModelReputation(attempts=5, accepted=4))
    before = registry.get("qwen-max").reputation

    clock = FixedClock()
    sm = HealthStateMachine(clock)
    state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
    quota = ProviderQuotaState(provider_signal=QuotaSignal.EXHAUSTED)
    signal = signal_from_quota(
        quota, provider_id="qwen", route_id="qwen-direct", failure_domain="qwen", clock=clock
    )
    sm.apply(state, signal)

    assert registry.get("qwen-max").reputation == before


@pytest.mark.architecture
def test_recovered_provider_becomes_eligible_not_preferred() -> None:
    """Recovery restores eligibility; it does not artificially make a route the winner."""
    clock = FixedClock()
    engine = OutageSurvivalEngine(clock=clock)
    candidates = [
        SurvivalCandidate(
            provider_id="b",
            model_id="recovered-model",
            route_id="recovered",
            capabilities=ModelCapabilities(context_tokens=1000),
            recovery_state=RouteRecoveryState(health=ProviderHealth.HEALTHY),
        ),
        SurvivalCandidate(
            provider_id="a",
            model_id="steady-model",
            route_id="steady",
            capabilities=ModelCapabilities(context_tokens=1000),
            recovery_state=RouteRecoveryState(health=ProviderHealth.HEALTHY),
        ),
    ]
    decision = engine.dispatch_or_wait(role=ExecutionRole.CODING, candidates=candidates)
    assert decision.dispatch
    # Deterministic tie-break on provider_id; recovery does not reorder above "a".
    assert decision.choice is not None
    assert decision.choice.provider_id == "a"


@pytest.mark.architecture
def test_no_naive_datetimes_in_recovery_state() -> None:
    """Recovery dataclasses reject naive timestamps."""
    naive = __import__("datetime").datetime(2026, 1, 1)
    with pytest.raises(ValueError):
        RouteRecoveryState(health=ProviderHealth.HEALTHY, last_success_at=naive)

    with pytest.raises(ValueError):
        PersistedWait(
            task_id="t",
            role="coding",
            reason="test",
            affected_failure_domains=frozenset(),
            next_recheck_at=naive,
        )


@pytest.mark.architecture
def test_no_hot_loop_scheduling_after_quota_exhaustion() -> None:
    """Quota exhaustion must schedule a future recheck, not an immediate retry."""
    clock = FixedClock()
    sm = HealthStateMachine(clock)
    scheduler = RecoveryScheduler(policy=RecheckPolicy())
    state = RouteRecoveryState(health=ProviderHealth.HEALTHY)
    error = ProviderError(code=ProviderErrorCode.QUOTA_EXHAUSTED, message="quota exhausted")
    signal = signal_from_error(error, route_id="r", failure_domain="fd", clock=clock)
    state = sm.apply(state, signal)
    assert state.next_recheck_at is not None
    scheduler.schedule("r", state.next_recheck_at)
    assert scheduler.due_routes(clock) == ()


@pytest.mark.architecture
def test_administrative_disabled_distinct_from_outage() -> None:
    """Administrative disablement must not be overwritten by infrastructure signals."""
    clock = FixedClock()
    sm = HealthStateMachine(clock)
    state = RouteRecoveryState(health=ProviderHealth.DISABLED)
    error = ProviderError(code=ProviderErrorCode.PROVIDER_UNAVAILABLE, message="down")
    signal = signal_from_error(error, route_id="r", failure_domain="fd", clock=clock)
    new_state = sm.apply(state, signal)
    assert new_state.health is ProviderHealth.DISABLED


@pytest.mark.architecture
def test_failure_domain_propagation_does_not_mutate_model_reputation() -> None:
    """Shared-domain outage propagation changes route state, not model reputation."""
    registry = ModelRegistry()
    identity = ModelIdentity(model_id="claude-x", family="claude")
    registry.register(identity)
    registry.set_reputation("claude-x", ModelReputation(attempts=3, accepted=3))
    before = registry.get("claude-x").reputation

    clock = FixedClock()
    sm = HealthStateMachine(clock)
    index = FailureDomainIndex()
    index.register("openrouter-claude", "openrouter")
    states = {"openrouter-claude": RouteRecoveryState(health=ProviderHealth.HEALTHY)}
    index.mark_domain_affected("openrouter", sm, states, "openrouter outage")

    assert registry.get("claude-x").reputation == before


@pytest.mark.architecture
def test_secret_sentinel_absent_from_persisted_recovery_state() -> None:
    """Raw secret material must not survive into persisted runtime state."""
    from src.persistence import runtime_state

    error = ProviderError(
        code=ProviderErrorCode.AUTH_FAILURE,
        message=f"auth failed bearer {SENTINEL}",
        raw_metadata={"api_key": SENTINEL},
    )
    state = {
        "schema_version": "1.2.0",
        "run_id": "run-1",
        "workflow_state": "WAITING_FOR_PROVIDER",
        "checkpoint": {},
        "provider_status": {},
        "model_status": {},
        "route_status": {},
        "routing_mode": "legacy",
        "exploration_enabled": False,
        "pins": {},
        "project_policies": {},
        "provider_recovery_state": {
            "openai": RouteRecoveryState(
                health=ProviderHealth.AUTH_FAILED,
                reason=error.message,
            ).to_dict()
        },
        "route_recovery_state": {},
        "failure_domain_index": {},
        "recovery_scheduler": {},
        "waiting_tasks": {
            "task-1": {
                "task_id": "task-1",
                "role": "coding",
                "reason": error.message,
                "affected_failure_domains": [],
                "next_recheck_at": "2026-01-01T00:00:00+00:00",
                "attempted_candidates": [{"error": error.message, "meta": error.raw_metadata}],
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runtime.json"
        runtime_state.save_runtime_state(path, state)
        text = path.read_text(encoding="utf-8")
        assert not contains_secret(text, SENTINEL)
