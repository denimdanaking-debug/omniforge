from __future__ import annotations

import pytest

from src.policy.risk import RiskLevel
from src.providers.identity import (
    ProviderHealth,
    ProviderOperationalState,
    ProviderQuotaState,
    QuotaSignal,
)
from src.recovery.reserve import ReserveCapacityPolicy
from src.recovery.state_machine import RouteRecoveryState
from src.routing.capabilities import CostMetadata, DeploymentMode, ModelCapabilities
from src.routing.dynamic.candidate import PerformanceEvidence, RoutingCandidate
from src.routing.dynamic.request import DynamicRoutingRequest
from src.routing.inference_route import (
    InferenceRouteIdentity,
    RouteHealth,
    RouteOperationalState,
    RouteType,
)
from src.routing.model_identity import ModelIdentity, ModelLifecycle
from src.routing.policy import RoutingPolicyEngine
from src.routing.roles import ExecutionRole


@pytest.fixture
def base_request() -> DynamicRoutingRequest:
    return DynamicRoutingRequest(
        task_id="task-1",
        project_id="project-a",
        role=ExecutionRole.CODING,
        risk=RiskLevel.R2_NORMAL,
        task_class="default",
    )


@pytest.fixture
def base_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        context_tokens=128_000,
        structured_output=True,
        tool_use=True,
        streaming=True,
        code_generation=True,
        deployment_mode=DeploymentMode.CLOUD,
        cost=CostMetadata(input_per_million=2.5, output_per_million=10.0),
        supported_roles=frozenset({ExecutionRole.CODING.value}),
    )


@pytest.fixture
def healthy_candidate(base_capabilities: ModelCapabilities) -> RoutingCandidate:
    model = ModelIdentity(model_id="gpt-4o", family="gpt", lifecycle=ModelLifecycle.HIGH_RISK)
    route = InferenceRouteIdentity(
        route_id="openai-direct",
        provider_id="openai",
        route_type=RouteType.DIRECT,
        endpoint_key="https://api.openai.com/v1",
        failure_domain="openai.com",
    )
    return RoutingCandidate(
        provider_id="openai",
        model_id="gpt-4o",
        route_id="openai-direct",
        model_identity=model,
        route_identity=route,
        capabilities=base_capabilities,
        recovery_state=RouteRecoveryState(health=ProviderHealth.HEALTHY),
        operational_state=ProviderOperationalState(health=ProviderHealth.HEALTHY),
        route_cost_state=RouteOperationalState(
            health=RouteHealth.HEALTHY,
            rolling_latency_ms=300.0,
            input_cost_per_million=2.5,
            output_cost_per_million=10.0,
        ),
        quota_state=ProviderQuotaState(provider_signal=QuotaSignal.AVAILABLE),
    )


@pytest.fixture
def cheap_unreliable_candidate(base_capabilities: ModelCapabilities) -> RoutingCandidate:
    model = ModelIdentity(model_id="tiny-model", family="tiny", lifecycle=ModelLifecycle.NORMAL)
    route = InferenceRouteIdentity(
        route_id="tiny-direct",
        provider_id="tiny-provider",
        route_type=RouteType.DIRECT,
        endpoint_key="https://tiny.example.com",
        failure_domain="tiny.example.com",
    )
    caps = ModelCapabilities(
        context_tokens=32_000,
        code_generation=True,
        deployment_mode=DeploymentMode.CLOUD,
        cost=CostMetadata(input_per_million=0.5, output_per_million=2.5),
        supported_roles=frozenset({ExecutionRole.CODING.value}),
    )
    evidence = PerformanceEvidence(
        attempts=100,
        successes=10,
        success_rate=0.1,
        retry_rate=0.5,
        repair_rate=0.4,
    )
    return RoutingCandidate(
        provider_id="tiny-provider",
        model_id="tiny-model",
        route_id="tiny-direct",
        model_identity=model,
        route_identity=route,
        capabilities=caps,
        recovery_state=RouteRecoveryState(health=ProviderHealth.HEALTHY),
        operational_state=ProviderOperationalState(health=ProviderHealth.HEALTHY),
        route_cost_state=RouteOperationalState(
            health=RouteHealth.HEALTHY,
            input_cost_per_million=0.5,
            output_cost_per_million=2.5,
        ),
        performance_evidence=evidence,
    )


@pytest.fixture
def expensive_reliable_candidate(base_capabilities: ModelCapabilities) -> RoutingCandidate:
    model = ModelIdentity(
        model_id="reliable-model", family="reliable", lifecycle=ModelLifecycle.HIGH_RISK
    )
    route = InferenceRouteIdentity(
        route_id="reliable-direct",
        provider_id="reliable-provider",
        route_type=RouteType.DIRECT,
        endpoint_key="https://reliable.example.com",
        failure_domain="reliable.example.com",
    )
    caps = ModelCapabilities(
        context_tokens=128_000,
        code_generation=True,
        deployment_mode=DeploymentMode.CLOUD,
        cost=CostMetadata(input_per_million=5.0, output_per_million=20.0),
        supported_roles=frozenset({ExecutionRole.CODING.value}),
    )
    evidence = PerformanceEvidence(
        attempts=100,
        successes=98,
        success_rate=0.98,
        retry_rate=0.01,
        repair_rate=0.01,
    )
    return RoutingCandidate(
        provider_id="reliable-provider",
        model_id="reliable-model",
        route_id="reliable-direct",
        model_identity=model,
        route_identity=route,
        capabilities=caps,
        recovery_state=RouteRecoveryState(health=ProviderHealth.HEALTHY),
        operational_state=ProviderOperationalState(health=ProviderHealth.HEALTHY),
        route_cost_state=RouteOperationalState(
            health=RouteHealth.HEALTHY,
            input_cost_per_million=5.0,
            output_cost_per_million=20.0,
        ),
        performance_evidence=evidence,
    )


@pytest.fixture
def policy_engine() -> RoutingPolicyEngine:
    return RoutingPolicyEngine(
        provider_enabled={
            "openai": True,
            "anthropic": True,
            "tiny-provider": True,
            "reliable-provider": True,
        },
        model_enabled={"gpt-4o": True, "claude": True},
        route_enabled={"openai-direct": True, "anthropic-direct": True},
    )


@pytest.fixture
def reserve_policy() -> ReserveCapacityPolicy:
    return ReserveCapacityPolicy(
        reserved_provider_ids=frozenset({"openai"}),
        reserved_roles=frozenset({ExecutionRole.HIGH_RISK_REVIEW.value}),
    )
