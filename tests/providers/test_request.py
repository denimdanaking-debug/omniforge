"""Unit tests for the normalized request contract (Step 2.2)."""

from __future__ import annotations

import pytest

from src.policy.risk import RiskLevel
from src.providers.identity import ProviderIdentity
from src.providers.request import (
    CapabilityRequirement,
    ContextPacket,
    Message,
    MessageRole,
    ProviderRequest,
    ReasoningMode,
    StructuredOutputRequirement,
    TaskLineage,
    ToolChoiceMode,
    ToolDefinition,
)
from src.routing.inference_route import InferenceRouteIdentity, RouteType
from src.routing.model_identity import ModelIdentity
from src.routing.roles import ExecutionRole


@pytest.fixture
def provider() -> ProviderIdentity:
    return ProviderIdentity("stub", "Stub Provider", "stub.example")


@pytest.fixture
def model() -> ModelIdentity:
    return ModelIdentity(model_id="stub-model", family="stub")


@pytest.fixture
def route(provider: ProviderIdentity) -> InferenceRouteIdentity:
    return InferenceRouteIdentity(
        route_id="stub-direct",
        provider_id=provider.provider_id,
        route_type=RouteType.DIRECT,
        endpoint_key="https://stub.example/v1",
        failure_domain="stub.example",
    )


@pytest.fixture
def sample_request(provider: ProviderIdentity, model: ModelIdentity) -> ProviderRequest:
    return ProviderRequest(
        request_id="req-1",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        target_provider=provider,
        target_model=model,
    )


def test_request_requires_identity() -> None:
    with pytest.raises(ValueError):
        ProviderRequest(
            request_id="",
            execution_role=ExecutionRole.CODING,
            risk_level=RiskLevel.R2_NORMAL,
        )


def test_request_stores_all_core_fields(
    provider: ProviderIdentity, model: ModelIdentity, route: InferenceRouteIdentity
) -> None:
    request = ProviderRequest(
        request_id="req-1",
        execution_role=ExecutionRole.PLANNING,
        risk_level=RiskLevel.R3_HIGH,
        system_instructions=["Be concise"],
        messages=[Message(role=MessageRole.USER, content="Hello")],
        context_packets=[ContextPacket(kind="authority", source="roadmap.md")],
        tools=[ToolDefinition(name="read_file", description="Read a file")],
        tool_choice=ToolChoiceMode.REQUIRED,
        structured_output=StructuredOutputRequirement(schema={"type": "object"}, name="plan"),
        temperature=0.2,
        reasoning=ReasoningMode.EFFORT_HIGH,
        max_output_tokens=4096,
        max_total_tokens=8192,
        stop_sequences=["END"],
        stream=True,
        target_model=model,
        target_route=route,
        target_provider=provider,
        capability_requirements=[CapabilityRequirement("streaming", required=True)],
        lineage=TaskLineage(project_id="omniforge", run_id="run-1", task_id="task-1"),
        correlation_id="corr-1",
        cancellation_id="cancel-1",
        metadata={"key": "value"},
    )

    assert request.request_id == "req-1"
    assert request.execution_role is ExecutionRole.PLANNING
    assert request.risk_level is RiskLevel.R3_HIGH
    assert request.system_instructions == ["Be concise"]
    assert request.messages[0].role is MessageRole.USER
    assert request.context_packets[0].kind == "authority"
    assert request.tools[0].name == "read_file"
    assert request.tool_choice is ToolChoiceMode.REQUIRED
    assert request.structured_output is not None
    assert request.temperature == 0.2
    assert request.reasoning is ReasoningMode.EFFORT_HIGH
    assert request.max_output_tokens == 4096
    assert request.max_total_tokens == 8192
    assert request.stop_sequences == ["END"]
    assert request.stream is True
    assert request.target_model is model
    assert request.target_route is route
    assert request.target_provider is provider
    assert request.required_capabilities() == ["streaming"]
    assert request.lineage == TaskLineage(project_id="omniforge", run_id="run-1", task_id="task-1")
    assert request.correlation_id == "corr-1"
    assert request.cancellation_id == "cancel-1"
    assert request.metadata == {"key": "value"}


def test_required_capabilities_fail_eligibility() -> None:
    request = ProviderRequest(
        request_id="req-2",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        capability_requirements=[
            CapabilityRequirement("tool_use", required=True),
            CapabilityRequirement("structured_output", preferred=True),
        ],
    )
    assert request.requires_tools() is True
    assert request.requires_structured_output() is False
    assert request.required_capabilities() == ["tool_use"]
    assert request.preferred_capabilities() == ["structured_output"]


def test_structured_output_enforced_counts_as_required() -> None:
    request = ProviderRequest(
        request_id="req-3",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        structured_output=StructuredOutputRequirement(schema={"type": "object"}),
    )
    assert request.requires_structured_output() is True


def test_unenforced_structured_output_is_not_required() -> None:
    request = ProviderRequest(
        request_id="req-4",
        execution_role=ExecutionRole.CODING,
        risk_level=RiskLevel.R2_NORMAL,
        structured_output=StructuredOutputRequirement(
            schema={"type": "object"}, enforce_schema=False
        ),
    )
    assert request.requires_structured_output() is False
