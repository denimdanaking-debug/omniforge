# OmniForge Architecture Baseline

OmniForge is structured around strict separation of concerns:

- `orchestration`: provider-agnostic workflow control
- `providers`: provider/model/route adapters and capability normalization
- `routing`: deterministic and learned model selection
- `policy`: risk, authority, independence, and project restrictions
- `planning`: immutable-authority planning and task decomposition
- `execution`: implementation and resumable task execution
- `validation`: deterministic checks that outrank model confidence
- `review`: independent review, arbitration, and repair evidence
- `integration`: Git safety, workspace lifecycle, integration, and authority advancement
- `telemetry`: explainable routing and complete task lineage
- `persistence`: versioned restart-safe state and learned statistics
- `webui`: operational visibility and controls

Provider-specific behavior must terminate at the provider abstraction boundary. Core orchestration must consume normalized contracts rather than branching on provider names.

The Workspace Manager owns managed task folders, branches, and worktrees. It may clean only resources proven `SAFE_TO_DELETE`; unresolved, unknown, or authoritative work must be preserved.
