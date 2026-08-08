# OmniForge — Full Authoritative Roadmap v1.0

**Project:** OmniForge  
**Repository:** `omniforge`  
**Purpose:** A provider-agnostic, self-optimizing AI software factory that plans, implements, validates, reviews, repairs, integrates, and advances software roadmaps continuously while dynamically routing work across a heterogeneous fleet of AI models and providers.

---

## 0. Authority and Non-Negotiable Invariants

This roadmap is authoritative for OmniForge v1.0.

No planner, coder, reviewer, repair agent, router, provider adapter, dashboard component, or automation may reinterpret or silently amend authoritative project state.

Only work that has been:
1. implemented,
2. deterministically validated,
3. independently reviewed when required,
4. repaired if required,
5. safely integrated,
6. and atomically recorded in authoritative project state

may advance roadmap progress.

### Core invariants

- OmniForge is generic and configurable; it must not be Black-Ledger-specific or Autonomous-specific.
- Authority, planning state, task state, execution state, review state, provider state, router-learning state, and integration state remain distinct.
- Provider/model selection must never bypass deterministic validation, review requirements, Git safety, or authority gates.
- Model capability and provider availability are separate concepts.
- Model identity and inference route are separate concepts.
- Infrastructure failures must not be scored as model-quality failures.
- Prompt/context construction failures must not be scored as model-quality failures unless the model actually caused them.
- Provider/model routing decisions must be explainable through structured factors, not hidden chain-of-thought.
- Useful branches, worktrees, evidence, and recovery state must not be destructively discarded.
- Human escalation occurs only for genuine human-only, safety, credential, policy, or irreducibly ambiguous authority blockers.
- Temporary provider exhaustion, rate limits, cooldowns, malformed fallback output, or recoverable planning failures are not human blockers.
- Runtime state and learned routing statistics must persist safely across restart.
- OmniForge must remain usable when intelligent/dynamic routing is disabled.
- Existing or future projects may opt into stricter policies than the global defaults.

---

# Phase 0 — Repository Bootstrap and Engineering Baseline

## 0.1 Create the OmniForge repository
Create a new GitHub repository named `omniforge` and initialize the local repository.

### Acceptance criteria
- Repository exists independently of Autonomous.
- Default branch is `main`.
- README identifies OmniForge as a separate product.
- Autonomous remains untouched.

## 0.2 Establish repository layout
Create a modular top-level structure for orchestration, providers, routing, policy, planning, execution, validation, review, integration, telemetry, persistence, web UI, tests, docs, and schemas.

### Acceptance criteria
- No provider-specific business logic is embedded in core orchestration modules.
- The layout supports future provider adapters without core rewrites.

## 0.3 Establish development standards
Define formatting, linting, typing, test, security, dependency, documentation, and CI expectations.

### Acceptance criteria
- CI has deterministic build/test/lint commands.
- Local and CI validation commands are documented.
- Failure output is actionable.

## 0.4 Create configuration schema versioning
Implement a versioned configuration model from the beginning.

### Acceptance criteria
- Configuration files contain an explicit schema version.
- Unknown/incompatible schema versions fail safely.
- Migration hooks exist.

## 0.5 Create runtime-state schema versioning
Implement versioned runtime/persistence schemas independent of configuration schemas.

### Acceptance criteria
- Runtime state survives restart.
- State migrations are explicit and testable.
- Corrupt state fails closed with recoverable diagnostics.

## 0.6 Define authoritative project contract
Specify how OmniForge consumes project roadmaps, project state, source/integration branches, validation profiles, and project-specific policies.

### Acceptance criteria
- Authority source is immutable during a planning/execution cycle.
- Current roadmap position is deterministic.
- Advancement cannot happen from planner output alone.

---

# Phase 1 — Core Domain Model

## 1.1 Define provider identity
Create a first-class provider identity separate from models and routes.

Examples include Anthropic, OpenAI, Moonshot/Kimi, Alibaba/Qwen, DeepSeek, Google, xAI, Z.AI, MiniMax, Mistral, OpenRouter, local endpoints, and future enterprise gateways.

### Acceptance criteria
- Provider has stable identifier.
- Provider health/quota is independently tracked.
- Provider can expose multiple models and routes.

## 1.2 Define model identity
Create a first-class model identity.

### Acceptance criteria
- Model identity includes family, version/revision when detectable, capability metadata, and lifecycle status.
- Model reputation is not conflated with provider health.

## 1.3 Define inference route identity
Represent the transport/path used to reach a model.

Examples:
- direct provider API
- OpenRouter
- local OpenAI-compatible endpoint
- enterprise gateway
- future Bedrock/Azure/Vertex route

### Acceptance criteria
- Same model can exist behind multiple routes.
- Route-specific latency, cost, errors, and health can be measured independently.

## 1.4 Define model capability metadata
Create normalized capabilities.

Include at least:
- context size
- structured output
- tool use
- streaming
- reasoning support
- code-generation suitability
- multimodal capability
- local/cloud
- cost metadata
- concurrency/rate metadata
- supported roles

### Acceptance criteria
- Router can filter candidates by required capability.
- Unsupported features never rely on prompt assumptions.

## 1.5 Define execution roles
Formalize roles such as:
- planning
- architecture
- coding
- debugging
- repair
- review
- high-risk review
- arbitration
- context analysis
- integration analysis

### Acceptance criteria
- A model may have different empirical performance by role.
- Role is part of every routing request.

## 1.6 Define risk taxonomy
Implement:
- `R0_TRIVIAL`
- `R1_LOW`
- `R2_NORMAL`
- `R3_HIGH`
- `R4_CRITICAL_AUTHORITY`

### Acceptance criteria
- Every executable task receives a risk classification.
- Risk can increase during execution.
- Risk influences model eligibility, review count, experimentation, and context policy.

## 1.7 Define normalized task outcome taxonomy
Separate successful execution, deterministic validation failures, authority violations, model-output invalidity, provider failures, and system failures.

### Acceptance criteria
- Telemetry can distinguish quality from infrastructure.
- Learning subsystem never treats quota exhaustion as poor coding ability.

---

# Phase 2 — Generic Provider Adapter Layer

## 2.1 Define provider adapter interface
Create a stable provider contract for request submission, streaming, tool calls, structured output, cancellation, health, quota/reset information, and normalized errors.

## 2.2 Define normalized request contract
Create provider-neutral request representation for prompts, context packets, tools, structured schemas, temperature/reasoning controls where supported, token limits, and metadata.

## 2.3 Define normalized response contract
Normalize text, tool calls, usage, finish reason, latency, model/version, provider, route, and errors.

## 2.4 Define normalized provider error taxonomy
Include at least:
- `RATE_LIMITED`
- `QUOTA_EXHAUSTED`
- `AUTH_FAILURE`
- `CONTEXT_OVERFLOW`
- `PROVIDER_UNAVAILABLE`
- `TRANSIENT_TRANSPORT`
- `UNSUPPORTED_CAPABILITY`
- `INVALID_MODEL_OUTPUT`
- `TASK_FAILURE`
- `CANCELLED`

## 2.5 Implement provider health interface
Adapters report current health and recovery metadata where possible.

## 2.6 Implement provider quota interface
Track quota pressure, reset time, and rate-limit signals where available.

## 2.7 Implement adapter contract tests
All adapters must pass the same provider-neutral test suite.

## 2.8 Enforce no provider conditionals in orchestration
Add tests/linting or architectural boundaries preventing scattered `if provider == ...` logic.

---

# Phase 3 — Core Production Provider Fleet

## 3.1 Implement Anthropic adapter
Support Claude models through the generic provider contract.

### Initial status
`HIGH_RISK`

## 3.2 Implement OpenAI adapter
Support OpenAI/Codex models through the generic provider contract.

### Initial status
`HIGH_RISK`

## 3.3 Implement Kimi adapter
Support Kimi K3 and compatible Kimi coding models.

### Initial status
`HIGH_RISK`

### Required behavior
- Full eligibility for planning, coding, repair, review, arbitration, architecture, and context analysis.
- No mandatory shadow stage.

## 3.4 Implement Qwen adapter
Support Qwen3.8-Max and compatible Qwen coding-capable models.

### Initial status
`HIGH_RISK`

### Required behavior
- Full eligibility for planning, coding, repair, review, arbitration, architecture, and context analysis.
- No mandatory shadow stage.

## 3.5 Implement DeepSeek adapter
Support current production DeepSeek coding/reasoning models.

### Acceptance criteria
- Multiple DeepSeek models may coexist with distinct identities.
- Cost-efficient variants can compete normally in routing.

## 3.6 Implement Google Gemini adapter
Support current Gemini coding/reasoning models generically rather than hardcoding one permanent model name.

## 3.7 Implement xAI adapter
Support current Grok coding/agentic models.

## 3.8 Implement Z.AI / GLM adapter
Support current GLM coding/reasoning models generically.

## 3.9 Implement Cursor route adapter
Preserve Cursor as a genuine execution route where automation access is available.

## 3.10 Validate first-class fleet interoperability
Run provider-neutral smoke tests across all enabled core adapters.

### Acceptance criteria
- Each configured provider can complete at least one normalized planning/coding/review-style request.
- Capability metadata and normalized errors are verified.

---

# Phase 4 — Extended Provider and Gateway Support

## 4.1 Implement MiniMax adapter

## 4.2 Implement Mistral/Devstral adapter

## 4.3 Implement OpenRouter gateway
Treat OpenRouter as an inference route/gateway rather than a model family.

### Acceptance criteria
- Underlying model identity is preserved.
- Gateway failures do not corrupt the underlying model's reputation.
- Direct and OpenRouter routes may coexist.

## 4.4 Implement generic OpenAI-compatible endpoint adapter
Support arbitrary compatible endpoints through configuration.

## 4.5 Implement local endpoint support
Support at least configurable routes compatible with:
- Ollama
- llama.cpp
- vLLM
- LM Studio
- SGLang
- generic OpenAI-compatible servers

## 4.6 Add local-model capability probing
Discover or configure model capabilities without assuming parity with cloud providers.

## 4.7 Add enterprise-route abstraction
Prepare model/route architecture for future:
- AWS Bedrock
- Azure AI
- Google Vertex AI

### Acceptance criteria
- No enterprise-specific code is required in core routing.

---

# Phase 5 — Credentials, Security, and Provider Configuration

## 5.1 Separate secrets from normal config
API keys and credentials must never be committed in project configuration.

## 5.2 Add environment/secret resolution
Create documented mechanisms for resolving provider secrets.

## 5.3 Add provider-level enable/disable controls

## 5.4 Add model-level enable/disable controls

## 5.5 Add route-level enable/disable controls

## 5.6 Add manual model/provider pinning
Allow administrative forcing of a provider/model/route for diagnostics.

## 5.7 Add global dynamic-routing switch
Support:
- `legacy`
- `dynamic`

## 5.8 Add global exploration switch

## 5.9 Add project-level routing overrides
Projects may prohibit providers, require specific review independence, or impose stricter risk policies.

## 5.10 Validate secret redaction
Logs, dashboard, diagnostics bundles, and failure traces must not leak credentials.

---

# Phase 6 — Provider Health, Quota, and Recovery Engine

## 6.1 Implement provider health state machine
States include:
- healthy
- degraded
- rate_limited
- quota_exhausted
- cooling
- unavailable
- auth_failed
- disabled

## 6.2 Persist provider recovery state
Cooldown/reset information survives restart.

## 6.3 Implement scheduled provider rechecks

## 6.4 Prevent hot-loop retries

## 6.5 Separate model health from provider health

## 6.6 Detect shared failure domains
Models/routes sharing provider/gateway/quota domains must not be treated as fully independent fallbacks.

## 6.7 Reintroduce recovered providers automatically
Recovery returns candidates to routing rather than blindly restoring them to first place.

## 6.8 Implement reserve-capacity policy
Preserve premium/high-reliability provider capacity for critical repair, review, arbitration, and integration events.

## 6.9 Implement quota-aware load balancing

## 6.10 Validate outage survival
Simulate full and partial provider outages.

### Acceptance criteria
- Active roadmap position is preserved.
- Work either routes elsewhere or enters a persisted wait.
- Automatic continuation occurs when eligible capacity returns.

---

# Phase 7 — Context Construction Engine

## 7.1 Define context packet schema
Context must identify:
- authority
- acceptance criteria
- relevant files
- current diff
- test evidence
- historical findings
- task metadata
- exclusions
- provenance

## 7.2 Implement targeted retrieval strategy

## 7.3 Implement large-context strategy

## 7.4 Implement hierarchical-summary strategy

## 7.5 Implement hybrid context strategy

## 7.6 Implement arbitration evidence packet
Arbitrators receive exact disputed findings and supporting primary evidence.

## 7.7 Track context provenance
Every supplied summary or extracted fact links back to authoritative/raw evidence.

## 7.8 Prevent lossy authority replacement
Summaries may assist navigation but cannot silently replace authoritative source.

## 7.9 Record context strategy in telemetry

## 7.10 Add context-quality outcome hooks
Learning can later evaluate which strategy worked best for which model/task class.

---

# Phase 8 — Deterministic Dynamic Router v1

## 8.1 Define candidate eligibility pipeline
Filter by:
- enabled state
- role capability
- risk eligibility
- provider health
- route health
- context capacity
- project restrictions
- independence rules

## 8.2 Define transparent scoring factors
Include at minimum:
- expected success
- role fit
- risk fit
- empirical reliability
- context suitability
- recent performance
- provider health
- quota pressure
- cost
- latency
- affinity
- diversity/reserve-capacity effects

## 8.3 Implement deterministic scoring
With exploration disabled and identical state, the same candidates must produce the same winner.

## 8.4 Implement seeded priors
Initial model/provider strengths may seed routing before sufficient empirical data exists.

## 8.5 Prevent permanent privileged models
No model remains globally preferred solely because of a hardcoded brand ranking.

## 8.6 Add emergency deterministic fallback order
If scoring cannot operate safely, role-specific fallback lists provide graceful degradation.

## 8.7 Record full routing decision metadata
Record candidates, exclusions, scores/factors, selected model, route, policy overrides, and runner-up.

## 8.8 Implement explainable routing summaries
Provide structured user-facing reasons without exposing hidden model reasoning.

## 8.9 Add cost-to-accepted-task estimate

## 8.10 Validate router determinism and policy compliance

---

# Phase 9 — Risk Engine

## 9.1 Implement initial risk classifier
Use deterministic task metadata and repository impact.

## 9.2 Detect authority-sensitive changes
Touching roadmap/project-state/integration-policy files escalates risk.

## 9.3 Detect security-sensitive changes

## 9.4 Detect broad architectural changes

## 9.5 Implement runtime risk escalation
Escalate based on:
- repeated test failures
- model disagreement
- unexpected touched files
- merge conflicts
- repair loops
- authority violations
- integration anomalies

## 9.6 Connect risk to model eligibility

## 9.7 Connect risk to review count

## 9.8 Connect risk to experimentation eligibility

## 9.9 Connect risk to context depth

## 9.10 Add project-specific risk overrides

---

# Phase 10 — Failure-Type-Aware Retry and Escalation

## 10.1 Build failure classifier

## 10.2 Handle transient provider failures
Retry or reroute without model-quality penalty.

## 10.3 Handle quota exhaustion
Fallback or wait based on eligible alternatives.

## 10.4 Handle invalid structured output
Allow bounded constrained retry before switching models.

## 10.5 Handle invalid planning output
Preserve rejection evidence and route to another planner or provider recovery path.

## 10.6 Handle deterministic implementation failure
Provide actual validation evidence to repair routing.

## 10.7 Handle conceptual/repeated implementation failure
Prefer cross-model/provider escalation.

## 10.8 Handle context overflow
Rebuild context or choose a context-capable model.

## 10.9 Handle authority violations
Apply strong task-class penalties and escalate risk.

## 10.10 Prevent retry storms
Repeated identical failures cannot bounce endlessly among providers.

---

# Phase 11 — Empirical Model Intelligence

## 11.1 Create performance-event ledger
Store immutable task outcome events.

## 11.2 Track success by role

## 11.3 Track first-pass acceptance

## 11.4 Track valid-plan rate

## 11.5 Track deterministic validation failures

## 11.6 Track repair effectiveness

## 11.7 Track reviewer precision
Measure supported, unsupported, stale, duplicate, and mis-severity findings.

## 11.8 Track reviewer false negatives
Later discovered material defects reduce review reliability.

## 11.9 Track authority-adherence rate

## 11.10 Track latency

## 11.11 Track token consumption

## 11.12 Track direct call cost

## 11.13 Track total cost to accepted integration
Include planner, implementation, retries, reviews, repairs, and arbitration.

## 11.14 Track time to accepted integration

## 11.15 Track context strategy performance

## 11.16 Track task difficulty/risk performance

## 11.17 Track language/framework performance

## 11.18 Track project-specific performance

## 11.19 Track provider-route performance separately from model quality

## 11.20 Persist statistics across restart

---

# Phase 12 — Hierarchical Learning and Confidence

## 12.1 Implement global model profile

## 12.2 Implement language/domain profile

## 12.3 Implement framework/toolchain profile

## 12.4 Implement task-role profile

## 12.5 Implement risk-level profile

## 12.6 Implement project-specific profile

## 12.7 Blend hierarchical evidence
Use broader priors when local sample size is small.

## 12.8 Add confidence-aware estimates
Two successes out of two must not outweigh hundreds of reliable observations.

## 12.9 Add minimum sample safeguards

## 12.10 Add performance aging/decay

## 12.11 Add model-version awareness

## 12.12 Add cold-start inheritance
New model revisions inherit reduced-confidence family/provider priors.

## 12.13 Add administrative reputation reset/down-weight

## 12.14 Ensure statistics are not ordinary editable config

---

# Phase 13 — Dynamic Value Optimization

## 13.1 Implement expected-success estimation

## 13.2 Implement quality/risk suitability estimate

## 13.3 Implement quota-impact estimate

## 13.4 Implement cost estimate

## 13.5 Implement latency estimate

## 13.6 Implement provider-diversity bonus

## 13.7 Implement reserve-capacity penalty

## 13.8 Implement affinity bonus
Recent subsystem understanding may improve ranking without creating ownership.

## 13.9 Implement total value score

## 13.10 Validate that risk can override savings
Critical work must not choose a materially weaker model merely because it is cheaper.

## 13.11 Validate that savings can override tiny quality differences
Routine work should not burn premium quota for negligible expected benefit.

---

# Phase 14 — Model Lifecycle, Promotion, Demotion, and Experimentation

## 14.1 Implement lifecycle states
- `SHADOW`
- `LOW_RISK`
- `NORMAL`
- `HIGH_RISK`
- `DISABLED`

## 14.2 Configure first-class trusted models
Initial Kimi K3 and Qwen3.8-Max status:
`HIGH_RISK`

Claude/OpenAI trusted production models:
`HIGH_RISK`

## 14.3 Support configurable starting status for future models

## 14.4 Implement evidence-based automatic demotion

## 14.5 Implement evidence-based recovery/promotion

## 14.6 Implement exploration budget
Default experimentation should be small and configurable.

## 14.7 Restrict exploration to eligible low-risk tasks

## 14.8 Implement shadow planning/review experiments

## 14.9 Prevent shadow outputs from modifying authority or Git state

## 14.10 Log all randomized exploration decisions explicitly

## 14.11 Add exploration kill switch

---

# Phase 15 — Planning Engine

## 15.1 Implement immutable authority snapshot for planning

## 15.2 Implement hierarchical planning
Support:
- phase plan
- step plan
- task graph

## 15.3 Version planning artifacts

## 15.4 Validate plan authority references

## 15.5 Reject future-step leakage

## 15.6 Reject premature project-state advancement

## 15.7 Reject forbidden scope expansion

## 15.8 Route planning dynamically

## 15.9 Persist rejected-plan evidence

## 15.10 Support recoverable planner waiting

## 15.11 Re-evaluate provider/model routing on every planning dispatch

## 15.12 Preserve exact planning position across restart

---

# Phase 16 — Task Graph and Execution Engine

## 16.1 Define task graph schema

## 16.2 Encode dependencies and acceptance criteria

## 16.3 Support isolated worktree execution

## 16.4 Preserve coherent task ownership when beneficial

## 16.5 Allow router to select different models for different tasks

## 16.6 Capture implementation evidence

## 16.7 Detect unexpected changed files

## 16.8 Prevent unauthorized authority-file changes

## 16.9 Persist interrupted task state

## 16.10 Support safe resume

## 16.11 Prepare for future multi-provider concurrency
Do not enable unsafe parallel Git mutation yet.

---

# Phase 17 — Deterministic Validation Engine

## 17.1 Define project validation profiles

## 17.2 Run build/test/lint/static checks independently of model opinion

## 17.3 Capture machine-readable validation evidence

## 17.4 Distinguish flaky/transient validation failures

## 17.5 Prevent validation bypass by high model score

## 17.6 Support project-specific validation gates

## 17.7 Persist validation evidence for review/arbitration

## 17.8 Require successful mandated validation before integration

---

# Phase 18 — Independent Review Engine

## 18.1 Enforce coder/reviewer separation
Coder cannot be its own final independent reviewer.

## 18.2 Implement risk-based review count
- low/normal risk: one independent reviewer
- high/critical risk: two independent reviewers

## 18.3 Enforce cross-provider independence for high-risk work
At least one reviewer must use a distinct provider/failure domain whenever possible.

## 18.4 Build review evidence packet

## 18.5 Normalize review findings

## 18.6 Require evidence references for blocking findings

## 18.7 Detect duplicate/stale findings

## 18.8 Feed reviewer accuracy into empirical profiles

## 18.9 Support review-provider waiting/fallback

## 18.10 Persist review state across restart

---

# Phase 19 — Evidence-First Arbitration

## 19.1 Detect material reviewer disagreement

## 19.2 Route arbitration to an independent eligible model

## 19.3 Build arbitration packet
Include:
- authority
- acceptance criteria
- exact diff
- deterministic validation
- disputed findings
- relevant source
- prior repair evidence

## 19.4 Classify each disputed finding
- supported
- unsupported
- stale
- unresolved

## 19.5 Prevent majority-vote shortcuts

## 19.6 Prevent prestige-based shortcuts
Highest-ranked reviewer does not automatically win.

## 19.7 Escalate unresolved material risk

## 19.8 Add `ARBITRATING` workflow state

## 19.9 Feed arbitration outcomes back into reviewer metrics

---

# Phase 20 — Repair Engine

## 20.1 Build repair packets from validated findings

## 20.2 Route repair dynamically

## 20.3 Prefer cross-model repair after conceptual coding failure

## 20.4 Permit same-model repair when failure type supports it

## 20.5 Re-run deterministic validation after repair

## 20.6 Require genuinely fresh review after material repair

## 20.7 Reconcile stale findings

## 20.8 Prevent no-change repair loops

## 20.9 Persist bounded repair state

## 20.10 Escalate risk after repeated repair cycles

---

# Phase 21 — Git, Integration, and Authority Advancement

## 21.1 Define source/integration/task branch policy

## 21.2 Use isolated worktrees

## 21.3 Detect branch divergence before integration

## 21.4 Preserve useful branches before reconciliation

## 21.5 Treat project-state files as protected authority

## 21.6 Never blindly choose ours/theirs for authority conflicts

## 21.7 Require integration verification

## 21.8 Support PR-based integration

## 21.9 Support configurable auto-merge policy

## 21.10 Atomically update authoritative project state only after successful integration

## 21.11 Ensure integrated validated work is the sole source of progress

## 21.12 Persist integration recovery state

## 21.13 Detect and recover from interrupted integration


## 21.14 Implement first-class Workspace Manager
Own repository task-workspace lifecycle from branch/worktree creation through execution, validation, integration, preservation, and cleanup.

## 21.15 Implement safe task-folder creation
Automatically create task, scratch, evidence, and temporary directories only inside configured workspace roots.

## 21.16 Implement safe worktree creation
Create isolated Git worktrees for tasks without requiring human directory setup.

## 21.17 Implement safe task-branch creation
Create uniquely named task/recovery branches with deterministic lineage metadata.

## 21.18 Classify workspace cleanup state
Every managed workspace must be classified as:
- `ACTIVE`
- `SAFE_TO_DELETE`
- `PRESERVE`
- `RECOVERY_REQUIRED`
- `UNKNOWN_DO_NOT_TOUCH`

## 21.19 Preserve unresolved work automatically
Validation failure, review failure, merge conflict, interrupted execution, provider exhaustion mid-task, useful uncommitted changes, or recovery evidence must prevent automatic cleanup.

## 21.20 Implement Workspace Janitor
Periodically identify stale managed branches, worktrees, scratch directories, and artifacts and clean only those explicitly classified `SAFE_TO_DELETE`.

## 21.21 Protect authoritative and unknown workspaces
Never automatically delete `main`, protected integration branches, authority files, unknown/unmanaged user directories, or paths outside configured workspace roots.

## 21.22 Implement remote branch cleanup
Delete disposable remote task branches only after successful integration and proof that no recovery/reference dependency remains.

## 21.23 Audit all workspace lifecycle actions
Creation, preservation, cleanup, refusal-to-delete, and recovery decisions must be structured, logged, and attributable to a task/run.

---

# Phase 22 — Continuous Roadmap Controller

## 22.1 Implement continuous roadmap loop

## 22.2 Separate controller liveness from active-task execution

## 22.3 Persist current roadmap position

## 22.4 Support `WAITING_FOR_PROVIDER`

## 22.5 Support `WAITING_FOR_RETRY`

## 22.6 Support `REVIEWING`

## 22.7 Support `ARBITRATING`

## 22.8 Support `REPAIRING`

## 22.9 Support `VALIDATING`

## 22.10 Support `INTEGRATING`

## 22.11 Support `BLOCKED`

## 22.12 Support `COMPLETE`

## 22.13 Schedule exact next wakeup

## 22.14 Resume automatically after provider recovery

## 22.15 Resume automatically after process restart

## 22.16 Stop only for genuine terminal/human-only blockers

---

# Phase 23 — Router Simulation and Historical Backtesting

## 23.1 Store replayable routing inputs

## 23.2 Build router dry-run mode

## 23.3 Build historical replay engine

## 23.4 Compare legacy/static selection with dynamic routing

## 23.5 Calculate counterfactual cost/latency estimates where possible

## 23.6 Detect policy regressions before activation

## 23.7 Support deterministic simulation with exploration disabled

## 23.8 Produce routing-change reports

---

# Phase 24 — Benchmark and Evaluation Suite

## 24.1 Create provider-neutral planning benchmark

## 24.2 Create coding benchmark

## 24.3 Create repair benchmark

## 24.4 Create review benchmark

## 24.5 Create authority-adherence benchmark

## 24.6 Create structured-output benchmark

## 24.7 Create long-context benchmark

## 24.8 Create arbitration benchmark

## 24.9 Use benchmark outcomes only as priors

## 24.10 Ensure production evidence eventually dominates benchmark priors

---

# Phase 25 — Observability and Audit

## 25.1 Implement structured event log

## 25.2 Implement routing-decision audit log

## 25.3 Implement provider-health history

## 25.4 Implement model-performance history

## 25.5 Implement task lineage
A user can trace roadmap step → plan → tasks → model calls → validation → review → repair → integration → authority advancement.

## 25.6 Add correlation/run IDs

## 25.7 Redact secrets and sensitive provider data

## 25.8 Build diagnostics bundle

## 25.9 Make diagnostics restart-safe

## 25.10 Ensure audit records do not store hidden chain-of-thought

---

# Phase 26 — Web Dashboard: Fleet and Routing

## 26.1 Build Model Fleet panel

## 26.2 Show provider status
- healthy
- degraded
- exhausted
- cooling
- unavailable
- next retry/reset

## 26.3 Show model lifecycle status
- shadow
- low-risk
- normal
- high-risk
- disabled

## 26.4 Show route status separately

## 26.5 Show current task routing
Include:
- task
- risk
- role
- selected model
- selected route
- runner-up
- structured selection reason

## 26.6 Show quota pressure

## 26.7 Show reserve capacity

## 26.8 Show next provider recovery check

---

# Phase 27 — Web Dashboard: Workflow State

## 27.1 Show continuous-controller heartbeat

## 27.2 Show active task separately from controller state

## 27.3 Show explicit workflow states
- RUNNING / PLANNING
- RUNNING / EXECUTING
- RUNNING / WAITING FOR PROVIDER
- RUNNING / VALIDATING
- RUNNING / REVIEWING
- RUNNING / ARBITRATING
- RUNNING / REPAIRING
- RUNNING / INTEGRATING
- STOPPED
- BLOCKED
- COMPLETE

## 27.4 Show next automatic action

## 27.5 Show next-wakeup countdown

## 27.6 Show current roadmap phase/step/task

## 27.7 Show blocking reason with evidence

---

# Phase 28 — Web Dashboard: Empirical Performance

## 28.1 Show task attempts by model

## 28.2 Show accepted integrations

## 28.3 Show first-pass success rate

## 28.4 Show repair rate

## 28.5 Show plan-validity rate

## 28.6 Show reviewer accuracy

## 28.7 Show authority-violation rate

## 28.8 Show average latency

## 28.9 Show token usage

## 28.10 Show cost per accepted task

## 28.11 Show total cost to accepted integration

## 28.12 Show performance by role

## 28.13 Show performance by language/framework

## 28.14 Show performance by project

## 28.15 Avoid presenting one opaque composite score as the only metric

---

# Phase 29 — Administrative Controls

## 29.1 Enable/disable providers live where safe

## 29.2 Enable/disable models live where safe

## 29.3 Enable/disable routes live where safe

## 29.4 Pin model/provider for diagnostics

## 29.5 Disable exploration

## 29.6 Reset/down-weight model reputation

## 29.7 Inspect routing policy

## 29.8 Inspect project overrides

## 29.9 Export empirical stats

## 29.10 Protect dangerous controls with confirmation/audit

---

# Phase 30 — Backward Compatibility and Project Migration

## 30.1 Define legacy-compatible project config

## 30.2 Support static/legacy routing mode

## 30.3 Support dynamic-routing opt-in

## 30.4 Provide configuration migration tooling

## 30.5 Validate existing project definitions

## 30.6 Ensure dynamic routing does not require roadmap-format changes

## 30.7 Provide project-specific migration report

## 30.8 Preserve authority semantics during migration

---

# Phase 31 — Safety, Integrity, and Chaos Testing

## 31.1 Simulate provider outage

## 31.2 Simulate all premium providers exhausted

## 31.3 Simulate shared-gateway outage

## 31.4 Simulate context overflow

## 31.5 Simulate malformed planner output

## 31.6 Simulate planner authority violation

## 31.7 Simulate coder unauthorized-file modification

## 31.8 Simulate reviewer false-positive storm

## 31.9 Simulate reviewer disagreement

## 31.10 Simulate repair no-change loop

## 31.11 Simulate integration-branch divergence

## 31.12 Simulate protected project-state conflict

## 31.13 Simulate process crash during planning

## 31.14 Simulate process crash during coding

## 31.15 Simulate process crash during review

## 31.16 Simulate process crash during integration

## 31.17 Verify restart recovery in every case

## 31.18 Verify no false authoritative advancement

---

# Phase 32 — Dynamic Routing Production Rollout

## 32.1 Run router in dry-run mode against real work

## 32.2 Compare selected candidates with expected expert choices

## 32.3 Enable dynamic routing for R0/R1 work

## 32.4 Validate cost, success, and latency outcomes

## 32.5 Enable R2 work

## 32.6 Validate cross-provider review behavior

## 32.7 Enable R3 work

## 32.8 Validate two-reviewer and arbitration behavior

## 32.9 Enable R4 authority-sensitive work

## 32.10 Confirm deterministic validation and integration gates remain mandatory

## 32.11 Enable controlled exploration for eligible low-risk work

## 32.12 Validate automatic demotion/promotion

---

# Phase 33 — Full-Fleet Production Validation

## 33.1 Validate Claude production routing

## 33.2 Validate OpenAI/Codex production routing

## 33.3 Validate Kimi K3 production routing

## 33.4 Validate Qwen3.8-Max production routing

## 33.5 Validate DeepSeek production routing

## 33.6 Validate Gemini production routing

## 33.7 Validate Grok production routing

## 33.8 Validate GLM production routing

## 33.9 Validate Cursor route behavior

## 33.10 Validate MiniMax where configured

## 33.11 Validate Mistral/Devstral where configured

## 33.12 Validate OpenRouter route behavior

## 33.13 Validate local OpenAI-compatible route

## 33.14 Validate provider fallback without progress loss

## 33.15 Validate preferred-candidate re-entry after recovery

## 33.16 Validate route-vs-model reputation separation

## 33.17 Validate cross-provider independence

---

# Phase 34 — End-to-End Unattended Software Factory Validation

## 34.1 Select a production-grade test project

## 34.2 Load authoritative roadmap

## 34.3 Run planning unattended

## 34.4 Run task decomposition unattended

## 34.5 Run multi-model coding unattended

## 34.6 Run deterministic validation unattended

## 34.7 Run independent review unattended

## 34.8 Trigger at least one repair cycle

## 34.9 Trigger at least one reviewer disagreement/arbitration cycle

## 34.10 Trigger at least one provider exhaustion/wait/fallback cycle

## 34.11 Trigger at least one process restart

## 34.12 Trigger at least one integration recovery condition

## 34.13 Verify exact roadmap-position preservation

## 34.14 Verify only integrated validated work advances authority

## 34.15 Verify automatic continuation through multiple roadmap steps

## 34.16 Verify dashboard accurately reflects every state

## 34.17 Verify complete audit lineage for all advanced steps

---

# Phase 35 — v1.0 Release Gate

## 35.1 Run full automated test suite

## 35.2 Run provider contract suite

## 35.3 Run router determinism suite

## 35.4 Run risk-policy suite

## 35.5 Run authority-integrity suite

## 35.6 Run restart/recovery suite

## 35.7 Run Git/integration safety suite

## 35.8 Run dashboard/observability suite

## 35.9 Run security/secret-redaction suite

## 35.10 Run migration/backward-compatibility suite

## 35.11 Complete independent architecture review

## 35.12 Complete independent security/integrity review

## 35.13 Resolve all blocking findings

## 35.14 Freeze v1.0 schemas

## 35.15 Publish operator documentation

## 35.16 Publish provider-adapter documentation

## 35.17 Publish model onboarding documentation

## 35.18 Publish project onboarding documentation

## 35.19 Tag OmniForge v1.0

## 35.20 Declare production readiness only after successful unattended validation

---

# Post-v1 Backlog — Explicitly Deferred

The following are architecturally anticipated but are not required to declare v1.0 complete:

- contextual-bandit or ML-based router
- full parallel multi-provider task execution
- autonomous benchmark generation
- enterprise Bedrock implementation
- Azure AI implementation
- Vertex AI enterprise implementation
- distributed multi-host execution
- remote worker pools
- organization/team tenancy
- billing/chargeback
- public marketplace for provider adapters
- fully autonomous model discovery from arbitrary public endpoints

These items must not expand v1.0 scope unless the authoritative roadmap is explicitly revised.

---

# OmniForge v1.0 Definition of Done

OmniForge v1.0 is complete only when all authoritative v1.0 steps are satisfied and a real project demonstrates the following unattended loop:

`authoritative roadmap`
→ `immutable authority snapshot`
→ `phase/step/task planning`
→ `risk classification`
→ `dynamic model/route selection`
→ `coding`
→ `deterministic validation`
→ `independent cross-provider review`
→ `evidence-first arbitration when needed`
→ `repair`
→ `safe Git integration`
→ `atomic authoritative-state advancement`
→ `next roadmap step`
→ `repeat unattended`

The system must remain operational through provider quota exhaustion, model failure, route failure, malformed model output, process restart, and recoverable Git/integration conditions without losing authoritative position or falsely advancing progress.

Kimi K3 and Qwen3.8-Max are first-class production models from initial provider rollout and are not required to complete a shadow/probation period before normal or high-risk eligibility.

No model, regardless of historical performance, may bypass deterministic validation, required independent review, protected authority handling, or integration gates.
