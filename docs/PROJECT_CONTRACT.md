# OmniForge Authoritative Project Contract

A project contract tells OmniForge where authority lives and which integration/validation rules apply to that project. It is configuration, not progress state.

## Required fields

- `schema_version`: contract schema version.
- `project_id`: stable project identity.
- `roadmap_file`: repository-relative authoritative roadmap path.
- `project_state_file`: repository-relative authoritative project-state path.
- `source_branch`: branch from which task work is derived.
- `integration_branch`: branch used for configured integration workflows.
- `validation_profile`: ordered deterministic commands required by the project.
- `policy`: optional project-specific restrictions or stricter rules.

Example:

```json
{
  "schema_version": "1.0.0",
  "project_id": "example-project",
  "roadmap_file": "docs/ROADMAP.md",
  "project_state_file": "docs/PROJECT_STATE.json",
  "source_branch": "main",
  "integration_branch": "integration",
  "validation_profile": [
    "python -m unittest discover -s tests -p 'test_*.py' -v"
  ],
  "policy": {
    "require_independent_review": true
  }
}
```

## Authority snapshot

At the beginning of a planning/execution cycle, OmniForge reads the roadmap and project-state bytes and records immutable SHA-256 fingerprints together with the deterministic current roadmap position. That snapshot remains the cycle's authority reference.

Before authority-sensitive actions, OmniForge compares the current authority files with the cycle snapshot. A changed roadmap or project-state file invalidates the cycle rather than being silently accepted.

## Deterministic roadmap position

The current position comes from the configured authoritative project-state file. OmniForge verifies that the referenced step exists in the configured roadmap. A planner is not allowed to invent or select an unrelated future step as the new authoritative position.

## Advancement gate

Planner output is never completion evidence by itself. Authority may advance only when all required evidence is present:

1. implementation completed;
2. deterministic validation passed;
3. required independent review was satisfied;
4. work was safely integrated.

Project policy may add stricter gates but may not weaken these v1.0 authority requirements.
