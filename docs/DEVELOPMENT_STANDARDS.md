# OmniForge Development Standards

These standards govern all code, configuration, documentation, tests, and automation added to OmniForge unless a stricter project-specific rule applies.

## Core principles

1. **Authority first.** Roadmap and project-state semantics may not be silently changed by implementation work.
2. **Deterministic validation.** Every change must have reproducible local and CI validation commands.
3. **Provider neutrality.** Core orchestration must not accumulate provider-specific conditionals.
4. **Safe Git isolation.** Feature work is developed on non-authoritative branches and integrated through reviewable pull requests.
5. **Restart safety.** Persistent state changes must be schema-versioned and recoverable.
6. **Evidence over model opinion.** Tests, schemas, diffs, and explicit evidence determine acceptance.
7. **No secret leakage.** Credentials, tokens, and secret-bearing environment files must not enter Git history or diagnostics.

## Formatting and text

- UTF-8 text files.
- LF line endings in the repository.
- Final newline required.
- No trailing whitespace in newly created or ordinarily editable files.
- Hash-pinned authoritative artifacts may retain pre-existing formatting that would otherwise violate a style rule; changing those bytes requires an explicit authority revision and corresponding hash update.
- Markdown headings should be hierarchical and stable enough for machine parsing.
- JSON committed as project/config/state data must parse with the standard JSON grammar; comments are not allowed in JSON authority files.

## Code quality

As implementation languages are introduced, each language must add its canonical formatter, linter/static analysis, and test runner to the repository validation profile. Tooling is configuration-driven rather than embedded in provider adapters.

New production modules must include tests for meaningful behavior and negative/failure paths. Error handling must preserve the normalized OmniForge failure taxonomy rather than flattening failures into generic strings.

## Architectural boundaries

- `src/providers` implements provider and route adapters.
- `src/routing` selects among normalized candidates.
- `src/orchestration` coordinates workflows but does not directly encode provider-specific behavior.
- `src/persistence` owns durable runtime-state access and migrations.
- `src/validation` owns deterministic validation execution/evidence.
- `src/review` owns independent review and arbitration behavior.
- `src/integration` owns Git/integration and authority-advancement mechanics.
- `src/telemetry` owns structured measurements and audit events.

Cross-boundary dependencies must be explicit. Provider-specific `if/else` routing logic outside provider configuration/adapters is prohibited.

## Pull requests and commits

- `main` is authoritative and should receive reviewed/integrated work.
- `integration` is the configured integration branch for workflows that require it.
- Task branches must be narrowly scoped and attributable to one roadmap step or repair lineage where practical.
- Pull requests must state the roadmap step(s), validation evidence, and authority impact.
- Material repairs require fresh validation and review.
- Authority state advances only after successful integration.

## Deterministic validation commands

Bootstrap validation is intentionally dependency-light and uses Python standard library only.

Run locally from repository root:

```text
python scripts/validate_bootstrap.py --mode lint
python scripts/validate_bootstrap.py --mode build
python scripts/validate_bootstrap.py --mode test
python scripts/validate_bootstrap.py --mode all
```

The same commands run in GitHub Actions. As OmniForge gains a concrete application toolchain, these commands remain stable entry points or are replaced through a documented versioned validation profile without weakening checks.

## Security and dependencies

- Pin or constrain dependencies through the chosen ecosystem's standard lock/manifest mechanism.
- New dependencies require a clear product or engineering purpose.
- Secrets belong in environment/secret-management facilities, never normal config.
- Diagnostics must redact known credential fields and secret-bearing environment values.

## Definition of a passing change

A change is eligible for integration only when:

1. required deterministic validation passes;
2. required independent review passes or disputes are resolved by evidence-first arbitration;
3. no unresolved authority or security blocker remains;
4. the Git diff matches intended scope; and
5. integration can occur without unsafe authority conflict resolution.
