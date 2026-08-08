# OmniForge Workspace Lifecycle Policy

OmniForge owns the lifecycle of managed task folders, Git branches, worktrees, scratch space, and recovery artifacts within configured workspace roots.

## Managed states

Every managed workspace must be classified as one of:

- `ACTIVE`
- `SAFE_TO_DELETE`
- `PRESERVE`
- `RECOVERY_REQUIRED`
- `UNKNOWN_DO_NOT_TOUCH`

## Automatic creation

OmniForge may automatically create task folders, scratch directories, evidence folders, uniquely named task/recovery branches, and isolated Git worktrees when required by an authorized task.

## Automatic preservation

A workspace must not be deleted when it contains unresolved validation failure, review failure, merge conflict, interrupted execution, provider exhaustion mid-task, useful uncommitted changes, recovery evidence, or any unknown/unmanaged content.

## Automatic cleanup

The Workspace Janitor may remove only resources classified `SAFE_TO_DELETE` after successful integration and after confirming that no recovery, audit, or reference dependency remains.

## Protected resources

OmniForge must never automatically delete `main`, protected integration branches, authoritative roadmap/state files, unknown user directories, or anything outside configured workspace roots.

## Remote branch cleanup

Disposable remote task branches may be removed after successful integration and proof that no recovery/reference dependency remains.

## Auditability

Every create, preserve, cleanup, refusal-to-delete, and recovery decision must be recorded with task/run lineage.
