# Cross-Repository Administration

Global repository registry, cross-domain status, and the master calendar are maintained in `krahd/tom-work-admin`.

This repository remains canonical for **Demokritos: Interactive, Community-Centred, Self-Improving Shader Generation using Large Language Models** as a project: implementation, project-specific technical state, shader-generation code, browser/WebGL behaviour, and repository-local evidence.

Paper manuscripts and publication strategy belong in `krahd/research/academic-writing/`. Submission-specific packages belong in `krahd/professional-opportunities`; grant/funding packages belong in `krahd/grant-applications`.

The central registry currently records the project's lifecycle as `unknown` unless and until direct evidence establishes a current state. Repository existence or historical grant support must not be used to infer that the project is currently active.

## Mandatory synchronisation rule

`krahd/tom-work-admin` **must be kept current** whenever work here materially changes the project's administratively meaningful state. Updating the administration repository is part of completing the change, not optional later cleanup.

Update this repository first for substantive project changes, then update `krahd/tom-work-admin` in the same work session when any of the following changes:

- project lifecycle state, working title, scope, or major artistic/research direction;
- release/version, deployment, public/private visibility, implementation milestone, or major validation state;
- relationship to a manuscript, submission, grant, collaborator, repository, dataset, compute resource, or other cross-domain dependency;
- submission/publication/award outcome where it materially affects global project status or next actions;
- deadline, event, exhibition, presentation, freeze date, or other material cross-domain date;
- current next action or major research/production gate.

A publication or submission outcome does not redefine the project's underlying status unless the project itself changes.

## Ownership boundary

Keep substantive implementation, tests, project-specific evidence, and technical state here. `tom-work-admin` stores only the concise cross-repository identity/status/relationship view and must point back to canonical project sources rather than duplicate them.

## Completion check

Before considering a material project-state change complete, verify that:

1. this repository reflects the substantive change;
2. `krahd/tom-work-admin` reflects any resulting global status, date, relationship, or next-action change;
3. related domain repositories are updated when the change affects manuscripts, submissions, or grants;
4. no stale cross-domain status or date remains in `tom-work-admin`.
