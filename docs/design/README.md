# Designs

Short design documents for changes to the public API, to physics or to a released task. A
design is written before the code, accepted by a maintainer, and then implemented as written;
deviations go under "Implementation notes" with their reason. See AGENTS.md, "Before you write
code".

Template (`docs/design/<slug>.md`):

```markdown
# <Title>

Status: Proposed | Accepted | Implemented | Superseded
Owner: @github-handle
Issue: #<number>

## Problem
## Decision
API, physics and the validation gates that must pass (closed forms, constraints, solver
convergence, QuTiP cross-check, mutation-tested where it matters).
## Out of scope
## Implementation notes
Deviations from the decision above, each with its reason.
```

Earlier maintainer notes in Korean live in `docs/superpowers/` and are not part of the site.
