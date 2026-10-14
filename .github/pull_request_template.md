## What and why

<!-- One paragraph: the change and the reason. Link the issue. -->

## Checklist

- [ ] `uv run pytest -n auto` passes; `uv run ruff check . && uv run ruff format --check .` is clean
- [ ] New or changed physics comes with checks in `src/qenvs/validation/<model>.py` (closed form, constraints, solver convergence, QuTiP cross-check) that fail on a realistic bug
- [ ] Every new check passes in float32 and float64 (`uv run qenvs validate`)
- [ ] A released task's reward, observation or defaults are unchanged, or its id is bumped (`-vN`)
- [ ] Docs updated (model page, validation table numbers measured, references verified)
- [ ] `CHANGELOG.md` entry under "Unreleased"
- [ ] Plain ASCII punctuation (AGENTS.md rule 8)
