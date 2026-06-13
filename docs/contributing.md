---
description: How to land a PR — hatch envs, ruff/mypy, lazy-import discipline, JSON registries, test layout.
---

# Contributing

Repo: [`strands-labs/robots`](https://github.com/strands-labs/robots). Requires **Python ≥ 3.12**.

```bash
git clone https://github.com/strands-labs/robots
cd robots
uv pip install -e '.[all,dev]'
```

## Commands

```bash
hatch run test                       # full suite
hatch run test --no-cov tests/       # fast, no coverage
hatch run lint                       # ruff check + ruff format --check + mypy
hatch run format                     # ruff fix + format
mkdocs serve                         # docs at http://localhost:8000
mkdocs build --strict                # CI gate
```

CI runs `hatch run test -x --strict-markers`.

## Rules

**Lazy imports** — heavy modules (`mujoco`, `lerobot`, `torch`, `zenoh`) must not load at top-level. Use PEP 562 `__getattr__`. Enforced by `tests/test_init.py`.

**Tests mirror source** — `tests/policies/test_groot.py` mirrors `strands_robots/policies/groot/`. Keep 1:1.

**No host paths** — `/Users/...` is CI-blocked. Use `tmp_path`, `~/.cache`, or env vars.

**JSON registries** — new robots and policies are JSON edits + tests. No hardcoded lookups in `.py` files.

**Tool errors return, don't raise:**
```python
{"status": "error", "content": [{"text": "human-readable error"}]}
```

## PR workflow

Branch from `main` → write tests first → keep PR ≤ 300 lines → update docs → `hatch run lint && hatch run test` → open PR → squash on merge.

Releases: `hatch version` + GitHub release. Semver: minor for additive, patch for fixes, major for breaking.

## Where to ask

| Topic | Where |
|-------|-------|
| Bug | [Issues](https://github.com/strands-labs/robots/issues) |
| Feature | [Issues](https://github.com/strands-labs/robots/issues) (feature template) |
| How-to | [Discussions](https://github.com/strands-labs/robots/discussions) |
| Security | SECURITY.md |

## See also

- [Architecture](architecture.md) — module conventions.
- [API reference](api-reference.md) — public symbols.
