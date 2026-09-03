---
description: How to land a PR - hatch envs, ruff/mypy, lazy-import discipline, JSON registries, test layout.
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

`-x` means a red run stops at the first failure with the rest of the suite
unexecuted, and its counts line is shaped exactly like a complete run's. So
`tests/conftest.py` registers the reporter in `tests/session_truncation.py`,
which states the size of the gap:

```
============= session truncated: 1926 of 4878 collected tests ran ==============
2952 collected tests never started, so the counts below are a floor, not a total.
```

A session that ran every test it collected prints nothing extra, so the section
appears only where it changes what the counts below it mean.

Coverage is collected through PEP 669 (`sys.monitoring`) rather than the default
C trace function - `core = "sysmon"` in `[tool.coverage.run]`. It reports the
same line coverage at roughly a tenth of the cost (measured +8.2% against
+76.6% over 17,102 tests), so `--no-cov` above is a smaller win than it used to
be. The setting is for line coverage only; enabling `branch` means re-measuring
that equivalence first.

## Rules

**Lazy imports** - heavy modules (`mujoco`, `lerobot`, `torch`, `zenoh`) must not load at top-level. Use PEP 562 `__getattr__`. Enforced by `tests/test_init.py`.

**Tests mirror source** - `tests/policies/test_groot.py` mirrors `strands_robots/policies/groot/`. Keep 1:1.

**No host paths** - `/Users/...` is CI-blocked. Use `tmp_path`, `~/.cache`, or env vars.

**JSON registries** - new robots and policies are JSON edits + tests. No hardcoded lookups in `.py` files.

**A dependency change and its relock are one commit** - editing `pyproject.toml` without running `uv lock` leaves the lock describing a manifest that no longer exists. `uv.lock` is one of the manifests GitHub's dependency graph parses, so a stale lock is a stale *security surface*, not just a stale install. Check it before pushing with `python scripts/check_lockfile_parity.py` (offline, no resolver) or `uv lock --check`.

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

- [Architecture](architecture.md) - module conventions.
- [API reference](api-reference.md) - public symbols.
