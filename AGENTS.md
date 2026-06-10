# AGENTS.md - strands-labs/robots

## Overview

`strands-robots` is a robot control library for [Strands Agents](https://strandsagents.com). It provides policy inference, teleoperation, calibration, and simulation tools for physical robots.

## Project Dashboard

**Board**: https://github.com/orgs/strands-labs/projects/2
**Project ID**: `PVT_kwDOD151Fs4BSRJP`

> **RULE**: ALWAYS use the project board to track work. When creating follow-up items,
> create GitHub issues and add them to this board with Status + Priority set.
> Never track work only in local markdown - the board is the source of truth.

## Repository Structure

```
strands_robots/
├── policies/              # Policy providers (pluggable via registry)
│   ├── base.py            # Abstract Policy base class
│   ├── factory.py         # create_policy() factory + registry
│   ├── mock.py            # MockPolicy for testing
│   ├── groot/             # NVIDIA GR00T N1.5/N1.6/N1.7 inference
│   │   ├── policy.py      # Gr00tPolicy (ZMQ + HTTP modes)
│   │   ├── client.py      # Gr00tInferenceClient
│   │   ├── data_config.py # Gr00tDataConfig + ModalityConfig
│   │   └── data_configs.json  # 25 robot embodiment configs
│   └── lerobot_local/     # HuggingFace LeRobot direct inference
│       ├── policy.py      # LerobotLocalPolicy (RTC support)
│       ├── processor.py   # ProcessorBridge (pre/post pipelines)
│       └── resolution.py  # Policy class resolution (v0.4/v0.5)
├── registry/              # JSON registry for policy discovery
├── tools/                 # Strands @tool functions
│   ├── gr00t_inference.py # GR00T inference tool
│   ├── lerobot_calibrate.py
│   ├── lerobot_camera.py
│   ├── lerobot_teleoperate.py
│   ├── pose_tool.py
│   └── serial_tool.py
├── robot.py               # Core Robot class
└── utils.py               # Shared utilities (require_optional, etc.)

tests/                     # Unit tests (run with: hatch run test)
tests_integ/               # Integration tests (run with: hatch run test-integ)
```

## Development

```bash
# Install with all optional deps
pip install -e ".[all,dev]"

# Run tests
hatch run test              # unit tests
hatch run test-integ        # integration tests (needs GPU + model weights)

# Lint & format
hatch run lint              # ruff check, ruff format --check, mypy
hatch run format            # ruff check --fix, ruff format
```

> **Note**: Hatch uses `uv` as installer (`installer = "uv"` in pyproject.toml) for faster
> environment creation. No manual uv install needed - hatch handles it.

## Key Conventions

1. **Python 3.12+** - `requires-python = ">=3.12"` (LeRobot >=0.5.0 requires 3.12)
2. **Dependency bounds** - `>=1.0` deps: cap major. `<1.0` deps: cap minor. E.g. `lerobot>=0.5.0,<0.6.0`
3. **`__init__.py` must be thin** - exports only, no logic
4. **Imports at file top** - unless lazy-loading heavy deps with documented reason
5. **Raise on fatal errors** - never warn-and-continue if the system will behave unexpectedly
6. **No silent defaults on error** - returning zero-valued actions on failure is forbidden
7. **Use `require_optional()`** - from `strands_robots/utils.py` for all optional deps
8. **Integration tests required** - each policy needs `tests_integ/` tests with real inference
9. **Test behavior, not implementation** - assert on outputs, not internal state
10. **No dead code** - if it's not called and not part of base class, delete it

## PR Workflow

1. Create feature branch from `main`
2. Make changes, run `hatch run format && hatch run lint && hatch run test`
3. All tests must pass, lint must be clean
4. Open PR from your fork, address all review comments
5. Track follow-up items as issues on the [project board](https://github.com/orgs/strands-labs/projects/2)
6. Squash merge into `main`


## Registry conventions (strands_robots/registry/robots.json)

- **Flat asset paths** (e.g. `"model_xml": "scene.xml"`) are the common case.
- **Nested asset paths** (e.g. `"model_xml": "xmls/asimov.xml"`) are allowed when
  the upstream source repo uses a subdir layout. Example: `asimov_v0` maps to
  `asimovinc/asimov-v0` which has `sim-model/xmls/asimov.xml` +
  `sim-model/assets/`. The `_safe_join` helper in `strands_robots/utils.py`
  guards against traversal (`..`).
- **Auto-download strategy** - every robot with an `asset` block must declare
  exactly one of:
    1. `asset.robot_descriptions_module` (preferred)
    2. `asset.source` with `type: "github"`
    3. `asset.auto_download: false` (explicit opt-out)
  Enforced by `tests/test_registry_integrity.py`.


## Review Learnings (PR #85 - MuJoCo Backend)

Corrections from code review that apply to all future contributions:

### Thread Safety
- **Lock ALL model/data mutations** - MuJoCo `model`/`data` are not thread-safe. Any method that writes `qpos`, `qvel`, `ctrl`, `qfrc_applied`, `body_mass`, `geom_friction`, or calls `mj_step`/`mj_forward`/`mj_resetData` MUST hold `self._lock`.
- **Guard scene mutations during policy** - Use `_require_no_running_policy()` before any action that recompiles or replaces the model/data objects.
- **Document the concurrency contract** - If a method is safe to call concurrently, say so. If not, say so.

### Error Handling Contracts
- **Return error dicts, never raise** - All `AgentTool` action handlers must return `{"status": "error", "content": [...]}` on failure. Never raise exceptions that bypass the structured response.
- **Clean up on failure** - If you register state (e.g., add to `self._world.objects`) before an operation that can fail, pop/undo it in the except path.
- **Fail-fast with `strict=True`** - Silent frame dropping or catch-all `except Exception` with logging is forbidden unless gated behind a `strict=False` parameter.

### API Consistency
- **Don't export private functions** - `_`-prefixed names must never appear in `__all__`.
- **Match docstrings to semantics** - If the docstring says "single-shot" but the code is "latched", one of them must change. Always verify by reading the underlying library docs.
- **Forward all advertised kwargs** - If `tool_spec.json` exposes a parameter, the dispatch chain must forward it all the way through. Silent drops are bugs.
- **Centralize import checks at init** - Prefer checking optional deps once in `__init__` over scattered `_ensure_X()` guards. Consumers catch issues at init time.

### Data Integrity
- **Per-name state copy, not flat index** - When recompiling MuJoCo models (inject/eject), copy qpos/qvel per-joint by name. Flat-index slicing breaks when body-tree order shifts.
- **Sanitize user inputs into XML** - Validate names against `^[a-zA-Z0-9_-]+$` before interpolating into MJCF. LLM-provided strings are untrusted.
- **Match schema and data keys** - If a feature is declared with sanitized names (e.g., `__`), the data producer must emit the same sanitized keys.

### Testing
- **Test import paths must match production** - If `src/` imports `from lerobot.datasets.X`, tests must use the same path. Mismatched paths cause silent skips via `except ImportError`.
- **Round-trip tests for recording** - Any recording feature needs: start -> write -> stop -> reopen -> assert non-empty. Schema-only tests miss silent data loss.
- **Pin regression tests for reviewed fixes** - Every review fix gets a test that fails on pre-fix code. Otherwise the next refactor silently reintroduces the bug.
- **No host paths in test files** - Never commit `/Users/<name>/` or `/home/<name>/` paths. CI test `test_no_host_paths.py` enforces this.

### Performance
- **Don't create executors in hot loops** - Reuse a single `ThreadPoolExecutor` instance instead of creating one per call at 50Hz.
- **Cache expensive JSON parsing** - If a `@property` re-parses a JSON file on every access, cache the result at module load or first access.


## Review Learnings (PR #86 - Robot() factory)

Corrections from code review that apply to all future contributions:

### Resource Cleanup on Partial Failure
- **Always destroy on failure** - If `create_world()` succeeds but `add_robot()` fails, you MUST call `sim.destroy()` before raising. The `Simulation` object owns a `ThreadPoolExecutor`, MuJoCo world, and temp directory — leaking these is silent damage.
- **Pattern**: every `_dispatch_action(...)` call that could mutate persistent state needs `if result["status"] == "error": sim.destroy(); raise RuntimeError(...)`.
- **Don't discard return values** - If a step returns `{"status": ...}`, check it. The compiler won't catch a silently-ignored failure.

### Exception Clauses Must Be Narrow
- **`except Exception` is forbidden** for non-recovery code paths. Use the smallest superset of expected exception types.
- **`except (ImportError, Exception)` is a bug** — `Exception` is a superclass of `ImportError`, so the tuple collapses to `except Exception`. Lint/review will catch this; don't write it.
- **USB / hardware probing** - use `except (ImportError, OSError)`. `PermissionError` is an `OSError`, `FileNotFoundError` is an `OSError`, etc.

### Module-Level Side Effects
- **If you must run code at import time, comment WHY it can't be lazy.** `MUJOCO_GL` is the canonical example: MuJoCo locks the GL backend at first `import mujoco`, so the env var must be set before any downstream import chain triggers it.
- **Cheap-guard optional imports** - `if importlib.util.find_spec("mujoco") is not None:` before doing `from strands_robots.simulation.mujoco.backend import _configure_gl_backend`. Users without the `[sim-mujoco]` extra shouldn't pay an import-attempt cost on every `import strands_robots`.

### Public API Hygiene
- **Never recommend a `_method` in user-facing docstrings or error messages.** If `Robot()`'s docstring says "use `sim._dispatch_action(...)` to add a camera", you've just locked in a private dependency. Promote it (rename `_dispatch_action` → `dispatch_action`) or add public shorthands (`Simulation.add_camera()` / `.create_world()` / `.add_robot()`) before merging.
- **Type factory returns precisely** - never return `Any` from a factory. Use `@typing.overload` keyed on `Literal` mode args so IDEs resolve `Simulation` vs `HardwareRobot` at the call site. `# noqa: N802` is acceptable on factory functions named like classes (`Robot`), with a comment.
- **Reject silently-dropped kwargs** - if `Robot("so100", cameras={...})` is called in `mode="sim"` and the sim branch ignores `cameras`, raise `ValueError` instead of producing a sim with no cameras. Silent drops are bugs masquerading as features.
- **Don't conflate identity with schema** - `data_config` (e.g. `so100_dualcam`) is a separate concept from robot name (`so100`). Defaulting `data_config=robot_name` silently locks out multi-cam configs. Use an explicit `data_config: str | None = None` kwarg that defaults to canonical name only when omitted.

### Env Vars
- **Warn on unrecognized values** - `STRANDS_ROBOT_MODE=foo` (typo) must `logger.warning(...)`, not silently fall through. Silent typo'd env vars surprise users hours later.
- **Document every env var in README.md** - if you introduce a new `STRANDS_*` variable, add it to the Configuration section in the same PR. The list is the single source of truth for users.
- **Currently tracked**: `STRANDS_ROBOT_MODE`, `STRANDS_TRUST_REMOTE_CODE`, `MUJOCO_GL`.

### Safety Defaults
- **Sim-by-default** - any factory that can return either real hardware or a simulator must default to the simulator. Real hardware affects the physical world; users must opt in explicitly with `mode="real"` or `STRANDS_ROBOT_MODE=real`.
- **Reject invalid modes loudly** - `Robot("so100", mode="virtual")` must raise `ValueError`, not coerce to "sim".
- **Document parameter scope** - if `backend=` only applies to `mode="sim"`, say so in the docstring AND log a debug message when it's passed in `mode="real"` so it doesn't appear silently ignored.

### Naming & Module Organization
- **`robot.py` is for the `Robot()` factory**, the user-facing entry point. Hardware-specific code lives in `hardware_robot.py`. Don't have two files both named "robot something" with different responsibilities.
- **Reference module names, not filenames, in docstrings** - `strands_robots.hardware_robot` not `robot.py`. Filenames change; module paths are the public contract.

### Unicode & String Hygiene
- **No emojis in user-facing strings** - this is a project rule. Tool result dicts (`{"content": [{"text": ...}]}`), log messages, error messages: plain ASCII only. Agents read these strings programmatically; emojis just add tokenizer noise.
- **Hunt orphan combining marks after any emoji sweep** - `⏱️` is `U+23F1` + `U+FE0F` (variation selector). Stripping `U+23F1` leaves a stray invisible `U+FE0F` in the output. Sweep with:
  ```bash
  grep -nP '[^\x00-\x7F]' path/to/file.py
  ```
  or a Python check: `unicodedata.category(ch).startswith("So") or ord(ch) == 0xFE0F`.

### Testing Patterns
- **Use `monkeypatch.setenv`, never `os.environ[...] = ...`** - direct mutation leaks if the test raises before `finally`, and `del os.environ[...]` can `KeyError` under parallel runs. The pytest fixture handles teardown atomically.
- **Happy-path tests, not just error-paths** - if you have `test_factory_raises_on_bad_xml`, you also need `test_factory_returns_working_sim` gated behind `pytest.importorskip("mujoco")`. Steps physics, asserts state, destroys cleanly.
- **Pin every reviewed fix with a regression test** - every behavioral fix in this PR (warning on bad env var, rejecting `cameras=` in sim, default `mode="sim"`, etc.) has a dedicated test. "Trust me, the diff fixes it" is not a review-pass condition.
- **`importlib.reload` for module-state tests** - if a test modifies module-level state (env vars read at import time), reload the module inside the test and restore in teardown.

### Reviewing & Iteration
- **Resolve threads as you fix them** - leaving 14 unresolved threads on a PR with all fixes pushed makes re-review painful. Mark threads resolved when the commit lands; reviewers can re-open if not satisfied.
- **Reference commits in resolution comments** - "Fixed in `376376b`" + the suggested code block is dramatically faster to re-review than "fixed".
- **Force-push invalidates approvals** - after a rebase, prior `APPROVED` reviews drop to `DISMISSED` automatically. Mention it in the PR comment so reviewers know to re-approve, not re-review the whole diff.

## Review Learnings (PR #92 - CI Security Baseline)

Corrections from code review that apply to all future contributions:

### LLM Input Safety
- **Validate before subprocess interpolation** — every parameter on an agent-callable
  tool (`@tool` decorated function, `AgentTool.stream` dispatch handler) that flows
  into `subprocess.run`, `subprocess.Popen`, MJCF / XML interpolation, or filesystem
  path construction MUST be validated up front via regex allowlist, enum match, or
  range check. Argv-style subprocess does not exempt you — defense-in-depth.
- **Centralise validation in one function** — pattern: a `validate_inputs(...)` helper
  at the top of the tool module that takes every user-supplied param as a keyword arg
  and raises `ValueError` with a clear message on any rejection. Single entry-point
  is independently testable. PR #90's `gr00t_inference.validate_inputs()` is the
  canonical example.
- **Allowlist enumerable values** — `data_config`, `embodiment_tag`, dtype strings,
  container names: all match `^[a-z][a-z0-9_]+$` or an explicit `{"fp16", "fp8", ...}`
  set. Never accept arbitrary strings into enumerable surfaces.
- **Reject shell metacharacters in paths** — `;`, `|`, `$`, backticks, `>`, `<`,
  `\n`, `\r`, `\x00`. Also reject `..` path traversal components. Apply even when
  using argv-style subprocess.
- **Bind to `127.0.0.1` by default**, not `0.0.0.0`. Users explicitly opt into
  network exposure.

### CI Security Baseline
- **CodeQL findings are not PR-blocking but ARE actionable** — check the Security
  tab after pushing to a branch. False-positives get dismissed with a reason;
  real findings get fixed.
- **Dependency Review hard-fails on high/critical CVEs in new deps.** If a PR
  needs a dep with a known critical CVE, the conversation is "do we need this
  dep" not "let's bypass the check."
- **The LLM-input-safety workflow is a hint, not a gate.** Inline annotations
  on `subprocess + f-string` and `name-into-XML` patterns flag code that needs
  validation review. Confirm validation is present, then ignore the annotation
  in review.

### Action Pinning
- **All `uses:` references in workflows pin to a full 40-character commit SHA**,
  with the version tag preserved as a trailing comment: `uses: actions/checkout@<sha>  # v4.2.2`.
- **Dependabot keeps these fresh** via the `github-actions` ecosystem entry.
  Do not manually bump tags; merge the Dependabot PR.
- **Especially `pypa/gh-action-pypi-publish`** — it uses a moving `release/v1`
  branch, which is exactly the supply-chain pattern that the `tj-actions/changed-files`
  incident exploited. This pin is non-negotiable.

### Operational Runbooks for Security Pins
- **A static security pin must ship with a rotation runbook, not just a recompute
  command.** A docstring one-liner that recomputes a pin is necessary but not
  sufficient; on-call at 3 AM needs a documented grace-period strategy. For the
  Amazon Root CA1 pin (`provision._AMAZON_ROOT_CA1_PINS`) the runbook lives in
  README.md > "CA Pin Rotation Runbook": dual-pin tuple during the overlap, ship
  the new pin first, drop the old pin in a follow-up release after fleet uptake,
  and use `STRANDS_MESH_CA_PINS` only as an emergency out-of-band override.
- **Make the accepted-pin set a collection, never a scalar.** `_resolve_ca_pins()`
  returns a `frozenset` so the dual-pin grace period is expressible. Any future
  pinned fingerprint (other roots, signing keys) should follow the same
  multi-value shape so rotation never requires a flag-day deploy.
