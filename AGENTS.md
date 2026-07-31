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
3. Record the change as a news fragment: `changelog.d/<pr-number>-<slug>.md`
   (see [`changelog.d/README.md`](changelog.d/README.md)). **Never append to
   `## [Unreleased]` in `CHANGELOG.md` directly** - every branch inserts at the
   same anchor, so two PRs open at once conflict on ordering alone, and because
   stale approvals are dismissed on push each resolution costs a re-approval
   round that reviews no changed behaviour. A fragment is its own file, so there
   is nothing to conflict on. `CHANGELOG.md` is assembled from the accumulated
   fragments when a tag is cut (`python scripts/assemble_changelog.py --apply`).
   This is enforced by `.github/workflows/changelog-fragment.yml`: the rule was
   documented in two places and enforced by nothing until #1784, and a pull
   request reached `APPROVED` / `SUCCESS` / `CLEAN` having appended to the log.
4. All tests must pass, lint must be clean
5. Open PR from your fork, address all review comments
6. Track follow-up items as issues on the [project board](https://github.com/orgs/strands-labs/projects/2)

   **Read the board with `PAT_TOKEN`, not the Actions `GITHUB_TOKEN`.** An
   installation token that cannot see an organization project does not fail the
   query - `issue.projectItems` comes back as an empty list, which is
   indistinguishable from an issue nobody has tracked. The same one query, run
   twice on #1762, #1768 and #1770:

   | token | `issue.projectItems` |
   |---|---|
   | `GITHUB_TOKEN` | `[]` for all three |
   | `PAT_TOKEN` | one item each; #1768 and #1770 already at `Done` |

   Two things follow. First, adding an issue to the board is almost never the
   missing step: all three items were created by `github-project-automation`
   within three seconds of the issue itself, and the automation also moves them
   to `Done` on close, so a merged fix needs no manual flip. What it cannot
   infer is a status like `In review` while a PR is open - that is the part
   worth setting by hand. Second, acting on the empty read looks harmless and
   is not: `addProjectV2ItemById` is idempotent and returns the *existing*
   item's id, so no duplicate ever appears and the false-empty read leaves no
   trace - until a `Status` written on the strength of it silently overwrites a
   value that was never read. Read a field before you set it, and treat an
   empty project read as unknown rather than as absent.
7. Squash merge into `main`
8. **Verify a PR's state by reading it back - before and after you change it.**
   Neither direction can be inferred:
   - *Before.* Query `timelineItems(itemTypes: [CLOSED_EVENT, REOPENED_EVENT])`
     before closing or reopening. A lone `CLOSED_EVENT` is safe to re-apply; an
     alternating run means something is undoing you, and a further flip only
     lengthens it. #1667 - the retired `pr/consolidated-local-work` staging PR -
     was closed and reopened **ten times in under fourteen hours**, each reopen 2-22
     minutes after the preceding close, because independent contributors read
     the same PR and reached opposite conclusions. Its extraction plan and
     terminal state live in #1723; do not flip #1667 again.
   - *After.* A `mergePullRequest` mutation can report `Pull Request is not
     mergeable` on a merge that in fact landed - observed on #1756, where the
     mutation returned that error and the squash was already on `main`. Confirm
     with `state`/`merged`, or `git log origin/main`, before concluding a merge
     failed and redoing the work.
   - *And on `main` afterwards.* A rollup of `FAILURE` on a merge commit is not
     evidence that the squash broke anything: a **cancelled** check aggregates
     into `FAILURE`. `pr-and-push.yml` keys its concurrency group on
     `github.event.pull_request.number || github.ref`, and on a push there is no
     PR number, so every push to `refs/heads/main` shares one group under
     `cancel-in-progress: true` - each merge kills the run of the merge before
     it. Read each context's own `conclusion` before you believe the rollup.
     Four PRs merged in the 22 minutes from 03:03:44 to 03:25:25 left three
     consecutive commits - #1788, #1794, #1796 - each reporting rollup
     `FAILURE` whose only non-`SUCCESS` context was
     `call-test-lint / Test and Lint` = `CANCELLED`, killed at 1m07s, 15m00s
     and 5m38s into their runs. Nothing had failed. See #1800.

     The same timings carry a cost that is not a misread: that suite had not
     finished in 15m00s, so merging faster than it runs leaves **only the tip
     verified** and no intermediate commit attributable. A batch is still
     defensible - each of those four was individually green, passed
     `Detect an untested overlap with the base branch`, and touched a file set
     disjoint from the others - but price it knowingly: a red tip then costs a
     manual bisect, and an intermediate commit's green is not available to lean
     on. Do not read the intermediate `FAILURE`s as the culprit; they are the
     batching, not a defect.
   And before merging, `reviewDecision: APPROVED` alone is not the gate: poll
   `statusCheckRollup.state == SUCCESS` and `mergeStateStatus == CLEAN`
   together, since `reviewDecision` flips before the checks finish.

   Those three together are still not sufficient. They are all evaluated against
   the base the branch was tested on, so none of them can see a **semantic**
   conflict with a PR that landed on `main` after those checks ran. #1766 and
   #1763 both edited `_recompile_preserving_state` for unrelated reasons: the
   text merged with no conflict, #1763 stayed `MERGEABLE`/`CLEAN` with
   `SUCCESS` checks after #1766 landed, and the squash still broke `main`,
   because #1763 carried a *premise* test asserting the very defect #1766 had
   just fixed. Neither PR's CI ever compiled the two together.

   So when a second approved PR touches a file - especially a function - that a
   just-merged PR also touched, do not merge on the green alone. Merge `main`
   into the branch (or check out the merge locally) and run the affected tests
   before issuing the mutation. A `CLEAN` status is a statement about text, not
   about meaning. This is cheap: the check that would have caught the above was
   one `pytest` invocation on two files.

   Read that run as a **delta, not an absolute**. The environment you verify in
   is almost never the one CI uses, and a partial one fails tests for reasons
   that have nothing to do with the merge. Composing #1786 and #1804 - both
   approved, both `CLEAN`, both editing `simulation/predicates.py`, neither ever
   compiled with the other - the affected suite reported **376 failed, 34
   errors** on the composition, which on its own reads as a broken merge and a
   reason to stop. The same command on the unmerged base reported the same
   **376 failed, 34 errors**, and 4185 passed against the composition's 4229:
   every failure pre-existed (a hosted runner has no GPU and only software
   OSMesa, so the rendering tests fail there whatever the diff), and the whole
   effect of the merge was **+44 passes** - exactly the 24 + 20 tests the two
   branches add. The reading matters in both directions: an absolute count can
   invent a regression and cost a good merge, and it can equally hide a real one
   inside the noise. Run the same command on the base *before* you read the
   number, and compare the two.

   Then confirm the tree you verified is the tree that landed -
   `git diff --name-only <local-composition> origin/main -- strands_robots/ tests/`
   should be empty. Squash rewrites the commits, so nothing but that equivalence
   ties your local run to `main`; and on a batch, where only the tip's
   `call-test-lint` survives the concurrency group above, it is the sole evidence
   the intermediate commits were ever compiled together.

   Fixing forward beats reverting here - the two production changes were both
   correct, and only an assertion and its justification were stale. Prefer a
   narrow follow-up that re-pins the invalidated premise over reverting a
   reviewed change. If a premise test is invalidated by a fix landing, replace it
   rather than deleting it: the conclusion it supported usually still holds for a
   different reason, and that reason is what the next reader needs.

   The converse happens too: every signal above satisfied, and the PR still
   refuses to merge with no field naming the reason. #1722 carried one current
   `APPROVED` review that post-dated its head commit, all four review threads
   resolved, `call-test-lint` `SUCCESS` - and `reviewDecision`
   `REVIEW_REQUIRED`. The `default` branch ruleset sets
   `require_last_push_approval: true`, so the most recent push must be approved
   by **someone other than whoever pushed it**, and the agent had pushed that
   head commit with `PAT_TOKEN`. GitHub attributes a push to the token's
   *owner*, which was the same account that then approved. No number of further
   approvals from that account can clear it.

   What makes this worth writing down is that the commit metadata asserts the
   opposite. `d938686`'s author *and* committer are `strands-robots`, an
   identity distinct from the approver, so reading the commit list says the rule
   is satisfied. The pusher is in none of the fields you would check:
   `reviewDecision` is `REVIEW_REQUIRED` and `mergeStateStatus` is `BLOCKED`,
   which is also exactly what a PR with no approval at all looks like. The one
   place it is legible:

   ```
   GET /repos/{owner}/{repo}/actions/runs?head_sha=<head>  ->  triggering_actor
   ```

   #1035 is the control - same author, same fork, same `strands_robots/mesh/`
   files, one approval from the same account post-dating its head commit,
   threads clear, checks green, and no CODEOWNERS file in the tree to make
   `require_code_owner_review` bite. It differs in exactly one input, and reads
   `APPROVED`:

   | PR | commit author | `triggering_actor` | approver | `reviewDecision` |
   |---|---|---|---|---|
   | #1035 | the contributor | the contributor | the maintainer | `APPROVED` |
   | #1722 | `strands-robots` | the maintainer | the maintainer | `REVIEW_REQUIRED` |

   So **pushing a fix to a contributor's branch consumes the approval of
   whoever owns the token you push with**, turning a PR one maintainer could
   merge into one that needs a second. It compounds with
   `dismiss_stale_reviews_on_push`, which drops the existing approval in the
   same motion that disqualifies that account from re-supplying it. Prefer
   leaving the change for the contributor to push, so they stay the last
   pusher; when the agent must push, that PR now requires a second approver,
   and saying so is the difference between a one-line request and a branch that
   never merges.

   The general rule behind all three: **a decision recorded only in a PR or
   issue comment is not durable** - the next contributor will not read the same
   comment. If a decision must survive, it belongs in this file.


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
- **Always destroy on failure** - If `create_world()` succeeds but `add_robot()` fails, you MUST call `sim.destroy()` before raising. The `Simulation` object owns a `ThreadPoolExecutor`, MuJoCo world, and temp directory - leaking these is silent damage.
- **Pattern**: every `_dispatch_action(...)` call that could mutate persistent state needs `if result["status"] == "error": sim.destroy(); raise RuntimeError(...)`.
- **Don't discard return values** - If a step returns `{"status": ...}`, check it. The compiler won't catch a silently-ignored failure.

### Exception Clauses Must Be Narrow
- **`except Exception` is forbidden** for non-recovery code paths. Use the smallest superset of expected exception types.
- **`except (ImportError, Exception)` is a bug** - `Exception` is a superclass of `ImportError`, so the tuple collapses to `except Exception`. Lint/review will catch this; don't write it.
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
- **Validate before subprocess interpolation** - every parameter on an agent-callable
  tool (`@tool` decorated function, `AgentTool.stream` dispatch handler) that flows
  into `subprocess.run`, `subprocess.Popen`, MJCF / XML interpolation, or filesystem
  path construction MUST be validated up front via regex allowlist, enum match, or
  range check. Argv-style subprocess does not exempt you - defense-in-depth.
- **Centralise validation in one function** - pattern: a `validate_inputs(...)` helper
  at the top of the tool module that takes every user-supplied param as a keyword arg
  and raises `ValueError` with a clear message on any rejection. Single entry-point
  is independently testable. PR #90's `gr00t_inference.validate_inputs()` is the
  canonical example.
- **Allowlist enumerable values** - `data_config`, `embodiment_tag`, dtype strings,
  container names: all match `^[a-z][a-z0-9_]+$` or an explicit `{"fp16", "fp8", ...}`
  set. Never accept arbitrary strings into enumerable surfaces.
- **Reject shell metacharacters in paths** - `;`, `|`, `$`, backticks, `>`, `<`,
  `\n`, `\r`, `\x00`. Also reject `..` path traversal components. Apply even when
  using argv-style subprocess.
- **Bind to `127.0.0.1` by default**, not `0.0.0.0`. Users explicitly opt into
  network exposure.

### CI Security Baseline
- **CodeQL findings are not PR-blocking but ARE actionable** - check the Security
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
- **Especially `pypa/gh-action-pypi-publish`** - it uses a moving `release/v1`
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

## Review Learnings (PR-6 - mesh core safety hardening)

Corrections from the mesh safety/audit hardening review trail (#221/#225). They
apply to all future work on `strands_robots/mesh/{core,audit,security}.py`.

### Safety-handler discipline
- **Hoist env-var reads out of the hot path and the lock.** Safety handlers
  (`_on_safety_estop` / `_on_safety_resume`) run per-envelope; resolve lazy env
  vars (`_resume_forward_skew_s`, `_resume_freshness_window_s`) into locals at
  handler entry, before taking the cache lock, so the lock holds for the minimum
  window and a hot path never re-parses the environment per call.
- **Lockout-engagement is decoupled from the per-issuer cache cap.** A bounded
  replay cache that is full (flood, or a tiny operator override) must still let a
  legitimate peer ENGAGE a lockout - the cap bounds memory, not safety. Pin both
  directions: `*_per_issuer_cap_exceeded_still_engages_lockout` and
  `*_low_cache_max_does_not_deny_safety`.
- **Domain-tag trust-boundary cache keys.** A TLS-bound `wire_zid` and an
  app-level `issuer_id` that happen to share a string must not collide into one
  replay-cache slot. Prefix keys with a trust-domain tag (`("wire", …)` vs
  `("body", …)`) so the two namespaces can never alias.

### Audit poison-record symmetry
- **Every degraded audit path writes a poison record, never a silent drop.** The
  poison `sig` discriminators (`PSK_DEGRADED`, `SIGN_FAILED`, `SEQ_LOCK_DEGRADED`,
  `NEXT_SEQ_DEGRADED`) let a `verify_audit_integrity` walker attribute a stream
  gap to a specific failure class. When you add a new `_next_seq`/sign/persist
  failure branch, add the matching poison `sig` instead of returning early.

### Replay-cache eviction
- **TTL purge runs unconditionally, not only when the cache is full.** On a
  low-traffic mesh the cache may never reach `max_size`, but stale entries still
  accumulate. `_evict_replay_cache` always runs the O(n) TTL pass first, then the
  over-budget trim.

## Review Learnings (PR-7 - robot_mesh HITL patterns)

From the `robot_mesh` human-in-the-loop review trail (#227). Apply to the
`robot_mesh` tool and any agent-facing tool that gates on an operator interrupt.

### Operator responses are not an LLM channel
- **Never echo the operator's literal interrupt response back to the LLM.**
  Record the full response in the LOCAL audit row for forensics, but return a
  flat, fixed sentinel to the model. Echoing the operator's typed reply turns the
  human into a prompt-injection content side-channel (the agent could phrase the
  approval reason so the operator's answer leaks data into the context).

### Audit completeness
- **Audit read-only/observation actions too, not just actuation.** `peers`,
  `status`, `inbox`, and `unsubscribe` each leave a `_audit_tool_action(...)` row
  so the audit log is a complete record of agent mesh access - operators get the
  "agent read N frames from sub X at time T" trail that raw telemetry access
  otherwise lacks.

### Rate-limit safety semantics
- **A declined HITL approval must NOT consume a rate-limit slot.** The slot is
  recorded only after approval is granted (or atomically via
  `_rate_limit_check_and_record` on the post-approval path). Otherwise nuisance
  prompts an operator declines would lock the agent out of issuing a genuine
  `emergency_stop` - the inverse of the intended safety property.
