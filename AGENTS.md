# AGENTS.md

Guide for coding agents (and people) working on strands-robots, a Python library for building robots.
Read this first. The tests enforce most of the rules below, so a rule broken here is a red CI run.

## What the library is

The same code drives a simulated robot or a real one. The package exports exactly eleven names plus
`__version__` (`strands_robots/__init__.py`, locked by `tests/test_public_surface.py`):

| Name | Is |
|---|---|
| `Simulation` | One world: MuJoCo on CPU, Isaac Sim or Newton (MuJoCo-Warp) on GPU |
| `Robot` | A body in a world, or a real robot through a driver (optionally with a digital twin) |
| `Policy` | Turns an `Observation` into an `Action` (functions, ONNX, LeRobot, replay) |
| `Mesh` | Robots on other machines over Zenoh or AWS IoT Core |
| `Body`, `Description` | The body protocol and a robot's model facts |
| `Observation`, `Action`, `JointState`, `Pose` | Plain data types |
| `RequestRefused` | The one refusal error (a `PermissionError`) |

Everything else (`Scene`, `Backend`, `RunResult`, `Units`, `Calibration`, ...) is imported from its subpackage, never
from the top level.

## Setup

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e ".[all,dev]"
uv pip install "lerobot[remote,feetech,dataset] @ git+https://github.com/huggingface/lerobot@main"
```

- Python 3.10+ for the package; CI, mypy and LeRobot use 3.12.
- `[all]` is every extra except `isaac`, `newton` and `dev`. Adding an extra means adding it to `all`
  (`tests/test_install_lines.py`).
- LeRobot comes from its main branch and is never a wheel dependency (a wheel can't carry a git ref). The one line
  lives in `strands_robots/_extras.py` (`LEROBOT`), the hatch `lerobot` env and the docs; a test keeps them equal.
- Headless Linux: set `MUJOCO_GL=egl` (CI does) or let `_gl.ensure_gl()` pick egl or osmesa. Import `strands_robots`
  before `mujoco`.

## Run what CI runs

The required check is `call-test-lint / Test and Lint` (`.github/workflows/test-lint.yml`). Locally:

```bash
hatch run lint          # ruff check, ruff format --check, mypy strands_robots, lint-imports
hatch run format        # auto-fix ruff findings and formatting
hatch run test -q       # pytest on tests/ (tests_integ/ is never collected by default)
mkdocs build --strict   # pip install -r docs/requirements.txt first
hatch run lerobot:test  # LeRobot policy and recording suites (Python 3.12, torch)
```

Without hatch, run the same commands in the venv: `ruff check strands_robots tests tests_integ`,
`ruff format --check strands_robots tests tests_integ`, `mypy strands_robots`, `lint-imports`, `pytest -q tests`.

- Ruff: line length 120, rules E, F, I, UP, B. mypy: `check_untyped_defs`. pytest: 300 s timeout per test.
- Tests skip, not fail, when an extra is missing (mujoco, fastapi, lerobot, zenoh, onnx, strands, playwright, ...)
  and when there is no CUDA machine (Isaac, Newton). A green local run without extras proves less than CI.
- The `network` marker (GitHub, Hugging Face) is not deselected in CI. Use `-m "not network"` offline.
- Opt-in suites: `STRANDS_ROBOTS_LIVE=1` (real Bedrock model), `STRANDS_IOT_LIVE=1` (real AWS IoT Core),
  `STRANDS_SO101_PORT=/dev/... pytest tests_integ/acceptance` (a physical SO-101).
- `tests/conftest.py` resets `Simulation.default()` around every test. Real hardware is faked with the memory driver.
- A second check, `Security` (CodeQL + dependency review), runs on PRs. Each new CodeQL alert opens a review thread,
  and unresolved threads block the merge.

## Rules the tests enforce

Layout (`tests/test_layout.py`, every file under `strands_robots/`):

- No file over 250 lines. Split by responsibility; several files already sit at exactly 250.
- No banner comments (any comment with `---`, or a four-space-indented all-lowercase comment line).
- `__init__.py` files only re-export: imports, docstrings and dunder assignments. No logic.
- No robot name or alias as a string literal outside docstrings. Names come from `registry/robots.json`. Only
  `robot/drivers/driver.py` and `robot/drivers/unitree.py` are exempt.

Imports:

- `import strands_robots` loads none of mujoco, torch, lerobot, zenoh, isaacsim, warp, onnxruntime,
  huggingface_hub (`tests/test_package.py`). Heavy imports go inside the function or class that needs them.
- A missing extra raises `ImportError` with `_extras.hint(...)`, whose text is pinned, e.g.
  `Simulation() needs MuJoCo: pip install 'strands-robots[mujoco]'`.
- Every module imports alone in a fresh interpreter with no cycle (`tests/test_modules_import_alone.py`).
- GPU backends import lazily (`tests/test_gpu_backends_import_lazily.py`); `strands` loads only when `.tool` is used;
  fastapi/uvicorn only in `dashboard/{app,connect_routes,guard,record_routes,server,voice_routes}.py` and
  `dashboard/auth/{guard,routes}.py`.
- `lint-imports` contracts in `pyproject.toml`: `types` imports nothing in the package; `description` only `types`,
  `_gl`, `_extras`; `units` only `types` and `description`; `simulation` never imports `robot` or `policy` (except the
  listed lazy edges); `policy` never imports `simulation` or `robot.drivers`; `mesh` never imports `policy`;
  `robot.safety` and `robot.device` import no policy and no simulation backend; `recording` only `types`,
  `description`, `units`.

Wording and safety text (`tests/test_safety_wording.py`): a run plan names its policy as `policy='<name>'`, never a
memory address; the e-stop refusal reads `e-stop is engaged by <who> (estop())`.

Docs (`docs/*.md`, `README.md`, `mkdocs.yml`, `overrides/`):

- The first `python` code block on each page runs as-is (`tests/test_docs_examples.py`). Mark it
  `<!-- test: skip, reason -->` or `<!-- test: network -->` on the line above if it can't. `my_policy` is the one
  allowed placeholder.
- Front matter is YAML with a string `description` (`tests/test_docs_front_matter.py`).
- No emoji, and no references to planning issues or status pages (`tests/test_docs_no_emoji.py`).
- Every user install line ends in `@ git+https://github.com/strands-labs/robots@main` (`tests/test_install_lines.py`).
- `mkdocs build --strict` fails on any broken link or nav entry.

Pins and versions: the `mesh` and `lerobot` extras pin the same `eclipse-zenoh>=1.9.0,<1.11.0`
(`tests/test_zenoh_pin.py`), the range LeRobot's `remote` extra resolves. `__version__` is the only version source
(`[tool.hatch.version]`); there is no static `version =` in `pyproject.toml`.

## Code rules (not all tested, all reviewed)

- No `**kwargs` swallowing. Unknown keyword arguments raise `TypeError` that lists what is accepted.
- Units: the robot always works in radians. Metres, quaternions as (w, x, y, z), sizes as half-extents, colours 0 to 1.
  A policy declares its units (`units` from `units.SCHEMES`); conversions go through `units/`, never ad hoc.
- NaN and inf never reach a body, a bus, gains, lights or the wire. Refuse them (`NonFiniteTarget`), don't clip.
- Joints an action leaves out hold their last target. A policy returning `None` holds the tick.
- Every world read or write takes `sim.lock` (a fair, reentrant `WorldLock`). Group scene edits in
  `with sim.editing():`; every edit otherwise recompiles the MuJoCo model.
- All MuJoCo rendering runs on the single `strands-gl` thread (`_gl.py`), except GLFW on macOS.
- Generated, not hand-written: device manifests (`robot/device/build.py`), agent tool specs (`tools/spec.py` from
  method signatures and docstrings), mesh keys (one `KEYS` table in `mesh/wire.py`).
- Errors say what to do next. A refusal is a `RequestRefused` with a sentence a person can act on.

## Safety model

A real robot moves only through its operator gate. Keep these invariants when you touch `robot/`, `mesh/`, `tools/` or
`dashboard/`:

- Two paths. The developer's own `robot.act()` / `robot.run()` is the operator: no chain, but still clipped to
  limits and refused on NaN. Agent, tool, mesh and dashboard writes go through `robot/safety/chain.py`: e-stop latch,
  joint range, the soft `safe` bound, interlocks, validators, then the operator's yes (`confirm=` / `confirm_act`).
- An agent run on a real body (`with from_agent():`) asks `approve(RunPlan)` once. Every step is still checked, and
  an out-of-range step ends the run instead of being clipped.
- The e-stop never takes the world lock or the run lock, and is never queued. It latches first, then stops the body.
  Over the mesh it is published on the tenant's `@estop` key and sent as an rpc. Only a named operator can resume,
  and only the stop they saw (`approved_by`, compare-and-release).
- A model never resumes, calibrates, picks a network address for a policy (`://` specs are refused unless the
  developer registered the handle), or sets `record`, `push` or `units`.
- A real robot without `hardware.on_disconnect` refuses a watchdog; a real robot on the mesh without a watchdog is
  observe-only. The dashboard binds loopback only, checks Host and Origin, answers `/api` and `/ws` only to a
  passkey session or a `dash.auth.token()` bearer (`dashboard/auth/`; a new route is guarded by being under those
  prefixes), and moves a real robot only with `X-Confirm: yes`, which chat and voice never send. Tests sign in through
  `tests/conftest.py` (per-test passkey store; `@pytest.mark.signed_out` sends no token).
- Feetech writes are speed-capped (`robot/drivers/slew.py`, 2 rad/s), and torque comes on only after every goal is
  computed, with every servo first held where it is (`FeetechBus.engage`: Goal := Present, then torque). Unitree and
  microduck drivers are observe-only.

## Where things live

All paths are under `strands_robots/`.

| Path | Owns |
|---|---|
| `robot/robot.py`, `robot/targets.py`, `robot/gate.py` | `Robot`, action coercion and hold, the gate |
| `robot/safety/` | Safety chain, validators, e-stop latch, watchdog |
| `robot/device/` | Manifest, rpcs, events (`DeviceVerbs`) |
| `robot/drivers/` | Driver protocol and `_DRIVERS`; feetech, unitree, microduck, memory; cameras, calibration |
| `robot/run/` | The run loop: bind, tick, pacing, lockstep, agent source, `RunResult` |
| `description/`, `registry/` | Model loading, asset fetching and caching, `robots.json` (76 robots) |
| `simulation/` | `Simulation`, scene data, `WorldLock`, backends (`mujoco`, `isaac`, `newton`) |
| `policy/` | Factory, ONNX presets, LeRobot local and remote (Zenoh), replay, wrappers |
| `recording/` | LeRobot v3 datasets plus `meta/units.json` |
| `units/` | Unit schemes and LeRobot semantics |
| `mesh/` | Mesh, wire format, transports (zenoh, mem, iot), authority, AWS IoT tooling |
| `tools/` | Strands Agents tools: `robot_<id>`, `sim_world`, `use_mesh` |
| `dashboard/` | FastAPI app, chat (`Agent`), voice (`BidiAgent`), Connect, Record, static front end |

## How to add things

- A robot: an entry in `registry/robots.json` (device type from `device_types`, asset, aliases, optional hardware
  and gripper). No Python. Update the hard-coded robot counts in `tests/test_description.py`.
- A driver: `robot/drivers/<kind>.py` implementing the `Driver` protocol with its SDK imported lazily, a line in
  `_DRIVERS`, `hardware.driver` in the registry, and `on_disconnect` / `position_hold` so the watchdog works. Extend
  the per-driver model in `tests/test_d3_passthrough.py`. A one-off can skip registration: `Robot(name, driver=obj)`.
- A unit scheme: `units/<name>.py` with `@register("<name>")`, imported in `units/__init__.py`.
- A policy: subclass `Policy`, declare `units`, implement `act` and `reset`. For a spec string, add an ONNX entry to
  `policy/onnx/presets.json` (with sha256) or a branch plus a `FORMS` line in `policy/factory.py`. Policies are
  chosen by the form of the spec, never by model family.
- A simulation backend: a `Backend` subclass (12 methods) and one `"module:Class"` entry in `_BACKENDS`.
- An agent tool action: a typed, documented public method declared in the manifest's rpcs. The tool spec follows.
  For the mesh, also add it to `mesh/serve.py` `ARGS`.
- A dashboard panel: front end only, `SR.registerPanel` / `SR.registerSlot` in `dashboard/static/panels/`
  (see its `README.md`). Never send `X-Confirm` without a `confirm()` that names the real robot.

## Pull requests

- `main` is the released, default branch. Branch from it and open the PR against it.
- Title in conventional form, a plain sentence: `fix(simulation): ...`, `feat(policy): ...`, `docs: ...`, `test: ...`.
- Every behaviour change adds `changelog.d/<PR number>-<slug>.md` with exactly one `### <Category>: <summary>`
  heading and no `##` heading. Rename the file once the PR number exists (0 and 9000+ are refused). Check with
  `python scripts/assemble_changelog.py --check`. Never edit `[Unreleased]` in `CHANGELOG.md` directly.
- Keep a PR to one change, with its tests, and run lint, tests and the strict docs build before pushing.
- Release: `python scripts/assemble_changelog.py --apply`, set the version, merge, then publish a GitHub release
  tagged `v<version>` on `main`. The publish workflow checks the tag equals `v<__version__>` and uploads to PyPI.
- Security issues go to the AWS vulnerability disclosure program, not a public issue (`SECURITY.md`).

## Docs and notebooks

- Pages are short, plain English: what it is, one runnable example, then facts in tables. No marketing, no emoji.
- Notebooks are generated: edit `notebooks/scripts/NN_name.py`, build with
  `python notebooks/_nb.py build notebooks/scripts/NN_name.py`, execute with
  `jupyter nbconvert --to notebook --execute --inplace notebooks/NN_name.ipynb`, and check outputs with
  `python notebooks/scripts/check_nb.py NN_name.ipynb`. Commit them without outputs
  (`jupyter nbconvert --clear-output --inplace`; `tests/test_notebooks_are_light.py`). Their GIFs and screenshots are
  assets of the `media` GitHub release, not files in git. Keep cells short and use the public API; helpers go in
  `notebooks/_nb.py`.
- Nothing large goes in the package: the dashboard's `.wasm` engines are fetched on first use
  (`dashboard/vendor.py`, pinned by sha256), so the installed package stays around 2 MB.
- Drawings: `docs/drawings/_tools/scene.py` (see `README.txt` next to it).

## Known gotchas

- A real `Robot` has no twin unless `twin=True`, which mirrors it in `Simulation.default()` and needs the `mujoco`
  extra. The dashboard draws a real robot through its twin.
- Registry entries with `driver: "strands"` or `"lerobot"` have no driver in `_DRIVERS`: `Robot(x, port=...)` fails
  with "no driver known".
- Without MuJoCo, the XML fallback ignores `angle="degree"` and `<default>` ranges, so limits can differ.
- GitHub-hosted models are pinned: `asset.source.commit` plus `asset.source.sha256` (`assets.tree_digest` of the
  subdir). Bumping a model means updating both; a cache from another pin is fetched again.
- Two robots with the same id clash as tool names. Pass `id=`.
- Over Zenoh, resume by name is refused unless the device has operator keys (`Authority(public_keys=...)`) or
  `trust_names=True`.
- Recording needs an integer `hz`, and refuses an existing dataset with a different fps, robot, units or features.
- Isaac: one world per process, `OMNI_KIT_ACCEPT_EULA=YES` required, no contacts, lights or randomisation yet, and
  runs are not bit-for-bit repeatable. Newton is deterministic only with one environment.
- Zenoh scouting uses 224.0.0.224:7447 (one port above Zenoh's default); `?listen=` / `?connect=` turn it off.
