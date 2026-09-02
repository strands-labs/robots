"""A scene mutation swaps the live model under ``self._lock``.

``Simulation._lock`` is declared by the backend's own module docstring as the
"RLock serializing ALL model/data access", and the render path takes it because
"the recorder daemon calls render() on its own thread, so this path is NOT
covered by the blanket dispatch lock". Seven scene-mutation verbs performed the
swap outside it and relied on ``_require_no_running_policy()`` instead - a guard
that refuses a *policy* worker and cannot see the recorder thread, which has no
policy.

The swap is also two assignments wide:
``scene_ops.install_compiled_model`` rebinds ``world._model`` and then
``world._data``, so a reader that is not excluded can observe a model from after
the swap paired with data from before it, and MuJoCo dereferences that pair
natively.

Measured with a reader of the recorder's shape - it takes the lock and, while
still holding it, re-reads ``(nq, qpos.size, nbody, ncam)``:

| verb | swap seen under lock (pre) | (post) |
| --- | --- | --- |
| ``replace_scene_mjcf`` | yes | no |
| ``patch_scene_mjcf`` | yes | no |
| ``add_object`` | yes | no |
| ``add_robot`` | yes | no |
| ``remove_object`` | yes | no |
| ``add_camera`` | yes | no |
| ``remove_camera`` | yes | no |
| ``load_scene`` (control) | no | no |
| ``remove_robot`` (control) | no | no |

The two controls already swapped under the lock and are unchanged by this
module's fix; they are what distinguishes the pin from "every mutation fails".
"""

import ast
import threading
import time
from pathlib import Path

import pytest

import strands_robots.simulation.mujoco as mujoco_pkg
from strands_robots.simulation import create_simulation

# The reader holds the lock for this long. Every unlocked swap measured well
# under 0.25 s, so a writer that waits is unambiguous at this hold.
HOLD_S = 1.0

_ONE_BODY_SCENE = """<mujoco>
  <worldbody>
    <body name="a" pos="0 0 1"><joint type="free"/><geom type="box" size=".1 .1 .1"/></body>
  </worldbody>
</mujoco>"""


@pytest.fixture(scope="module")
def scene_file(tmp_path_factory):
    """An MJCF on disk for the ``load_scene`` control row."""
    path = tmp_path_factory.mktemp("scenes") / "one_body.xml"
    path.write_text(_ONE_BODY_SCENE)
    return str(path)


@pytest.fixture(scope="module", autouse=True)
def _warm_robot_assets():
    """Resolve both robot descriptions once, before anything is timed.

    ``add_robot`` may fetch a robot description on first use. That fetch inside
    a timed row would be indistinguishable from a writer waiting on the lock,
    so it happens here instead.
    """
    sim = create_simulation("mujoco")
    try:
        sim.create_world()
        sim.add_robot(name="so100")
        sim.add_robot(name="so101", position=[1, 0, 0])
    finally:
        sim.destroy()


# (label, call, already_locked_before_this_fix)
MUTATIONS = [
    ("replace_scene_mjcf", lambda s, scene: s.replace_scene_mjcf(_ONE_BODY_SCENE), False),
    (
        "patch_scene_mjcf",
        lambda s, scene: s.patch_scene_mjcf(
            [{"op": "add_body", "parent": "world", "name": "patched", "pos": [0, 0, 3]}]
        ),
        False,
    ),
    ("add_object", lambda s, scene: s.add_object(name="crate", shape="box", position=[0.4, 0, 0.1]), False),
    ("add_robot", lambda s, scene: s.add_robot(name="so101", position=[1, 0, 0]), False),
    ("remove_object", lambda s, scene: s.remove_object("box0"), False),
    ("add_camera", lambda s, scene: s.add_camera(name="side", position=[0, -1, 0.5], target=[0, 0, 0.2]), False),
    ("remove_camera", lambda s, scene: s.remove_camera("side"), False),
    # Controls: these two already held the lock across the swap.
    ("load_scene", lambda s, scene: s.load_scene(scene), True),
    ("remove_robot", lambda s, scene: s.remove_robot("so100"), True),
]


def _observe_swap_under_lock(sim, call, scene):
    """Run ``call`` while a reader holds ``sim._lock``; report what it saw.

    Returns:
        A dict with ``status`` (the verb's envelope status), ``swapped`` (did the
        live scene change while the reader held the lock) and ``writer_waited``
        (did the verb block until the reader let go).
    """
    entered, writer_done = threading.Event(), threading.Event()
    seen: dict[str, object] = {}

    def reader() -> None:
        # The shape of the recorder daemon: it renders under this same lock.
        with sim._lock:
            model, data = sim._world._model, sim._world._data
            before = (model.nq, data.qpos.size, model.nbody, model.ncam)
            entered.set()
            writer_done.wait(timeout=HOLD_S)
            model, data = sim._world._model, sim._world._data
            after = (model.nq, data.qpos.size, model.nbody, model.ncam)
            seen["swapped"] = before != after
            # A torn pair: the model was rebound but its data was not.
            seen["pair_torn"] = model.nq != data.qpos.size

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        assert entered.wait(timeout=10), "reader never acquired the lock"
        started = time.monotonic()
        result = call(sim, scene)
        elapsed = time.monotonic() - started
    finally:
        writer_done.set()
        thread.join(timeout=10)

    return {
        "status": result.get("status"),
        "swapped": seen.get("swapped"),
        "pair_torn": seen.get("pair_torn"),
        "writer_waited": elapsed > HOLD_S * 0.8,
    }


@pytest.mark.parametrize(("label", "call", "already_locked"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_a_scene_mutation_is_not_visible_to_a_thread_holding_the_lock(label, call, already_locked, scene_file):
    """No scene swap becomes visible to a reader that holds ``sim._lock``.

    ``already_locked`` marks the two control verbs, which behaved this way
    before the fix. Both groups must pass: the verb still succeeds, and the
    reader's view of the live scene is stable for as long as it holds the lock.
    """
    sim = create_simulation("mujoco")
    try:
        sim.create_world()
        sim.add_robot(name="so100")
        sim.add_object(name="box0", shape="box", position=[0.3, 0, 0.1])
        if label == "remove_camera":
            sim.add_camera(name="side", position=[0, -1, 0.5], target=[0, 0, 0.2])

        observed = _observe_swap_under_lock(sim, call, scene_file)
    finally:
        sim.destroy()

    assert observed["status"] == "success", f"{label} did not succeed: {observed}"
    assert observed["swapped"] is False, (
        f"{label} swapped the live scene while a reader held sim._lock; "
        f"the recorder daemon renders under that lock and is not covered by "
        f"_require_no_running_policy()"
    )
    assert observed["pair_torn"] is False, f"{label} left model/data mismatched for a lock holder"
    assert observed["writer_waited"] is True, f"{label} did not wait for the lock holder"


def _live_model_swappers() -> set[str]:
    """Every ``scene_ops`` function that (transitively) swaps the live model.

    Derived rather than listed, so a new helper that recompiles the scene joins
    the graded set on the commit that introduces it.
    """
    source = Path(mujoco_pkg.__file__).parent / "scene_ops.py"
    tree = ast.parse(source.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    calls = {
        name: {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(fn)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        for name, fn in functions.items()
    }
    swappers = {"install_compiled_model"}
    changed = True
    while changed:
        changed = False
        for name, called in calls.items():
            if name not in swappers and called & swappers:
                swappers.add(name)
                changed = True
    return swappers


def _lock_covered_lines(fn: ast.AST) -> list[tuple[int, int]]:
    """Line ranges inside ``fn`` that a ``with <...>lock<...>:`` statement covers."""
    ranges = []
    for node in ast.walk(fn):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            header = " ".join(ast.unparse(item.context_expr) for item in node.items)
            if "lock" in header.lower():
                ranges.append((node.body[0].lineno, node.end_lineno or node.body[0].lineno))
    return ranges


def test_every_call_that_swaps_the_live_model_holds_the_lock():
    """No caller in the MuJoCo backend recompiles the live scene unlocked.

    The behaviour pins above cover today's verbs one by one. This closes the
    family: an eighth verb that recompiles the scene outside ``self._lock`` is
    reported here rather than waiting for someone to record while it runs.
    """
    swappers = _live_model_swappers()
    assert "replace_scene_mjcf" in swappers, "swapper derivation found nothing to grade"

    unlocked = []
    for path in sorted(Path(mujoco_pkg.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Skip the swappers' own definitions, which live in scene_ops and
            # take no lock by design (the lock belongs to Simulation, and they
            # call each other). Keyed on the FILE, not the function name: the
            # backend's ``replace_scene_mjcf`` / ``patch_scene_mjcf`` verbs share
            # a name with the scene_ops helper they delegate to, so a name-only
            # skip would silently exempt the two verbs most in need of grading.
            if path.name == "scene_ops.py" and fn.name in swappers:
                continue
            covered = _lock_covered_lines(fn)
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name in swappers and not any(lo <= node.lineno <= hi for lo, hi in covered):
                    unlocked.append(f"{path.name}:{node.lineno} {fn.name}() -> {name}()")

    assert not unlocked, (
        "these call sites recompile the live scene without holding self._lock, so the "
        "recorder thread can render a half-swapped scene:\n  " + "\n  ".join(unlocked)
    )
