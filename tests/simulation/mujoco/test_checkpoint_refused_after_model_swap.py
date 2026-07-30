"""A physics checkpoint is refused once the model it was taken against is gone.

``save_state`` stamps a checkpoint with a fingerprint - ``nq``/``nv``/``na``/``nu``
plus ``SimWorld._recompile_generation`` - and ``load_state`` refuses to apply a
checkpoint whose fingerprint no longer matches the live model. The counts catch a
swap that resizes the state vector. The generation exists for the swap that does
NOT: two models can agree on every count and still mean different things at the
same state indices.

The established contract, set by the dominant mutation path, is that ANY model
swap invalidates an outstanding checkpoint - ``add_camera`` refuses one even
though adding a camera leaves the state vector byte-identical. These tests pin
the sibling swap paths to the same contract:

* ``replace_scene_mjcf`` installs a whole new scene, so a checkpoint from the old
  one is written index by index into whatever those indices now mean.
* ``remove_camera`` is ``add_camera``'s mirror and leaves the state size
  untouched, so nothing but the generation distinguishes the two models.
* ``patch_scene_mjcf`` edits the live spec; an op that adds no body (``add_geom``)
  leaves the state size untouched while changing what the model describes.

``eject_robot_from_scene`` is deliberately not pinned behaviourally here: removing
a robot always drops ``nbody``, and the state vector includes ``xfrc_applied``
(``nbody`` x 6), so the size check refuses it on its own. It is covered by the
structural test at the bottom, which is what keeps every swap site consistent.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.scene_ops import install_compiled_model  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Two scenes that agree on nq/nv/na/nu/nbody and disagree on what qpos[0] means:
# in PAN the hinge turns the arm about z, in LIFT it swings it about y. A
# checkpoint taken from one describes a pose the other never held.
_PAN_SCENE = """<mujoco model="pan">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="post" pos="0 0 0.3">
      <joint name="swing" type="hinge" axis="0 0 1" range="-3 3" limited="true" damping="1"/>
      <geom name="arm" type="box" size="0.30 0.05 0.05" pos="0.30 0 0" rgba="0.2 0.5 0.9 1"/>
    </body>
  </worldbody>
  <actuator><position name="swing_act" joint="swing" kp="30" ctrlrange="-3 3"/></actuator>
</mujoco>"""

_LIFT_SCENE = _PAN_SCENE.replace('model="pan"', 'model="lift"').replace('axis="0 0 1"', 'axis="0 1 0"')

_SAVED_SWING = 0.9


# The fixture parameter is left un-annotated in these helpers: ``sim._world`` is
# ``SimWorld | None`` on the class, and every probe below runs against a world the
# fixture already created.
def _state_size(sim) -> int:
    """Length of the state vector a checkpoint stores for the live model."""
    return int(mj.mj_stateSize(sim._world._model, mj.mjtState.mjSTATE_INTEGRATION))


def _arm_position(sim) -> list[float]:
    """World position of the arm geom - where the restored pose actually put it."""
    model, data = sim._world._model, sim._world._data
    mj.mj_forward(model, data)
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "arm")
    assert gid >= 0, "fixture scene must expose the 'arm' geom"
    return [float(v) for v in data.geom_xpos[gid]]


def _set_swing(sim, value: float) -> None:
    """Drive the single hinge to ``value`` and refresh derived state."""
    sim._world._data.qpos[0] = value
    mj.mj_forward(sim._world._model, sim._world._data)


@pytest.fixture
def sim():
    s = Simulation(tool_name="checkpoint_swap_sim", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


class TestReplaceSceneMjcfInvalidatesCheckpoints:
    def test_a_checkpoint_from_the_replaced_scene_is_refused(self, sim):
        """The new scene keeps every count, so only the generation can refuse it.

        Pre-fix ``replace_scene_mjcf`` rebound the live model without bumping the
        generation, so the fingerprint still matched and the checkpoint was
        applied to a different scene.
        """
        assert sim.replace_scene_mjcf(_PAN_SCENE)["status"] == "success"
        _set_swing(sim, _SAVED_SWING)
        size_before = _state_size(sim)
        assert sim.save_state(name="panned")["status"] == "success"

        assert sim.replace_scene_mjcf(_LIFT_SCENE)["status"] == "success"
        assert _state_size(sim) == size_before, (
            "fixture must keep the state size identical across the swap, otherwise "
            "the size check refuses the checkpoint and the generation is untested"
        )

        result = sim.load_state(name="panned")
        assert result["status"] == "error"
        assert "stale" in result["content"][0]["text"].lower()

    def test_the_refused_checkpoint_leaves_the_new_scene_at_rest(self, sim):
        """A refusal must not half-apply the state it declined.

        Pre-fix the saved hinge value was written into the new scene's hinge,
        which swings the arm about a different axis: the arm ended up in a pose
        no checkpoint ever recorded.
        """
        assert sim.replace_scene_mjcf(_PAN_SCENE)["status"] == "success"
        _set_swing(sim, _SAVED_SWING)
        panned_arm = _arm_position(sim)
        assert sim.save_state(name="panned")["status"] == "success"

        assert sim.replace_scene_mjcf(_LIFT_SCENE)["status"] == "success"
        at_rest = _arm_position(sim)

        assert sim.load_state(name="panned")["status"] == "error"

        assert _arm_position(sim) == pytest.approx(at_rest), "the new scene must be untouched by a refused checkpoint"
        # And the pose the stale vector would have produced is a real, different
        # place - so the assertion above is not vacuous.
        assert _arm_position(sim) != pytest.approx(panned_arm)

    def test_a_checkpoint_saved_after_the_replacement_still_loads(self, sim):
        """The refusal is scoped to checkpoints older than the swap."""
        assert sim.replace_scene_mjcf(_PAN_SCENE)["status"] == "success"
        assert sim.replace_scene_mjcf(_LIFT_SCENE)["status"] == "success"

        _set_swing(sim, 0.5)
        lifted = _arm_position(sim)
        assert sim.save_state(name="lifted")["status"] == "success"

        _set_swing(sim, 0.0)
        assert _arm_position(sim) != pytest.approx(lifted)

        assert sim.load_state(name="lifted")["status"] == "success"
        assert _arm_position(sim) == pytest.approx(lifted)


class TestCameraRemovalMatchesCameraAddition:
    def test_remove_camera_invalidates_a_checkpoint_exactly_as_add_camera_does(self, sim):
        """Both directions swap the model and neither changes the state size.

        Pre-fix ``add_camera`` refused the checkpoint and ``remove_camera``
        accepted it - the same checkpoint, opposite verdicts, for two halves of
        one operation.
        """
        assert sim.add_camera(name="look", position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])["status"] == "success"
        assert sim.save_state(name="two_cams")["status"] == "success"
        size_before = _state_size(sim)

        added = sim.add_camera(name="second", position=[0.0, 2.0, 1.0], target=[0.0, 0.0, 0.0])
        assert added["status"] == "success"
        assert _state_size(sim) == size_before, "adding a camera must not resize the state vector"
        after_add = sim.load_state(name="two_cams")

        assert sim.save_state(name="three_cams")["status"] == "success"
        assert sim.remove_camera("second")["status"] == "success"
        assert _state_size(sim) == size_before, "removing a camera must not resize the state vector"
        after_remove = sim.load_state(name="three_cams")

        assert after_add["status"] == "error"
        assert after_remove["status"] == after_add["status"], (
            "adding and removing a camera must reach the same verdict on a "
            f"checkpoint taken before it (add={after_add['status']}, "
            f"remove={after_remove['status']})"
        )
        assert "stale" in after_remove["content"][0]["text"].lower()

    def test_a_checkpoint_saved_after_the_removal_still_loads(self, sim):
        assert sim.add_camera(name="look", position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])["status"] == "success"
        assert sim.remove_camera("look")["status"] == "success"
        assert sim.save_state(name="fresh")["status"] == "success"
        assert sim.load_state(name="fresh")["status"] == "success"


class TestPatchSceneMjcfInvalidatesCheckpoints:
    def test_a_patch_that_adds_no_body_still_invalidates_a_checkpoint(self, sim):
        """``add_geom`` keeps the state size and changes what the body describes.

        A body without an explicit ``<inertial>`` derives its mass and inertia
        from the geoms it owns, so this patch produces a different model at the
        same state indices. Pre-fix nothing distinguished the two.
        """
        assert (
            sim.add_object(name="cube", shape="box", size=[0.1, 0.1, 0.1], position=[0.0, 0.0, 0.5])["status"]
            == "success"
        )
        size_before = _state_size(sim)
        assert sim.save_state(name="one_geom")["status"] == "success"

        patched = sim.patch_scene_mjcf(
            ops=[{"op": "add_geom", "body": "cube", "type": "sphere", "size": [0.04], "name": "bump"}]
        )
        assert patched["status"] == "success"
        assert _state_size(sim) == size_before, (
            "fixture must keep the state size identical, otherwise the size check "
            "refuses the checkpoint and the generation is untested"
        )

        result = sim.load_state(name="one_geom")
        assert result["status"] == "error"
        assert "stale" in result["content"][0]["text"].lower()

    def test_a_checkpoint_saved_after_the_patch_still_loads(self, sim):
        assert (
            sim.add_object(name="cube", shape="box", size=[0.1, 0.1, 0.1], position=[0.0, 0.0, 0.5])["status"]
            == "success"
        )
        assert (
            sim.patch_scene_mjcf(
                ops=[{"op": "add_geom", "body": "cube", "type": "sphere", "size": [0.04], "name": "bump"}]
            )["status"]
            == "success"
        )
        assert sim.save_state(name="fresh")["status"] == "success"
        assert sim.load_state(name="fresh")["status"] == "success"


# The live model/data pair a checkpoint is fingerprinted against.
_LIVE_MODEL_ATTRS = frozenset({"_model", "_data"})

# The one function allowed to install them, which is also what bumps the
# generation. Any other site would be a swap that leaves the fingerprint behind.
_INSTALLER = "install_compiled_model"


def _assignment_sites(source: str) -> dict[str, list[int]]:
    """Map enclosing function name -> lines assigning a live model attribute.

    Attributes assignments to :data:`_LIVE_MODEL_ATTRS` are reported against the
    innermost enclosing function (``"<module>"`` for a top-level statement), so a
    nested helper is never charged to its parent.
    """
    sites: dict[str, list[int]] = {}

    def visit(node: ast.AST, enclosing: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                visit(child, child.name)
                continue
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    elements = target.elts if isinstance(target, ast.Tuple) else [target]
                    for element in elements:
                        if isinstance(element, ast.Attribute) and element.attr in _LIVE_MODEL_ATTRS:
                            sites.setdefault(enclosing, []).append(child.lineno)
            visit(child, enclosing)

    visit(ast.parse(source), "<module>")
    return sites


class TestEverySwapSiteGoesThroughTheInstaller:
    def test_only_the_installer_assigns_the_live_model(self):
        """A swap site that rebinds the model directly cannot bump the generation.

        The generation is the only part of the fingerprint that changes when a
        swap keeps every count, so a site that assigns ``_model`` outside the
        installer silently accepts checkpoints taken against the previous model.
        Scoped to the MuJoCo backend: it is the only backend whose ``SimWorld``
        carries a compiled model and data pair.
        """
        backend = Path(inspect.getfile(install_compiled_model)).parent
        modules = sorted(backend.glob("*.py"))
        assert modules, f"no backend modules found under {backend.name}"

        offenders: dict[str, dict[str, list[int]]] = {}
        for module in modules:
            found = _assignment_sites(module.read_text(encoding="utf-8"))
            extra = {name: lines for name, lines in found.items() if name != _INSTALLER}
            if extra:
                offenders[module.name] = extra

        assert not offenders, (
            f"these sites assign the live model outside {_INSTALLER}(), so a "
            f"checkpoint taken against the previous model would still be "
            f"accepted: {offenders}"
        )

    def test_the_installer_is_the_site_the_scan_finds(self):
        """Guard against a scan that matches nothing and therefore proves nothing."""
        source = Path(inspect.getfile(install_compiled_model)).read_text(encoding="utf-8")
        assert _INSTALLER in _assignment_sites(source)

    def test_a_planted_swap_site_is_detected(self):
        """The scan must fail for a site it is meant to catch."""
        planted = "def swap_it(world, model, data):\n    world._model = model\n    world._data = data\n"
        assert _assignment_sites(planted) == {"swap_it": [2, 3]}

        tuple_form = "def swap_it(world, spec):\n    world._model, world._data = spec.recompile()\n"
        assert _assignment_sites(tuple_form) == {"swap_it": [2, 2]}
