"""A converted MJCF's floating base is recorded, so ``base_*`` reaches the robots that have one.

A composition-only defect, and the most instructive one in this series: two changes
that are each correct alone combine into a bug, with no textual conflict between
them.

* One added ``fixed_base`` to ``_RobotState`` and made ``get_observation`` emit
  ``base_pos`` / ``base_quat`` / ``base_lin_vel`` / ``base_ang_vel`` only for a
  robot whose base is free. On that branch the only way to get a floating base is
  ``add_robot(urdf_path=..., fix_base=False)``, and it records what the caller
  passed.
* The other made ``add_robot`` resolve a robot name through the registry, convert
  the resulting MJCF to USD, and load it by the native USD path. On that branch
  there is no ``fixed_base`` field at all.

Composed, the USD construction site records the ``_RobotState`` **default** -
``fixed_base=True`` - for every converted MJCF. So a registry humanoid or quadruped
lands on the stage with a genuinely free root, because its MJCF declares
``<freejoint/>``, and is reported as bolted down. ``get_observation`` then omits all
four ``base_*`` keys for exactly the robots whose base is the thing a locomotion
policy is conditioned on. 18 of the shipped registry's 64 MJCF robots are humanoids
(``unitree_g1``, ``unitree_h1``, ``talos``, ``cassie``, ``op3`` and the rest), plus
9 mobile bases and 2 aerial.

The answer is in the file rather than in a flag, and that asymmetry is the whole
design: URDF **cannot** declare a floating base - its universal convention for a
mobile robot is a root link with no parent joint, byte-identical to how a bolted
arm declares its base - so that path has to ask the caller. MJCF **can**, with
``<freejoint/>`` or ``<joint type="free">``, so asking the caller there would be
asking them to restate what they already handed over.

A plain USD asset keeps the fixed-base default: this backend did not import it, so
the caller's ``fix_base`` describes nothing about it and the asset's own
articulation root is the only truth - which is why ``add_robot`` refuses
``fix_base=False`` on that path rather than recording a claim it cannot check.
"""

from __future__ import annotations

import os
import tempfile
import threading
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac import simulation as sim_mod  # noqa: E402
from strands_robots.simulation.isaac.config import IsaacConfig  # noqa: E402
from strands_robots.simulation.isaac.loaders import mjcf_declares_floating_base  # noqa: E402
from strands_robots.simulation.isaac.simulation import IsaacSimulation  # noqa: E402

#: The four entries the ``SimEngine.get_observation`` schema reserves for a base.
_BASE_KEYS = ("base_pos", "base_quat", "base_lin_vel", "base_ang_vel")

#: A legged robot the way every quadruped and humanoid in the corpus declares one.
_LEGGED = """<mujoco model="legged">
  <worldbody>
    <body name="torso" pos="0 0 0.8">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="thigh">
        <joint name="hip" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.03 0.15"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""

#: The same model bolted down - the only difference is the root joint.
_BOLTED = _LEGGED.replace("      <freejoint/>\n", "")

#: MJCF's other spelling of the same joint, which MuJoCo compiles identically.
_LEGGED_TYPE_FREE = _LEGGED.replace("<freejoint/>", '<joint type="free"/>')


class _Articulation:
    dof_names = ["hip"]

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        return None

    def get_joint_positions(self) -> Any:
        return np.zeros(1, dtype=np.float32)

    def get_world_pose(self) -> Any:
        return (
            np.array([0.0, 0.0, 0.8], dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    def get_linear_velocity(self) -> Any:
        return np.zeros(3, dtype=np.float32)

    def get_angular_velocity(self) -> Any:
        return np.zeros(3, dtype=np.float32)


def _write(xml: str, name: str = "robot.xml") -> str:
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(xml)
    return path


def _engine() -> Any:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(render_mode="headless")
    engine._world = types.SimpleNamespace()
    engine._world_created = True
    engine._robots = {}
    engine._objects = {}
    engine._cameras = {}
    engine._prim_registry = []
    engine._action_controllers = {}
    engine._replicated = False
    engine._recording_state_dict = {}
    engine._main_tid = threading.get_ident()
    engine._pump_running = False
    # The USD leaf is stood in: what these grade is what ``fixed_base`` RECORDS,
    # not the stage authoring, which needs Kit.
    engine._load_usd_robot = lambda prim_path, usd_path, position: (  # type: ignore[method-assign]
        ["hip"],
        _Articulation(),
    )
    return engine


@pytest.fixture
def stub_conversion(monkeypatch) -> None:
    """Stand in for the MJCF -> USD conversion, which needs a running Kit."""
    monkeypatch.setattr(sim_mod, "convert_mjcf_to_usd", lambda path, **kwargs: f"{path}.usd", raising=True)


class TestTheBaseIsReadFromTheMjcf:
    def test_a_legged_mjcf_is_recorded_as_floating(self, stub_conversion: None) -> None:
        engine = _engine()

        result = engine.add_robot("bot", mjcf_path=_write(_LEGGED))

        assert result["status"] == "success", result
        assert engine._robots["bot"].fixed_base is False

    def test_a_bolted_mjcf_is_recorded_as_fixed(self, stub_conversion: None) -> None:
        """The control. Without it, "always floating" would pass the case above."""
        engine = _engine()

        result = engine.add_robot("bot", mjcf_path=_write(_BOLTED))

        assert result["status"] == "success", result
        assert engine._robots["bot"].fixed_base is True

    def test_both_mjcf_spellings_of_a_free_joint_are_read(self, stub_conversion: None) -> None:
        """MuJoCo compiles ``<freejoint/>`` and ``<joint type="free">`` to the same
        joint, and a reader that consults only one reports no base for the models
        that use the other."""
        engine = _engine()

        assert engine.add_robot("bot", mjcf_path=_write(_LEGGED_TYPE_FREE))["status"] == "success"

        assert engine._robots["bot"].fixed_base is False


class TestTheObservationFollows:
    """The consequence, which is the reason the recording matters at all."""

    def test_a_legged_robot_reports_every_base_key(self, stub_conversion: None) -> None:
        engine = _engine()
        assert engine.add_robot("bot", mjcf_path=_write(_LEGGED))["status"] == "success"

        obs = engine.get_observation("bot")

        for key in _BASE_KEYS:
            assert key in obs, f"{key} missing from {sorted(obs)}"

    def test_a_bolted_robot_reports_none_of_them(self, stub_conversion: None) -> None:
        engine = _engine()
        assert engine.add_robot("bot", mjcf_path=_write(_BOLTED))["status"] == "success"

        obs = engine.get_observation("bot")

        assert [k for k in obs if k.startswith("base_")] == []
        assert "hip" in obs, "joint state is owed either way"


class TestThePredicateReadsWhatMujocoReads:
    """The vocabulary has one owner in ``loaders``, which already parses MJCF."""

    @pytest.mark.parametrize(
        ("xml", "expected"),
        [
            (_LEGGED, True),
            (_LEGGED_TYPE_FREE, True),
            (_BOLTED, False),
            (_LEGGED.replace("<freejoint/>", '<joint type="hinge" axis="0 0 1"/>'), False),
            (_LEGGED.replace("<freejoint/>", '<freejoint name="floating_base_joint"/>'), True),
        ],
        # Named, because the parameter is a whole XML document: the default ids
        # inline the model into the test name and make a failure unreadable.
        ids=["freejoint-element", "joint-type-free", "no-root-joint", "hinge-root", "named-freejoint"],
    )
    def test_the_root_joint_decides(self, xml: str, expected: bool) -> None:
        assert mjcf_declares_floating_base(_write(xml)) is expected

    def test_a_free_joint_deeper_in_the_tree_is_not_the_base(self) -> None:
        """A free-flying CHILD is a MuJoCo idiom for a detached payload. Reporting
        it as a floating base would put ``base_pos`` on a robot bolted to a table."""
        xml = """<mujoco model="table">
          <worldbody>
            <body name="table" pos="0 0 0.4">
              <geom type="box" size="0.5 0.5 0.02"/>
              <body name="payload"><freejoint/><geom type="sphere" size="0.05"/></body>
            </body>
          </worldbody>
        </mujoco>"""

        assert mjcf_declares_floating_base(_write(xml)) is False

    def test_a_commented_out_free_joint_is_not_read(self) -> None:
        """A commented-out ``type="free"`` must not count, and this is not
        hypothetical: the shipped ``rby1`` model carries exactly that line -
        ``<!-- <joint name="world_j" type="free" .../> -->`` - above a base that is
        genuinely PLANAR (``joint_x`` slide, ``joint_y`` slide, ``joint_th``
        hinge). A text scan for the attribute would report that wheeled base as
        floating and put ``base_pos`` on it. Reading the parsed tree gets it right
        for free, because ElementTree discards comments.
        """
        xml = """<mujoco model="planar">
          <worldbody>
            <body name="base" pos="0 0 0.1">
              <!-- <joint name="world_j" type="free" limited="false"/> -->
              <joint name="joint_x" type="slide" axis="1 0 0"/>
              <joint name="joint_y" type="slide" axis="0 1 0"/>
              <joint name="joint_th"/>
              <geom type="box" size="0.2 0.2 0.05"/>
            </body>
          </worldbody>
        </mujoco>"""

        assert mjcf_declares_floating_base(_write(xml)) is False

    def test_a_planar_base_is_not_a_floating_base(self) -> None:
        """Three prismatic/hinge joints are a wheeled base, not a 6-DoF root, and
        the ``base_*`` schema entries describe the latter."""
        xml = _LEGGED.replace(
            "<freejoint/>",
            '<joint name="jx" type="slide" axis="1 0 0"/><joint name="jy" type="slide" axis="0 1 0"/>',
        )

        assert mjcf_declares_floating_base(_write(xml)) is False

    def test_a_root_body_in_an_included_fragment_is_read(self) -> None:
        """MuJoCo splices ``<include>`` textually, so a root body need not live in
        the named file - which is how the shipped corpus is organised."""
        directory = tempfile.mkdtemp()
        with open(os.path.join(directory, "frag.xml"), "w", encoding="utf-8") as handle:
            handle.write(f"<mujoco>{_LEGGED.split('<mujoco model="legged">')[1]}")
        top = os.path.join(directory, "top.xml")
        with open(top, "w", encoding="utf-8") as handle:
            handle.write('<mujoco model="m"><include file="frag.xml"/></mujoco>')

        assert mjcf_declares_floating_base(top) is True

    @pytest.mark.parametrize("path", ["/nonexistent/robot.xml"])
    def test_an_unreadable_file_answers_false_rather_than_raising(self, path: str) -> None:
        """The caller is deciding whether to REPORT four keys. A robot that loads
        with no ``base_*`` is a smaller error than an ``add_robot`` that refuses
        over a file MuJoCo itself may accept, and the load path parses the same
        file immediately afterwards and reports any real problem there."""
        assert mjcf_declares_floating_base(path) is False

    def test_a_malformed_file_answers_false_rather_than_raising(self) -> None:
        assert mjcf_declares_floating_base(_write("<mujoco><worldbody>")) is False


class TestTheOtherPathsAreUnchanged:
    """The asymmetry between the three loaders is deliberate, so it is pinned."""

    def test_a_urdf_import_still_takes_the_callers_flag(self) -> None:
        """URDF cannot declare a floating base, so that path asks."""
        import inspect

        assert "fix_base" in inspect.signature(IsaacSimulation.add_robot).parameters
        assert inspect.signature(IsaacSimulation.add_robot).parameters["fix_base"].default is True

    def test_a_plain_usd_import_still_refuses_a_free_base(self) -> None:
        """This backend did not import the asset, so the caller's flag describes
        nothing about it - refusing beats recording a claim it cannot check."""
        engine = _engine()

        result = engine.add_robot("bot", usd_path="/assets/robot.usda", fix_base=False)

        assert result["status"] == "error", result
        assert "fix_base=False" in result["content"][0]["text"]

    def test_a_plain_usd_import_records_a_fixed_base(self, stub_conversion: None) -> None:
        """And with the default it records fixed, because nothing says otherwise."""
        engine = _engine()

        assert engine.add_robot("bot", usd_path="/assets/robot.usda")["status"] == "success"

        assert engine._robots["bot"].fixed_base is True
