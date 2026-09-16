"""``render_video.py`` places the ball where the kick weights were trained to find it.

``scene_ball.xml`` declares its ball 0.3 m straight ahead of the duck. The two
kick weights were trained with the ball 0.09 m ahead and 0.042 m to the side of
the kicking foot, in the robot's yaw frame (Pollen's ``microduck_rl``
``scripts/infer_policy.py``: ``BALL_OFFSET_X``, ``BALL_OFFSET_ABS_Y``, and the
``_place_ball`` teleport its runtime performs before every kick).
``docs/policies/microduck.md`` has said so since the ball-scene placement fix,
and the example's own docstring recipe - ``--onnx ball_kick_left.onnx --scene
scene_ball.xml`` - went on rendering four seconds of the duck kicking air.

Measured in MuJoCo 3.13 with ``ball_kick_left.onnx``, 4 s at 50 Hz, before and
after the placement this file pins - the gap is the smallest surface distance
between any robot geom and the ball over the rollout (``mj_geomDistance``)::

    declared (0.300, 0.000): ball travel 0.000 m, min robot-geom gap  0.189 m, 0 contact ticks
    placed   (0.090, 0.042): ball travel 0.485 m, min robot-geom gap -0.004 m, 1 contact tick

These cells exercise the placement on a small MuJoCo scene of their own (a
trunk with a free joint at an arbitrary yaw, and a ball), so they need
``mujoco`` but no GL, no asset download and no ONNX; the foot inference and the
command line are checked without ``mujoco`` at all.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "microduck" / "render_video.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_video_kick_under_test", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene(yaw: float, with_ball: bool = True) -> str:
    qw, qz = np.cos(yaw / 2), np.sin(yaw / 2)
    ball = (
        '<body name="microduck/ball" pos="0.3 0 0.035"><freejoint name="microduck/ball_free"/>'
        '<geom type="sphere" size="0.035" mass="0.015"/></body>'
        if with_ball
        else ""
    )
    return f"""
<mujoco>
  <worldbody>
    <geom type="plane" size="2 2 0.1"/>
    <body name="microduck/trunk_base" pos="0.5 -0.2 0.12" quat="{qw} 0 0 {qz}">
      <freejoint name="microduck/trunk_base_freejoint"/>
      <geom type="box" size="0.03 0.03 0.03" mass="0.5"/>
    </body>
    {ball}
  </worldbody>
</mujoco>"""


@pytest.fixture
def mujoco():
    return pytest.importorskip("mujoco")


class TestThePlacement:
    @pytest.mark.parametrize("foot, sign", [("left", 1.0), ("right", -1.0)])
    def test_the_ball_lands_at_the_trained_offset_in_the_trunks_yaw_frame(self, mujoco, foot, sign) -> None:
        module = _load()
        yaw = 0.7
        model = mujoco.MjModel.from_xml_string(_scene(yaw))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        placed = module.place_ball_for_kick(mujoco, model, data, foot)

        off = np.array([module.BALL_OFFSET_X, sign * module.BALL_OFFSET_ABS_Y])
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        expected = np.array([0.5, -0.2]) + rot @ off
        assert placed is not None
        assert np.allclose(placed, expected, atol=1e-9)
        # In the trunk frame: 0.09 ahead, 0.042 toward the kicking foot.
        rel = rot.T @ (np.array(placed) - np.array([0.5, -0.2]))
        assert np.allclose(rel, [0.09, sign * 0.042], atol=1e-9)

    def test_the_state_is_written_and_the_kinematics_recomputed(self, mujoco) -> None:
        module = _load()
        model = mujoco.MjModel.from_xml_string(_scene(0.0))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "microduck/ball_free")
        qadr, vadr = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
        data.qvel[vadr : vadr + 6] = 1.0  # a rolling ball from a previous rollout

        module.place_ball_for_kick(mujoco, model, data, "left")

        assert np.allclose(data.qpos[qadr : qadr + 7], [0.59, -0.158, 0.035, 1, 0, 0, 0])
        assert np.all(data.qvel[vadr : vadr + 6] == 0.0)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "microduck/ball")
        assert np.allclose(data.xpos[body][:2], [0.59, -0.158]), "mj_forward ran: the frame sees the ball there"

    def test_a_scene_without_a_ball_moves_nothing(self, mujoco) -> None:
        module = _load()
        model = mujoco.MjModel.from_xml_string(_scene(0.0, with_ball=False))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        before = data.qpos.copy()

        assert module.place_ball_for_kick(mujoco, model, data, "left") is None
        assert np.array_equal(data.qpos, before)

    def test_the_ball_joint_is_matched_with_or_without_the_robot_prefix(self, mujoco) -> None:
        module = _load()
        bare = mujoco.MjModel.from_xml_string(_scene(0.0).replace("microduck/ball_free", "ball_free"))
        prefixed = mujoco.MjModel.from_xml_string(_scene(0.0))

        assert module._ball_joint(mujoco, bare) >= 0
        assert module._ball_joint(mujoco, prefixed) >= 0
        assert module._ball_joint(mujoco, mujoco.MjModel.from_xml_string(_scene(0.0, with_ball=False))) == -1


class TestWhichFoot:
    @pytest.mark.parametrize(
        "onnx, expected",
        [
            ("../microduck/policies/ball_kick_left.onnx", "left"),
            ("/weights/ball_kick_right.onnx", "right"),
            ("alpha_walking.onnx", None),
            ("leftover_stand.onnx", None),  # a stem that merely contains the word
        ],
    )
    def test_it_is_read_off_the_weights_file_name(self, onnx, expected) -> None:
        module = _load()
        assert module._kick_foot(SimpleNamespace(onnx=onnx, kick_foot=None)) == expected

    def test_the_flag_wins_over_the_file_name(self) -> None:
        module = _load()
        assert module._kick_foot(SimpleNamespace(onnx="ball_kick_left.onnx", kick_foot="right")) == "right"

    def test_the_command_line_offers_the_flag_with_two_feet(self, monkeypatch) -> None:
        module = _load()
        captured: dict[str, argparse.ArgumentParser] = {}

        def _capture(self, args=None, namespace=None):
            captured["parser"] = self
            raise SystemExit(0)

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _capture)
        with pytest.raises(SystemExit):
            module.main()
        action = next(a for a in captured["parser"]._actions if "--kick-foot" in a.option_strings)
        assert action.choices is not None
        assert tuple(action.choices) == ("left", "right")
        assert action.default is None


class TestWhatItSaysAboutTheBall:
    """One line covers the three outcomes: placed, refused, or kicking air out loud."""

    def test_the_placed_line_names_where_it_went_and_which_foot(self) -> None:
        module = _load()

        line = module._placement_report(SimpleNamespace(kick_foot=None), "left", (0.59, -0.158))

        assert "0.590" in line and "-0.158" in line
        assert "left foot" in line

    def test_a_ball_less_scene_says_the_kick_swings_at_nothing(self) -> None:
        """A foot read off the weight's name is not a request, so this is said, not refused."""
        module = _load()

        line = module._placement_report(SimpleNamespace(kick_foot=None), "left", None)

        assert "swings at nothing" in line
        assert "--scene scene_ball.xml" in line

    def test_the_flag_on_a_ball_less_scene_is_refused_and_names_the_scene(self) -> None:
        module = _load()

        with pytest.raises(SystemExit) as excinfo:
            module._placement_report(SimpleNamespace(kick_foot="right"), "right", None)

        assert "scene_ball.xml" in str(excinfo.value)


class TestTheRecipeSaysSo:
    def test_the_docstring_recipe_explains_the_placement(self) -> None:
        text = _EXAMPLE.read_text(encoding="utf-8")
        assert "0.09 m ahead" in text
        assert "--kick-foot" in text
        assert "ball_kick_left.onnx --scene scene_ball.xml" in text
