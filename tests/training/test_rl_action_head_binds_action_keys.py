"""The RL action head is sized and named from the robot's action keys.

``SimEnv.step`` sends a numeric **vector**, and ``SimEngine.send_action`` binds a
vector positionally to ``robot_action_keys(robot_name)`` - not to
``robot_joint_names``. The base class says so in both directions:
``robot_joint_names`` is documented as naming the ``observation.state`` vector
with "Action-vector binding ... uses :meth:`robot_action_keys` instead", and
``robot_action_keys`` warns that "a caller must not assume it has the same width
as :meth:`robot_joint_names`".

The two lists coincide only when a robot's actuator set matches its joint set.
Two shipped shapes make them disagree:

* a **tendon-driven gripper** - one actuator over two mimic finger joints, so the
  actuator names are not joint names at all. The builtin MuJoCo ``panda`` has
  nine joints and eight action keys.
* a **floating base** on the Newton backend - a 6-DoF free joint that is a joint
  with no commandable scalar, so the action keys are the joint names minus one.

Sizing the head from the joint list on either shape produced a vector one wider
than ``send_action`` binds, which every backend refuses with a structured error.
``SimEnv.step`` does not read that result, so the refusal was invisible: no
target was written, the world never advanced, and the reward term was still
evaluated and banked. A rollout of any length could be collected for a robot
that never moved.

The checkpoint metadata is the same contract on the deployment side. It carries
``num_actions`` and, beside it, the names those outputs drive; a list of joint
names cannot serve that purpose when the widths differ, so it names the action
keys and ``len(action_keys) == num_actions`` holds.

The unit-level engine here is a purpose-built double rather than a backend, so
these pins run on every CI job. The behavioural half uses the real MuJoCo
``panda`` and the narrowing half the real ``NewtonSimEngine``, built solver-free
via ``__new__`` the way ``tests/simulation/newton/test_free_base_is_not_an_actuator.py``
does.
"""

from __future__ import annotations

import ast
import inspect
import threading
from pathlib import Path
from typing import Any

import pytest
import torch

import strands_robots.training.rl as rl_pkg
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.models import SimRobot, SimWorld
from strands_robots.training.rl.env import SimEnv

_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
_FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
_ARM_ACTUATORS = [f"actuator{i}" for i in range(1, 8)]
_GRASP_TENDON = "actuator8"


class _TwoVocabEngine(SimEngine):
    """A concrete engine whose action keys are not its joint names.

    Models the tendon-gripper shape: two mimic finger joints driven by one
    tendon actuator, so the actuator vocabulary is both differently spelled and
    one narrower than the joint list. ``send_action`` reproduces the width rule
    every backend publishes - a vector must match the action-key count - so a
    mis-sized head is refused here exactly as it is on a real backend, without
    needing one.
    """

    def __init__(self, *, joints: list[str], action_keys: list[str]) -> None:
        self._joints = list(joints)
        self._action_keys = list(action_keys)
        self.applied: list[list[float]] = []
        self.refused: list[int] = []

    # The action vocabulary under test.
    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._joints)

    def robot_action_keys(self, robot_name: str) -> list[str]:
        return list(self._action_keys)

    def list_robots(self) -> list[str]:
        return ["arm"]

    def get_observation(self, robot_name: str | None = None, skip_images: bool = False) -> dict[str, Any]:
        return {j: 0.0 for j in self._joints} | {f"{j}.vel": 0.0 for j in self._joints}

    def send_action(
        self, action: Any, robot_name: str | None = None, n_substeps: int = 1, **kwargs: Any
    ) -> dict[str, Any]:
        width = len(action)
        if width != len(self._action_keys):
            self.refused.append(width)
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"send_action: action vector length {width} does not match robot "
                            f"'arm' action-key count {len(self._action_keys)}."
                        )
                    }
                ],
            }
        self.applied.append([float(v) for v in action])
        return {"status": "success", "content": [{"text": f"Action applied to 'arm' ({width} keys)."}]}

    # Inert scaffolding: present only so the class is concrete. Permissive
    # signatures keep them Liskov-safe without restating each contract.
    def add_object(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    def add_robot(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    def create_world(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    def destroy(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    def get_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    def remove_object(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    def remove_robot(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    def render(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    def reset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    def step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}


def _env(engine: SimEngine, **kwargs: Any) -> SimEnv:
    return SimEnv(engine, actor_obs_keys=["joint1"], reward_terms=[lambda _e: 1.0], robot_name="arm", **kwargs)


class TestTheHeadIsSizedFromTheActionKeys:
    """``num_actions`` is the count ``send_action`` binds a vector against."""

    def test_the_default_width_is_the_action_key_count(self) -> None:
        engine = _TwoVocabEngine(joints=_ARM_JOINTS + _FINGER_JOINTS, action_keys=_ARM_ACTUATORS + [_GRASP_TENDON])
        # Premise: the two vocabularies really do disagree, so a width taken
        # from the wrong one is observable rather than accidentally correct.
        assert len(engine.robot_joint_names("arm")) == 9
        assert len(engine.robot_action_keys("arm")) == 8
        assert _env(engine).num_actions == 8

    def test_the_default_width_is_accepted_by_send_action(self) -> None:
        engine = _TwoVocabEngine(joints=_ARM_JOINTS + _FINGER_JOINTS, action_keys=_ARM_ACTUATORS + [_GRASP_TENDON])
        env = _env(engine)
        env.step(torch.zeros(1, env.num_actions))
        assert engine.refused == []
        assert len(engine.applied) == 1

    def test_a_narrower_action_list_is_honored(self) -> None:
        """The floating-base shape: same spelling, one fewer key."""
        engine = _TwoVocabEngine(joints=["free_base"] + _ARM_JOINTS, action_keys=list(_ARM_JOINTS))
        env = _env(engine)
        assert env.num_actions == len(_ARM_JOINTS)
        env.step(torch.zeros(1, env.num_actions))
        assert engine.refused == []

    def test_an_explicit_action_dim_still_wins(self) -> None:
        """The override is the escape hatch for a robot the default mis-sizes."""
        engine = _TwoVocabEngine(joints=_ARM_JOINTS + _FINGER_JOINTS, action_keys=_ARM_ACTUATORS + [_GRASP_TENDON])
        assert _env(engine, action_dim=3).num_actions == 3

    def test_a_robot_whose_actuators_match_its_joints_is_unchanged(self) -> None:
        """The overwhelmingly common shape: the two lists already agreed."""
        engine = _TwoVocabEngine(joints=list(_ARM_JOINTS), action_keys=list(_ARM_JOINTS))
        assert _env(engine).num_actions == len(_ARM_JOINTS)

    def test_no_robot_and_no_action_dim_is_still_refused(self) -> None:
        engine = _TwoVocabEngine(joints=[], action_keys=[])
        engine.list_robots = lambda: []  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="action_dim must be given"):
            SimEnv(engine, actor_obs_keys=["joint1"], reward_terms=[lambda _e: 1.0])


class TestARefusedActionWritesNoTarget:
    """Why the mis-sized head was silent: ``step`` does not read the result."""

    def test_a_mis_sized_head_banks_reward_without_applying_anything(self) -> None:
        engine = _TwoVocabEngine(joints=_ARM_JOINTS + _FINGER_JOINTS, action_keys=_ARM_ACTUATORS + [_GRASP_TENDON])
        env = _env(engine, action_dim=9)  # the width the joint list would have given
        total = 0.0
        for _ in range(20):
            _obs, reward, done, _info = env.step(torch.zeros(1, 9))
            total += float(reward.item())
            assert not bool(done.item())
        # Every action was refused, yet a full rollout of reward was collected.
        assert engine.applied == []
        assert engine.refused == [9] * 20
        assert total == 20.0


class TestAFloatingBaseRobotOnTheNewtonBackend:
    """The real ``NewtonSimEngine``, solver-free, excludes the free root."""

    @staticmethod
    def _engine() -> Any:
        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        base, scalars = "floating_base_joint", ["hip_yaw", "hip_pitch", "knee", "ankle"]
        world = SimWorld()
        world.robots["g1"] = SimRobot(name="g1", urdf_path="g1.xml", data_config="g1", joint_names=[base] + scalars)
        engine = NewtonSimEngine.__new__(NewtonSimEngine)
        engine._world = world
        engine._model = object()  # non-None sentinel: "world created"
        engine.default_width, engine.default_height = 64, 48
        engine._robot_free_base_joint = {"g1": base}
        engine._lock = threading.RLock()
        engine._targets = {}
        engine._write_targets = lambda: None  # type: ignore[method-assign]
        engine._advance = lambda n_steps: None  # type: ignore[method-assign]

        def _observe(robot_name: str | None = None, skip_images: bool = False) -> dict[str, Any]:
            # The scalar observation the backend really emits: the free joint is
            # absent from it, which is the disagreement under test.
            return {j: 0.0 for j in scalars}

        engine.get_observation = _observe  # type: ignore[method-assign]
        return engine, scalars

    def test_the_head_excludes_the_free_joint(self) -> None:
        engine, scalars = self._engine()
        assert len(engine.robot_joint_names("g1")) == len(scalars) + 1
        env = SimEnv(engine, actor_obs_keys=scalars[:1], reward_terms=[lambda _e: 1.0], robot_name="g1")
        assert env.num_actions == len(scalars)

    def test_a_default_sized_action_is_applied(self) -> None:
        engine, scalars = self._engine()
        env = SimEnv(engine, actor_obs_keys=scalars[:1], reward_terms=[lambda _e: 1.0], robot_name="g1", n_substeps=1)
        env.step(torch.zeros(1, env.num_actions))
        assert set(k[1] for k in engine._targets) == set(scalars)


class TestATendonGripperArmIsDriven:
    """End to end on the default backend with a builtin registry robot."""

    @pytest.fixture
    def panda(self):  # type: ignore[no-untyped-def]
        mujoco = pytest.importorskip("mujoco")
        assert mujoco is not None
        from strands_robots import Simulation

        sim = Simulation(backend="mujoco", tool_name="rl_action_head_sim", mesh=False)
        sim.create_world()
        assert sim.add_robot(name="panda", data_config="panda")["status"] == "success"
        yield sim
        sim.cleanup()

    def test_the_two_vocabularies_disagree_on_this_robot(self, panda) -> None:  # type: ignore[no-untyped-def]
        """Premise for the two tests below, on the shipped asset itself."""
        assert len(panda.robot_joint_names("panda")) == 9
        assert len(panda.robot_action_keys("panda")) == 8

    def test_a_default_sized_head_drives_the_arm(self, panda) -> None:  # type: ignore[no-untyped-def]
        obs = panda.get_observation(robot_name="panda", skip_images=True)
        env = SimEnv(
            panda,
            actor_obs_keys=["joint1"],
            reward_terms=[lambda _e: 1.0],
            robot_name="panda",
            n_substeps=10,
            max_episode_steps=500,
        )
        assert env.num_actions == 8
        start = float(obs["joint1"])
        action = torch.zeros(1, env.num_actions)
        action[0, 0] = 0.9
        for _ in range(60):
            env.step(action)
        end = float(panda.get_observation(robot_name="panda", skip_images=True)["joint1"])
        # Sized from the joint list this vector is refused and joint1 never
        # leaves 0.0 however many steps are taken.
        assert abs(end - start) > 0.5

    def test_the_joint_list_width_moves_nothing(self, panda) -> None:  # type: ignore[no-untyped-def]
        """The consequence, on the real backend, of the width this used to take.

        Holds on either side of the fix - it characterises ``send_action``'s
        width rule and ``step``'s silence rather than the sizing - and is the
        reproduction the sizing test above is the fix for.
        """
        wide = len(panda.robot_joint_names("panda"))
        env = SimEnv(
            panda,
            actor_obs_keys=["joint1"],
            reward_terms=[lambda _e: 1.0],
            robot_name="panda",
            action_dim=wide,
            n_substeps=10,
            max_episode_steps=500,
        )
        start = float(panda.get_observation(robot_name="panda", skip_images=True)["joint1"])
        action = torch.zeros(1, wide)
        action[0, 0] = 0.9
        banked = 0.0
        for _ in range(60):
            _obs, reward, done, _info = env.step(action)
            banked += float(reward.item())
            assert not bool(done.item())
        end = float(panda.get_observation(robot_name="panda", skip_images=True)["joint1"])
        # Sixty steps, every one refused, and the reward collected regardless.
        assert end == start
        assert banked == 60.0
        assert panda.send_action([0.0] * wide, robot_name="panda", n_substeps=1)["status"] == "error"

    def test_send_action_accepts_the_default_width(self, panda) -> None:  # type: ignore[no-untyped-def]
        env = SimEnv(panda, actor_obs_keys=["joint1"], reward_terms=[lambda _e: 1.0], robot_name="panda")
        result = panda.send_action([0.0] * env.num_actions, robot_name="panda", n_substeps=1)
        assert result["status"] == "success"


def _training_modules() -> list[Path]:
    """Every module of the training package, rooted at an imported symbol."""
    root = Path(inspect.getfile(rl_pkg)).resolve().parent.parent
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _joint_name_reads(source: str) -> list[str]:
    """Names of functions calling ``robot_joint_names``, at any receiver."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef):
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                if call.func.attr == "robot_joint_names":
                    found.append(node.name)
    return found


class TestNoTrainingModuleReadsTheJointList:
    """A joint list cannot size or name an action, so nothing here reads one.

    Stated structurally rather than left to review: the package's every use of
    a robot's joint vocabulary was an action-shaped one, so the honest rule is
    that it reads the action vocabulary instead. A module that genuinely needs
    the joint list - to name an observation, say - will fail this and should
    say why in the same change rather than reintroducing the read silently.
    """

    def test_no_training_module_reads_the_joint_list(self) -> None:
        offenders = {
            path.name: names
            for path in _training_modules()
            if (names := _joint_name_reads(path.read_text(encoding="utf-8")))
        }
        assert offenders == {}, (
            f"these training functions read robot_joint_names: {offenders}; an action vector binds to robot_action_keys"
        )

    def test_the_scanner_detects_a_planted_read(self) -> None:
        planted = "def f(self):\n    return len(self.engine.robot_joint_names('r'))\n"
        assert _joint_name_reads(planted) == ["f"]

    def test_the_scan_root_is_the_training_package(self) -> None:
        names = {p.name for p in _training_modules()}
        assert {"env.py", "ppo.py", "fast_sac.py", "vec_env.py", "gym_env.py"} <= names


def _metadata_action_key_reads(source: str) -> set[str]:
    """Functions building a metadata dict with an ``action_keys`` entry."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Dict):
                continue
            keys = {k.value for k in sub.keys if isinstance(k, ast.Constant)}
            if {"num_actions", "action_keys"} <= keys:
                found.add(node.name)
    return found


class TestTheCheckpointNamesWhatTheHeadDrives:
    """Both trainers write the action vocabulary beside ``num_actions``."""

    @pytest.mark.parametrize("module", ["ppo.py", "fast_sac.py"])
    def test_the_metadata_carries_action_keys(self, module: str) -> None:
        path = next(p for p in _training_modules() if p.name == module)
        assert _metadata_action_key_reads(path.read_text(encoding="utf-8"))

    def test_the_metadata_scanner_detects_a_missing_entry(self) -> None:
        planted = 'def save(self):\n    return {"num_actions": 1, "joint_names": []}\n'
        assert _metadata_action_key_reads(planted) == set()
