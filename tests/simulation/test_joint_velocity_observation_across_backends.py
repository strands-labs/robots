"""Every backend emits the per-joint ``"<name>.vel"`` keys its consumers read.

The ``SimEngine.get_observation`` schema now documents a per-joint velocity
entry, ``"<joint_name>.vel"``, additive beside the position key. It was
previously undocumented at the ABC and lived only in the MuJoCo implementation
(``mujoco/rendering.py``, emitted since #761) - which is exactly how two
backends shipped without it: Isaac read ``get_joint_positions()`` and never
``get_joint_velocities()``, and Newton read ``joint_qd`` into a local for its
floating-base twist and never emitted a scalar-joint entry from it.

The gap was not cosmetic, because the key has consumers that a missing entry
breaks in three different ways, each on a policy that works unchanged on
MuJoCo:

* the WBC balance controller degrades to zero joint velocities with a one-time
  "Gait stability may degrade" warning (``policies/wbc/policy.py``) - open-loop
  on the quantity it exists to feed back;
* the microduck and ProtoMotions observation packers raise ``KeyError``
  (``policies/microduck/observation.py``, ``policies/protomotions/policy.py``);
* an RL ``SimEnv`` with ``.vel`` in its ``actor_obs_keys`` refuses at reset
  (``training/rl/env.py``).

Two adjacent defects are pinned alongside, because the fix exposed them:

* **Newton's ``joint_vel_std`` configured a channel that did not exist.**
  ``set_obs_noise`` accepted and documented it all along, while the noise pass
  (``_apply_joint_pos_noise``) applied ``joint_pos_std`` to every entry it was
  handed. Now that ``.vel`` entries flow through that pass, it splits by suffix
  exactly as MuJoCo's ``_apply_obs_noise`` does - without the split, position
  noise would land on velocities and ``joint_vel_std`` would stay inert.
* **Newton's velocity index is ``_joint_dof_index``, not
  ``_joint_coord_index``.** A free joint upstream shifts the two apart (7
  position coords vs 6 velocity dofs); indexing ``joint_qd`` by the coord map
  would read a neighbouring joint's velocity, silently, for every joint after
  the free one.

Scope: Isaac and Newton are graded on ``__new__`` skeletons (their runtimes are
not importable here; the seams are the articulation handle and ``_state_0``),
MuJoCo on a real compiled model since it is installed. The live-Isaac half - a
falling robot reporting nonzero ``.vel`` from the real
``get_joint_velocities`` - is verified on GPU.
"""

from __future__ import annotations

import importlib.util
import threading
import types
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.newton.simulation import NewtonSimEngine

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    IsaacConfig,
    IsaacSimulation,
    _RobotState,
)


# --------------------------------------------------------------------------- #
# Isaac                                                                        #
# --------------------------------------------------------------------------- #
class _IsaacArticulation:
    def __init__(self, positions: Any = (0.1, 0.2), velocities: Any = (1.5, -2.5)) -> None:
        self._q, self._qd = positions, velocities

    def get_joint_positions(self) -> Any:
        return np.asarray(self._q, dtype=np.float32)

    def get_joint_velocities(self) -> Any:
        return np.asarray(self._qd, dtype=np.float32)


def _isaac_engine(articulation: Any = None) -> Any:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(render_mode="headless")
    engine._world = types.SimpleNamespace()
    engine._world_created = True
    engine._cameras = {}
    engine._objects = {}
    engine._robots = {
        "arm": _RobotState(
            name="arm",
            prim_path="/World/Robots/arm",
            joint_names=["shoulder", "elbow"],
            articulation=_IsaacArticulation() if articulation is None else articulation,
        )
    }
    return engine


class TestIsaacEmitsJointVelocities:
    def test_a_vel_key_per_joint_beside_the_position(self) -> None:
        obs = _isaac_engine().get_observation()
        assert obs["shoulder"] == pytest.approx(0.1)
        assert obs["elbow"] == pytest.approx(0.2)
        assert obs["shoulder.vel"] == pytest.approx(1.5)
        assert obs["elbow.vel"] == pytest.approx(-2.5)

    def test_the_values_are_plain_floats(self) -> None:
        """Dataset columns and JSON consume this dict; a numpy scalar leaking
        through fails serialisation far from here."""
        obs = _isaac_engine().get_observation()
        assert type(obs["shoulder.vel"]) is float

    def test_a_velocity_read_failure_degrades_to_positions_only(self) -> None:
        """The schema: joint state MUST still be returned when other reads fail.

        A handle predating ``get_joint_velocities`` raises ``AttributeError``;
        were the velocity read inside the position ``try`` without its own
        handler, that would take the already-read positions down with it.
        """

        class _OldHandle:
            def get_joint_positions(self) -> Any:
                return np.asarray([0.1, 0.2], dtype=np.float32)

            def __getattr__(self, name: str) -> Any:
                raise AttributeError(name)

        obs = _isaac_engine(articulation=_OldHandle()).get_observation()
        assert obs["shoulder"] == pytest.approx(0.1)
        assert obs["elbow"] == pytest.approx(0.2)
        assert not any(k.endswith(".vel") for k in obs)

    def test_a_none_velocity_read_is_omitted_not_zeroed(self) -> None:
        """``None`` (uninitialised view) must not become four zero keys: WBC
        reads a present-but-zero velocity as a real measurement, and the
        warning that guards the absent case never fires."""

        class _NoneVel(_IsaacArticulation):
            def get_joint_velocities(self) -> Any:
                return None

        obs = _isaac_engine(articulation=_NoneVel()).get_observation()
        assert obs["shoulder"] == pytest.approx(0.1)
        assert not any(k.endswith(".vel") for k in obs)

    def test_a_short_velocity_buffer_emits_only_what_it_covers(self) -> None:
        obs = _isaac_engine(articulation=_IsaacArticulation(velocities=(1.5,))).get_observation()
        assert obs["shoulder.vel"] == pytest.approx(1.5)
        assert "elbow.vel" not in obs


# --------------------------------------------------------------------------- #
# Newton (module imports without warp; the engine is a skeleton)               #
# --------------------------------------------------------------------------- #
class _Buffer:
    def __init__(self, values: list[float]) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    def numpy(self) -> Any:
        return self._values


def _newton_engine(
    *,
    joint_q: list[float],
    joint_qd: list[float],
    coord_index: dict[str, int],
    dof_index: dict[str, int],
    free_base_joint: str | None = None,
) -> Any:
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._lock = threading.RLock()
    engine._model = object()
    engine._state_0 = types.SimpleNamespace(joint_q=_Buffer(joint_q), joint_qd=_Buffer(joint_qd))
    joints = list(coord_index)
    if free_base_joint is not None:
        joints = [free_base_joint, *joints]
    engine._world = types.SimpleNamespace(  # type: ignore[assignment]
        robots={"bot": types.SimpleNamespace(joint_names=joints)},
        cameras={},
        _backend_state={},
    )
    engine._robot_free_base_joint = {"bot": free_base_joint} if free_base_joint else {}
    engine._joint_coord_index = {("bot", j): i for j, i in coord_index.items()}
    engine._joint_dof_index = {("bot", j): i for j, i in dof_index.items()}
    if free_base_joint is not None:
        engine._joint_coord_index[("bot", free_base_joint)] = 0
        engine._joint_dof_index[("bot", free_base_joint)] = 0
    engine._obs_noise = None
    engine._obs_noise_rng = None
    return engine


class TestNewtonEmitsJointVelocities:
    def test_a_vel_key_per_joint_from_the_dof_index(self) -> None:
        engine = _newton_engine(
            joint_q=[0.1, 0.2],
            joint_qd=[1.5, -2.5],
            coord_index={"j1": 0, "j2": 1},
            dof_index={"j1": 0, "j2": 1},
        )
        obs = engine.get_observation("bot", skip_images=True)
        assert obs["j1"] == pytest.approx(0.1)
        assert obs["j1.vel"] == pytest.approx(1.5)
        assert obs["j2.vel"] == pytest.approx(-2.5)

    def test_a_free_joint_upstream_shifts_the_two_indices_apart(self) -> None:
        """The reason ``_joint_dof_index`` exists: a floating base takes 7
        position coords but 6 velocity dofs, so a scalar joint after it sits at
        coord 7 and dof 6. Indexing ``joint_qd`` by the coord map would read
        the wrong slot - here 99.0 instead of 3.25 - silently, for every joint
        downstream of the free one."""
        joint_q = [0.0] * 7 + [0.4]  # free base (7) + one hinge
        joint_qd = [0.0] * 6 + [3.25, 99.0]  # free base (6) + hinge dof + a trap value
        engine = _newton_engine(
            joint_q=joint_q,
            joint_qd=joint_qd,
            coord_index={"hinge": 7},
            dof_index={"hinge": 6},
            free_base_joint="base_free",
        )
        obs = engine.get_observation("bot", skip_images=True)
        assert obs["hinge"] == pytest.approx(0.4)
        assert obs["hinge.vel"] == pytest.approx(3.25)
        # And the free joint itself never appears as a scalar ``.vel`` entry.
        assert "base_free.vel" not in obs

    def test_velocity_noise_lands_on_vel_keys_and_position_noise_does_not(self) -> None:
        """``joint_vel_std`` was accepted and documented by Newton's
        ``set_obs_noise`` all along, while the noise pass applied
        ``joint_pos_std`` to every entry - so the parameter configured a channel
        that did not exist, and once ``.vel`` entries exist, the unsplit pass
        would put position noise on them."""
        engine = _newton_engine(
            joint_q=[0.1],
            joint_qd=[1.5],
            coord_index={"j1": 0},
            dof_index={"j1": 0},
        )
        engine._obs_noise = {"joint_pos_std": 0.5, "joint_vel_std": 0.0}
        engine._obs_noise_rng = np.random.default_rng(7)
        noisy = engine.get_observation("bot", skip_images=True)
        assert noisy["j1"] != pytest.approx(0.1), "position noise configured but not applied"
        assert noisy["j1.vel"] == pytest.approx(1.5), "joint_pos_std leaked onto a .vel key"

        engine._obs_noise = {"joint_pos_std": 0.0, "joint_vel_std": 0.5}
        noisy = engine.get_observation("bot", skip_images=True)
        assert noisy["j1"] == pytest.approx(0.1), "joint_vel_std leaked onto a position key"
        assert noisy["j1.vel"] != pytest.approx(1.5), "joint_vel_std configured but not applied"


# --------------------------------------------------------------------------- #
# MuJoCo control + the contract itself                                         #
# --------------------------------------------------------------------------- #
class TestTheContractHasOneOwner:
    def test_the_abc_documents_the_vel_entry(self) -> None:
        """The key lived only in MuJoCo's implementation, which is how two
        backends shipped without it - the schema is where a backend author
        reads what to emit."""
        import inspect

        from strands_robots.simulation.base import SimEngine

        doc = inspect.getdoc(SimEngine.get_observation) or ""
        assert '"<joint_name>.vel"' in doc

    @pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco not installed")
    def test_mujoco_still_emits_the_key_these_two_now_match(self) -> None:
        """Control: the reference backend's spelling, measured rather than
        assumed, so a rename on either side fails here instead of shipping
        three vocabularies."""
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        sim = MuJoCoSimEngine()
        try:
            assert sim.create_world()["status"] == "success"
            assert sim.add_robot("so100")["status"] == "success"
            sim.step(5)
            obs = sim.get_observation(skip_images=True)
            joint_keys = [k for k in obs if isinstance(obs[k], float) and not k.endswith(".vel")]
            scalar_joints = [k for k in joint_keys if not k.startswith("base_") and "." not in k]
            assert scalar_joints, "premise: the robot reports scalar joints"
            for k in scalar_joints:
                assert f"{k}.vel" in obs, f"MuJoCo stopped emitting {k}.vel"
        finally:
            sim.destroy()

    def test_the_wbc_docstring_no_longer_blames_the_wrong_backend(self) -> None:
        """It claimed MuJoCo's observation exposes positions only - false since
        #761 - sending anyone diagnosing a zero-velocity gait toward the one
        backend that was fine."""
        import inspect

        from strands_robots.policies.wbc import policy as wbc_policy

        source = inspect.getsource(wbc_policy)
        assert "exposes joint *positions*" not in source
