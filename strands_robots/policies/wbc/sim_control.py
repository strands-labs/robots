"""WBC torque action-controller: drive WBC through ``sim.run_policy`` so it walks.

:class:`WBCPolicy` emits joint-**position** targets. The MuJoCo backend's default
``_apply_sim_action`` writes those targets straight to ``data.ctrl`` - which, on
the stock Unitree G1 scene, drives **position-servo** actuators with a single
stiff gain (``kp = 500`` uniform). That gain overrides SONIC's tuned per-joint PD
(``kp`` 40-250, ``kd`` 2-5) and the gait diverges: the robot falls within a
fraction of a second.

This module provides the missing piece - a controller installed via the same
``world._backend_state["action_controller"]`` hook the LIBERO adapter uses
(see :class:`strands_robots.benchmarks.libero.adapter._LiberoOSCController`).
When installed it:

1. Flips the G1's leg+waist+arm actuators to **torque (motor) mode** in the
   compiled model (restored on :meth:`uninstall`), so writing a torque to
   ``data.ctrl`` applies that torque directly.
2. On each policy step, converts WBC's position-target action dict to joint
   **torques** via the upstream SONIC PD law (:meth:`WBCPolicy.compute_torques`)
   and advances physics by ``control_decimation`` substeps, recomputing the PD
   torque each substep (``owns_stepping = True``).

The arm joints WBC does not drive are held at their nominal pose with a light PD,
matching the reference deploy loop. With this controller installed,
``sim.run_policy(robot_name="unitree_g1", policy_object=WBCPolicy(...), ...)``
produces a real walking / balancing gait on the standard ``Robot("unitree_g1")``
model - no upstream model swap, no mesh download.

Verified on the Menagerie ``unitree_g1`` model: walk command -> +2.3 m forward
upright; zero command -> balanced standing (< 0.1 m drift).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.policies.wbc.policy import WBC_G1_ALL_JOINTS, WBCPolicy

if TYPE_CHECKING:
    from strands_robots.simulation.base import SimEngine

logger = logging.getLogger(__name__)

# Upstream g1_gear_wbc.yaml: 0.005 s physics step, one inference per 4 steps
# (50 Hz control). The PD->torque law runs every physics substep.
_SIM_DT = 0.005
_CONTROL_DECIMATION = 4
# Arm joints (not driven by WBC) are held at nominal 0 with a light PD so they
# do not flail; matches the reference deploy loop's arm hold.
_ARM_KP = 100.0
_ARM_KD = 0.5


class WBCTorqueController:
    """Action controller converting WBC position targets to G1 joint torques.

    Mirrors the :class:`_LiberoOSCController` contract: exposes
    ``apply(action_dict, model, data, robot_name)`` and declares
    ``owns_stepping = True`` so :meth:`_apply_sim_action` does not double-step.

    Construct via :meth:`from_sim`, which resolves the actuators by name and
    flips them to torque mode. Call :meth:`uninstall` to restore the original
    actuator gains (e.g. when reusing the world for a non-WBC policy).
    """

    # Tell the SimEngine this controller advances physics itself (one apply()
    # runs ``physics_substeps_per_control`` mj_step calls); skip the outer loop.
    owns_stepping: bool = True

    def __init__(
        self,
        policy: WBCPolicy,
        *,
        leg_waist_actuator_ids: list[int],
        arm_actuator_ids: list[int],
        leg_waist_qpos_addrs: list[int],
        leg_waist_dof_addrs: list[int],
        arm_qpos_addrs: list[int],
        arm_dof_addrs: list[int],
        saved_actuator_gains: dict[int, tuple[Any, Any, Any, Any, Any]],
        model: Any,
        physics_substeps_per_control: int = _CONTROL_DECIMATION,
    ) -> None:
        self.policy = policy
        self.leg_waist_actuator_ids = list(leg_waist_actuator_ids)
        self.arm_actuator_ids = list(arm_actuator_ids)
        self.leg_waist_qpos_addrs = list(leg_waist_qpos_addrs)
        self.leg_waist_dof_addrs = list(leg_waist_dof_addrs)
        self.arm_qpos_addrs = list(arm_qpos_addrs)
        self.arm_dof_addrs = list(arm_dof_addrs)
        self._saved_actuator_gains = dict(saved_actuator_gains)
        self._model = model
        self.physics_substeps_per_control = max(1, int(physics_substeps_per_control))
        # The default-angle hold target, used until the policy returns its first
        # action (a stable first step: PD against the init pose -> ~0 torque).
        # Use the policy's RESOLVED default_angles (config or G1 SONIC fallback),
        # not config.default_angles (empty when the checkpoint ships no config).
        n = policy.config.num_actions
        self._target_q = np.asarray(policy.default_angles, dtype=np.float64).copy()
        if self._target_q.shape[0] != n:
            self._target_q = np.zeros(n, dtype=np.float64)

    # ------------------------------------------------------------------
    # Install / teardown
    # ------------------------------------------------------------------

    @classmethod
    def from_sim(
        cls,
        sim: SimEngine,
        policy: WBCPolicy,
        robot_name: str,
    ) -> WBCTorqueController:
        """Build a controller for ``robot_name`` and flip its actuators to torque.

        Resolves the leg+waist (driven) and arm (held) joints by name within the
        robot's namespace, records the original actuator gains so they can be
        restored, and switches each driven actuator to torque (motor) mode.

        Raises:
            RuntimeError: If the world is absent, or an expected WBC joint /
                its driving actuator cannot be found in the model.
        """
        import mujoco as mj

        world = getattr(sim, "_world", None)
        if world is None or getattr(world, "_model", None) is None:
            raise RuntimeError("WBCTorqueController.from_sim: no compiled world/model on the sim.")
        model = world._model

        robot = world.robots.get(robot_name)
        pfx = robot.namespace if robot is not None else ""

        n_act = policy.config.num_actions
        n_obs = policy.config.n_obs_joints
        driven_names = list(WBC_G1_ALL_JOINTS[:n_act])
        all_names = list(WBC_G1_ALL_JOINTS[:n_obs])
        arm_names = all_names[n_act:]

        def _joint_id(name: str) -> int:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, pfx + name)
            if jid < 0:
                jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            return int(jid)

        def _actuator_for_joint(jid: int) -> int:
            for ai in range(model.nu):
                # transmission target 0 is the joint id for JOINT-type actuators.
                if int(model.actuator_trnid[ai, 0]) == jid:
                    return ai
            return -1

        def _resolve(names: list[str]) -> tuple[list[int], list[int], list[int]]:
            act_ids, qpos_addrs, dof_addrs = [], [], []
            for name in names:
                jid = _joint_id(name)
                if jid < 0:
                    raise RuntimeError(
                        f"WBCTorqueController: joint {name!r} not found in the model "
                        f"(looked for {pfx + name!r} and {name!r})."
                    )
                ai = _actuator_for_joint(jid)
                if ai < 0:
                    raise RuntimeError(f"WBCTorqueController: joint {name!r} (id {jid}) has no driving actuator.")
                act_ids.append(ai)
                qpos_addrs.append(int(model.jnt_qposadr[jid]))
                dof_addrs.append(int(model.jnt_dofadr[jid]))
            return act_ids, qpos_addrs, dof_addrs

        leg_act, leg_qpos, leg_dof = _resolve(driven_names)
        arm_act, arm_qpos, arm_dof = _resolve(arm_names) if arm_names else ([], [], [])

        # Save the original gains, then flip every controlled actuator to torque
        # (motor) mode: gaintype FIXED gainprm=[1,0,0], biastype NONE biasprm=0,
        # widened ctrlrange so the PD torque is not clipped. Restored on uninstall.
        saved: dict[int, tuple[Any, Any, Any, Any, Any]] = {}
        for ai in [*leg_act, *arm_act]:
            saved[ai] = (
                int(model.actuator_gaintype[ai]),
                int(model.actuator_biastype[ai]),
                np.array(model.actuator_gainprm[ai], copy=True),
                np.array(model.actuator_biasprm[ai], copy=True),
                np.array(model.actuator_ctrlrange[ai], copy=True),
            )
            model.actuator_gaintype[ai] = mj.mjtGain.mjGAIN_FIXED
            model.actuator_biastype[ai] = mj.mjtBias.mjBIAS_NONE
            model.actuator_gainprm[ai][:3] = [1.0, 0.0, 0.0]
            model.actuator_biasprm[ai][:3] = [0.0, 0.0, 0.0]
            model.actuator_ctrlrange[ai] = [-1000.0, 1000.0]

        # Match the SONIC training physics rate (the stock scene ships a finer
        # 0.002 step). control_frequency in run_policy should be 1/(dt*decim)=50.
        model.opt.timestep = _SIM_DT

        # Set the SONIC initial stance. The policy was trained from the nominal
        # crouch (default_angles) at the commanded base height; starting from
        # the scene's neutral pose (legs straight at qpos=0) is out-of-
        # distribution and the controller collapses on the first steps. Seed the
        # driven joints to their defaults and lift the floating base to
        # height_cmd, then refresh derived quantities.
        data = world._data
        default_angles = np.asarray(policy.default_angles, dtype=np.float64)
        if default_angles.shape[0] == len(leg_qpos):
            for adr, angle in zip(leg_qpos, default_angles, strict=True):
                data.qpos[adr] = float(angle)
        # The free joint's qpos is the first 7 entries (pos[3] + quat[4]); lift
        # the base to the target height and set an upright orientation.
        free_jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, (pfx + "floating_base_joint"))
        if free_jid < 0:
            free_jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        if free_jid >= 0 and int(model.jnt_type[free_jid]) == int(mj.mjtJoint.mjJNT_FREE):
            base_adr = int(model.jnt_qposadr[free_jid])
            data.qpos[base_adr + 2] = float(policy.config.height_cmd)
            data.qpos[base_adr + 3 : base_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0
        mj.mj_forward(model, data)

        logger.info(
            "WBCTorqueController installed on %r: %d driven + %d arm actuators -> torque mode, dt=%.4f, decim=%d",
            robot_name,
            len(leg_act),
            len(arm_act),
            _SIM_DT,
            _CONTROL_DECIMATION,
        )
        return cls(
            policy,
            leg_waist_actuator_ids=leg_act,
            arm_actuator_ids=arm_act,
            leg_waist_qpos_addrs=leg_qpos,
            leg_waist_dof_addrs=leg_dof,
            arm_qpos_addrs=arm_qpos,
            arm_dof_addrs=arm_dof,
            saved_actuator_gains=saved,
            model=model,
        )

    def uninstall(self) -> None:
        """Restore the original actuator gains saved at install time."""
        model = self._model
        for ai, (gaintype, biastype, gainprm, biasprm, ctrlrange) in self._saved_actuator_gains.items():
            model.actuator_gaintype[ai] = gaintype
            model.actuator_biastype[ai] = biastype
            model.actuator_gainprm[ai] = gainprm
            model.actuator_biasprm[ai] = biasprm
            model.actuator_ctrlrange[ai] = ctrlrange
        logger.debug("WBCTorqueController uninstalled: restored %d actuator gains.", len(self._saved_actuator_gains))

    # ------------------------------------------------------------------
    # Action-controller hook
    # ------------------------------------------------------------------

    def apply(
        self,
        action_dict: dict[str, Any],
        model: Any,
        data: Any,
        robot_name: str,  # noqa: ARG002 - kept for hook signature parity
    ) -> None:
        """Convert WBC position targets to torques and advance physics.

        ``action_dict`` maps the WBC leg+waist joint names to absolute position
        targets (the policy's output). We update the held target, then run the
        SONIC PD law (:meth:`WBCPolicy.compute_torques`) every physics substep
        for ``physics_substeps_per_control`` steps, recomputing the torque from
        the integrated state each substep. The arm joints are held at nominal 0
        with a light PD.

        ``owns_stepping = True`` tells the SimEngine not to call ``mj_step``
        after this returns - we have advanced physics by the full control step.
        """
        import mujoco as mj

        # Refresh the target from this step's action (bare joint-name keys, in
        # WBC output order). Missing keys keep the previous target.
        driven_names = WBC_G1_ALL_JOINTS[: len(self.leg_waist_actuator_ids)]
        for i, name in enumerate(driven_names):
            v = action_dict.get(name)
            if v is not None:
                try:
                    self._target_q[i] = float(v)
                except (TypeError, ValueError):
                    # Non-numeric action value for this joint: keep the previous
                    # target rather than aborting the whole control step (one bad
                    # key degrades to a hold, the rest of the action still applies).
                    continue

        leg_q_adr = np.asarray(self.leg_waist_qpos_addrs, dtype=int)
        leg_d_adr = np.asarray(self.leg_waist_dof_addrs, dtype=int)
        leg_act = self.leg_waist_actuator_ids
        arm_q_adr = np.asarray(self.arm_qpos_addrs, dtype=int)
        arm_d_adr = np.asarray(self.arm_dof_addrs, dtype=int)
        arm_act = self.arm_actuator_ids

        for _ in range(self.physics_substeps_per_control):
            q = data.qpos[leg_q_adr]
            dq = data.qvel[leg_d_adr]
            tau = self.policy.compute_torques(self._target_q, q, dq)
            for ai, t in zip(leg_act, tau, strict=True):
                data.ctrl[ai] = float(t)
            if arm_act:
                qa = data.qpos[arm_q_adr]
                dqa = data.qvel[arm_d_adr]
                arm_tau = -qa * _ARM_KP - dqa * _ARM_KD
                for ai, t in zip(arm_act, arm_tau, strict=True):
                    data.ctrl[ai] = float(t)
            mj.mj_step(model, data)


def install_wbc_torque_control(sim: SimEngine, policy: WBCPolicy, robot_name: str) -> WBCTorqueController:
    """Install a :class:`WBCTorqueController` on ``sim`` for ``robot_name``.

    Registers the controller in ``world._backend_state["action_controller"]``,
    where :meth:`_apply_sim_action` dispatches to it. After this call,
    ``sim.run_policy(robot_name=robot_name, policy_object=policy, ...)`` drives
    the G1 with the SONIC PD->torque law and produces a real gait.

    Use ``control_frequency=50.0`` in ``run_policy`` to match the controller's
    physics step (dt=0.005) x decimation (4).

    Returns the installed controller (call :meth:`WBCTorqueController.uninstall`
    to restore the original actuators).

    Raises:
        RuntimeError: If the world is absent or the actuators cannot be resolved.
    """
    controller = WBCTorqueController.from_sim(sim, policy, robot_name)
    world = getattr(sim, "_world", None)
    if world is None:
        raise RuntimeError("install_wbc_torque_control: no world on the sim.")
    backend_state = getattr(world, "_backend_state", None)
    if not isinstance(backend_state, dict):
        raise RuntimeError("install_wbc_torque_control: world has no _backend_state dict.")
    backend_state["action_controller"] = controller
    return controller


__all__ = ["WBCTorqueController", "install_wbc_torque_control"]
