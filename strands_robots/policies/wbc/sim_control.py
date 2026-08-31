"""WBC torque action-controller: drive WBC through ``sim.run_policy`` so it walks.

:class:`WBCPolicy` emits joint-**position** targets. The MuJoCo backend's default
``_apply_sim_action`` writes those targets straight to ``data.ctrl`` - which, on
the stock Unitree G1 scene, drives **position-servo** actuators with a single
stiff gain (``kp = 500`` uniform). That gain overrides SONIC's tuned per-joint PD
(``kp`` 40-250, ``kd`` 2-5) and the gait diverges: the robot falls within a
fraction of a second.

This module provides the missing piece - a controller installed via the same
``world._backend_state["action_controller"]`` hook a benchmark adapter uses to
install its own controller.
When installed it:

1. Flips the G1's leg+waist+arm actuators to **torque (motor) mode** in the
   compiled model (restored on :meth:`uninstall`), so writing a torque to
   ``data.ctrl`` applies that torque directly.
2. On each policy step, converts WBC's position-target action dict to joint
   **torques** via the upstream SONIC PD law (:meth:`WBCPolicy.compute_torques`)
   and advances physics by ``control_decimation`` substeps, recomputing the PD
   torque each substep (``owns_stepping = True``).

The arm joints WBC does not drive track whatever target the action dict names for
them, under a light PD, and hold their nominal pose while unnamed - so the same
controller carries the upper body of a
:class:`~strands_robots.policies.composite.CompositePolicy` (legs+waist from WBC,
arms from a manipulation policy) instead of overriding it. With this controller
installed,
``sim.run_policy(robot_name="unitree_g1", policy_object=WBCPolicy(...), ...)``
produces a real walking / balancing gait on the standard ``Robot("unitree_g1")``
model - no upstream model swap, no mesh download.

Verified on the Menagerie ``unitree_g1`` model: walk command -> +2.3 m forward
upright; zero command -> balanced standing (< 0.1 m drift).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
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
# Arm joints (not driven by WBC) run a light PD toward their commanded target,
# defaulting to the nominal 0 hold of the reference deploy loop when nothing
# commands them.
_ARM_KP = 100.0
_ARM_KD = 0.5


class WBCTorqueController:
    """Action controller converting WBC position targets to G1 joint torques.

    Mirrors the :class:`_LiberoOSCController` contract: exposes
    ``apply(action_dict, model, data, robot_name)`` and declares
    ``owns_stepping = True`` so :meth:`_apply_sim_action` does not double-step.

    Construct via :meth:`from_sim`, which resolves the actuators by name and
    flips them to torque mode. Call :meth:`uninstall` to hand the world back: it
    drops this controller from ``world._backend_state["action_controller"]`` and
    restores the original actuator gains (e.g. when reusing the world for a
    non-WBC policy). Releasing both is what makes that reuse real -- see
    :meth:`uninstall`.
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
        world: Any = None,
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
        # The world whose ``_backend_state`` registered us, so :meth:`uninstall`
        # can release that registration too. ``None`` when built through the
        # constructor directly: nothing registered it, so there is nothing to
        # release.
        self._world = world
        self.physics_substeps_per_control = max(1, int(physics_substeps_per_control))
        # The default-angle hold target, used until the policy returns its first
        # action (a stable first step: PD against the init pose -> ~0 torque).
        # Use the policy's RESOLVED default_angles (config or G1 SONIC fallback),
        # not config.default_angles (empty when the checkpoint ships no config).
        n = policy.config.num_actions
        self._target_q = np.asarray(policy.default_angles, dtype=np.float64).copy()
        if self._target_q.shape[0] != n:
            self._target_q = np.zeros(n, dtype=np.float64)
        # Arm targets, in ``arm_actuator_ids`` order. Zero is the nominal hold of
        # the reference deploy loop, and stays the target for any arm joint no
        # action dict ever names - so a bare WBC rollout is unchanged.
        self._arm_target_q = np.zeros(len(self.arm_actuator_ids), dtype=np.float64)
        # Bare joint names of the held arm joints, positionally aligned with
        # ``_arm_target_q`` (WBC drives ``num_actions``; the rest are the arms).
        self._arm_joint_names: tuple[str, ...] = tuple(WBC_G1_ALL_JOINTS[n : n + len(self.arm_actuator_ids)])

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
            world=world,
        )

    def uninstall(self) -> None:
        """Release both halves of the install: the registration, then the gains.

        :func:`install_wbc_torque_control` acquires two things - it flips the
        driven actuators to torque mode *and* registers this controller in
        ``world._backend_state["action_controller"]``, the seam
        ``_apply_sim_action`` dispatches through. Restoring only the gains leaves
        the registration behind, and that leftover is not inert: it is the value
        ``MuJoCoSimEngine._maybe_install_wbc_torque_control`` reads to decide a
        controller is already present, where a present controller is treated as
        a manual install that wins. The next rollout on the same world therefore
        skips the install and dispatches every action through this finished
        controller - writing PD torques into actuators whose position-servo gains
        this method has just restored.

        The registration goes first, so a failure restoring the gains cannot
        leave a controller dispatching into actuators that are already servos
        again. Only *this* controller's registration is dropped: one installed
        since (a manual install, or the LIBERO adapter, which shares the seam)
        is never clobbered.
        """
        backend_state = getattr(self._world, "_backend_state", None) if self._world is not None else None
        deregistered = False
        if isinstance(backend_state, dict) and backend_state.get("action_controller") is self:
            del backend_state["action_controller"]
            deregistered = True
        model = self._model
        for ai, (gaintype, biastype, gainprm, biasprm, ctrlrange) in self._saved_actuator_gains.items():
            model.actuator_gaintype[ai] = gaintype
            model.actuator_biastype[ai] = biastype
            model.actuator_gainprm[ai] = gainprm
            model.actuator_biasprm[ai] = biasprm
            model.actuator_ctrlrange[ai] = ctrlrange
        logger.debug(
            "WBCTorqueController uninstalled: restored %d actuator gains, deregistered=%s.",
            len(self._saved_actuator_gains),
            deregistered,
        )

    # ------------------------------------------------------------------
    # Action-controller hook
    # ------------------------------------------------------------------

    @staticmethod
    def _refresh_targets(
        action_dict: dict[str, Any],
        names: Sequence[str],
        targets: np.ndarray,
    ) -> None:
        """Overwrite ``targets`` in place from the values ``action_dict`` names.

        ``names`` is positionally aligned with ``targets``. A name the action
        dict omits, or whose value is not a number, keeps its previous target:
        one bad or absent key degrades to a hold rather than aborting the whole
        control step, and the rest of the action still applies.
        """
        for i, name in enumerate(names):
            v = action_dict.get(name)
            if v is None:
                continue
            try:
                targets[i] = float(v)
            except (TypeError, ValueError):
                continue

    def apply(
        self,
        action_dict: dict[str, Any],
        model: Any,
        data: Any,
        robot_name: str,  # noqa: ARG002 - kept for hook signature parity
    ) -> None:
        """Convert WBC position targets to torques and advance physics.

        ``action_dict`` maps joint names to absolute position targets (the
        policy's output). We update the held targets, then run the SONIC PD law
        (:meth:`WBCPolicy.compute_torques`) every physics substep for
        ``physics_substeps_per_control`` steps, recomputing the torque from the
        integrated state each substep.

        Arm joints - the ones WBC does not drive - track any target the action
        dict names for them under a light PD, and hold their previous target
        (nominal 0 until something commands them) otherwise. That is what lets
        one :class:`~strands_robots.policies.composite.CompositePolicy` put WBC
        on the legs and a manipulation policy on the arms: pinning the arms to 0
        here would discard every upper-body command without a word.

        ``owns_stepping = True`` tells the SimEngine not to call ``mj_step``
        after this returns - we have advanced physics by the full control step.
        """
        import mujoco as mj

        # Refresh the target from this step's action (bare joint-name keys, in
        # WBC output order). Missing keys keep the previous target.
        driven_names = WBC_G1_ALL_JOINTS[: len(self.leg_waist_actuator_ids)]
        self._refresh_targets(action_dict, driven_names, self._target_q)
        # Same refresh for the arm joints, so an upper-body policy composed on
        # top of WBC reaches the actuators instead of being overwritten.
        self._refresh_targets(action_dict, self._arm_joint_names, self._arm_target_q)

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
                arm_tau = (self._arm_target_q - qa) * _ARM_KP - dqa * _ARM_KD
                for ai, t in zip(arm_act, arm_tau, strict=True):
                    data.ctrl[ai] = float(t)
            mj.mj_step(model, data)


def wbc_uses_position_servo(sim: SimEngine, policy: WBCPolicy, robot_name: str) -> bool:
    """Return True if the WBC-driven actuators on ``robot_name`` are position-servo.

    :class:`WBCPolicy` emits joint-**position** targets. On a position-servo
    scene (the stock Menagerie Unitree G1 ships a uniform ``kp=500`` servo) those
    targets fight the servo gain and override SONIC's tuned per-joint PD, so the
    gait diverges and the robot falls. This predicate lets ``run_policy`` decide
    whether the torque shim (:func:`install_wbc_torque_control`) is needed.

    A MuJoCo *position* actuator has ``biastype == mjBIAS_AFFINE`` (the ``-kp``
    bias term); a *motor* (pure torque) actuator has ``biastype == mjBIAS_NONE``.
    Returns ``True`` as soon as one driven actuator is position-servo, ``False``
    when they are already torque actuators or cannot be resolved (no world,
    unknown joints) - the conservative answer that leaves an already-correct or
    unrecognised scene untouched.
    """
    import mujoco as mj

    world = getattr(sim, "_world", None)
    if world is None or getattr(world, "_model", None) is None:
        return False
    model = world._model
    robot = world.robots.get(robot_name) if getattr(world, "robots", None) else None
    pfx = robot.namespace if robot is not None else ""

    n_act = policy.config.num_actions
    driven_names = list(WBC_G1_ALL_JOINTS[:n_act])
    for name in driven_names:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, (pfx or "") + name)
        if jid < 0:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        for ai in range(model.nu):
            if int(model.actuator_trnid[ai, 0]) == jid:
                if int(model.actuator_biastype[ai]) == int(mj.mjtBias.mjBIAS_AFFINE):
                    return True
                break
    return False


def install_wbc_torque_control(sim: SimEngine, policy: WBCPolicy, robot_name: str) -> WBCTorqueController:
    """Install a :class:`WBCTorqueController` on ``sim`` for ``robot_name``.

    Registers the controller in ``world._backend_state["action_controller"]``,
    where :meth:`_apply_sim_action` dispatches to it. After this call,
    ``sim.run_policy(robot_name=robot_name, policy_object=policy, ...)`` drives
    the G1 with the SONIC PD->torque law and produces a real gait.

    Use ``control_frequency=50.0`` in ``run_policy`` to match the controller's
    physics step (dt=0.005) x decimation (4).

    Returns the installed controller. Call
    :meth:`WBCTorqueController.uninstall` to undo *both* halves of this install:
    it drops the registration made here and restores the original actuators.

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


__all__ = ["WBCTorqueController", "install_wbc_torque_control", "wbc_uses_position_servo"]
