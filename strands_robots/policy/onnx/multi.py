"""An ONNX policy with many named inputs and outputs, carry buffers and per-joint gains.

Low-level walk policies (most IsaacLab exports and vendor examples) are not one-tensor-in,
one-tensor-out: they read joints, IMU and a command, keep a history they hand back each step, and output
targets plus stiffness and damping. ``MultiOnnxPolicy`` maps each input to an observation field and each
output to targets / kp / kd / carry, so such a graph runs in the normal ``robot.run`` loop.

Input sources: ``joints.positions``, ``joints.velocities``, ``pose.orientation`` (wxyz),
``pose.angular_velocity``, ``command`` (the ``command()`` callable, e.g. joystick axes) and ``carry`` (zeros
at reset, then the named output). Outputs: ``targets`` (required), ``kp``, ``kd``. ``joint_order`` is the
order the graph wants joints in (default the robot's action_keys); targets come back in action_keys order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np

from ...types import Observation
from ..base import Policy, PolicyFault

SOURCES = ("joints.positions", "joints.velocities", "pose.orientation", "pose.angular_velocity", "command", "carry")


class MultiOnnxPolicy(Policy):
    units = "rad"

    def __init__(
        self,
        path: str,
        *,
        inputs: Mapping[str, str],
        outputs: Mapping[str, str],
        carry: Mapping[str, str] | None = None,
        joint_order: Sequence[str] | None = None,
        command: Callable[[], Sequence[float]] | None = None,
        hz: float = 50.0,
        blend_steps: int = 50,
        session: Any = None,
    ) -> None:
        bad = {k: v for k, v in inputs.items() if v not in SOURCES}
        if bad:
            raise ValueError(f"unknown input sources {bad}; known: {SOURCES}")
        if "targets" not in outputs or ("kp" in outputs) != ("kd" in outputs):
            raise ValueError("outputs must name 'targets', and 'kp' and 'kd' together or neither")
        self.path, self.inputs, self.outputs = path, dict(inputs), dict(outputs)
        self.carry_map = dict(carry or {})  # input name -> output name that feeds it next step
        self.joint_order, self.command, self.hz, self.blend_steps = joint_order, command, float(hz), int(blend_steps)
        self._sess = session
        self._robot: Any = None
        self._perm: np.ndarray | None = None
        self.reset()

    def attach(self, robot: Any) -> None:
        if self._sess is None:
            try:
                import onnxruntime as ort
            except ImportError:
                raise ImportError("MultiOnnxPolicy needs onnxruntime: pip install 'strands-robots[onnx]'") from None
            self._sess = ort.InferenceSession(self.path, providers=["CPUExecutionProvider"])
        keys = list(robot.action_keys)
        order = list(self.joint_order or keys)
        missing = [k for k in order if k not in keys]
        if missing or len(order) != len(keys):
            raise ValueError(f"joint_order must be a permutation of the robot's action_keys; bad: {missing}")
        self._perm = np.array([keys.index(k) for k in order])
        takes_gains = robot.sim is not None or hasattr(robot.body, "set_gains")
        if ("kp" in self.outputs or "kd" in self.outputs) and not takes_gains:
            raise ValueError(f"{robot.id}'s body takes no per-tick gains, but this policy outputs kp/kd")
        self._robot = robot

    def reset(self) -> None:
        self._carry: dict[str, np.ndarray] = {}
        self._step, self._start = 0, cast("np.ndarray | None", None)

    def _feed(self, obs: Observation) -> dict[str, np.ndarray]:
        sess, perm = self._sess, self._perm
        shapes = {i.name: [d if isinstance(d, int) else 1 for d in i.shape] for i in sess.get_inputs()}
        feed = {}
        for name, src in self.inputs.items():
            if src == "joints.positions":
                v = np.asarray(obs.joints.positions)[perm]
            elif src == "joints.velocities":
                v = np.asarray(obs.joints.velocities)[perm]
            elif src.startswith("pose."):
                if obs.pose is None or getattr(obs.pose, src[5:]) is None:
                    raise PolicyFault(f"input {name} needs {src}, and this body reports none")
                v = np.asarray(getattr(obs.pose, src[5:]))
            elif src == "command":
                v = np.asarray(self.command() if self.command else np.zeros(int(np.prod(shapes[name]))))
            else:
                v = self._carry.get(name, np.zeros(shapes[name]))
            feed[name] = np.asarray(v, dtype=np.float32).reshape(shapes[name])
        return feed

    def act(self, obs: Observation, *, instruction: str | None = None) -> Any:
        if self._sess is None or self._perm is None:
            raise PolicyFault("MultiOnnxPolicy.act before attach(robot)")
        names = [o.name for o in self._sess.get_outputs()]
        out = dict(zip(names, self._sess.run(None, self._feed(obs)), strict=True))
        for inp, outname in self.carry_map.items():
            self._carry[inp] = out[outname]
        targets = np.empty(len(self._perm))
        targets[self._perm] = np.asarray(out[self.outputs["targets"]], dtype=float).reshape(-1)
        if not np.isfinite(targets).all():
            raise PolicyFault("the ONNX graph returned non-finite targets")
        if self._start is None:
            self._start = np.asarray(obs.joints.positions, dtype=float).copy()
        a = min(1.0, (self._step + 1) / max(1, self.blend_steps))
        self._step += 1
        if "kp" in self.outputs:  # per-tick PD: Robot.set_gains, before the targets go out
            kp, kd = np.empty(len(self._perm)), np.empty(len(self._perm))
            kp[self._perm] = np.asarray(out[self.outputs["kp"]], dtype=float).reshape(-1)
            kd[self._perm] = np.asarray(out[self.outputs["kd"]], dtype=float).reshape(-1)
            self._robot.set_gains(kp, kd)
        return a * targets + (1 - a) * self._start  # crossfade from the measured pose over blend_steps
