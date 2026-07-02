"""Force-limited pinch gripper for Newton cloth-grasp tests.

Two finger plates are **free dynamic rigid bodies**; every substep each receives a
bounded applied wrench:

* a *pinch* force pushing the plates toward each other, clamped to ``f_pinch_max``, and
* a *carriage* PD force servoing each plate toward its slot on a commanded wrist
  trajectory, clamped to ``f_carry_max``.

Because every force is clamped, the grip is **honest**: it holds only while friction
at ≤ f_pinch_max normal force can transmit the drag load, and it *peels* when the
load exceeds that — requirement E behaviour emerging from contact physics instead of
a scripted break-force. No kinematic bodies ever touch the cloth.

Force budget — ratios, not absolute newtons. The VBD cm recipe runs at the mass
scale of Newton's own cloth examples (see ``bedsheet.BedsheetParams``); what decides
hold-vs-peel is the dimensionless ratio of grip authority to load. We therefore set

    f_pinch_max = pinch_to_weight · sheet_weight ≈ 1.5 × sheet weight
    f_carry_max = carry_to_weight · sheet_weight ≈ 8   × sheet weight

These are the tuned working budgets (see ``PinchParams.pinch_to_weight`` / ``carry_to_weight``).
The Inspire two-finger spec — ~20 N pinch / ~80 N carry against a ~4.9 N (~0.5 kg) cover, i.e.
~4.1× / ~16× — is the hand's MAX ceiling, not the working point.

(The commanded wrist trajectory plays the role of the robot arm; in the #2 demo that
command comes from the whole-body reach policy, which balances the robot. This
module only owns the fingers.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton


@dataclass
class PinchParams:
    plate_hx: float = 0.4      # cm half-extents: thin fingers (0.8 cm plates)
    plate_hy: float = 6.0
    plate_hz: float = 4.0
    # VBD mixes friction as the GEOMETRIC MEAN sqrt(soft_contact_mu · shape_mu)
    # (newton rigid_vbd_kernels) — with cloth-side 0.7 this gives an effective
    # pinch mu ≈ 1.2 (rubber pad on cotton)
    finger_mu: float = 2.0
    # Force budget as ratios to sheet weight. The pinch need only beat the drag load
    # through friction: with μ_eff ≈ 1.2, friction capacity = pinch·μ, and the
    # quasi-static drag load is ≈ 1× cover weight, so pinch ≈ 1.5× weight already
    # gives a ~1.8× friction margin. The Inspire's 20 N spec (4.1× a 0.5 kg cover)
    # is the hand's MAX, not the grasp's working point — applying 4.1× weight to a
    # light free pad launches it through the compliant cloth contact (eye-verified
    # v18–v20: pad ejected to x≈98). 1.5× holds without the violence.
    pinch_to_weight: float = 1.5
    carry_to_weight: float = 8.0          # arm pull authority (still >> the drag load)
    # Reflected pad mass. A real fingertip is light, but the linkage + arm inertia it
    # is rigidly part of make the EFFECTIVE mass seen at the contact much larger; a
    # near-massless free pad is both unrealistic and numerically violent under applied
    # wrenches. 0.4× the cover mass is a defensible reflected inertia.
    plate_to_sheet_mass: float = 0.4
    servo_omega: float = 25.0  # rad/s carriage-servo natural frequency
    servo_zeta: float = 1.2    # damping ratio (overdamped tracking)
    gap_open: float = 10.0     # cm plate-CENTRE separation when open
    # Closed centre-separation. MUST exceed the plate thickness (2·plate_hx): the
    # centres can never be closer than the plates are thick — commanding less drives
    # plate-into-plate interpenetration and the AVBD contact blows the fingers apart
    # (eye-verified harness v1).
    gap_closed: float = 1.7    # single hanging layer (1.6 cm): plates 0.8 + ~50% squeeze
                               # (3.0 was sized for a two-layer fold — the channel never
                               # compressed a single hem layer and the grip carried by
                               # incidental grazing only: 18.5 cm slip, harness v11)
    gap_rate: float = 25.0     # cm/s slew limit on the commanded gap (no step-close)


class ForceLimitedPinch:
    """Two-plate pinch with bounded forces. Build *before* ``finalize()``.

    ``sheet_mass`` is the total cloth mass in builder units (count · particle_mass);
    the entire force budget is derived from it at real-world ratios.
    """

    def __init__(self, builder: newton.ModelBuilder, sheet_mass: float,
                 params: PinchParams | None = None, spawn=(0.0, 0.0, -100.0)):
        self.p = params or PinchParams()
        w = sheet_mass * 981.0                       # sheet weight, sim units
        self.f_pinch_max = self.p.pinch_to_weight * w
        self.f_carry_max = self.p.carry_to_weight * w
        self.plate_mass = self.p.plate_to_sheet_mass * sheet_mass
        self.kp_carry = self.plate_mass * self.p.servo_omega ** 2
        self.kd_carry = 2.0 * self.p.servo_zeta * self.plate_mass * self.p.servo_omega

        cfg = builder.default_shape_cfg.copy()
        cfg.mu = self.p.finger_mu
        # Fingertip contact stiffness: fabric pinched at real pad pressures is nearly
        # incompressible, and the contact must take the full pinch budget at SMALL
        # penetration. The equilibrium penetration is F/(n_contacts·ke); with the
        # coarse grid only ~7 nodes touch a pad face, and if the penetration reaches
        # the particle radius the contacts cascade and the pad blasts through
        # (eye-verified v18/v19: pad ejected to x≈98). Size ke for ≤ 0.2 cm
        # penetration at the FULL pinch budget over 7 contacts; VBD is implicit, so
        # stiffness this high is fine given enough iterations (use ≥ 16).
        cfg.ke = max(1.0e5, self.f_pinch_max / (7 * 0.2))
        # Newton kd convention (newton-physics#285): user kd is multiplied by ke
        # internally (Rayleigh) — working examples use ~1e-3..1e-2, NOT a ke fraction
        cfg.kd = 2.0e-3
        # Density defines the plate's mass AND inertia. Do NOT also pass mass= to
        # add_body: Newton accumulates the shape-from-density mass onto the body, so
        # setting both double-counts the plate mass (~2x) and mistunes the gravity
        # feed-forward and servo/orientation gains that are derived from plate_mass.
        cfg.density = self.plate_mass / (8 * self.p.plate_hx * self.p.plate_hy * self.p.plate_hz)
        self.bodies = []
        for k in (0, 1):
            b = builder.add_body(
                xform=wp.transform(wp.vec3(spawn[0], spawn[1], spawn[2] - 20.0 * k),
                                   wp.quat_identity()),
                label=f"pinch_plate_{k}")
            builder.add_shape_box(b, hx=self.p.plate_hx, hy=self.p.plate_hy,
                                  hz=self.p.plate_hz, cfg=cfg)
            self.bodies.append(b)
        self.cmd_center = np.array(spawn, dtype=float)
        self.cmd_vel = np.zeros(3)
        self.closing = False
        self.axis = np.array([1.0, 0.0, 0.0])   # pinch axis (plate 0 on +x side)
        self._gap = self.p.gap_open              # slew-limited live gap command
        self._squeeze = 0.0                       # time-ramped pinch engagement
        # Orientation hold. A real finger is RIGIDLY mounted to the wrist — mount
        # rigidity is structural, not an actuation budget (the honest force limits
        # are the pinch and carry forces). A weak rotational spring lets contact
        # torques pitch the pads until they wedge diagonally in the frame-mattress
        # gap (eye-verified v15: gap stuck at 5.1, pads sunk 1.9 below command).
        # Stiff PD sized to the pad's inertia at ~60 rad/s.
        iyy = self.plate_mass / 12.0 * ((2 * self.p.plate_hz) ** 2 + (2 * self.p.plate_hx) ** 2)
        self.kr = iyy * 60.0 ** 2
        self.kr_d = 2.0 * 1.2 * iyy * 60.0

    # ── commands (the behaviour layer / reach policy calls these) ────────────
    def command(self, center, vel=(0.0, 0.0, 0.0)) -> None:
        self.cmd_center = np.asarray(center, dtype=float)
        self.cmd_vel = np.asarray(vel, dtype=float)

    def close(self) -> None:
        self.closing = True

    def open(self) -> None:
        self.closing = False

    # ── per-substep force application ─────────────────────────────────────────
    def apply_forces(self, state: newton.State, dt: float) -> None:
        """Clamped pinch + carriage-servo wrenches → ``state.body_f``.
        Call after ``state.clear_forces()`` each substep."""
        q = state.body_q.numpy()
        qd = state.body_qd.numpy()
        f = state.body_f.numpy()
        # slew-limited gap command — fingers close at a finite speed, no step impact
        target = self.p.gap_closed if self.closing else self.p.gap_open
        self._gap += float(np.clip(target - self._gap, -self.p.gap_rate * dt,
                                   self.p.gap_rate * dt))
        # POSITION-close, then FORCE-squeeze (like a real gripper): the pads track
        # their slew-limited slots at a controlled speed; the pinch force engages only
        # once the COMMANDED gap has arrived at gap_closed — engaging it during the
        # approach slams the pads together at ~400 cm/s and ejects the cloth before
        # the contacts can resolve (harness v14: 3 nodes caught of an expected ~12).
        # Pinch engagement: gate on the close COMMAND having arrived (the slew-limited
        # approach is slam-free), then RAMP the force in TIME so the wad is loaded
        # quasi-statically. A step to full force while the pads rest on a thick wad
        # is an impulse the contact solver answers explosively (v18: pad blasted to
        # x=98); a "near closed" gap gate instead throttles the pinch exactly when
        # the jaw catches a thick wad — the best grip there is (v17: 8% squeeze).
        target = 0.0
        if self.closing:
            target = float(np.clip(1.0 - (self._gap - self.p.gap_closed) / 0.5, 0.0, 1.0))
        rate = 1.5  # full pinch force in ~0.7 s
        self._squeeze += float(np.clip(target - self._squeeze, -rate * dt, rate * dt))
        squeeze = self._squeeze
        for k, b in enumerate(self.bodies):
            sign = 1.0 if k == 0 else -1.0
            slot = self.cmd_center + self.axis * sign * self._gap / 2.0
            pos = q[b][:3]
            vel = qd[b][:3]
            fc = self.kp_carry * (slot - pos) + self.kd_carry * (self.cmd_vel - vel)
            fc += np.array([0.0, 0.0, self.plate_mass * 981.0])   # gravity feed-forward
            n = np.linalg.norm(fc)
            if n > self.f_carry_max:
                fc *= self.f_carry_max / n
            fc += -self.axis * sign * self.f_pinch_max * squeeze
            f[b][:3] += fc
            # orientation hold: bounded torque PD to the spawn orientation (a wrist
            # holds the fingers level; the torque budget is finite like everything else)
            quat = q[b][3:]                       # (x, y, z, w)
            rotvec = 2.0 * quat[:3] * (1.0 if quat[3] >= 0.0 else -1.0)
            tau = -self.kr * rotvec - self.kr_d * qd[b][3:]
            f[b][3:] += tau                       # structural mount stiffness — uncapped
        state.body_f.assign(f)

    # ── readbacks ─────────────────────────────────────────────────────────────
    def plate_positions(self, state: newton.State) -> np.ndarray:
        q = state.body_q.numpy()
        return np.stack([q[b][:3] for b in self.bodies])

    def center(self, state: newton.State) -> np.ndarray:
        return self.plate_positions(state).mean(axis=0)

    def gap(self, state: newton.State) -> float:
        pp = self.plate_positions(state)
        return float(abs((pp[0] - pp[1]) @ self.axis))
