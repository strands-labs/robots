"""Hot-swap FSM over several Microduck ONNX policies.

A single Microduck robot ships many skills as separate ONNX files
(``alpha_walking``, ``alpha_stand``, ``roulade``, ``ball_kick_*`` ...). A
:class:`MicroduckPolicyBundle` holds several :class:`MicroduckPolicy` instances
warm and delegates each tick to the ACTIVE one, so a controller can switch skill
mid-rollout (walk -> kick -> walk) without tearing down and rebuilding sessions.

Switching is explicit: the caller names the next skill via
``get_actions(select=...)`` (or :meth:`switch`). When ``switch_on_velocity`` is
set, the bundle also auto-selects between a ``move_key`` and an ``idle_key`` by
the magnitude of the twist command - the same walking<->standing gate Pollen's
``infer_policy.py`` uses. Both gate keys must name held skills for that gate to
fire, so they are checked when it is enabled. The previously active child's
``last_action`` history is left intact so returning to a skill resumes cleanly.

Pollen's reference ``infer_policy.py`` also supports **episodic behaviors** -
short skills like ``kick_left`` / ``kick_right`` / ``roulade`` that run for a
fixed wall-clock duration and then auto-return to the default skill. Callers
declare these via ``episodic_skills={"kick_left": 1.2, "roulade": 2.0}`` and
activate them with :meth:`trigger`. Each tick decrements a timer by
``1/control_frequency``; when it reaches zero, the bundle reverts to
``default_skill``. This matches ``_end_behavior`` in the upstream FSM.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from strands_robots.policies.base import Policy
from strands_robots.utils import finite_vector_error, positive_finite_number_error

from .policy import TARGET_VELOCITY_WIDTHS, MicroduckPolicy


def _target_velocity_error(source: str, value: object) -> str | None:
    """Return why ``value`` cannot be a ``target_velocity``, or ``None`` if it can.

    The two questions the tick itself will ask, asked before the velocity gate
    reads a magnitude out of the value: the shared per-component vector domain,
    and the component counts :class:`~strands_robots.policies.microduck.MicroduckPolicy`
    documents for the same key. Both come from the child's own module, so the
    gate and the tick cannot drift apart on what a velocity is.

    Args:
        source: The surface to name in the message - the bundle's own
            ``get_actions``, since that is the method the caller called.
        value: The candidate ``target_velocity``.

    Returns:
        A message naming the parameter and the reason, or ``None`` if the value
        is a velocity both readers accept.
    """
    if error := finite_vector_error(source, "target_velocity", value):
        return error
    width = int(np.asarray(value, dtype=np.float32).reshape(-1).shape[0])
    if width not in TARGET_VELOCITY_WIDTHS:
        widths = " or ".join(str(w) for w in sorted(TARGET_VELOCITY_WIDTHS, reverse=True))
        return (
            f"{source}: target_velocity has {width} component(s), expected {widths} "
            f"([vx, vy, omega] or [vx, vy]). The gate reads a magnitude from these "
            f"components, so a width the tick refuses must not move the selection."
        )
    return None


class MicroduckPolicyBundle(Policy):
    """A named collection of Microduck skills with a single active policy.

    Args:
        policies: Mapping of skill name -> :class:`MicroduckPolicy`.
        active: The initially selected skill name. Defaults to the first key.
        switch_on_velocity: If set, auto-switch between ``move_key`` and
            ``idle_key`` by ``|twist|`` against this threshold each tick. Must
            be a positive finite number - the gate compares a magnitude, so a
            threshold of ``0`` or below can never select ``idle_key`` and a
            non-finite one can never select ``move_key``. Pass ``None`` (the
            default) to leave the gate off.
        move_key / idle_key: Skill names the velocity gate selects between.
            Each must be one of ``policies`` whenever ``switch_on_velocity`` is
            set, because the gate reads both every tick: a key that names no
            held skill leaves the gate inert rather than failing, so a bundle
            keyed by the shipped weight names (``alpha_walking`` /
            ``alpha_stand``) never switches under the defaults. They are not
            read at all when the gate is off, and are not checked then.
    """

    requires_images = False

    def __init__(
        self,
        policies: dict[str, MicroduckPolicy],
        *,
        active: str | None = None,
        switch_on_velocity: float | None = None,
        move_key: str = "walk",
        idle_key: str = "stand",
        episodic_skills: dict[str, float] | None = None,
        default_skill: str | None = None,
    ) -> None:
        if not policies:
            raise ValueError("MicroduckPolicyBundle requires at least one policy.")
        for name, pol in policies.items():
            if not isinstance(pol, MicroduckPolicy):
                raise TypeError(
                    f"MicroduckPolicyBundle: policy {name!r} is {type(pol).__name__}, expected MicroduckPolicy."
                )
        self._policies = dict(policies)
        first = next(iter(self._policies))
        self._active = active or first
        if self._active not in self._policies:
            raise ValueError(
                f"MicroduckPolicyBundle: active skill {self._active!r} is not one of {list(self._policies)}."
            )
        if switch_on_velocity is not None and (
            error := positive_finite_number_error(switch_on_velocity, "switch_on_velocity", "MicroduckPolicyBundle")
        ):
            raise ValueError(error)
        self._switch_on_velocity = float(switch_on_velocity) if switch_on_velocity is not None else None
        if self._switch_on_velocity is not None:
            # Only the gate reads these two, so a caller who left it off is
            # never refused for a key the bundle does not look at. With the gate
            # on, `_auto_switch` returns early for a key that names no held
            # skill - the whole gate goes inert, including the direction whose
            # key IS a skill - so the membership `active` is already held to
            # belongs here too, at the same construction-time seam.
            unknown = [
                f"{param}={key!r}"
                for param, key in (("move_key", move_key), ("idle_key", idle_key))
                if key not in self._policies
            ]
            if unknown:
                raise ValueError(
                    f"MicroduckPolicyBundle: {' and '.join(unknown)} names no held skill; "
                    f"have {list(self._policies)}. switch_on_velocity is set, so the velocity "
                    "gate reads both keys every tick and cannot select a skill the bundle "
                    "does not hold."
                )
        self._move_key = move_key
        self._idle_key = idle_key
        # Episodic-behavior FSM state (mirrors Pollen `infer_policy.py`).
        # A caller who omits ``episodic_skills`` gets exactly the previous
        # velocity-gated bundle: :meth:`trigger` refuses, no timer runs,
        # ``get_actions`` never decrements anything, and :meth:`reset` leaves
        # the active skill alone. So an ONNX bundle that never means to kick
        # pays no cost per tick and cannot revert unexpectedly.
        self._episodic_durations: dict[str, float] = {}
        if episodic_skills is not None:
            unknown = [name for name in episodic_skills if name not in self._policies]
            if unknown:
                raise ValueError(
                    f"MicroduckPolicyBundle: episodic_skills names no held skill(s) {unknown!r}; "
                    f"have {list(self._policies)}. Every episodic skill must be one of policies."
                )
            for name, duration in episodic_skills.items():
                if error := positive_finite_number_error(
                    duration, f"episodic_skills[{name!r}]", "MicroduckPolicyBundle"
                ):
                    raise ValueError(error)
                self._episodic_durations[name] = float(duration)
        if default_skill is not None and default_skill not in self._policies:
            raise ValueError(
                f"MicroduckPolicyBundle: default_skill={default_skill!r} names no held skill; "
                f"have {list(self._policies)}."
            )
        # Fallback for :meth:`_end_episode`: the caller's declared default,
        # else the initially-active skill (the same identity the bundle started
        # in). Never an episodic skill itself, because auto-returning INTO an
        # episodic skill would immediately arm its timer and never terminate.
        resolved_default = default_skill if default_skill is not None else self._active
        if resolved_default in self._episodic_durations:
            raise ValueError(
                f"MicroduckPolicyBundle: default_skill={resolved_default!r} is itself an episodic skill; "
                "auto-return would re-arm the timer immediately."
            )
        self._default_skill = resolved_default
        self._episodic_active: str | None = None
        self._episodic_time_left: float = 0.0

    @property
    def provider_name(self) -> str:
        """Registry key for this provider (``"microduck_bundle"``)."""
        return "microduck_bundle"

    @property
    def active(self) -> str:
        """The currently selected skill name."""
        return self._active

    @property
    def children(self) -> tuple[Policy, ...]:
        """Every held skill, so a capability probe can walk to the leaf policy."""
        return tuple(self._policies.values())

    def switch(self, name: str) -> None:
        """Select ``name`` as the active skill.

        Cancels any running episodic behavior: :meth:`switch` is the explicit
        override path, so a caller who names a skill wins over the FSM timer.
        """
        if name not in self._policies:
            raise ValueError(f"MicroduckPolicyBundle: unknown skill {name!r}; have {list(self._policies)}.")
        self._active = name
        self._episodic_active = None
        self._episodic_time_left = 0.0

    @property
    def episodic_active(self) -> str | None:
        """The currently running episodic skill, or ``None`` if none is armed."""
        return self._episodic_active

    def trigger(self, name: str) -> None:
        """Arm an episodic behavior; the bundle will run it until its timer ends.

        Args:
            name: The episodic skill to activate. Must be a key of
                ``episodic_skills`` passed at construction time.

        Raises:
            ValueError: If ``name`` is not a declared episodic skill, or if
                another episodic behavior is already running (matches
                ``_end_behavior`` gating in Pollen ``infer_policy.py`` - only
                one behavior at a time; the caller must let it finish or use
                :meth:`switch` to cancel).
        """
        if name not in self._episodic_durations:
            declared = list(self._episodic_durations)
            raise ValueError(
                f"MicroduckPolicyBundle: {name!r} is not a declared episodic skill; "
                f"have {declared!r}. Pass episodic_skills={{{name!r}: <duration>}} at construction time."
            )
        if self._episodic_active is not None:
            raise ValueError(
                f"MicroduckPolicyBundle: episodic skill {self._episodic_active!r} is already running "
                f"({self._episodic_time_left:.3f}s left); let it finish or call switch() to override."
            )
        self._episodic_active = name
        self._episodic_time_left = self._episodic_durations[name]
        self._active = name

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Forward the robot's joint list to every held skill."""
        for pol in self._policies.values():
            pol.set_robot_state_keys(robot_state_keys)

    def set_control_frequency(self, hz: float) -> None:
        """Forward the control rate to every held skill."""
        super().set_control_frequency(hz)
        for pol in self._policies.values():
            pol.set_control_frequency(hz)

    def reset(self, seed: int | None = None) -> None:
        """Reset every held skill's per-episode state and clear the episodic timer.

        A bundle that declares episodic skills also normalises the active
        skill here: an episode may be mid-flight, and the runtime calls this
        between episodes, so leaving a half-run kick selected would start the
        next episode inside a behaviour whose timer has been cleared.

        A bundle that declares none is left exactly as it was. The active
        skill is the caller's to choose there - ``switch`` is the only way it
        moves - and this seam is called once per episode by the rollout, so
        normalising unconditionally would silently discard that choice at
        every episode boundary of a multi-episode run.
        """
        for pol in self._policies.values():
            pol.reset(seed)
        if self._episodic_durations:
            self._episodic_active = None
            self._episodic_time_left = 0.0
            self._active = self._default_skill

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Delegate this tick to the active skill, after any requested switch.

        Ordering per tick (mirrors Pollen ``infer_policy.py``): an explicit
        ``select=`` cancels any running episodic behavior (like :meth:`switch`);
        otherwise if an episodic behavior is running it stays active and the
        velocity gate is skipped (so the FSM cannot yank a kick mid-episode);
        otherwise the velocity gate fires. After the active child runs, the
        episodic timer decrements by ``1/control_frequency`` and, if it hits
        zero, the bundle reverts to ``default_skill`` for the NEXT tick. The
        last tick of the episode still executes the episodic skill, so a
        1.2s / 60-step kick at 50Hz produces 60 kick actions and the 61st
        tick runs default_skill.

        When the velocity gate is on, a ``target_velocity`` this tick cannot
        honor is refused before the gate arbitrates, naming this bundle and the
        parameter - so a refused tick leaves the active skill exactly as it was.
        """
        select = kwargs.get("select")
        if select is not None:
            self.switch(str(select))
        elif self._episodic_active is None and self._switch_on_velocity is not None:
            self._auto_switch(kwargs.get("target_velocity"))
        result = await self._policies[self._active].get_actions(observation_dict, instruction, **kwargs)
        if self._episodic_active is not None:
            self._tick_episodic()
        return result

    def _tick_episodic(self) -> None:
        """Decrement the episodic timer by one control step; revert on expiry.

        A missing ``control_frequency`` refuses (loudly) rather than silently
        assuming 50Hz - an assumed rate would mis-time every kick at any other
        loop rate. The runtime is expected to call :meth:`set_control_frequency`
        before the rollout, same contract as every other Policy consumer.
        """
        if self.control_frequency is None:
            raise RuntimeError(
                "MicroduckPolicyBundle: episodic timer requires control_frequency; "
                "call set_control_frequency(hz) before running an episodic behavior."
            )
        dt = 1.0 / self.control_frequency
        self._episodic_time_left -= dt
        if self._episodic_time_left <= 0.0:
            self._end_episode()

    def _end_episode(self) -> None:
        """Complete the running episodic behavior and revert to ``default_skill``."""
        self._episodic_active = None
        self._episodic_time_left = 0.0
        self._active = self._default_skill

    def _auto_switch(self, target_velocity: Any) -> None:
        """Arbitrate between ``move_key`` and ``idle_key`` by twist magnitude.

        The gate selects BETWEEN those two skills. It does not select INTO them
        from a third one, so an explicit :meth:`switch` (or ``select=``) to a
        skill outside the pair is not undone by the next tick that carries a
        ``target_velocity``.

        Every other skill the provider ships reads the same twist slots for
        something that is not a velocity, which is why the pair is the gate's
        whole domain. ``alpha_sitstand`` is the sharpest case: ``twist[0]`` is a
        posture flag there (``1`` sit, ``0`` stand, the same policy sitting,
        holding and standing back up), so both of its commands have a magnitude
        the gate would read as a walk request or an idle one - the sit routing to
        ``move_key`` at 1.0 and the stand to ``idle_key`` at 0.0. Neither ever
        reached the skill that was asked for. Pollen's ``infer_policy.py`` draws
        the same boundary in ``_update_policy_session``, which returns early for
        each of its non-pair modes ("Don't switch while sitting") before it looks
        at the magnitude.

        A ``target_velocity`` the tick will refuse is refused here first, before
        the selection moves, so no failed tick leaves the bundle running a skill
        the caller did not ask for. An absent one (``None``) is still simply "no
        goal this tick" and leaves the selection alone.

        Raises:
            ValueError: If ``target_velocity`` is present but is not a velocity
                both this gate and the active child can read - a non-numeric or
                non-finite component, or a component count outside
                :data:`~strands_robots.policies.microduck.policy.TARGET_VELOCITY_WIDTHS`.
        """
        if self._move_key not in self._policies or self._idle_key not in self._policies:
            return
        if self._active not in (self._move_key, self._idle_key):
            return
        if target_velocity is None or self._switch_on_velocity is None:
            return
        # Asked before the coercion below, and before the selection moves. This
        # is the third reader of the well-known ``target_velocity`` key in the
        # family and was the only one not held to a domain - the other operand of
        # the very comparison two lines down, ``switch_on_velocity``, is held to
        # ``positive_finite_number_error`` at construction. The child's
        # ``_apply_command_kwargs`` states both reasons for its own guard, and
        # both of them apply here one layer up. A non-numeric value otherwise
        # surfaced as a bare ``could not convert string to float`` out of the
        # ``np.asarray`` below (a ``TypeError`` for a mapping, where this family
        # documents ``ValueError``), naming neither the bundle the caller called
        # nor the parameter it passed. And a non-finite one made ``mag`` ``nan``,
        # which is ``>=`` nothing, so the gate silently selected ``idle_key``: a
        # caller asking the robot to move at a ``nan`` velocity got the standing
        # skill, the tick was then refused by the child against a name the caller
        # never used, and the moved selection outlived that failed call.
        if error := _target_velocity_error(f"{type(self).__name__}.get_actions", target_velocity):
            raise ValueError(error)
        mag = float(np.linalg.norm(np.asarray(target_velocity, dtype=np.float32).reshape(-1)[:3]))
        self._active = self._move_key if mag >= self._switch_on_velocity else self._idle_key
