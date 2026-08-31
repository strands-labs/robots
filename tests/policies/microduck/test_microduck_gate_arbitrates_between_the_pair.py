"""The velocity gate arbitrates between its two keys instead of selecting into them.

``MicroduckPolicyBundle``'s ``switch_on_velocity`` gate exists to choose between
a ``move_key`` and an ``idle_key`` by ``|twist|`` each tick. Its own docstring
said "between", and it read the magnitude whatever skill was active - so an
explicit :meth:`~MicroduckPolicyBundle.switch` to any third skill was undone by
the very next tick that carried a ``target_velocity``, before that skill was
ticked even once.

Every skill the provider ships beyond the pair reads the same twist slots for
something that is not a velocity, so there is no value of ``target_velocity`` a
caller could have passed to stay. ``alpha_sitstand`` is the sharpest case, and
the one that makes the failure a motion fault rather than a lost tick: for that
policy ``twist[0]`` is a posture flag, ``1`` sit and ``0`` stand, with the same
policy sitting, holding and standing back up. So the documented sit command has
magnitude 1.0 and routed to ``move_key`` - a sit executed as a forward walk at
the flag's own value - and the stand command has magnitude 0.0 and routed to
``idle_key``. Both directions of the only control that policy has left it
unreachable, and the walking policy received the flag as a velocity.

Pollen's ``infer_policy.py`` draws the boundary in the same place. Its
``_update_policy_session`` is documented "Switch between walking and standing
sessions based on vel_cmd magnitude" and returns early for each of its non-pair
modes - ``ground_pick_mode``, ``sit_mode`` ("Don't switch while sitting"),
``slope_mode`` and an active ``behavior_mode`` - before it computes the
magnitude. Our gate carried only the first of those guards, the one that checks
both sessions are loaded.

Nothing graded it because the only cell exercising the gate builds a bundle
holding exactly the pair, where the missing guard cannot change an outcome: with
no third skill to be active, "arbitrates between the pair" and "selects into the
pair from anywhere" agree on every input. The cells below hold a bundle that
also carries third skills, which is the population the guard is about.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap

import numpy as np
import pytest

from strands_robots.policies.microduck import (
    MICRODUCK_JOINT_NAMES,
    MicroduckPolicy,
    MicroduckPolicyBundle,
)
from strands_robots.policies.microduck import composite as composite_mod

from .test_microduck_policy import _obs_dict, _StubSession

#: The command block's offset in the observation vector, derived from the layout
#: rather than restated: base_ang_vel(3) + projected_gravity(3) + three
#: per-joint blocks (joint_pos, joint_vel, last_action).
_COMMAND_OFFSET = 3 + 3 + 3 * len(MICRODUCK_JOINT_NAMES)

#: The gate's own two keys, and the skills that are not either of them. Kept as
#: names the provider actually ships so the population is the real one.
_MOVE_KEY = "walk"
_IDLE_KEY = "stand"
_THIRD_SKILLS = ("sitstand", "kick", "roulade", "ground_pick")

#: The two commands ``alpha_sitstand`` accepts, by Pollen's convention: the flag
#: rides ``twist[0]`` and the rest of the twist stays zero. Their magnitudes are
#: what the gate reads, which is why both of them routed away.
_SIT = [1.0, 0.0, 0.0]
_STAND = [0.0, 0.0, 0.0]

_GATE_THRESHOLD = 0.1


def _bundle(*, active: str, gate: float | None = _GATE_THRESHOLD, skills=None) -> MicroduckPolicyBundle:
    """A bundle holding the gate's pair plus third skills, each with its own stub."""
    names = skills if skills is not None else (_MOVE_KEY, _IDLE_KEY, *_THIRD_SKILLS)
    return MicroduckPolicyBundle(
        {name: MicroduckPolicy(session=_StubSession()) for name in names},  # type: ignore[arg-type]
        active=active,
        switch_on_velocity=gate,
        move_key=_MOVE_KEY,
        idle_key=_IDLE_KEY,
    )


def _recorded_input(policy: MicroduckPolicy) -> np.ndarray | None:
    """The observation a held skill's stub last received, or ``None`` if unticked.

    ``MicroduckPolicy._session`` is typed ``MicroduckSession | None`` and the
    stub's recording attribute is not on that protocol, so the narrowing lives
    here once instead of at every read - the same ``union-attr`` ignore the
    sibling suites carry, in one place.
    """
    return policy._session.last_input  # type: ignore[union-attr]


def _tick(bundle: MicroduckPolicyBundle, **kwargs: object) -> list[str]:
    """Run one tick and report which held skills the graph was actually run for."""
    asyncio.run(bundle.get_actions(_obs_dict(), "", **kwargs))
    return [name for name, pol in bundle._policies.items() if _recorded_input(pol) is not None]


def _command_block(bundle: MicroduckPolicyBundle, name: str) -> np.ndarray:
    """The command a held skill was handed, read off its own stub."""
    recorded = _recorded_input(bundle._policies[name])
    assert recorded is not None, f"{name} was never ticked"
    return recorded.reshape(-1)[_COMMAND_OFFSET:]


class TestAnExplicitSelectionSurvivesTheGate:
    """The regression: a third skill stays selected and is the one that runs."""

    @pytest.mark.parametrize("skill", _THIRD_SKILLS)
    @pytest.mark.parametrize("velocity,label", [(_SIT, "sit-flag"), (_STAND, "stand-flag")])
    def test_the_selected_skill_is_still_active_after_the_tick(self, skill, velocity, label):
        bundle = _bundle(active=_IDLE_KEY)
        bundle.switch(skill)
        _tick(bundle, target_velocity=velocity)
        assert bundle.active == skill

    @pytest.mark.parametrize("skill", _THIRD_SKILLS)
    @pytest.mark.parametrize("velocity,label", [(_SIT, "sit-flag"), (_STAND, "stand-flag")])
    def test_the_selected_skill_is_the_one_that_runs(self, skill, velocity, label):
        bundle = _bundle(active=_IDLE_KEY)
        bundle.switch(skill)
        assert _tick(bundle, target_velocity=velocity) == [skill]

    @pytest.mark.parametrize("skill", _THIRD_SKILLS)
    def test_the_pair_is_not_run_instead(self, skill):
        """The gate's own keys must not receive a tick meant for a third skill."""
        bundle = _bundle(active=_IDLE_KEY)
        bundle.switch(skill)
        ticked = _tick(bundle, target_velocity=_SIT)
        assert _MOVE_KEY not in ticked
        assert _IDLE_KEY not in ticked


class TestThePostureFlagReachesTheSitstandPolicy:
    """The sharpest case: both values of the flag reach the skill that reads it."""

    @pytest.mark.parametrize("velocity,expected", [(_SIT, 1.0), (_STAND, 0.0)])
    def test_the_flag_lands_in_the_first_twist_slot_of_that_policy(self, velocity, expected):
        bundle = _bundle(active=_IDLE_KEY)
        bundle.switch("sitstand")
        _tick(bundle, target_velocity=velocity)
        assert _command_block(bundle, "sitstand")[0] == pytest.approx(expected)

    def test_the_rest_of_the_twist_stays_zero(self):
        """Pollen writes the flag alone: ``cmd[0] = 1.0 if sit else 0.0``."""
        bundle = _bundle(active=_IDLE_KEY)
        bundle.switch("sitstand")
        _tick(bundle, target_velocity=_SIT)
        assert list(_command_block(bundle, "sitstand")[1:3]) == [0.0, 0.0]

    def test_the_walking_policy_never_receives_the_flag(self):
        """A sit command reaching ``move_key`` is the motion fault, not a lost tick."""
        bundle = _bundle(active=_IDLE_KEY)
        bundle.switch("sitstand")
        _tick(bundle, target_velocity=_SIT)
        assert _recorded_input(bundle._policies[_MOVE_KEY]) is None


class TestTheGateStillArbitratesBetweenItsPair:
    """Over-reach controls: the documented walk<->stand behaviour is unchanged."""

    def test_a_moving_command_selects_the_move_key_from_idle(self):
        bundle = _bundle(active=_IDLE_KEY)
        _tick(bundle, target_velocity=[0.5, 0.0, 0.0])
        assert bundle.active == _MOVE_KEY

    def test_a_zero_command_selects_the_idle_key_from_move(self):
        bundle = _bundle(active=_MOVE_KEY)
        _tick(bundle, target_velocity=_STAND)
        assert bundle.active == _IDLE_KEY

    def test_the_threshold_is_still_inclusive_at_its_own_value(self):
        bundle = _bundle(active=_IDLE_KEY)
        _tick(bundle, target_velocity=[_GATE_THRESHOLD, 0.0, 0.0])
        assert bundle.active == _MOVE_KEY

    def test_a_bundle_holding_only_the_pair_is_unaffected(self):
        """The guard is a no-op for the two-skill bundle the docs example builds."""
        bundle = _bundle(active=_IDLE_KEY, skills=(_MOVE_KEY, _IDLE_KEY))
        _tick(bundle, target_velocity=[0.5, 0.0, 0.0])
        assert bundle.active == _MOVE_KEY
        _tick(bundle, target_velocity=_STAND)
        assert bundle.active == _IDLE_KEY

    def test_returning_to_the_pair_re_enables_the_gate(self):
        """Selecting a gate key again is how a caller hands arbitration back.

        Whether the third skill survived the intervening tick is the regression
        above; this holds either way, and pins that the guard does not latch.
        """
        bundle = _bundle(active=_IDLE_KEY)
        bundle.switch("sitstand")
        _tick(bundle, target_velocity=_SIT)
        bundle.switch(_IDLE_KEY)
        _tick(bundle, target_velocity=[0.5, 0.0, 0.0])
        assert bundle.active == _MOVE_KEY


class TestWhatIsUnchangedEitherWay:
    """Premises and neighbours: true before and after, recorded so they stay so."""

    @pytest.mark.parametrize("skill", _THIRD_SKILLS)
    def test_an_explicit_per_tick_selection_still_wins(self, skill):
        """``select=`` is checked before the gate and is not affected by it."""
        bundle = _bundle(active=_IDLE_KEY)
        assert _tick(bundle, select=skill, target_velocity=_SIT) == [skill]
        assert bundle.active == skill

    @pytest.mark.parametrize("skill", _THIRD_SKILLS)
    def test_the_gate_being_off_already_left_a_third_skill_alone(self, skill):
        """With no threshold there is no gate to guard against."""
        bundle = _bundle(active=_IDLE_KEY, gate=None)
        bundle.switch(skill)
        assert _tick(bundle, target_velocity=_SIT) == [skill]

    def test_a_tick_with_no_velocity_leaves_the_active_skill_alone(self):
        bundle = _bundle(active=_IDLE_KEY)
        bundle.switch("sitstand")
        assert _tick(bundle) == ["sitstand"]

    def test_a_bundle_missing_a_gate_key_is_refused_before_it_can_tick(self):
        """Why the pair-held check the gate already carried is not the one at issue.

        A key naming no held skill is refused at construction, so that guard can
        no longer be reached through the constructor and the gate always has both
        of its skills by the time it runs. The membership this cell's siblings are
        about is the ACTIVE skill's, which construction says nothing about.
        """
        with pytest.raises(ValueError, match=r"names no held skill"):
            _bundle(active="sitstand", skills=("sitstand", _IDLE_KEY))

    def test_the_flag_magnitudes_are_what_the_gate_would_have_read(self):
        """Why both posture commands routed away, as arithmetic rather than prose."""
        assert float(np.linalg.norm(_SIT)) >= _GATE_THRESHOLD
        assert float(np.linalg.norm(_STAND)) < _GATE_THRESHOLD


class TestTheGuardIsAskedBeforeTheMagnitude:
    """Structural: the pair check precedes the arithmetic it exists to skip."""

    @staticmethod
    def _auto_switch_body() -> ast.FunctionDef:
        source = textwrap.dedent(inspect.getsource(MicroduckPolicyBundle._auto_switch))
        node = ast.parse(source).body[0]
        assert isinstance(node, ast.FunctionDef)
        return node

    def test_the_active_skill_is_checked_against_both_gate_keys(self):
        """A guard reading only one key would leave the other direction open."""
        node = self._auto_switch_body()
        reads = {
            ast.unparse(sub) for statement in node.body for sub in ast.walk(statement) if isinstance(sub, ast.Attribute)
        }
        assert "self._move_key" in reads
        assert "self._idle_key" in reads

    def test_the_pair_membership_check_precedes_the_magnitude(self):
        node = self._auto_switch_body()
        guard_line = next(
            (
                statement.lineno
                for statement in node.body
                if isinstance(statement, ast.If) and "self._active" in ast.unparse(statement.test)
            ),
            None,
        )
        assert guard_line is not None, "no statement guards on the active skill"
        magnitude_line = next(
            (statement.lineno for statement in node.body if "linalg.norm" in ast.unparse(statement)),
            None,
        )
        assert magnitude_line is not None, "the gate no longer computes a magnitude"
        assert guard_line < magnitude_line

    def test_the_module_records_why_the_pair_is_the_whole_domain(self):
        doc = " ".join((composite_mod.MicroduckPolicyBundle._auto_switch.__doc__ or "").split())
        assert "sitstand" in doc
        assert "posture flag" in doc
