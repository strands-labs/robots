"""IK-model resolution refusals on the Isaac ``move_to`` (follow-up to GH #2155).

Isaac's ``move_to`` cannot solve on the articulation alone: it resolves the
robot's ``data_config`` to a MuJoCo model, builds the mink damped-least-squares
bridge on it, reconciles the solved joints with the articulation by NAME, and
only then drives PD targets. Every step of that resolution can fail on a real
install, and the adapter documents seven distinct refusals for it.

``tests/simulation/isaac/test_move_to_ik.py`` owns the SOLVE - convergence,
timeout, unreachable, mid-run abort, joint-name reconciliation - and reaches
them all through one healthy model. This module owns the RESOLUTION: the seven
ways the IK model or the articulation read can fail before or during a solve,
each mapped to the reason the caller is handed.

The distinction that matters for an agent driving this blind: a resolution
failure is not a convergence failure. "The arm did not reach the target" tells
the caller to raise ``max_steps`` or move the target; "the IK model for this
data_config has no non-gripper joints to solve" tells them the model is wrong.
Six of the seven refuse before a single PD target is written; the seventh
(a mid-run read failure) aborts a rollout in progress, so it is the one that
does command the arm first. Both halves are pinned below.

Like its sibling, this module needs no NVIDIA Isaac Sim: the articulation and
world are faked and the IK side runs on real inline MJCF. The refusals that
answer before the solver is built run anywhere; the rest ``importorskip`` on
``mujoco`` / ``mink`` (the dev env ships both).
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.simulation import IsaacSimulation

from .test_move_to_ik import (  # noqa: F401 - fake_articulation_action is an autouse fixture
    ARM_XML,
    MJCF_JOINTS,
    REACHABLE_LOCAL,
    _FakeArticulation,
    _FakeWorld,
    _make_sim,
    _text,
    fake_articulation_action,
)

# --- IK models that fail to resolve ----------------------------------------
# A model whose XML does not parse at all: MuJoCo refuses to compile it.
BROKEN_MJCF = '<mujoco model="prim_arm"><worldbody><body name="a"><joint name="j" type="hinge"/>'

# A scene fragment - geometry, no bodies. EE-frame discovery has no TCP site,
# no hand/tool body and no kinematic chain to take a leaf from.
NO_EE_FRAME_MJCF = """
<mujoco model="prim_arm">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <geom type="box" pos="0 0 0.1" size="0.1 0.1 0.1"/>
  </worldbody>
</mujoco>
"""

# A model whose ONLY joint is a gripper: an EE frame is discoverable, but there
# is nothing the solver may move to reach a Cartesian target.
GRIPPER_ONLY_MJCF = """
<mujoco model="prim_arm">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="base">
      <geom type="cylinder" size="0.04 0.02"/>
      <body name="hand" pos="0 0 0.05">
        <joint name="jaw" type="hinge" axis="0 0 1" range="-0.2 1.5"/>
        <geom type="box" size="0.01 0.01 0.02"/>
        <site name="ee_site" pos="0 0 0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# A registry ``gripper`` block that exists but is not the documented shape.
# Reachable through the user-local registry overlay; the shipped robots.json is
# shape-checked by its own tests.
MALFORMED_GRIPPER_METADATA: dict[str, Any] = {"gripper": {"actuators": [], "closed": "low", "open": "high"}}


def _point_resolve_model(tmp_path, monkeypatch, xml: str) -> str:
    """Write *xml* as the IK model ``data_config='prim_arm'`` resolves to."""
    path = tmp_path / "prim_arm.xml"
    path.write_text(xml)
    monkeypatch.setattr(
        "strands_robots.simulation.isaac.motion_primitives.resolve_model",
        lambda name, _p=str(path): _p if name == "prim_arm" else None,
    )
    return str(path)


class _UnreadableArticulation(_FakeArticulation):
    """Articulation whose joint-position read stops answering after N calls.

    ``bad_after=0`` fails the first read (before the solve); a positive value
    lets the rollout start and fails mid-run. ``bad_value`` is what the read
    returns instead - a too-short vector or nothing at all, the two shapes a
    real articulation reports when its DOF view is not ready.
    """

    def __init__(self, *args: Any, bad_after: int = 0, bad_value: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.bad_after = bad_after
        self.bad_value = bad_value
        self.reads = 0

    def get_joint_positions(self) -> Any:
        self.reads += 1
        if self.reads > self.bad_after:
            return self.bad_value
        return super().get_joint_positions()


def _sim_with_unreadable_state(**kwargs: Any) -> tuple[IsaacSimulation, _UnreadableArticulation]:
    """``_make_sim``'s wiring with an articulation that stops answering reads."""
    sim, _ = _make_sim()
    art = _UnreadableArticulation(MJCF_JOINTS, **kwargs)
    sim._world = _FakeWorld(art)
    sim._robots["arm"].articulation = art
    return sim, art


def _move(sim: IsaacSimulation) -> dict[str, Any]:
    return sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, tol=0.02, max_steps=60)


# Every scenario below is one documented resolution failure. The value builds a
# simulation primed for it; the loop in TestTheseRefusalsShareOneContract drives
# the whole family, so the table is the single source of truth for what the
# family IS.
def _sc_stack_absent(tmp_path, monkeypatch):
    _point_resolve_model(tmp_path, monkeypatch, ARM_XML)
    monkeypatch.setitem(sys.modules, "mujoco", None)
    return _make_sim()


def _sc_does_not_compile(tmp_path, monkeypatch):
    _point_resolve_model(tmp_path, monkeypatch, BROKEN_MJCF)
    return _make_sim()


def _sc_no_ee_frame(tmp_path, monkeypatch):
    _point_resolve_model(tmp_path, monkeypatch, NO_EE_FRAME_MJCF)
    return _make_sim()


def _sc_nothing_to_solve(tmp_path, monkeypatch):
    _point_resolve_model(tmp_path, monkeypatch, GRIPPER_ONLY_MJCF)
    return _make_sim(joint_names=["jaw"])


def _sc_malformed_metadata(tmp_path, monkeypatch):
    _point_resolve_model(tmp_path, monkeypatch, ARM_XML)
    monkeypatch.setattr(
        IsaacSimulation,
        "_get_registry_robot",
        staticmethod(lambda _dc: MALFORMED_GRIPPER_METADATA),
    )
    return _make_sim()


def _sc_read_short(tmp_path, monkeypatch):
    _point_resolve_model(tmp_path, monkeypatch, ARM_XML)
    return _sim_with_unreadable_state(bad_value=np.zeros(1))


def _sc_read_missing(tmp_path, monkeypatch):
    _point_resolve_model(tmp_path, monkeypatch, ARM_XML)
    return _sim_with_unreadable_state(bad_value=None)


def _sc_read_fails_mid_run(tmp_path, monkeypatch):
    _point_resolve_model(tmp_path, monkeypatch, ARM_XML)
    return _sim_with_unreadable_state(bad_after=2, bad_value=np.zeros(1))


# label -> (builder, needs mink, commands the arm before refusing)
_SCENARIOS: dict[str, tuple[Any, bool, bool]] = {
    "stack-absent": (_sc_stack_absent, False, False),
    "does-not-compile": (_sc_does_not_compile, False, False),
    "no-ee-frame": (_sc_no_ee_frame, False, False),
    "nothing-to-solve": (_sc_nothing_to_solve, False, False),
    "malformed-gripper-metadata": (_sc_malformed_metadata, False, False),
    "read-short": (_sc_read_short, True, False),
    "read-missing": (_sc_read_missing, True, False),
    "read-fails-mid-run": (_sc_read_fails_mid_run, True, True),
}


def _provoke(label: str, tmp_path, monkeypatch) -> tuple[dict[str, Any], Any]:
    """Drive *label*'s scenario and return ``(result, articulation)``."""
    builder, needs_mink, _ = _SCENARIOS[label]
    if needs_mink:
        pytest.importorskip("mujoco")
        pytest.importorskip("mink")
    sim, art = builder(tmp_path, monkeypatch)
    return _move(sim), art


class TestTheIkModelCannotBeResolved:
    """The model ``data_config`` names is missing, unusable or has no EE frame.

    Each of these answers before the solver exists, so none of them needs the
    IK stack installed - which is the install where a caller most needs the
    reason rather than a traceback.
    """

    def test_the_ik_stack_is_not_importable(self, tmp_path, monkeypatch) -> None:
        result, art = _provoke("stack-absent", tmp_path, monkeypatch)
        assert result["status"] == "error"
        text = _text(result)
        assert "needs the 'mujoco' + 'mink' stack" in text
        assert "sim-mujoco" in text or "install" in text, text
        assert art.applied == []

    def test_the_ik_model_does_not_compile(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("mujoco")
        result, art = _provoke("does-not-compile", tmp_path, monkeypatch)
        assert result["status"] == "error"
        text = _text(result)
        assert "could not compile the IK model" in text
        # The reason names the data_config AND the file, so the caller knows
        # which registry entry to fix rather than which primitive failed.
        assert "prim_arm" in text and "prim_arm.xml" in text
        assert art.applied == []

    def test_no_end_effector_frame_can_be_discovered(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("mujoco")
        result, art = _provoke("no-ee-frame", tmp_path, monkeypatch)
        assert result["status"] == "error"
        text = _text(result)
        assert "could not auto-discover an end-effector frame" in text
        assert art.applied == []


class TestTheIkModelHasNothingToSolve:
    """The model compiles and has an EE frame, but no solvable joint set.

    Both refusals are about the JOINT MAP rather than the model file, and both
    are reachable on a real install: a robot whose only joint is its gripper,
    and a user-local registry overlay whose ``gripper`` block is not the
    documented shape.
    """

    def test_every_joint_is_classified_as_a_gripper(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("mujoco")
        result, art = _provoke("nothing-to-solve", tmp_path, monkeypatch)
        assert result["status"] == "error"
        text = _text(result)
        assert "no non-gripper hinge/slide joints to solve" in text
        # It names the data_config and says why nothing can move, so the caller
        # knows the model is wrong rather than the target.
        assert "prim_arm" in text and "nothing can move the end-effector" in text
        assert art.applied == []

    def test_registry_gripper_metadata_is_malformed(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("mujoco")
        result, art = _provoke("malformed-gripper-metadata", tmp_path, monkeypatch)
        assert result["status"] == "error"
        text = _text(result)
        assert "is malformed" in text
        # The reason quotes the offending block and names where to fix it, so
        # the caller is not left guessing which registry entry is at fault.
        assert "prim_arm" in text and "user_robots.json" in text
        assert art.applied == []

    def test_a_well_formed_gripper_block_still_solves(self, tmp_path, monkeypatch) -> None:
        """The refusal above is about the SHAPE, not about having metadata."""
        pytest.importorskip("mujoco")
        pytest.importorskip("mink")
        _point_resolve_model(tmp_path, monkeypatch, ARM_XML)
        monkeypatch.setattr(
            IsaacSimulation,
            "_get_registry_robot",
            staticmethod(lambda _dc: {"gripper": {"actuators": ["jaw"], "closed": "low", "open": "high"}}),
        )
        sim, art = _make_sim()
        result = _move(sim)
        assert result["status"] == "success", _text(result)
        assert art.applied, "a usable gripper block must not stop the solve"


class TestTheArticulationStateCannotBeRead:
    """The model is fine; the articulation does not report a usable state.

    A real articulation answers with a short vector or nothing at all while its
    DOF view is still being built. Before the solve that is a refusal with no
    command written; mid-rollout it aborts a run already in progress, which is
    a different report and the only member of this family that has commanded
    the arm.
    """

    @pytest.mark.parametrize("label", ["read-short", "read-missing"], ids=["short-vector", "no-vector"])
    def test_an_unusable_pre_solve_read_refuses_before_commanding(self, label, tmp_path, monkeypatch) -> None:
        result, art = _provoke(label, tmp_path, monkeypatch)
        assert result["status"] == "error"
        assert "did not report a usable joint-position vector" in _text(result)
        assert art.applied == []

    def test_a_read_that_fails_mid_run_aborts_and_says_so(self, tmp_path, monkeypatch) -> None:
        result, art = _provoke("read-fails-mid-run", tmp_path, monkeypatch)
        assert result["status"] == "error"
        text = _text(result)
        assert "mid-run" in text and "aborting" in text
        # This is the one refusal that has already moved the arm: the caller
        # needs to know the rollout was interrupted, not that it never started.
        assert art.applied, "a mid-run abort must be distinguishable from a pre-flight refusal"

    def test_the_mid_run_report_is_not_the_pre_solve_report(self, tmp_path, monkeypatch) -> None:
        """Same cause, two moments - and the caller can tell them apart."""
        pre, _ = _provoke("read-short", tmp_path, monkeypatch)
        mid, _ = _provoke("read-fails-mid-run", tmp_path, monkeypatch)
        assert _text(pre) != _text(mid)


class TestTheseRefusalsShareOneContract:
    """What holds across the whole family - and the one thing that does not.

    Driven from ``_SCENARIOS`` so a refusal added to the adapter has to be
    added here to be covered, rather than silently joining an untested set.
    """

    def test_the_scenario_table_covers_every_documented_refusal(self) -> None:
        assert len(_SCENARIOS) == 8, "one scenario per documented resolution failure (+1 read shape)"

    @pytest.mark.parametrize("label", sorted(_SCENARIOS))
    def test_every_resolution_failure_is_a_structured_error(self, label, tmp_path, monkeypatch) -> None:
        """Never a raise past the envelope, and never a false success."""
        result, _ = _provoke(label, tmp_path, monkeypatch)
        assert result["status"] == "error", result
        assert _text(result).strip(), "a refusal with no reason is a dead end"

    @pytest.mark.parametrize("label", sorted(_SCENARIOS))
    def test_only_a_mid_run_abort_has_commanded_the_arm(self, label, tmp_path, monkeypatch) -> None:
        """Pre-flight refusals cost nothing; the mid-run one interrupts a run."""
        _, _, commands_first = _SCENARIOS[label]
        _, art = _provoke(label, tmp_path, monkeypatch)
        assert bool(art.applied) is commands_first

    def test_the_reasons_are_pairwise_distinguishable(self, tmp_path, monkeypatch) -> None:
        """Eight failures, eight reports - a caller can act on which one it is."""
        texts = {}
        for label in sorted(_SCENARIOS):
            # Each scenario needs its OWN patch scope: 'stack-absent' hides
            # mujoco and 'malformed-gripper-metadata' patches the registry
            # reader, so a shared monkeypatch would make every later scenario
            # answer with an earlier one's refusal.
            with pytest.MonkeyPatch.context() as mp:
                result, _ = _provoke(label, tmp_path, mp)
            texts[label] = _text(result)
        # read-short and read-missing are the same refusal reached two ways.
        unique = {t for lbl, t in texts.items() if lbl != "read-missing"}
        assert len(unique) == len(texts) - 1, texts

    def test_the_gripper_metadata_refusal_names_the_gripper_not_the_primitive(self, tmp_path, monkeypatch) -> None:
        """Observed asymmetry, pinned so a wording change is deliberate.

        Six of the seven reasons are prefixed with the primitive that failed;
        the malformed-metadata one leads with the gripper it could not resolve
        because that message is raised by the shared registry reader and is
        worded for every primitive that consults it.
        """
        pytest.importorskip("mujoco")
        with pytest.MonkeyPatch.context() as mp:
            result, _ = _provoke("malformed-gripper-metadata", tmp_path, mp)
        text = _text(result)
        assert text.startswith("Cannot resolve the gripper for 'arm'"), text
        for label in ["does-not-compile", "no-ee-frame", "nothing-to-solve"]:
            with pytest.MonkeyPatch.context() as mp:
                other, _ = _provoke(label, tmp_path, mp)
            assert _text(other).startswith("move_to:"), other

    def test_a_resolvable_model_still_converges(self, tmp_path, monkeypatch) -> None:
        """The over-reach control: none of the above refuses a healthy setup."""
        pytest.importorskip("mujoco")
        pytest.importorskip("mink")
        _point_resolve_model(tmp_path, monkeypatch, ARM_XML)
        sim, art = _make_sim()
        result = _move(sim)
        assert result["status"] == "success", _text(result)
        assert art.applied, "the control must actually command the arm"
