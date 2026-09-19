"""A floating-base robot's dataset carries its base columns, as on the other backends.

``get_observation`` reports a floating base as ``base_pos`` / ``base_quat`` /
``base_lin_vel`` / ``base_ang_vel``, but ``observation.state``'s schema is derived
from scalar **joint** names - so those four vector signals are dropped unless the
recorder declares them. ``DatasetRecorder.create`` takes ``extra_state_specs`` for
exactly that, and both sibling backends pass it, each with the same recorded reason:

    ... those base signals would be dropped and a locomotion / velocity-tracking /
    whole-body-control policy trained on the dataset would be base-blind.

Isaac was the only backend that did not. Adding ``base_*`` to ``get_observation``
without this left an Isaac humanoid dataset **13 columns short** of the MuJoCo one
for the same robot - and silently, because a missing column is not an error.

The gap only became reachable with this series: before it, ``add_robot`` on Isaac
created no articulation at all, so no floating-base robot existed to record.

The base is detected from ``fixed_base`` - the field the MJCF free-joint read
records - rather than from a joint id, because this backend has no compiled model to
interrogate. A fixed-base arm declares no base columns, so its schema is byte-for-byte
what it was.
"""

from __future__ import annotations

import threading
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.config import IsaacConfig  # noqa: E402
from strands_robots.simulation.isaac.simulation import IsaacSimulation  # noqa: E402

#: The four signals and their components, in the order the siblings declare them.
_EXPECTED = [
    ("base_pos", ["x", "y", "z"]),
    ("base_quat", ["w", "x", "y", "z"]),
    ("base_lin_vel", ["x", "y", "z"]),
    ("base_ang_vel", ["x", "y", "z"]),
]


class _Articulation:
    dof_names = ["j0", "j1"]

    def get_joint_positions(self) -> Any:
        return np.zeros(2, dtype=np.float32)

    def get_joint_velocities(self) -> Any:
        return np.zeros(2, dtype=np.float32)


def _robot(name: str, *, fixed_base: bool) -> Any:
    return types.SimpleNamespace(
        name=name,
        joint_names=["j0", "j1"],
        articulation=_Articulation(),
        fixed_base=fixed_base,
        data_config=name,
        usd_to_urdf_joint_names=None,
    )


def _engine(robots: dict[str, Any]) -> Any:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(render_mode="headless")
    engine._world_created = True
    engine._world = types.SimpleNamespace()
    engine._robots = robots
    engine._cameras = {}
    engine._objects = {}
    engine._pump_running = False
    engine._main_tid = threading.get_ident()
    engine.robot_action_keys = lambda robot_name: ["j0", "j1"]  # type: ignore[method-assign]
    return engine


def _specs(engine: Any) -> list[tuple[str, list[str]]]:
    """The seventh element of the schema tuple: the base specs."""
    return engine._collect_recording_schema({})[6]


class TestAFloatingBaseRobotDeclaresItsBaseColumns:
    def test_all_four_signals_are_declared(self) -> None:
        engine = _engine({"g1": _robot("g1", fixed_base=False)})

        assert _specs(engine) == _EXPECTED

    def test_that_is_thirteen_scalar_columns(self) -> None:
        """The count the dataset gains, stated so a change to it is visible."""
        engine = _engine({"g1": _robot("g1", fixed_base=False)})

        assert sum(len(components) for _src, components in _specs(engine)) == 13

    def test_the_component_order_matches_the_sibling_backends(self) -> None:
        """``base_quat`` is (w,x,y,z) - scalar-first. A dataset recorded with the
        components in a different order would train a policy on a silently
        different quaternion convention."""
        engine = _engine({"g1": _robot("g1", fixed_base=False)})
        by_src = dict(_specs(engine))

        assert by_src["base_quat"] == ["w", "x", "y", "z"]
        assert by_src["base_pos"] == ["x", "y", "z"]

    def test_the_specs_reach_the_recorder(self) -> None:
        """Declaring them and not passing them is the same as not declaring them."""
        import inspect

        source = inspect.getsource(IsaacSimulation.start_recording)

        assert "extra_state_specs=base_state_specs" in source

    def test_the_resume_check_uses_the_expanded_names(self) -> None:
        """A resumed dataset's on-disk state includes the base columns, so
        validating against the bare joint list reports a mismatch on every
        floating-base append."""
        import inspect

        source = inspect.getsource(IsaacSimulation.start_recording)

        # Asserted on the CALL, not on the name merely existing: a version that
        # computed state_names_full and then verified against joint_names anyway
        # passed the weaker check.
        import re

        call = re.search(r"_verify_resume_schema\(\s*([^)]*)\)", source, re.S)
        assert call is not None, "the resume verification call is gone"
        args = [a.strip() for a in call.group(1).split(",")]
        assert args[1] == "state_names_full", f"resume verified against {args[1]!r}, not the expanded names"


class TestAFixedBaseRobotDeclaresNone:
    """The control. Declaring base columns for an arm would add 13 dead columns."""

    def test_no_specs_for_a_fixed_base_arm(self) -> None:
        engine = _engine({"arm": _robot("arm", fixed_base=True)})

        assert _specs(engine) == []

    def test_no_specs_when_fixed_base_is_absent(self) -> None:
        """A robot state built before this field existed must read as fixed, not
        floating - the safe direction, since a fixed-base arm emits no base_* and
        the columns would be dead."""
        engine = _engine({"arm": _robot("arm", fixed_base=True)})
        del engine._robots["arm"].fixed_base

        assert _specs(engine) == []

    def test_the_joint_columns_are_unchanged(self) -> None:
        engine = _engine({"arm": _robot("arm", fixed_base=True)})

        joint_names = engine._collect_recording_schema({})[0]

        assert joint_names == ["j0", "j1"]


class TestMultiRobotColumnsArePrefixed:
    """Matching the prefixed observation keys the recording hook emits, and the
    prefixing the sibling backends apply to their base columns."""

    def test_two_robots_prefix_the_base_columns(self) -> None:
        engine = _engine({"alice": _robot("alice", fixed_base=False), "bob": _robot("bob", fixed_base=False)})

        sources = [src for src, _c in _specs(engine)]

        assert "alice__base_quat" in sources
        assert "bob__base_quat" in sources

    def test_only_the_floating_robot_of_a_mixed_pair_contributes(self) -> None:
        engine = _engine({"arm": _robot("arm", fixed_base=True), "g1": _robot("g1", fixed_base=False)})

        sources = [src for src, _c in _specs(engine)]

        assert all(src.startswith("g1__") for src in sources), sources
        assert len(sources) == 4

    def test_a_single_robot_is_not_prefixed(self) -> None:
        engine = _engine({"g1": _robot("g1", fixed_base=False)})

        assert [src for src, _c in _specs(engine)] == ["base_pos", "base_quat", "base_lin_vel", "base_ang_vel"]


class TestTheSiblingBackendsDoTheSame:
    """Derived from their source, so a divergence shows up here rather than in a
    dataset. This is the parity the whole series claims."""

    @pytest.mark.parametrize("backend", ["mujoco", "newton"])
    def test_the_sibling_passes_extra_state_specs(self, backend: str) -> None:
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parents[3] / "strands_robots" / "simulation" / backend / "recording.py"
        ).read_text()

        assert "extra_state_specs=" in source, f"{backend} no longer declares base columns"

    @pytest.mark.parametrize("backend", ["mujoco", "newton"])
    def test_the_sibling_declares_the_same_four_signals(self, backend: str) -> None:
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parents[3] / "strands_robots" / "simulation" / backend / "recording.py"
        ).read_text()

        for src, _components in _EXPECTED:
            assert f'"{src}"' in source or f"base_{src.split('base_')[-1]}" in source, (
                f"{backend} does not declare {src}"
            )
