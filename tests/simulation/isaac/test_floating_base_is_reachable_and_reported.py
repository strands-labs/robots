"""A floating base is reachable on Isaac, and reported the way the schema says.

Two halves of one defect, because neither is usable without the other.

**Nothing on this backend could have a floating base.** ``fix_base = True`` was
hardcoded at *both* URDF import paths - the modern ``URDFImporterConfig`` and the
legacy ``_urdf.ImportConfig`` - with no way for a caller to change it. So every
URDF robot was welded to the world: a humanoid could not fall, a quadruped could
not walk, and nothing said so. Measured on ``nvcr.io/nvidia/isaac-sim:6.0.1``
(A10G), a two-link URDF spawned at ``z=1.0`` read base ``z 0.0 -> 0.0`` across 120
steps.

**And the ``base_*`` observation keys were absent.** The
:meth:`~strands_robots.simulation.base.SimEngine.get_observation` schema requires
that a robot whose root is a 6-DoF free joint surface ``base_pos``, ``base_quat``,
``base_lin_vel`` and ``base_ang_vel`` rather than reporting the free joint as a
scalar. MuJoCo emits all four (``mujoco/rendering.py``) and Newton emits all four
(``newton/simulation.py``); across all 11 files of the Isaac package there were
zero occurrences. A locomotion policy reading ``base_lin_vel`` - the base twist
every walking controller is conditioned on - got nothing on this backend and a
value on the other two, which is exactly what makes the page's "policies and
observation mappings transfer unchanged between backends" false for a legged robot.

The two are one change because the absence was *consistent*: with every base
welded, the four keys would have reported four constants. Fixing the observation
without the knob emits constants; fixing the knob without the observation leaves a
falling robot no one can observe.

``fix_base`` is a parameter rather than something read out of the file because URDF
cannot answer it. The format has a ``floating`` joint type, but the universal
convention for a mobile robot is a root link with no parent joint - byte-identical
to how a bolted-down arm declares its base - so the consumer chooses. That is why
Isaac's own importer takes the flag. MJCF *can* say (``<freejoint>``), which is why
MuJoCo and Newton have no such parameter.

Scope: these pin the plumbing, the posture-flag domain, the refusal on the paths
that cannot honour the flag, and the emission rule. That a robot imported with
``fix_base=False`` actually *falls* is a physics claim, verified on GPU.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import textwrap
import threading
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    IsaacConfig,
    IsaacSimulation,
    _RobotState,
)
from strands_robots.utils import boolean_flag_error  # noqa: E402

#: The four keys the schema reserves for a floating base.
_BASE_KEYS = ("base_pos", "base_quat", "base_lin_vel", "base_ang_vel")


class _Articulation:
    """A handle answering the three readers the base state is built from."""

    def __init__(
        self,
        pos: Any = (1.0, 2.0, 3.0),
        quat: Any = (1.0, 0.0, 0.0, 0.0),
        lin: Any = (0.1, 0.2, 0.3),
        ang: Any = (0.4, 0.5, 0.6),
        joints: Any = (0.7,),
    ) -> None:
        self._pos, self._quat, self._lin, self._ang, self._joints = pos, quat, lin, ang, joints

    def get_joint_positions(self) -> Any:
        return np.asarray(self._joints, dtype=np.float32)

    def get_world_pose(self) -> Any:
        return np.asarray(self._pos, dtype=np.float32), np.asarray(self._quat, dtype=np.float32)

    def get_linear_velocity(self) -> Any:
        return np.asarray(self._lin, dtype=np.float32)

    def get_angular_velocity(self) -> Any:
        return np.asarray(self._ang, dtype=np.float32)


def _engine(*, fixed_base: bool, articulation: Any = None) -> Any:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(render_mode="headless")
    engine._world = types.SimpleNamespace()
    engine._world_created = True
    engine._cameras = {}
    engine._objects = {}
    engine._replicated = False
    engine._prim_registry = []
    robot = _RobotState(
        name="arm",
        prim_path="/World/Robots/arm",
        joint_names=["j0"],
        articulation=_Articulation() if articulation is None else articulation,
        fixed_base=fixed_base,
    )
    engine._robots = {"arm": robot}
    return engine


class TestTheFlagReachesTheImporter:
    """It was hardcoded at both paths; a knob that reaches neither is decoration."""

    def _urdf_loader_source(self) -> str:
        return inspect.getsource(IsaacSimulation._load_urdf_robot)

    def test_add_robot_accepts_the_flag(self) -> None:
        assert "fix_base" in inspect.signature(IsaacSimulation.add_robot).parameters

    def test_the_default_is_a_welded_base(self) -> None:
        """The historical behaviour, and what the shipped LIBERO Franka needs."""
        assert inspect.signature(IsaacSimulation.add_robot).parameters["fix_base"].default is True

    def test_the_loader_takes_it(self) -> None:
        assert "fix_base" in inspect.signature(IsaacSimulation._load_urdf_robot).parameters

    def test_neither_import_path_hardcodes_it(self) -> None:
        """Both sites, because a knob honoured on one path is worse than none: the
        behaviour would then depend on which importer the runtime happens to
        expose."""
        source = self._urdf_loader_source()
        assert '("fix_base", True)' not in source
        assert "import_config.fix_base = True" not in source
        assert source.count("fix_base") >= 2, "both import paths must read the parameter"

    def test_add_robot_forwards_it(self) -> None:
        """Read off the call, so a parameter that is accepted and dropped fails."""
        source = inspect.getsource(IsaacSimulation.add_robot)
        tree = ast.parse(textwrap.dedent(source))
        forwarded = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_load_urdf_robot"
            ):
                names = [a.id for a in node.args if isinstance(a, ast.Name)]
                names += [k.value.id for k in node.keywords if isinstance(k.value, ast.Name)]
                forwarded = forwarded or "fix_base" in names
        assert forwarded, "_load_urdf_robot is called without forwarding fix_base"


class TestTheFlagIsCheckedNotReadByTruthiness:
    """A posture flag: it selects which base the robot gets."""

    @pytest.mark.parametrize("bad", ["false", "no", "off", "0", "", None, 0, 1, [], "true"])
    def test_a_non_boolean_is_refused(self, bad: Any) -> None:
        """``fix_base="false"`` is the one that costs: a non-empty string is
        truthy, so it would WELD the base of a caller who spelled out that they
        wanted it free, and the only symptom is a humanoid that never falls under
        a success envelope."""
        engine = _engine(fixed_base=True)
        result = engine.add_robot("r", urdf_path="/tmp/x.urdf", fix_base=bad)
        assert result["status"] == "error", result
        assert "fix_base" in result["content"][0]["text"]

    def test_the_domain_is_the_shared_one(self) -> None:
        """Parametrized over the shared domain rather than a copied spelling list,
        so a spelling added there is covered without an edit here."""
        for bad in ("false", "no", "off", "0", None, 0, 1):
            assert boolean_flag_error(bad, "fix_base", "add_robot") is not None
        for good in (True, False, np.bool_(True), np.bool_(False)):
            assert boolean_flag_error(good, "fix_base", "add_robot") is None


class TestAPathThatCannotHonourItRefuses:
    """Accept-and-ignore is the failure this avoids."""

    def test_a_usd_import_refuses_a_free_base(self) -> None:
        engine = _engine(fixed_base=True)
        result = engine.add_robot("r", usd_path="/tmp/x.usda", fix_base=False)
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "fix_base=False" in text
        assert "urdf_path" in text

    def test_a_procedural_add_refuses_a_free_base(self) -> None:
        engine = _engine(fixed_base=True)
        result = engine.add_robot("r", fix_base=False)
        assert result["status"] == "error", result
        assert "fix_base=False" in result["content"][0]["text"]

    def test_a_usd_import_still_accepts_the_default(self) -> None:
        """Control: the refusal is about the free base, not about the flag being
        present."""
        engine = _engine(fixed_base=True)
        result = engine.add_robot("r", usd_path="/tmp/x.usda", fix_base=True)
        assert "fix_base" not in result["content"][0]["text"]


class TestTheObservationReportsAFloatingBase:
    def test_all_four_keys_are_emitted(self) -> None:
        obs = _engine(fixed_base=False).get_observation()
        for key in _BASE_KEYS:
            assert key in obs, f"{key} missing from {sorted(obs)}"

    def test_the_values_are_the_ones_the_handle_reports(self) -> None:
        engine = _engine(
            fixed_base=False,
            articulation=_Articulation(pos=(1.5, -2.5, 0.75), quat=(0.0, 1.0, 0.0, 0.0), lin=(9.0, 8.0, 7.0)),
        )
        obs = engine.get_observation()
        assert obs["base_pos"] == pytest.approx([1.5, -2.5, 0.75])
        assert obs["base_quat"] == pytest.approx([0.0, 1.0, 0.0, 0.0])
        assert obs["base_lin_vel"] == pytest.approx([9.0, 8.0, 7.0])

    def test_the_values_are_plain_floats(self) -> None:
        """The schema is consumed by dataset columns and by JSON, so a numpy
        scalar leaking through is a serialisation failure somewhere far away."""
        obs = _engine(fixed_base=False).get_observation()
        for key in _BASE_KEYS:
            assert all(type(v) is float for v in obs[key]), f"{key} carries a non-float: {obs[key]}"

    def test_the_joint_entries_are_unaffected(self) -> None:
        obs = _engine(fixed_base=False).get_observation()
        assert obs["j0"] == pytest.approx(0.7)

    def test_a_fixed_base_emits_none_of_them(self) -> None:
        """The schema reserves these for a robot that HAS a base to report:
        "Absent for fixed-base arms". A welded root would report four constants."""
        obs = _engine(fixed_base=True).get_observation()
        for key in _BASE_KEYS:
            assert key not in obs, f"{key} emitted for a welded base"
        assert obs["j0"] == pytest.approx(0.7)

    def test_a_handle_that_cannot_answer_omits_rather_than_substitutes(self) -> None:
        """Zeros would report a base at the origin, at rest - which is a reading a
        consumer cannot tell from a real one."""

        class _Raising(_Articulation):
            def get_world_pose(self) -> Any:
                raise RuntimeError("physics view not created yet")

        obs = _engine(fixed_base=False, articulation=_Raising()).get_observation()
        for key in _BASE_KEYS:
            assert key not in obs
        # The joint state still comes back: the schema says joint state MUST be
        # returned even when other reads fail.
        assert obs["j0"] == pytest.approx(0.7)


class TestParityWithTheBackendsThatAlreadyDidThis:
    def test_the_key_names_match_the_other_backends(self) -> None:
        """Derived from the two backends that already emit them, so a rename on
        either side is caught rather than leaving three spellings of one schema."""
        import strands_robots.simulation.mujoco.rendering as mj_rendering
        import strands_robots.simulation.newton.simulation as newton_sim

        for module in (mj_rendering, newton_sim):
            path = module.__file__
            assert path is not None, f"{module.__name__} has no source file to read"
            source = pathlib.Path(path).read_text(encoding="utf-8")
            for key in _BASE_KEYS:
                assert f'"{key}"' in source, f"{key} is not the spelling {module.__name__} uses"

    def test_the_abc_documents_these_keys(self) -> None:
        from strands_robots.simulation.base import SimEngine

        doc = inspect.getdoc(SimEngine.get_observation) or ""
        for key in _BASE_KEYS:
            assert key in doc

    def test_the_isaac_backend_now_emits_them_somewhere(self) -> None:
        """The census that made this a defect: zero occurrences across the whole
        Isaac package, against four in each of the other two."""
        import strands_robots.simulation.isaac.simulation as isaac_sim

        source = pathlib.Path(isaac_sim.__file__).read_text(encoding="utf-8")
        for key in _BASE_KEYS:
            assert f'"{key}"' in source
