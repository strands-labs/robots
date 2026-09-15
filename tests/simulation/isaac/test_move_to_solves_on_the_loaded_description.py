"""``move_to``'s IK model is the description the robot was built from.

The IK solve runs on a compiled MuJoCo model, and it used to resolve one from
exactly one place: the robot's ``data_config``, through the registry. The file
the robot was actually BUILT from - the URDF ``add_robot`` imported, or the
MJCF an imported USD was converted from, both of which MuJoCo compiles - was
discarded at add time. Two costs, one loud and one silent:

* **loud** - a robot added via a bare ``urdf_path`` (no registry entry) was
  refused ``move_to`` outright, and a robot added by registry NAME
  (``add_robot("so100")``) was refused too, because the name path stored
  ``data_config=None`` unless the caller spelled it a second time;
* **silent** - a ``data_config`` naming a registry model that DIFFERS from the
  loaded asset while sharing every joint name solved on the wrong kinematics,
  and the convergence check runs by FK on the IK model itself
  (``pose_residuals``), so the solve CONFIRMED a pose the stage end-effector
  does not hold. "reached" with the arm somewhere else, and nothing anywhere
  disagreed.

``_RobotState`` now records ``description_path`` (URDF, or the pre-conversion
MJCF; ``None`` for a plain USD, which MuJoCo cannot compile), and
``_load_ik_mjcf`` prefers it over the registry lookup - the IK model is the
simulating file by construction. The registry path survives as the fallback
for description-less robots, and a divergence between the two is a WARNING
naming both files. The seven resolution refusals pinned by
:mod:`tests.simulation.isaac.test_move_to_ik_model_resolution` are unchanged
(their fixtures carry a ``data_config`` and no description).

Fixtures are the ``move_to`` siblings' (real inline MJCF for the IK side, a
fake articulation and world for the stage side); the ``add_robot`` threading is
pinned with the loaders stood in, the same seam
:mod:`tests.simulation.isaac.test_add_robot_resolves_a_real_description` uses.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from strands_robots.simulation.isaac import motion_primitives
from strands_robots.simulation.isaac.simulation import IsaacSimulation

from .test_move_to_ik import (  # noqa: F401 - fake_articulation_action is an autouse fixture
    ARM_XML,
    REACHABLE_LOCAL,
    _make_sim,
    fake_articulation_action,
)

pytestmark = pytest.mark.usefixtures("fake_articulation_action")


def _write_arm(tmp_path, name: str = "arm.xml") -> str:
    path = tmp_path / name
    path.write_text(ARM_XML)
    return str(path)


def _explode(name: str) -> Any:
    raise AssertionError(f"resolve_model was consulted for {name!r}; the description must win")


class TestTheDescriptionWins:
    def test_a_described_robot_solves_with_no_data_config_and_no_registry(self, tmp_path, monkeypatch) -> None:
        """The loud half: a bare-urdf_path (or name-added) robot gets move_to.

        ``resolve_model`` is patched to explode, so a pass proves the registry
        is not merely deprioritised but out of the path entirely.
        """
        monkeypatch.setattr(motion_primitives, "resolve_model", _explode)
        sim, _art = _make_sim(data_config=None)
        sim._robots["arm"].description_path = _write_arm(tmp_path)

        result = sim.move_to(position=REACHABLE_LOCAL, robot_name="arm")

        assert result["status"] == "success", result

    def test_the_description_beats_a_divergent_data_config_and_warns(self, tmp_path, monkeypatch, caplog) -> None:
        """The silent half. The registry resolves to a file that does not
        compile; solving on it would have been the wrong-kinematics solve, and
        failing on it would show the registry still won. Success proves the
        description was compiled, and the WARNING names both files so the
        caller who believes the registry model is solving learns otherwise."""
        described = _write_arm(tmp_path, "loaded.xml")
        divergent = tmp_path / "registry.xml"
        divergent.write_text("<mujoco><this does not compile></mujoco>")
        monkeypatch.setattr(
            motion_primitives,
            "resolve_model",
            lambda name: str(divergent) if name == "prim_arm" else None,
        )
        sim, _art = _make_sim(data_config="prim_arm")
        sim._robots["arm"].description_path = described

        with caplog.at_level(logging.WARNING, logger=motion_primitives.logger.name):
            result = sim.move_to(position=REACHABLE_LOCAL, robot_name="arm")

        assert result["status"] == "success", result
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert described in warnings[0]
        assert str(divergent) in warnings[0]
        assert "prim_arm" in warnings[0]

    def test_no_warning_when_both_name_the_same_file(self, tmp_path, monkeypatch, caplog) -> None:
        """The common case by construction: the name path resolves the registry
        file and records it as the description. A warning here would fire on
        every registry robot and train callers to ignore it."""
        described = _write_arm(tmp_path)
        monkeypatch.setattr(
            motion_primitives,
            "resolve_model",
            lambda name: described if name == "prim_arm" else None,
        )
        sim, _art = _make_sim(data_config="prim_arm")
        sim._robots["arm"].description_path = described

        with caplog.at_level(logging.WARNING, logger=motion_primitives.logger.name):
            result = sim.move_to(position=REACHABLE_LOCAL, robot_name="arm")

        assert result["status"] == "success", result
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_a_description_that_does_not_compile_is_an_error_not_a_fallback(self, tmp_path, monkeypatch) -> None:
        """Falling back to the registry after a compile failure would
        reintroduce, knowingly, the divergent-model solve this exists to
        remove. The file that is simulating failing to compile is a fact the
        caller must see."""
        broken = tmp_path / "loaded.xml"
        broken.write_text("<mujoco><not xml")
        healthy = _write_arm(tmp_path, "registry.xml")
        monkeypatch.setattr(
            motion_primitives,
            "resolve_model",
            lambda name: healthy if name == "prim_arm" else None,
        )
        sim, _art = _make_sim(data_config="prim_arm")
        sim._robots["arm"].description_path = str(broken)

        result = sim.move_to(position=REACHABLE_LOCAL, robot_name="arm")

        assert result["status"] == "error", result
        assert "compile" in result["content"][0]["text"].lower()


class TestTheRegistryFallbackSurvives:
    def test_a_description_less_robot_still_resolves_through_the_registry(self, tmp_path, monkeypatch) -> None:
        """Control: the pre-existing path, byte-for-byte - the seven refusals
        pinned by the sibling module all live on it."""
        consulted: list[str] = []
        healthy = _write_arm(tmp_path)

        def _resolve(name: str) -> str | None:
            consulted.append(name)
            return healthy if name == "prim_arm" else None

        monkeypatch.setattr(motion_primitives, "resolve_model", _resolve)
        sim, _art = _make_sim(data_config="prim_arm")
        assert getattr(sim._robots["arm"], "description_path", None) is None

        result = sim.move_to(position=REACHABLE_LOCAL, robot_name="arm")

        assert result["status"] == "success", result
        assert consulted == ["prim_arm"]

    def test_neither_source_is_a_refusal_naming_every_remedy(self, monkeypatch) -> None:
        """The reworded refusal: it must say WHY there is nothing to compile
        (a plain USD) and name all three ways out."""
        monkeypatch.setattr(motion_primitives, "resolve_model", _explode)
        sim, _art = _make_sim(data_config=None)

        result = sim.move_to(position=REACHABLE_LOCAL, robot_name="arm")

        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "USD" in text
        assert "urdf_path" in text
        assert "data_config" in text
        assert "send_action" in text


class TestAddRobotRecordsTheDescription:
    """The threading half: without it the preference above never fires."""

    def _engine(self, monkeypatch) -> Any:
        sim = IsaacSimulation()
        sim._world = object()
        sim._world_created = True

        def fake_load_urdf(prim_path: str, urdf_path: str, position: Any, *a: Any, **k: Any) -> tuple[Any, Any]:
            return ["j0"], None

        def fake_load_usd(prim_path: str, usd_path: str, position: Any, *a: Any, **k: Any) -> tuple[Any, Any]:
            return ["j0"], None

        monkeypatch.setattr(sim, "_load_urdf_robot", fake_load_urdf)
        monkeypatch.setattr(sim, "_load_usd_robot", fake_load_usd)
        return sim

    def test_a_urdf_robot_records_its_urdf(self, tmp_path, monkeypatch) -> None:
        sim = self._engine(monkeypatch)
        urdf = tmp_path / "r.urdf"
        urdf.write_text("<robot name='r'/>")

        assert sim.add_robot("r", urdf_path=str(urdf))["status"] == "success"
        assert sim._robots["r"].description_path == str(urdf)

    def test_an_mjcf_robot_records_the_mjcf_not_the_derived_usd(self, tmp_path, monkeypatch) -> None:
        """The USD is a cache artifact; the MJCF is the source of truth MuJoCo
        can compile."""
        sim = self._engine(monkeypatch)
        mjcf = tmp_path / "r.xml"
        mjcf.write_text(ARM_XML)
        monkeypatch.setattr(
            "strands_robots.simulation.isaac.simulation.convert_mjcf_to_usd",
            lambda path, **k: str(tmp_path / "r.usda"),
        )

        assert sim.add_robot("r", mjcf_path=str(mjcf))["status"] == "success"
        assert sim._robots["r"].description_path == str(mjcf)

    def test_a_plain_usd_robot_records_none(self, tmp_path, monkeypatch) -> None:
        """MuJoCo cannot compile USD, so recording it would send the IK loader
        into a compile failure instead of the registry fallback."""
        sim = self._engine(monkeypatch)
        usd = tmp_path / "r.usda"
        usd.write_text("#usda 1.0")

        assert sim.add_robot("r", usd_path=str(usd))["status"] == "success"
        assert sim._robots["r"].description_path is None

    def test_a_name_added_robot_records_the_resolved_description(self, tmp_path, monkeypatch) -> None:
        """The name path resolves a registry file and records it - which is
        what makes ``add_robot("so100"); move_to(...)`` work without spelling
        ``data_config`` a second time (it used to refuse: the name path stored
        ``data_config=None``)."""
        sim = self._engine(monkeypatch)
        urdf = tmp_path / "so_arm.urdf"
        urdf.write_text("<robot name='so_arm'/>")
        monkeypatch.setattr(
            "strands_robots.simulation.isaac.simulation._resolve_registry_description",
            lambda data_config, lookup: (str(urdf), None),
        )

        assert sim.add_robot("so_arm")["status"] == "success"
        assert sim._robots["so_arm"].description_path == str(urdf)
