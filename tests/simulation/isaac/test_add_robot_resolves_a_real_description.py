"""``add_robot`` with no asset path loads the description MuJoCo loads.

What this replaces was a branch whose comment read "Build procedurally via USD
API" and which made no USD call at all. Measured on live Isaac Sim 6.0.1 (A10G),
``add_robot("so100")`` reported::

    {"status": "success", ... "Robot 'so100' added (procedural: so100, 6 joints:
     ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll',
      'gripper'])"}

while creating **zero prims** under ``/World/Robots/so100``, leaving
``_RobotState.articulation`` as ``None`` before *and* after ``world.reset()``, and
returning ``{}`` from ``get_observation()`` at every point in the lifecycle.

It was also wrong as metadata, which is the half a caller could have checked. For
the same robot name the MuJoCo backend reports a different vocabulary, and for
``panda`` a different *count*:

============  ==============================================================  ===================
robot         the deleted table                                               MuJoCo, real asset
============  ==============================================================  ===================
``so100``     6: ``shoulder_pan shoulder_lift elbow_flex wrist_flex           6: ``Rotation Pitch
              wrist_roll gripper``                                            Elbow Wrist_Pitch
                                                                              Wrist_Roll Jaw``
``panda``     **7**: ``panda_joint1``..``panda_joint7``                       **9**: ``joint1``..
                                                                              ``joint7``,
                                                                              ``finger_joint1/2``
============  ==============================================================  ===================

So ``docs/simulation/isaac.md``'s promise that "the joint-name and observation
contract matches the MuJoCo backend, [so] policies and observation mappings
transfer unchanged between backends" was false before any physics was involved.

Both halves are answered by resolving through
:func:`~strands_robots.simulation.model_registry.resolve_model` - the same
resolver the MuJoCo backend's ``add_robot`` uses - and importing the result. On
live Isaac Sim that reproduces MuJoCo's names exactly, 9 of 9 for ``panda`` and 6
of 6 for ``so100``, with a wired articulation and a non-empty observation.

Scope: the Kit leaves are stood in, as in
:mod:`tests.simulation.isaac.test_mesh_size_is_discarded_for_the_asset_extent`.
What is graded here is the resolution, the dispatch by file format, and what the
report says - not the vendor importer, which has no Python API contract to pin
and is exercised on GPU.
"""

from __future__ import annotations

import os
import sys
import threading
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac import simulation as isaac_module  # noqa: E402
from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    IsaacConfig,
    IsaacSimulation,
)

#: Joint names the MuJoCo backend reports for ``trs_so_arm100/scene.xml``, the
#: description ``resolve_model("so100")`` returns. Spelled out because the point
#: of this whole change is that the two backends agree on them.
MUJOCO_SO100_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

#: The names the deleted table reported for the same robot. Kept as the negative
#: control: a report carrying these is the fiction, whatever its status field says.
FICTION_SO100_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class _Articulation:
    """A live-enough articulation handle: the loader's second return value."""

    def __init__(self, dof_names: list[str]) -> None:
        self.dof_names = list(dof_names)

    def get_joint_positions(self) -> list[float]:
        return [0.0] * len(self.dof_names)


def _engine() -> Any:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig()
    engine._world = types.SimpleNamespace(physics_sim_view=object())
    engine._world_created = True
    engine._robots = {}
    engine._objects = {}
    engine._cameras = {}
    engine._scene_objects = set()
    engine._prim_registry = []
    engine._action_controllers = {}
    engine._replicated = False
    return engine


@pytest.fixture
def loaded(monkeypatch) -> dict[str, Any]:
    """Record what the loaders were asked for, and answer as a live load would."""
    seen: dict[str, Any] = {"usd": [], "urdf": [], "converted": []}

    def fake_convert(mjcf_path: str, cache_dir: str | None = None, **kwargs: Any) -> str:
        seen["converted"].append(mjcf_path)
        return f"/cache/usd_robots/deadbeef/{os.path.basename(mjcf_path)}.usda"

    def fake_load_usd(
        self: Any, prim_path: str, usd_path: str, position: list[float], *args: Any, **kwargs: Any
    ) -> tuple[list[str], Any]:
        seen["usd"].append((prim_path, usd_path, list(position)))
        return list(MUJOCO_SO100_JOINTS), _Articulation(MUJOCO_SO100_JOINTS)

    def fake_load_urdf(
        self: Any, prim_path: str, urdf_path: str, position: list[float], *args: Any, **kwargs: Any
    ) -> tuple[list[str], Any]:
        # ``*args, **kwargs`` because this stands in for a PRIVATE method whose
        # signature grows: pinning its arity here makes an unrelated parameter
        # addition fail as "Failed to load URDF robot" - a production-looking
        # error from a stand-in. What these tests are about is which loader the
        # dispatch picks and with which path, so only those are captured.
        seen["urdf"].append((prim_path, urdf_path, list(position)))
        return ["j0"], _Articulation(["j0"])

    monkeypatch.setattr(isaac_module, "convert_mjcf_to_usd", fake_convert)
    monkeypatch.setattr(IsaacSimulation, "_load_usd_robot", fake_load_usd)
    monkeypatch.setattr(IsaacSimulation, "_load_urdf_robot", fake_load_urdf)
    return seen


def _resolves_to(monkeypatch, path: str | None) -> None:
    """Make ``resolve_model`` answer ``path`` for every name."""
    from strands_robots.simulation import model_registry

    monkeypatch.setattr(model_registry, "resolve_model", lambda name, prefer_scene=True: path)


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result["content"])


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    return next(block["json"] for block in result["content"] if "json" in block)


class TestABareNameResolvesTheDescriptionMuJoCoLoads:
    def test_the_registry_mjcf_is_converted_and_loaded(self, monkeypatch, loaded) -> None:
        _resolves_to(monkeypatch, "/assets/trs_so_arm100/scene.xml")
        engine = _engine()

        result = engine.add_robot("so100")

        assert result["status"] == "success", result
        # Resolution happened, and what was resolved is what was converted.
        assert loaded["converted"] == ["/assets/trs_so_arm100/scene.xml"]
        # The converted USD is what the proven native loader was handed.
        assert len(loaded["usd"]) == 1
        assert loaded["usd"][0][1].endswith("scene.xml.usda")
        assert loaded["urdf"] == []

    def test_the_joint_names_come_from_the_description(self, monkeypatch, loaded) -> None:
        """The whole point: they are MuJoCo's, not a table's."""
        _resolves_to(monkeypatch, "/assets/trs_so_arm100/scene.xml")
        engine = _engine()

        result = engine.add_robot("so100")

        assert _payload(result)["joint_names"] == MUJOCO_SO100_JOINTS
        assert engine._robots["so100"].joint_names == MUJOCO_SO100_JOINTS
        for fabricated in FICTION_SO100_JOINTS:
            assert fabricated not in _text(result)

    def test_the_articulation_is_wired(self, monkeypatch, loaded) -> None:
        """``articulation is None`` was the defect's signature; it must be gone."""
        _resolves_to(monkeypatch, "/assets/trs_so_arm100/scene.xml")
        engine = _engine()

        result = engine.add_robot("so100")

        assert result["status"] == "success", result
        assert engine._robots["so100"].articulation is not None
        # Reported too, so a caller sees it without reading private state.
        assert _payload(result)["articulation_wired"] is True

    def test_the_report_names_the_description_not_only_the_cache_path(self, monkeypatch, loaded) -> None:
        """A content-addressed cache path identifies nothing to a reader."""
        _resolves_to(monkeypatch, "/assets/trs_so_arm100/scene.xml")
        engine = _engine()

        result = engine.add_robot("so100")

        assert "/assets/trs_so_arm100/scene.xml" in _text(result)
        assert _payload(result)["mjcf_path"] == "/assets/trs_so_arm100/scene.xml"

    def test_the_report_no_longer_claims_to_be_procedural(self, monkeypatch, loaded) -> None:
        _resolves_to(monkeypatch, "/assets/trs_so_arm100/scene.xml")
        engine = _engine()

        assert "procedural" not in _text(engine.add_robot("so100")).lower()

    def test_data_config_is_what_gets_resolved(self, monkeypatch, loaded) -> None:
        """``data_config`` names the model; ``name`` is only the instance label."""
        asked: list[str] = []
        from strands_robots.simulation import model_registry

        def _record(name: str, prefer_scene: bool = True) -> str:
            asked.append(name)
            return "/assets/franka_emika_panda/scene.xml"

        monkeypatch.setattr(model_registry, "resolve_model", _record)
        engine = _engine()

        assert engine.add_robot("left_arm", data_config="panda")["status"] == "success"

        assert asked == ["panda"]
        assert list(engine._robots) == ["left_arm"]


class TestTheDispatchFollowsTheResolvedFormat:
    """``resolve_model`` can hand back a URDF or a USD, not only an MJCF.

    It consults user-registered URDFs before the Menagerie assets, so routing
    everything through the MJCF importer would fail on a file the URDF branch
    beside it loads correctly.
    """

    def test_a_resolved_urdf_takes_the_urdf_loader(self, monkeypatch, loaded) -> None:
        _resolves_to(monkeypatch, "/registered/arm.urdf")
        engine = _engine()

        result = engine.add_robot("arm")

        assert result["status"] == "success", result
        assert loaded["urdf"] and loaded["urdf"][0][1] == "/registered/arm.urdf"
        assert loaded["converted"] == []
        assert loaded["usd"] == []

    def test_a_resolved_usd_needs_no_conversion(self, monkeypatch, loaded) -> None:
        _resolves_to(monkeypatch, "/registered/arm.usda")
        engine = _engine()

        result = engine.add_robot("arm")

        assert result["status"] == "success", result
        assert loaded["usd"] and loaded["usd"][0][1] == "/registered/arm.usda"
        assert loaded["converted"] == []

    def test_an_explicit_path_is_not_overridden_by_resolution(self, monkeypatch, loaded) -> None:
        """A caller who named an asset gets that asset, resolvable name or not."""
        _resolves_to(monkeypatch, "/assets/trs_so_arm100/scene.xml")
        engine = _engine()

        assert engine.add_robot("so100", usd_path="/mine/robot.usda")["status"] == "success"

        assert loaded["usd"][0][1] == "/mine/robot.usda"
        assert loaded["converted"] == []


class TestAnExplicitMjcfPathIsSupported:
    """``mjcf_path`` was refused outright on the claim that Isaac has no MJCF
    importer. Isaac Sim 6.0.1 registers ``isaacsim.asset.importer.mjcf``."""

    def test_it_is_converted_and_loaded(self, monkeypatch, loaded) -> None:
        engine = _engine()

        result = engine.add_robot("arm", mjcf_path="/mine/robot.xml")

        assert result["status"] == "success", result
        assert loaded["converted"] == ["/mine/robot.xml"]
        assert loaded["usd"] and loaded["usd"][0][1].endswith("robot.xml.usda")

    def test_the_old_refusal_wording_is_gone(self, monkeypatch, loaded) -> None:
        engine = _engine()
        text = _text(engine.add_robot("arm", mjcf_path="/mine/robot.xml"))
        assert "no MJCF" not in text
        assert "not supported" not in text

    def test_a_conversion_failure_is_a_structured_error(self, monkeypatch, loaded) -> None:
        """The importer is a Kit extension, so its absence is the common case."""

        def _explode(mjcf_path: str, cache_dir: str | None = None, **kwargs: Any) -> str:
            raise ImportError("no importer here", name="isaacsim.asset.importer.mjcf")

        monkeypatch.setattr(isaac_module, "convert_mjcf_to_usd", _explode)
        engine = _engine()

        result = engine.add_robot("arm", mjcf_path="/mine/robot.xml")

        assert result["status"] == "error"
        text = _text(result)
        assert "/mine/robot.xml" in text
        # Both escapes are named, because neither is discoverable from the failure.
        assert "usd_path" in text
        assert "mujoco" in text.lower()
        # A failed load registers nothing, so a retry is not blocked by a
        # half-registered robot.
        assert engine._robots == {}
        assert engine._prim_registry == []


class TestTwoAssetPathsAreRefusedRatherThanRanked:
    """Each format is loaded by a different route, so naming two drops one.

    There was no such combination to refuse while ``mjcf_path`` was rejected
    outright and the other two were ordered by an ``elif``. With all three live
    the ambiguity is reachable, and ranking them silently would tell a caller
    nothing about which asset reached the stage.
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mjcf_path": "/a.xml", "urdf_path": "/b.urdf"},
            {"mjcf_path": "/a.xml", "usd_path": "/b.usda"},
            {"urdf_path": "/a.urdf", "usd_path": "/b.usda"},
            {"mjcf_path": "/a.xml", "urdf_path": "/b.urdf", "usd_path": "/c.usda"},
        ],
    )
    def test_two_paths_are_refused(self, monkeypatch, loaded, kwargs: dict[str, str]) -> None:
        engine = _engine()

        result = engine.add_robot("arm", **kwargs)

        assert result["status"] == "error", result
        text = _text(result)
        # Every path the caller named is quoted, so they can see the conflict.
        for value in kwargs.values():
            assert value in text
        # And nothing was loaded on the strength of one of them.
        assert loaded["converted"] == []
        assert loaded["usd"] == []
        assert loaded["urdf"] == []
        assert engine._robots == {}

    @pytest.mark.parametrize("key", ["mjcf_path", "urdf_path", "usd_path"])
    def test_one_path_is_still_accepted(self, monkeypatch, loaded, key: str) -> None:
        """The control: the guard refuses a combination, not a path."""
        engine = _engine()
        suffix = {"mjcf_path": "/a.xml", "urdf_path": "/a.urdf", "usd_path": "/a.usda"}[key]

        assert engine.add_robot("arm", **{key: suffix})["status"] == "success"


class TestAnUnresolvableNameIsRefusedWithADiagnosis:
    def test_the_refusal_uses_the_shared_message(self, monkeypatch, loaded) -> None:
        """One owner for the wording, so two backends cannot diagnose one
        registry differently."""
        from strands_robots.simulation.base import unknown_model_msg

        _resolves_to(monkeypatch, None)
        engine = _engine()

        result = engine.add_robot("not_a_robot_at_all")

        assert result["status"] == "error"
        assert unknown_model_msg("not_a_robot_at_all") in _text(result)

    def test_it_names_the_escapes_this_signature_has(self, monkeypatch, loaded) -> None:
        _resolves_to(monkeypatch, None)
        engine = _engine()

        text = _text(engine.add_robot("not_a_robot_at_all"))

        assert "data_config" in text
        assert "usd_path" in text
        assert "urdf_path" in text

    def test_a_data_config_miss_does_not_advise_data_config(self, monkeypatch, loaded) -> None:
        """The caller already passed it, so repeating it is not a remedy."""
        _resolves_to(monkeypatch, None)
        engine = _engine()

        text = _text(engine.add_robot("arm", data_config="not_a_model"))

        assert "Or pass data_config=" not in text

    def test_the_key_that_failed_is_what_is_named(self, monkeypatch, loaded) -> None:
        _resolves_to(monkeypatch, None)
        engine = _engine()

        text = _text(engine.add_robot("left_arm", data_config="not_a_model"))

        assert "not_a_model" in text

    def test_nothing_is_registered_by_a_refused_call(self, monkeypatch, loaded) -> None:
        _resolves_to(monkeypatch, None)
        engine = _engine()

        assert engine.add_robot("not_a_robot_at_all")["status"] == "error"

        assert engine._robots == {}
        assert engine._prim_registry == []


class TestTheSharedMessageHasOneOwner:
    """The MuJoCo backend had this wording inline until Isaac became a second
    caller. A second copy is how two backends drift apart on one registry."""

    @pytest.mark.parametrize("name", ["so100", "definitely_not_a_robot", "earthrover"])
    def test_mujoco_delegates_rather_than_repeating(self, name: str) -> None:
        from strands_robots.simulation.base import unknown_model_msg
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        assert MuJoCoSimEngine._unknown_model_msg(name) == unknown_model_msg(name)

    def test_the_mujoco_method_body_is_a_delegation(self) -> None:
        """Pinned structurally: an inline reimplementation passes the equality
        cell above on the day it is written and drifts afterwards."""
        import inspect

        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        body = inspect.getsource(MuJoCoSimEngine._unknown_model_msg)
        assert "return unknown_model_msg(requested)" in body
        # The three-way diagnosis must not have been copied back in.
        assert "is registered but its model file is not on disk" not in body

    def test_a_backend_can_name_its_own_discovery_surface(self) -> None:
        """The one backend-specific sentence is a parameter, not a fork."""
        from strands_robots.simulation.base import unknown_model_msg

        default = unknown_model_msg("definitely_not_a_robot")
        custom = unknown_model_msg("definitely_not_a_robot", discovery_hint=" Use list_robots().")
        assert default.endswith(" Use action='list_urdfs' to see all available robots.")
        assert custom.endswith(" Use list_robots().")
        # Only the trailing advice differs.
        assert (
            default[: -len(" Use action='list_urdfs' to see all available robots.")]
            == (custom[: -len(" Use list_robots().")])
        )


class TestTheFictionCannotComeBack:
    def test_no_module_reaches_for_a_procedural_lookup(self) -> None:
        """A structural pin: the lookup is deleted, and so is every caller."""
        import pathlib

        root = pathlib.Path(isaac_module.__file__).parent
        offenders = []
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for symbol in ("get_procedural_robot", "list_procedural_robots"):
                # A mention in prose explaining the deletion is fine; a call is not.
                if f"{symbol}(" in text:
                    offenders.append(f"{path.name}: {symbol}(")
        assert offenders == [], offenders

    def test_the_importer_module_is_not_imported_eagerly(self) -> None:
        """The Kit extension must stay behind the function that needs it, or
        importing this backend on a host without Isaac Sim would fail."""
        assert "isaacsim.asset.importer.mjcf" not in sys.modules
        import strands_robots.simulation.isaac.mjcf_assets as mjcf_assets  # noqa: F401

        assert "isaacsim.asset.importer.mjcf" not in sys.modules
