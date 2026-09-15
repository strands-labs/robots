"""Isaac's ``get_contacts``: the PhysX report in the cross-backend record shape.

Until this feature the method was the ``SimEngine`` raising stub, which is what
made ``eval_policy(success_fn="contact")`` unmeasurable on Isaac (now refused up
front by the resolver) and every ``contact_*`` predicate answer ``False`` on a
backend that simulates contacts perfectly well.

The mechanism, measured on ``isaacsim`` 6.0.1 before any code was written:

* a ``PhysxContactReportAPI`` applied to a prim BEFORE the reset that builds
  the physics view produces per-pair headers (decodable actor paths,
  ``CONTACT_FOUND``/``PERSIST``/``LOST`` types) with per-point
  ``position``/``separation``/``impulse`` - and one applied mid-simulation
  produces nothing, 0 headers over every later step. Hence enrollment lives in
  ``add_object``.
* the handle-level contact APIs (``get_net_contact_forces``,
  ``get_contact_force_data``) raise without a constructor-time contact view,
  so they are not the read path.

What is pinned here without a GPU: the pure translation
(``_translate_contact_report``) from the report's (headers, data) sequences to
the record shape the predicate DSL and the MuJoCo backend already speak; the
per-step cache (PhysX hands the events once - a second read without a step must
not answer "no contacts"); the refusals; the enrollment call in ``add_object``;
and that the DSL's ``contact_any`` reads the result end-to-end. The live half -
a resting cube reporting its ground pair, a floating one reporting none - is
verified on GPU.
"""

from __future__ import annotations

import sys
import threading
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    IsaacConfig,
    IsaacSimulation,
    _ObjectState,
    _translate_contact_report,
)


class _Header:
    def __init__(self, a0: int, a1: int, n: int, event: str = "ContactEventType.CONTACT_PERSIST") -> None:
        self.actor0, self.actor1, self.num_contact_data, self.type = a0, a1, n, event


class _Point:
    def __init__(
        self, pos: tuple[float, float, float], sep: float, impulse: tuple[float, float, float] = (0.0, 0.0, 3e-3)
    ) -> None:
        self.position, self.separation, self.impulse = pos, sep, impulse


#: encoded-actor-id -> path, standing in for PhysicsSchemaTools.intToSdfPath.
_PATHS = {
    1: "/World/Objects/cube",
    2: "/World/defaultGroundPlane/GroundPlane/CollisionPlane",
    3: "/World/Robots/arm/gripper_link",
}


def _decode(i: int) -> str:
    return _PATHS[i]


class TestTheTranslation:
    def test_a_pair_becomes_one_named_record(self) -> None:
        headers = [_Header(1, 2, 2)]
        data = [_Point((0.24, 0.06, 0.0), -1e-8), _Point((0.36, -0.06, 0.0), 4e-9)]
        records = _translate_contact_report(headers, data, _decode, {"/World/Objects/cube": "cube"})

        assert len(records) == 1
        r = records[0]
        assert r["geom1"] == "cube", "a registered object is named by its object name"
        assert r["geom2"] == "CollisionPlane", "an unregistered actor falls back to the path leaf"
        assert r["active"] is True
        assert r["dist"] == pytest.approx(-1e-8), "dist is the MINIMUM separation over the points"
        assert r["pos"] == pytest.approx([0.24, 0.06, 0.0])
        assert r["n_points"] == 2

    def test_data_is_consumed_per_header_in_order(self) -> None:
        """Each header owns the next num_contact_data entries; an offset bug
        would hand pair B pair A's points."""
        headers = [_Header(1, 2, 1), _Header(1, 3, 1)]
        data = [_Point((0.0, 0.0, 0.0), -0.5), _Point((9.0, 9.0, 9.0), -0.25)]
        records = _translate_contact_report(headers, data, _decode, {})

        assert records[0]["dist"] == pytest.approx(-0.5)
        assert records[1]["dist"] == pytest.approx(-0.25)
        assert records[1]["pos"] == pytest.approx([9.0, 9.0, 9.0])

    def test_a_speculative_pair_is_reported_inactive(self) -> None:
        """PhysX reports a pair inside the contact offset as CONTACT_PERSIST at
        a plainly positive separation with ZERO impulse - measured live, a cube
        resting on another cube "persisted" against the ground plane 0.12 m
        below it. Event-only ``active`` answers touching for bodies visibly
        apart; the impulse is the solver's own touch signal."""
        headers = [_Header(1, 2, 1)]
        data = [_Point((0.3, 0.0, 0.12), 0.12, impulse=(0.0, 0.0, 0.0))]
        records = _translate_contact_report(headers, data, _decode, {})
        assert records[0]["active"] is False
        assert records[0]["impulse"] == 0.0

    def test_a_lost_event_is_reported_inactive(self) -> None:
        """CONTACT_LOST is the pair separating - one final record with
        active=False, so a caller polling per tick sees the transition rather
        than a silent disappearance."""
        headers = [_Header(1, 2, 0, event="ContactEventType.CONTACT_LOST")]
        records = _translate_contact_report(headers, [], _decode, {})
        assert records[0]["active"] is False
        assert records[0]["dist"] == 0.0 and records[0]["pos"] == []

    def test_the_dsl_reads_the_records(self) -> None:
        """End-to-end into the shared consumer: contact_any and contact_between
        answer from these records exactly as they do from MuJoCo's."""
        from strands_robots.simulation import predicates

        headers = [_Header(1, 3, 1)]
        data = [_Point((0.1, 0.0, 0.2), -0.001)]
        records = _translate_contact_report(headers, data, _decode, {"/World/Objects/cube": "cube"})

        sim = types.SimpleNamespace(
            get_contacts=lambda: {"status": "success", "content": [{"json": {"contacts": records}}]}
        )
        assert predicates.make_predicate("contact_any")(sim) is True  # type: ignore[arg-type]
        assert (
            predicates.make_predicate("contact_between", geom_a="cube", geom_b="gripper_link")(sim)  # type: ignore[arg-type]
            is True
        )
        assert (
            predicates.make_predicate("contact_between", geom_a="cube", geom_b="nothing")(sim)  # type: ignore[arg-type]
            is False
        )


def _engine() -> Any:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(render_mode="headless")
    engine._world = types.SimpleNamespace()
    engine._world_created = True
    engine._objects = {"cube": _ObjectState(name="cube", prim_path="/World/Objects/cube", shape="box", is_static=False)}
    engine._robots = {}
    engine._step_count = 4
    return engine


def _fake_physx(monkeypatch, reports: list[Any]) -> list[int]:
    """Install omni.physx + pxr stand-ins; each get_contact_report() pops the
    next (headers, data) from *reports*. Returns the call log."""
    calls: list[int] = []

    def get_report():
        calls.append(1)
        return reports.pop(0) if reports else ([], [])

    physx_mod = types.ModuleType("omni.physx")
    physx_mod.get_physx_simulation_interface = lambda: types.SimpleNamespace(get_contact_report=get_report)  # type: ignore[attr-defined]
    omni_mod = types.ModuleType("omni")
    omni_mod.physx = physx_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omni", omni_mod)
    monkeypatch.setitem(sys.modules, "omni.physx", physx_mod)

    pxr = sys.modules.get("pxr") or types.ModuleType("pxr")
    monkeypatch.setattr(pxr, "PhysicsSchemaTools", types.SimpleNamespace(intToSdfPath=_decode), raising=False)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    return calls


class TestTheMethod:
    def test_no_world_is_refused(self) -> None:
        engine = _engine()
        engine._world_created = False
        assert engine.get_contacts()["status"] == "error"

    def test_a_resting_pair_is_reported_in_the_envelope(self, monkeypatch) -> None:
        _fake_physx(monkeypatch, [([_Header(1, 2, 1)], [_Point((0.3, 0.0, 0.0), -1e-8)])])
        result = _engine().get_contacts()

        assert result["status"] == "success", result
        payload = next(b["json"] for b in result["content"] if "json" in b)
        assert payload["contacts"][0]["geom1"] == "cube"
        text = next(b["text"] for b in result["content"] if "text" in b)
        assert "1 contacts (1 touching)" in text
        assert "cube <-> CollisionPlane" in text

    def test_the_report_is_cached_per_step(self, monkeypatch) -> None:
        """PhysX hands the events ONCE per fetch: without the cache, the second
        call in the same step reads an empty report and answers 'No contacts.'
        for a cube that is still resting on the ground."""
        calls = _fake_physx(monkeypatch, [([_Header(1, 2, 1)], [_Point((0.3, 0.0, 0.0), -1e-8)])])
        engine = _engine()

        first = engine.get_contacts()
        second = engine.get_contacts()

        assert len(calls) == 1, "the second same-step call must serve the cache, not re-fetch"
        p1 = next(b["json"] for b in first["content"] if "json" in b)
        p2 = next(b["json"] for b in second["content"] if "json" in b)
        assert p1 == p2 and p1["contacts"], "the cached answer must be the populated one"

    def test_a_step_invalidates_the_cache(self, monkeypatch) -> None:
        calls = _fake_physx(
            monkeypatch,
            [([_Header(1, 2, 1)], [_Point((0.3, 0.0, 0.0), -1e-8)]), ([], [])],
        )
        engine = _engine()
        engine.get_contacts()
        engine._step_count += 1
        result = engine.get_contacts()

        assert len(calls) == 2
        assert "No contacts." in next(b["text"] for b in result["content"] if "text" in b)

    def test_a_missing_runtime_is_a_structured_error(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "omni", None)
        monkeypatch.setitem(sys.modules, "omni.physx", None)
        result = _engine().get_contacts()
        assert result["status"] == "error"
        assert "get_contacts" in result["content"][0]["text"]


class TestEnrollmentAndDiscovery:
    @pytest.fixture
    def kit_stubs(self, monkeypatch) -> list[Any]:
        """The primitive-construction fakes add_object's success path imports,
        plus a recording PhysxSchema.PhysxContactReportAPI. Returns the list of
        prims enrolled."""
        names = (
            "isaacsim",
            "isaacsim.core",
            "isaacsim.core.api",
            "isaacsim.core.api.objects",
            "isaacsim.core.prims",
            "isaacsim.core.utils",
            "isaacsim.core.utils.stage",
        )
        mods: dict[str, types.ModuleType] = {}
        for name in names:
            module = types.ModuleType(name)
            monkeypatch.setitem(sys.modules, name, module)
            mods[name] = module
        mods["isaacsim"].core = mods["isaacsim.core"]  # type: ignore[attr-defined]
        mods["isaacsim.core"].api = mods["isaacsim.core.api"]  # type: ignore[attr-defined]
        mods["isaacsim.core"].prims = mods["isaacsim.core.prims"]  # type: ignore[attr-defined]
        mods["isaacsim.core"].utils = mods["isaacsim.core.utils"]  # type: ignore[attr-defined]
        mods["isaacsim.core.api"].objects = mods["isaacsim.core.api.objects"]  # type: ignore[attr-defined]
        mods["isaacsim.core.utils"].stage = mods["isaacsim.core.utils.stage"]  # type: ignore[attr-defined]

        class _Prim:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.prim = object()
                self.name = kwargs.get("name")

            def __getattr__(self, attribute: str) -> Any:
                if not attribute.startswith(("set_", "apply_")):
                    raise AttributeError(attribute)
                return lambda *a, **k: None

        for primitive in (
            "FixedCuboid",
            "DynamicCuboid",
            "FixedSphere",
            "DynamicSphere",
            "FixedCylinder",
            "DynamicCylinder",
            "FixedCapsule",
            "DynamicCapsule",
        ):
            setattr(mods["isaacsim.core.api.objects"], primitive, _Prim)

        enrolled: list[Any] = []

        class _ReportAPI:
            @staticmethod
            def Apply(prim: Any) -> Any:  # noqa: N802 - pxr spelling
                enrolled.append(prim)
                return types.SimpleNamespace(CreateThresholdAttr=lambda: types.SimpleNamespace(Set=lambda v: None))

        pxr = sys.modules.get("pxr") or types.ModuleType("pxr")
        monkeypatch.setattr(pxr, "PhysxSchema", types.SimpleNamespace(PhysxContactReportAPI=_ReportAPI), raising=False)
        monkeypatch.setitem(sys.modules, "pxr", pxr)
        return enrolled

    def test_add_object_enrolls_the_prim_for_reporting(self, kit_stubs: list[Any]) -> None:
        """Applied at add time because it must precede the view-building reset:
        applied mid-simulation the report API produces nothing (measured on
        isaacsim 6.0.1 - 0 headers over every later step)."""
        engine = _engine()
        engine._objects = {}
        engine._prim_registry = []
        engine._world.scene = types.SimpleNamespace(add=lambda h: None)
        engine._world.stop = lambda: None

        result = engine.add_object("box2", shape="cuboid", position=[0.1, 0.0, 0.2], size=[0.05] * 3)

        assert result["status"] == "success", result
        assert len(kit_stubs) == 1, "the object prim was not enrolled for contact reporting"
        assert kit_stubs[0] is engine._objects["box2"].handle.prim

    def test_the_resolver_now_accepts_isaac_for_contact_success(self) -> None:
        """The structural check in _resolve_success_fn keys on overriding the
        SimEngine stub; this feature is what flips it for Isaac."""
        from strands_robots.simulation.base import SimEngine

        assert IsaacSimulation.get_contacts is not SimEngine.get_contacts
