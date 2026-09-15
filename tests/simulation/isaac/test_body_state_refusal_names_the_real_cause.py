"""A registered object whose pose cannot be read is not reported as "not found".

``get_body_state`` resolves a name against the object registry first, then the
stage. When both miss it reported one message for every cause:

    Body 'mug' not found on the Isaac stage. Known objects: [mug]. Robots: []
    -- address robot links as '<robot>/<link>' (e.g. 'robot/panda_hand') or pass
    an absolute prim path ('/World/...').

The searched-for name is printed inside that message's own list of *known*
objects, so the sentence contradicts itself. Worse than the contradiction is the
advice: both remedies it offers - respell the name as ``<robot>/<link>``, or pass
an absolute prim path - are the remedies for a **misnamed** body. The name was
already right, so following either produces the identical refusal, and the actual
cause is named nowhere.

Three distinct states reach that branch with the name in the registry, and all
three are about the object's prim rather than its name:

* the object never got a rigid-prim handle,
* the handle raised on ``get_world_pose`` - the invalidate-on-reset family, where
  a scene change since the last ``reset()`` leaves the handle stale,
* the pose came back in an unusable shape.

The refusal now distinguishes "registered but unreadable" from "unknown", says
that respelling will not help, and points at ``reset()``. The unknown-name
message is unchanged, which is the control: a fix that widened the new wording to
every miss would lose the close-match guidance a genuinely misnamed body needs.

Nothing here needs Isaac Sim: the object handles are stand-ins and the stage read
is stubbed to miss, which is the state under test.
"""

from __future__ import annotations

import threading
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import IsaacSimulation  # noqa: E402


class _DeadHandle:
    """A handle whose prim is gone - what a post-scene-change read hits."""

    def get_world_pose(self) -> Any:
        raise RuntimeError("physics tensor view invalidated")


class _UnusablePoseHandle:
    """Returns a pose of the wrong shape, which ``_to_float_list`` rejects."""

    def get_world_pose(self) -> Any:
        return ("not", "a")


class _LiveHandle:
    def get_world_pose(self) -> Any:
        return ([0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0])


def _engine(objects: dict[str, Any]) -> Any:
    """A skeleton engine whose stage read always misses."""
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._world_created = True
    engine._world = types.SimpleNamespace()
    engine._robots = {}
    engine._pump_running = False
    engine._main_tid = threading.get_ident()
    engine._prim_body_state = lambda body_name: None  # type: ignore[method-assign]
    engine._objects = objects
    return engine


def _obj(handle: Any, name: str = "mug") -> Any:
    return types.SimpleNamespace(handle=handle, prim_path=f"/World/{name}", name=name)


def _text(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


class TestARegisteredObjectIsNotReportedAsUnknown:
    """The three unreadable states, each of which used to say "not found"."""

    @pytest.mark.parametrize(
        ("handle", "label"),
        [
            (None, "no-handle"),
            (_DeadHandle(), "handle-raises"),
            (_UnusablePoseHandle(), "unusable-pose"),
        ],
    )
    def test_the_refusal_says_registered_rather_than_not_found(self, handle: Any, label: str) -> None:
        engine = _engine({"mug": _obj(handle)})

        result = engine.get_body_state("mug")

        assert result["status"] == "error", result
        text = _text(result)
        assert "registered object" in text
        assert "not found" not in text

    @pytest.mark.parametrize("handle", [None, _DeadHandle(), _UnusablePoseHandle()])
    def test_it_says_respelling_the_name_will_not_help(self, handle: Any) -> None:
        """The precise harm: the old advice sent the caller to fix the name."""
        engine = _engine({"mug": _obj(handle)})

        text = _text(engine.get_body_state("mug"))

        assert "will not help" in text

    @pytest.mark.parametrize("handle", [None, _DeadHandle(), _UnusablePoseHandle()])
    def test_it_offers_the_prim_path_route_which_does_work(self, handle: Any) -> None:
        """Of main's two remedies, one genuinely applies here and one cannot.

        Reading the prim path bypasses the dead handle and goes to the stage, so it
        succeeds for exactly this state - demonstrated by
        ``test_the_prim_path_route_really_resolves`` below. An earlier version of
        this refusal said only that respelling "will not help" and dropped both,
        which removed the working remedy along with the useless one.
        """
        engine = _engine({"mug": _obj(handle)})

        text = _text(engine.get_body_state("mug"))

        assert "get_body_state('/World/mug')" in text

    def test_the_prim_path_route_really_resolves(self) -> None:
        """The premise, executable. Without this the advice above is a guess."""
        engine = _engine({"mug": _obj(None)})
        engine._prim_body_state = lambda name: (
            {
                "position": [1.0, 2.0, 3.0],
                "quaternion": [1.0, 0.0, 0.0, 0.0],
                "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "source": "prim",
                "prim_path": "/World/mug",
            }
            if name.startswith("/")
            else None
        )

        assert engine.get_body_state("mug")["status"] == "error", "the bare name still cannot resolve"
        assert engine.get_body_state("/World/mug")["status"] == "success", "the advised route must work"

    @pytest.mark.parametrize("handle", [None, _DeadHandle(), _UnusablePoseHandle()])
    def test_it_does_not_list_the_searched_name_as_a_known_object(self, handle: Any) -> None:
        """The self-contradiction, pinned directly."""
        engine = _engine({"mug": _obj(handle)})

        assert "Known objects: [mug]" not in _text(engine.get_body_state("mug"))

    def test_it_points_at_reset(self) -> None:
        """The one remedy that does apply, for the stale-handle case."""
        engine = _engine({"mug": _obj(_DeadHandle())})

        assert "reset()" in _text(engine.get_body_state("mug"))

    def test_it_names_the_prim_path(self) -> None:
        engine = _engine({"mug": _obj(None)})

        assert "/World/mug" in _text(engine.get_body_state("mug"))

    def test_the_two_causes_are_distinguishable(self) -> None:
        """A missing handle and a raising one need different investigations, so
        one wording for both would be the same collapse one level down."""
        no_handle = _text(_engine({"mug": _obj(None)}).get_body_state("mug"))
        raised = _text(_engine({"mug": _obj(_DeadHandle())}).get_body_state("mug"))

        assert "no rigid-prim handle" in no_handle
        assert "could not be read" in raised
        assert no_handle != raised


class TestTheUnknownNameMessageIsUnchanged:
    """The control: widening the new wording to every miss loses real guidance."""

    def test_an_unknown_name_still_says_not_found(self) -> None:
        engine = _engine({"cup": _obj(None, name="cup")})

        text = _text(engine.get_body_state("mug"))

        assert "not found on the Isaac stage" in text
        assert "registered object" not in text

    def test_an_unknown_name_still_lists_the_known_objects(self) -> None:
        engine = _engine({"cup": _obj(None, name="cup"), "plate": _obj(None, name="plate")})

        text = _text(engine.get_body_state("mug"))

        assert "cup" in text
        assert "plate" in text

    def test_an_unknown_name_still_advises_the_two_addressing_forms(self) -> None:
        engine = _engine({"cup": _obj(None, name="cup")})

        text = _text(engine.get_body_state("mug"))

        assert "<robot>/<link>" in text
        assert "absolute prim path" in text


class TestAReadableObjectStillReads:
    """The other control: none of this touches the success path."""

    def test_a_live_handle_returns_a_pose(self) -> None:
        engine = _engine({"mug": _obj(_LiveHandle())})

        result = engine.get_body_state("mug")

        assert result["status"] == "success", result
        assert result["content"][1]["json"]["position"] == [0.1, 0.2, 0.3]

    def test_the_stage_is_still_consulted_for_a_registered_name(self) -> None:
        """The registry miss must not short-circuit the stage read: an object
        may be registered without a handle and still be resolvable as a prim."""
        engine = _engine({"mug": _obj(None)})
        engine._prim_body_state = lambda name: {
            "position": [1.0, 2.0, 3.0],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "source": "prim",
            "prim_path": "/World/mug",
        }

        result = engine.get_body_state("mug")

        assert result["status"] == "success", result
        assert result["content"][1]["json"]["position"] == [1.0, 2.0, 3.0]
