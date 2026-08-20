# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: ``remove_camera`` refuses a name its own ``render`` resolves past.

:func:`~strands_robots.utils.reserved_camera_name_error` states one rule -- a
free-camera routing token is not a camera *name* -- and until this fix the
token-routing backends applied it at ``add_camera`` only. The name's other
lifecycle end was unguarded, so the token could be surrendered but never
reclaimed.

``remove_camera`` cannot remove the free view. ``render``, ``render_depth``,
``get_frame`` and ``get_camera_params`` resolve the token by an explicit check
that never consults the registry, and ``list_cameras`` names it
unconditionally -- its docstring calls it "the built-in ``default`` free view",
"independent of whether the loaded MJCF happens to bake a camera literally
named ``default``". What the call actually removed was the *registry entry*,
i.e. only the recordable/observable alias. Measured on MuJoCo 3.10.0, one
``create_world`` then ``remove_camera("default")``:

* ``status="success"``, text ``Camera 'default' removed.``;
* ``list_cameras()`` still answered ``['default']`` and ``render()`` still
  produced a frame, so every read surface kept advertising the name;
* ``start_recording(cameras=["default"])`` then refused it -- ``Unknown
  camera(s) ['default'] ... Available scene cameras: []`` -- a refusal whose own
  list contradicts ``list_cameras``;
* and the remedy that refusal implies is ``add_camera("default", ...)``, which
  :mod:`~strands_robots.utils`'s reserved rule refuses. There was no way back.

So the reservation guard added at creation turned "ends with an unreachable
camera" into "ends with an unrecoverable inconsistency": the advertised
free-view alias could be destroyed permanently, and the recording surface that
depends on it could not be restored by any public call.

The fix reads the same shared domain at both ends of the name's life, so the
side that *routes* a token and the side that *refuses* it stay in agreement. It
precedes the existence test for the reason the creation-side check precedes the
duplicate test: that test can answer for the token (``create_world`` registers
the free view under ``"default"``, and Newton does not) and either answer is
misleading -- ``"removed"`` or ``"not found. Registered: [...]"`` for a name
``list_cameras`` advertises.

The Isaac backend is deliberately unguarded: its ``get_frame`` looks the name up
in its camera map with no token check, so ``"default"`` there is an ordinary
addressable name and removing it is coherent.
"""

from typing import Any, cast

import pytest

from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.utils import FREE_CAMERA_TOKENS

pytest.importorskip("mujoco")

# ``FREE_CAMERA_TOKENS`` is ``(None, "", "default", "free")``: ``None`` is the
# "caller named no camera" sentinel rather than a name, and the shared domain
# deliberately accepts it, so only the string spellings are reserved *names*.
RESERVED_NAMES = sorted(t for t in FREE_CAMERA_TOKENS if isinstance(t, str))


@pytest.fixture
def sim() -> Any:
    """A compiled MuJoCo world holding only the built-in free view."""
    from strands_robots.simulation import create_simulation

    engine = create_simulation(backend="mujoco", tool_name="reserved_removal")
    engine.create_world()
    yield engine
    engine.destroy()


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


def _compiled_camera_names(sim: Any) -> list[str]:
    """The cameras the compiled MJCF actually holds.

    ``create_world`` bakes a camera literally named ``"default"`` (``ncam == 1``)
    as well as registering it, and the render path's token check resolves that
    name to the free view without consulting either. Removing it therefore
    deleted a compiled camera the scene ships with.
    """
    import mujoco

    model = sim._world._model
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]


class TestTheFreeViewAliasCannotBeSurrendered:
    """The token is refused at removal, so the alias survives."""

    @pytest.mark.parametrize("token", RESERVED_NAMES)
    def test_removing_a_routing_token_is_refused(self, sim, token: str) -> None:
        """Every token the render path resolves past is refused as a name here."""
        result = sim.remove_camera(token)
        assert result["status"] == "error", result
        text = _text(result)
        assert "reserved" in text
        assert "remove_camera" in text

    def test_the_advertised_alias_survives_the_attempt(self, sim) -> None:
        """``remove_camera('default')`` reported success and dropped the entry."""
        assert "default" in sim._world.cameras
        sim.remove_camera("default")
        assert "default" in sim._world.cameras
        assert sim.list_cameras() == ["default"]
        assert _compiled_camera_names(sim) == ["default"]

    def test_the_recordable_alias_survives_the_attempt(self, sim) -> None:
        """The consequence: recording the advertised name stopped working.

        ``list_cameras`` kept naming ``'default'`` while ``start_recording``
        refused it, and no public call could put the entry back.
        """
        sim.remove_camera("default")
        started = sim.start_recording(
            repo_id="local/reserved_removal",
            task="t",
            fps=30,
            root="/tmp/strands-reserved-removal",
            cameras=["default"],
        )
        assert started["status"] == "success", started
        sim.stop_recording()

    def test_the_refusal_names_the_surface_and_the_rule(self, sim) -> None:
        """A caller who tries gets the reason, not a bookkeeping answer."""
        text = _text(sim.remove_camera("default"))
        assert "removed" not in text
        assert "not found" not in text


class TestTheGuardDoesNotReachPastTheTokens:
    """Controls: these hold on both sides of the fix."""

    def test_an_addressable_camera_still_removes(self, sim) -> None:
        """The rule is membership, so an ordinary camera is unaffected."""
        sim.add_camera("cam1", position=[0.5, 0.5, 0.5], target=[0.0, 0.0, 0.0])
        removed = sim.remove_camera("cam1")
        assert removed["status"] == "success"
        assert "cam1" not in sim._world.cameras

    def test_a_name_that_merely_resembles_a_token_still_removes(self, sim) -> None:
        """``'default_cam'`` is not the token ``'default'``."""
        sim.add_camera("default_cam", position=[0.5, 0.5, 0.5], target=[0.0, 0.0, 0.0])
        removed = sim.remove_camera("default_cam")
        assert removed["status"] == "success"

    def test_an_unknown_name_still_reports_not_found(self, sim) -> None:
        """The existence test is still reached for every non-token name."""
        result = sim.remove_camera("nope")
        assert result["status"] == "error"
        assert "not found" in _text(result)


class TestTheRuleIsReadFromOneOwner:
    """Both token-routing backends refuse through the same shared domain."""

    @pytest.mark.parametrize("token", RESERVED_NAMES)
    def test_newton_refuses_the_same_tokens(self, token: str) -> None:
        """Newton registers no ``'default'`` entry, so its existence test
        answered ``"not found. Registered: []"`` for a name its own
        ``list_cameras`` advertises. The rule has to win there too.
        """
        import threading
        from types import SimpleNamespace

        stub = SimpleNamespace(
            _world=SimpleNamespace(cameras={}),
            _model=object(),
            _lock=threading.RLock(),
        )
        # ``stub`` is a duck-typed stand-in: the method reads only ``_world``.
        result = NewtonSimEngine.remove_camera(cast(Any, stub), token)
        assert result["status"] == "error"
        text = _text(result)
        assert "reserved" in text
        assert "not found" not in text
