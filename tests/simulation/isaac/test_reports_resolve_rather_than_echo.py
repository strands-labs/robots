"""What the Isaac backend reports about itself is read back, not echoed from the request.

The device-reporting fix established the rule: a status field naming a runtime
property must come from the runtime, not from the config that asked for it. It left
two surfaces still echoing, and both matter for a different reason.

**``__repr__``** is what a traceback and a failing assertion render first, so a
requested value presented as fact there is the worst placement of this shape. It
echoed ``config.device`` - ``"cuda:0"`` by default, while PhysX resolves to
``"cpu"`` - so a repr in a stack trace asserted the opposite of where physics was
running. It also echoed ``config.num_envs``, which is the *default for* ``replicate()``
and not a count of anything that exists ("setting it alone creates nothing"), where
``_num_envs_active`` is the count actually built.

**``physics_dt``** was reported as the value this backend *passed* to ``World``. That
is a lie in one specific and reachable case: ``World`` is a ``SimulationContext``
singleton, so a second construction returns the first instance with its original
``physics_dt`` intact - measured on an A10G, ``World(physics_dt=1/60)`` then
``World(physics_dt=1/120)`` reports ``get_physics_dt()`` of 1/60. Reading it back is
what makes that visible instead of confirming the caller's belief.

Reading it back also surfaces a separate, still-open defect rather than hiding it:
``create_world(timestep=X)`` is honoured by ``World`` but never written to
``IsaacConfig.physics_dt``, which is where ``physics_timestep()`` and the sim-time
accumulators read from. So ``physics_dt`` and ``physics_dt_requested`` disagreeing is
a real signal, and reporting only one of them is what kept it invisible.

Both readers answer ``None`` rather than raising when there is no world or the runtime
does not expose them, because they feed status reads and a repr - neither may become
the thing that fails.
"""

from __future__ import annotations

import threading
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.config import IsaacConfig  # noqa: E402
from strands_robots.simulation.isaac.simulation import IsaacSimulation  # noqa: E402


def _resolved_dt():
    """Imported inside the tests so this file's behavioural assertions still collect
    against a tree where the reader does not exist yet."""
    from strands_robots.simulation.isaac.simulation import _resolved_physics_dt

    return _resolved_physics_dt


def _world(*, device: str | None = None, dt: float | None = None, raise_dt: bool = False) -> Any:
    ctx = types.SimpleNamespace(device=device) if device is not None else types.SimpleNamespace()

    def _get_dt() -> Any:
        if raise_dt:
            raise RuntimeError("physics not initialized")
        return dt

    world = types.SimpleNamespace(get_physics_context=lambda: ctx)
    if dt is not None or raise_dt:
        world.get_physics_dt = _get_dt
    return world


def _engine(*, world: Any = None, requested_device: str = "cuda:0", requested_dt: float = 1 / 120) -> Any:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(device=requested_device, physics_dt=requested_dt, render_mode="headless")
    engine._world_created = world is not None
    engine._world = world
    engine._num_envs_active = 4
    engine._robots = {}
    engine._cameras = {}
    engine._objects = {}
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._main_tid = threading.get_ident()
    engine._pump_running = False
    return engine


class TestTheReprReportsWhatResolved:
    def test_it_reports_the_resolved_device(self) -> None:
        engine = _engine(world=_world(device="cpu"))

        text = repr(engine)

        assert "device='cpu'" in text

    def test_it_names_the_request_when_the_two_differ(self) -> None:
        """The live case: configured cuda:0, PhysX on cpu. Both belong in a repr a
        reader is using to work out why something is slow."""
        engine = _engine(world=_world(device="cpu"))

        text = repr(engine)

        assert "requested='cuda:0'" in text

    def test_it_stays_quiet_when_they_agree(self) -> None:
        """A repr is read at a glance; a redundant field costs attention."""
        engine = _engine(world=_world(device="cuda:0"))

        text = repr(engine)

        assert "device='cuda:0'" in text
        assert "requested=" not in text

    def test_it_reports_the_active_env_count_not_the_configured_default(self) -> None:
        engine = _engine(world=_world(device="cpu"))
        engine._config = IsaacConfig(num_envs=1024, render_mode="headless")
        engine._num_envs_active = 4

        assert "num_envs=4" in repr(engine)

    def test_it_falls_back_to_the_request_with_no_world(self) -> None:
        """Before create_world there is nothing to ask, and a repr must still work."""
        engine = _engine(world=None)

        text = repr(engine)

        assert "device='cuda:0'" in text
        assert "world=none" in text

    def test_it_never_raises_on_a_half_constructed_engine(self) -> None:
        """The property the original docstring is built around, preserved."""
        half = IsaacSimulation.__new__(IsaacSimulation)

        text = repr(half)

        assert "partially constructed" in text


class TestPhysicsDtIsReadBack:
    def test_get_state_reports_the_resolved_dt(self) -> None:
        engine = _engine(world=_world(device="cpu", dt=1 / 60), requested_dt=1 / 120)

        state = engine.get_state()["content"][0]["json"]

        assert state["physics_dt"] == pytest.approx(1 / 60)

    def test_get_state_reports_the_requested_dt_beside_it(self) -> None:
        """The pair is the signal. create_world(timestep=) is honoured by World and
        never written back to the config, so these disagreeing is a real finding."""
        engine = _engine(world=_world(device="cpu", dt=1 / 60), requested_dt=1 / 120)

        state = engine.get_state()["content"][0]["json"]

        assert state["physics_dt_requested"] == pytest.approx(1 / 120)
        assert state["physics_dt"] != state["physics_dt_requested"]

    def test_the_reader_answers_none_with_no_world(self) -> None:
        assert _resolved_dt()(None) is None

    def test_the_reader_answers_none_when_the_world_lacks_the_getter(self) -> None:
        assert _resolved_dt()(types.SimpleNamespace()) is None

    def test_the_reader_answers_none_when_the_getter_raises(self) -> None:
        assert _resolved_dt()(_world(dt=None, raise_dt=True)) is None

    @pytest.mark.parametrize("bad", ["not-a-number", None, object()])
    def test_the_reader_answers_none_for_an_uncoercible_value(self, bad: Any) -> None:
        world = types.SimpleNamespace(get_physics_dt=lambda: bad)

        assert _resolved_dt()(world) is None

    def test_the_reader_coerces_to_float(self) -> None:
        """Isaac may hand back a numpy scalar; the envelope is JSON."""
        world = types.SimpleNamespace(get_physics_dt=lambda: 1)

        value = _resolved_dt()(world)

        assert isinstance(value, float)
        assert value == 1.0


class TestTheEchoedFieldsThatStayEchoed:
    """``headless`` and ``render_mode`` keep coming from the config, deliberately.

    The process-wide ``SimulationApp`` offers no cheap resolved answer for either, and
    a status read must not become the thing that raises to find one. Pinned so the
    asymmetry with ``device`` and ``physics_dt`` reads as a decision.
    """

    def test_headless_comes_from_the_config(self) -> None:
        engine = _engine(world=_world(device="cpu"))

        assert engine.get_state()["content"][0]["json"]["headless"] == engine._config.headless

    def test_render_mode_comes_from_the_config(self) -> None:
        engine = _engine(world=_world(device="cpu"))

        assert engine.get_state()["content"][0]["json"]["render_mode"] == engine._config.render_mode
