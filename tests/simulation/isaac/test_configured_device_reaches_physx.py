"""The Isaac backend reports the device PhysX resolved, and does not claim the one it was asked for.

``IsaacConfig.device`` defaults to ``"cuda:0"`` and its ``__post_init__`` refuses
anything that does not start with ``cuda`` - "Isaac Sim requires a CUDA device".
``create_world`` builds the world **without** passing it, and ``World``'s own
``device`` default is ``None``, which resolves to ``"cpu"``. So PhysX solves on
the CPU, and it is expensive: measured through ``IsaacSimulation`` on an A10G
under Isaac Sim 6.0.1 - 40 cuboids, 300 ``step()`` calls after a 30-step warmup,
one container per arm - 9.9 steps/s on the CPU against 112.3 with
``device="cuda:0"``.

The device is nonetheless **deliberately not forwarded**, and that is pinned here
rather than left to look like an oversight, because forwarding it is the obvious
one-line change and it breaks the backend. Measured on the same A10G, same tree,
only the argument differing:

    no device arg     gpu_pipeline=False   add_robot -> success
    device="cuda:0"   gpu_pipeline=True    add_robot -> CUDA error: an illegal
        memory access was encountered (GpuArticulationView.cpp:631)

PhysX's GPU pipeline pre-sizes its tensor buffers at ``world.reset()``.
``create_world`` resets immediately, sizing them for a stage with a ground plane
and ZERO articulations, so the first ``add_robot`` initializes an articulation
into a view with no room. The illegal access poisons the CUDA context, so the
following ``add_object`` fails too. Adding while stopped instead fails with
``'NoneType' object has no attribute 'create_articulation'``. Making it work needs
Isaac Lab's build-the-whole-scene-then-reset-once pattern, which this backend's
incremental contract - an agent calling ``add_robot`` one tool call at a time -
does not have.

What this fixes is the **reporting**. All five surfaces that named the device -
``create_world``'s result, ``get_state``, ``replicate``'s message, the init log
line and ``__repr__`` - echoed ``self._config.device``, the value that had been
*requested*. Nothing read what PhysX resolved, so there was no field in which the
two could disagree, and a wholly CPU-bound run reported ``device=cuda:0``
everywhere a user would look. The only symptom was being slow, which reads as
"Isaac is heavyweight" rather than as a defect.

Now ``device`` is the resolved value and ``device_requested`` sits beside it, so
the gap is legible instead of invisible.

Nothing here needs Isaac Sim: a fake ``isaacsim`` tree records the kwargs
``World`` was constructed with and what its physics context reports.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from strands_robots.simulation.isaac import simulation as isaac_simulation
from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation


def _resolver():
    """Imported inside the tests, not at module scope, so this file's *behavioural*
    assertions still collect and run against a tree where the resolver does not
    exist yet. A module-scope import turns every test in the file into one
    collection error, which proves the symbol is new and says nothing about
    whether the device was forwarded - the thing actually under test."""
    from strands_robots.simulation.isaac.simulation import _resolved_physics_device

    return _resolved_physics_device


class _PhysicsContext:
    """Reports a device the way ``PhysicsContext`` does, from what World resolved."""

    def __init__(self, device: str | None) -> None:
        # None is what a real World resolves to "cpu" for; the fake mirrors that
        # so the omitted-argument case is reproduced rather than asserted about.
        self.device = device if device is not None else "cpu"

    def set_gravity(self, magnitude: object) -> None:
        return None


class _FakeScene:
    def add_default_ground_plane(self) -> None:
        return None


class _FakeWorld:
    """Stands in for ``isaacsim.core.api.World``, recording its own kwargs."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        type(self).last_kwargs = dict(kwargs)
        self.physics_context = _PhysicsContext(kwargs.get("device"))
        self.scene = _FakeScene()

    def get_physics_context(self) -> _PhysicsContext:
        return self.physics_context

    def reset(self) -> None:
        return None


@pytest.fixture()
def fake_isaacsim(monkeypatch):
    """Fake ``isaacsim`` tree covering every import ``create_world`` performs."""
    monkeypatch.setattr(isaac_simulation, "_SIMULATION_APP", None)
    monkeypatch.setattr(isaac_simulation, "_SIMULATION_APP_LAUNCH", None)
    _FakeWorld.last_kwargs = {}
    mods = {}
    for name in ("isaacsim", "isaacsim.core", "isaacsim.core.api"):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        mods[name] = module
    mods["isaacsim"].SimulationApp = lambda launch=None: types.SimpleNamespace(launch=launch)
    mods["isaacsim"].core = mods["isaacsim.core"]
    mods["isaacsim.core"].api = mods["isaacsim.core.api"]
    mods["isaacsim.core.api"].World = _FakeWorld
    return mods


class TestTheDeviceIsDeliberatelyNotForwarded:
    """Pinned as a decision, not left to look like an oversight.

    Forwarding ``device`` is a one-line change that turns the GPU pipeline on and
    makes the first ``add_robot`` fail with a CUDA illegal memory access, because
    PhysX pre-sizes its GPU tensor buffers at the ``world.reset()`` that
    ``create_world`` performs - before any articulation exists. Whoever reads
    ``World`` accepting ``device`` and this code not passing it needs to find the
    reason attached to the code, not rediscover it on a GPU.
    """

    def test_world_is_built_without_a_device_argument(self, fake_isaacsim) -> None:
        sim = IsaacSimulation(config=IsaacConfig(device="cuda:0"))

        assert sim.create_world()["status"] == "success"

        assert "device" not in _FakeWorld.last_kwargs

    def test_the_reason_is_recorded_next_to_the_call(self) -> None:
        """A bare omission reads as a bug and invites the one-line "fix". The
        measurement and the GpuArticulationView failure must travel with it."""
        import inspect

        source = inspect.getsource(IsaacSimulation.create_world)

        assert "GpuArticulationView" in source
        assert "illegal memory access" in source
        assert "pre-sizes" in source

    def test_the_config_still_carries_the_request(self, fake_isaacsim) -> None:
        """The value is not silently dropped from the config - it is reported as
        device_requested, which is what makes the CPU reality visible."""
        sim = IsaacSimulation(config=IsaacConfig(device="cuda:1"))
        sim.create_world()

        assert sim._config.device == "cuda:1"
        assert sim.get_state()["content"][0]["json"]["device_requested"] == "cuda:1"


class TestWhatIsReportedIsWhatResolved:
    """The half that makes a recurrence visible instead of benchmark-only."""

    def test_create_world_reports_the_resolved_device_not_the_request(self, fake_isaacsim) -> None:
        """The live situation, asserted directly: PhysX resolved cpu, the config
        asked for cuda:0, and the result says both. Before this, it said cuda:0
        twice and there was no field in which the truth could appear."""
        sim = IsaacSimulation(config=IsaacConfig(device="cuda:0"))

        info = sim.create_world()["content"][0]["json"]

        assert info["device"] == "cpu"
        assert info["device_requested"] == "cuda:0"

    def test_get_state_reports_both(self, fake_isaacsim) -> None:
        sim = IsaacSimulation(config=IsaacConfig(device="cuda:0"))
        sim.create_world()

        state = sim.get_state()["content"][0]["json"]

        assert state["device"] == "cpu"
        assert state["device_requested"] == "cuda:0"

    def test_the_two_fields_disagree_and_that_is_the_point(self, fake_isaacsim) -> None:
        """A reader who compares them learns where physics is running. A reader
        of the pre-fix reports could not, because one value was printed twice."""
        sim = IsaacSimulation(config=IsaacConfig(device="cuda:0"))
        sim.create_world()

        state = sim.get_state()["content"][0]["json"]

        assert state["device"] != state["device_requested"]

    def test_the_text_line_carries_the_resolved_device(self, fake_isaacsim) -> None:
        """The human-readable line is what most callers actually read."""
        sim = IsaacSimulation(config=IsaacConfig(device="cuda:0"))

        text = sim.create_world()["content"][0]["text"]

        assert "device=cpu" in text

    def test_a_divergence_is_legible(self, fake_isaacsim, monkeypatch) -> None:
        """The whole point. With the physics context reporting cpu against a
        cuda request, the two fields must disagree - that is the signal the
        pre-fix code could not emit, because it read one value twice."""
        sim = IsaacSimulation(config=IsaacConfig(device="cuda:0"))
        sim.create_world()
        sim._world.physics_context.device = "cpu"  # what the pre-fix world resolved

        state = sim.get_state()["content"][0]["json"]

        assert state["device"] == "cpu"
        assert state["device_requested"] == "cuda:0"
        assert state["device"] != state["device_requested"]


class TestTheResolverNeverRaises:
    """It feeds status reads, so it answers None instead of failing."""

    def test_no_world_answers_none(self) -> None:
        assert _resolver()(None) is None

    def test_a_world_without_a_physics_context_answers_none(self) -> None:
        assert _resolver()(types.SimpleNamespace()) is None

    def test_a_physics_context_without_a_device_answers_none(self) -> None:
        context = types.SimpleNamespace()  # a physics context declaring no device
        world = types.SimpleNamespace(get_physics_context=lambda: context)

        assert _resolver()(world) is None

    def test_a_raising_physics_context_answers_none(self) -> None:
        def _raise() -> Any:
            raise RuntimeError("physics not initialized")

        assert _resolver()(types.SimpleNamespace(get_physics_context=_raise)) is None

    def test_a_resolved_device_is_returned_as_a_string(self) -> None:
        world = types.SimpleNamespace(get_physics_context=lambda: types.SimpleNamespace(device="cuda:0"))

        assert _resolver()(world) == "cuda:0"

    def test_it_is_a_module_level_function_not_a_method(self) -> None:
        """Two of its three callers reach it with a ``types.SimpleNamespace``
        standing in for ``self`` - the ``replicate`` suites call the unbound
        method with a stub - so a method raised ``AttributeError`` there for a
        reporting concern. Pinned because moving it onto the class is the
        natural-looking refactor and it fails only in those suites."""
        assert not hasattr(IsaacSimulation, "_resolved_physics_device")
        assert callable(_resolver())


class TestTheConfigStillRefusesANonCudaDevice:
    """The control: this fix forwards the value, it does not widen the domain."""

    @pytest.mark.parametrize("device", ["cpu", "mps", "", "gpu:0"])
    def test_a_non_cuda_device_is_refused_by_the_config(self, device: str) -> None:
        with pytest.raises(ValueError, match="requires a CUDA device"):
            IsaacConfig(device=device)

    @pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1", "cuda:7"])
    def test_a_cuda_device_is_accepted(self, device: str) -> None:
        assert IsaacConfig(device=device).device == device
