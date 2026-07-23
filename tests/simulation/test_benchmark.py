"""Tests for ``strands_robots.simulation.benchmark``.

Covers:

* :class:`BenchmarkProtocol` ABC contract (cannot instantiate abstract,
  required methods must be implemented, optional hooks have usable defaults).
* :class:`StepInfo` dataclass.
* Registry operations (:func:`register_benchmark` / :func:`get_benchmark` /
  :func:`list_benchmarks` / :func:`unregister_benchmark`), including
  idempotent-overwrite and thread safety.
* Robot compatibility validation via :meth:`BenchmarkProtocol.on_episode_start`,
  including reconciliation when ``sim.list_robots()`` and ``world.robots`` diverge.
"""

from __future__ import annotations

import random
import threading
from typing import Any

import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.benchmark import (
    _BENCHMARK_REGISTRY,
    BenchmarkCompatibilityError,
    BenchmarkProtocol,
    StepInfo,
    get_benchmark,
    list_benchmarks,
    register_benchmark,
    unregister_benchmark,
)


class _MinimalBenchmark(BenchmarkProtocol):
    """Concrete benchmark used across tests."""

    max_steps = 42

    def __init__(
        self,
        *,
        supported: list[str] | None = None,
        default: str = "so100",
        success: bool = False,
        failure: bool = False,
        reward: float = 0.0,
    ):
        self._supported = list(supported if supported is not None else ["so100"])
        self._default = default
        self._success = success
        self._failure = failure
        self._reward = reward

    @property
    def supported_robots(self) -> list[str]:
        return list(self._supported)

    @property
    def default_robot(self) -> str:
        return self._default

    def on_step(self, sim: SimEngine, obs: dict[str, Any], action: dict[str, Any]) -> StepInfo:
        return StepInfo(reward=self._reward)

    def is_success(self, sim: SimEngine) -> bool:
        return self._success

    def is_failure(self, sim: SimEngine) -> bool:
        return self._failure


# Registry fixtures


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot + restore the registry around every test so they stay isolated."""
    snapshot = dict(_BENCHMARK_REGISTRY)
    _BENCHMARK_REGISTRY.clear()
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(snapshot)


# StepInfo


class TestStepInfo:
    def test_defaults(self):
        info = StepInfo()
        assert info.reward == 0.0
        assert info.done is False
        assert info.info == {}

    def test_custom_values(self):
        info = StepInfo(reward=3.5, done=True, info={"k": "v"})
        assert info.reward == 3.5
        assert info.done is True
        assert info.info == {"k": "v"}

    def test_is_frozen(self):
        info = StepInfo()
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            info.reward = 1.0  # type: ignore[misc]


# BenchmarkProtocol ABC contract


class TestBenchmarkProtocolContract:
    def test_cannot_instantiate_abstract(self):
        """ABC with abstract methods must not be instantiable."""
        with pytest.raises(TypeError):
            BenchmarkProtocol()  # type: ignore[abstract]

    def test_concrete_instantiates(self):
        bench = _MinimalBenchmark()
        assert bench.supported_robots == ["so100"]
        assert bench.default_robot == "so100"
        assert bench.max_steps == 42

    def test_is_failure_default_false(self):
        """The default is_failure returns False, so sparse-success benchmarks
        don't need to override it."""

        class _Sparse(_MinimalBenchmark):
            pass

        # Don't set failure=True; default should return False.
        bench = _Sparse()
        assert bench.is_failure(None) is False  # type: ignore[arg-type]

    def test_on_episode_start_has_default_impl(self):
        """on_episode_start is NOT abstract - base impl handles empty-sim + compat checks."""

        # Fake sim with no robots - should call add_robot with default_robot.
        class FakeSim:
            def __init__(self):
                self._robots: list[str] = []
                self.add_robot_calls: list[dict[str, Any]] = []

            def list_robots(self):
                return list(self._robots)

            def add_robot(self, *, name, data_config):
                self.add_robot_calls.append({"name": name, "data_config": data_config})
                self._robots.append(name)

        sim = FakeSim()
        bench = _MinimalBenchmark(supported=["so100"], default="so100")
        bench.on_episode_start(sim, random.Random(0))  # type: ignore[arg-type]
        assert len(sim.add_robot_calls) == 1
        assert sim.add_robot_calls[0]["data_config"] == "so100"


# augment_observation hook (#156)


class TestAugmentObservation:
    def test_default_is_pass_through(self):
        bench = _MinimalBenchmark()
        obs = {"j0": 1.0, "j1": 2.0, "image": "fake_array"}
        # Default impl returns the same dict (or an equivalent dict) without
        # mutating it. Identity by value, not by reference, is the contract.
        out = bench.augment_observation(None, obs)  # type: ignore[arg-type]
        assert out == obs

    def test_subclass_can_inject_keys(self):
        """A subclass override is the supported extension point."""

        class _AugBench(_MinimalBenchmark):
            def augment_observation(self, sim, obs):  # type: ignore[override]
                merged = dict(obs)
                merged["x"] = 1.5
                return merged

        bench = _AugBench()
        out = bench.augment_observation(None, {"j0": 0.0})  # type: ignore[arg-type]
        assert out["x"] == 1.5
        assert out["j0"] == 0.0


# Robot compatibility


class TestRobotCompatibility:
    def test_raises_when_loaded_robot_incompatible(self):
        """Loading a robot whose data_config is not in supported_robots raises
        BenchmarkCompatibilityError."""

        class _Robot:
            data_config = "panda"  # not in supported

        class FakeSimWithWorld:
            _world: Any = None

            def __init__(self):
                self._world = type(
                    "World",
                    (),
                    {"robots": {"arm1": _Robot()}},
                )()

            def list_robots(self):
                return ["arm1"]

            def add_robot(self, **kw):  # pragma: no cover
                raise AssertionError("should not be called when robot already present")

        sim = FakeSimWithWorld()
        bench = _MinimalBenchmark(supported=["so100", "so101"], default="so100")
        with pytest.raises(BenchmarkCompatibilityError) as excinfo:
            bench.on_episode_start(sim, random.Random(0))  # type: ignore[arg-type]
        assert excinfo.value.robot_name == "arm1"
        assert excinfo.value.data_config == "panda"
        assert excinfo.value.supported == ["so100", "so101"]

    def test_passes_when_supported_robots_empty(self):
        """Empty supported_robots means 'any robot' - no compat check."""

        class _Robot:
            data_config = "any_weird_thing"

        class FakeSim:
            def __init__(self):
                self._world = type(
                    "World",
                    (),
                    {"robots": {"arm1": _Robot()}},
                )()

            def list_robots(self):
                return ["arm1"]

            def add_robot(self, **kw):  # pragma: no cover
                raise AssertionError("should not be called")

        sim = FakeSim()
        bench = _MinimalBenchmark(supported=[], default="any_weird_thing")
        # Must not raise.
        bench.on_episode_start(sim, random.Random(0))  # type: ignore[arg-type]

    def test_skips_check_when_sim_has_no_world_attr(self):
        """Backends without a ``_world`` attribute are treated as "cannot verify"
        and skip the compat check rather than false-positive."""

        class FakeSim:
            def list_robots(self):
                return ["arm1"]

            def add_robot(self, **kw):  # pragma: no cover
                raise AssertionError("should not be called")

        sim = FakeSim()
        bench = _MinimalBenchmark(supported=["so100"], default="so100")
        # Must not raise even though arm1 has unknown data_config.
        bench.on_episode_start(sim, random.Random(0))  # type: ignore[arg-type]

    def test_skips_robot_listed_but_absent_from_world(self):
        """A robot returned by ``sim.list_robots()`` but missing from
        ``world.robots`` is skipped, not treated as a compat failure.

        The compatibility gate reconciles two independent robot views: the
        engine's ``list_robots()`` and the backend world's ``robots`` mapping.
        Those can legitimately diverge (a robot registered with the engine
        before it is composed into the world). Looking the name up with
        ``world.robots.get(name)`` and skipping a ``None`` result keeps the
        gate robust; a direct ``world.robots[name]`` index would raise
        ``KeyError`` and abort every episode on such a divergence.
        """

        class _Robot:
            data_config = "panda"  # not in supported -- would raise if reached

        class FakeSimWithGhost:
            def __init__(self):
                # world knows only "present"; list_robots also reports "ghost".
                self._world = type(
                    "World",
                    (),
                    {"robots": {"present": _Robot()}},
                )()

            def list_robots(self):
                return ["ghost", "present"]

            def add_robot(self, **kw):  # pragma: no cover
                raise AssertionError("should not be called")

        sim = FakeSimWithGhost()
        # supported includes "present"'s config so only the divergence path is
        # exercised: "ghost" (absent from world.robots) must be skipped, and
        # "present" (config in supported) must pass -- neither raises.
        bench = _MinimalBenchmark(supported=["so100", "panda"], default="so100")
        bench.on_episode_start(sim, random.Random(0))  # type: ignore[arg-type]


# Registry


class TestRegistry:
    def test_register_and_get(self):
        bench = _MinimalBenchmark()
        register_benchmark("my-task", bench)
        assert get_benchmark("my-task") is bench

    def test_get_unknown_returns_none(self):
        assert get_benchmark("nonexistent") is None

    def test_register_rejects_non_string_name(self):
        with pytest.raises(ValueError):
            register_benchmark("", _MinimalBenchmark())
        with pytest.raises(ValueError):
            register_benchmark(None, _MinimalBenchmark())  # type: ignore[arg-type]

    def test_register_rejects_non_benchmark(self):
        with pytest.raises(TypeError):
            register_benchmark("x", "not a benchmark")  # type: ignore[arg-type]

    def test_register_overwrites_and_warns(self, caplog):
        """Re-registering the same name replaces the entry and logs a warning."""
        first = _MinimalBenchmark()
        second = _MinimalBenchmark()
        register_benchmark("dup", first)
        with caplog.at_level("WARNING"):
            register_benchmark("dup", second)
        assert get_benchmark("dup") is second
        assert any("Overwriting existing" in rec.message for rec in caplog.records)

    def test_unregister_removes(self):
        bench = _MinimalBenchmark()
        register_benchmark("rm-me", bench)
        removed = unregister_benchmark("rm-me")
        assert removed is bench
        assert get_benchmark("rm-me") is None

    def test_unregister_unknown_returns_none(self):
        assert unregister_benchmark("never-registered") is None

    def test_list_benchmarks_metadata(self):
        bench = _MinimalBenchmark(supported=["so100", "so101"], default="so100")
        register_benchmark("listed", bench)
        listed = list_benchmarks()
        assert "listed" in listed
        meta = listed["listed"]
        assert meta["class"] == "_MinimalBenchmark"
        assert meta["supported_robots"] == ["so100", "so101"]
        assert meta["default_robot"] == "so100"
        assert meta["max_steps"] == 42

    def test_list_benchmarks_empty(self):
        assert list_benchmarks() == {}


class TestRegistryThreadSafety:
    """The registry guard is an RLock so concurrent registrations don't race."""

    def test_concurrent_registrations_all_land(self):
        benches = [_MinimalBenchmark() for _ in range(50)]
        barrier = threading.Barrier(len(benches))

        def register(i: int):
            barrier.wait()
            register_benchmark(f"thread-{i}", benches[i])

        threads = [threading.Thread(target=register, args=(i,)) for i in range(len(benches))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        listed = list_benchmarks()
        for i in range(len(benches)):
            assert f"thread-{i}" in listed


class TestBenchmarkCompatibilityError:
    def test_carries_context(self):
        e = BenchmarkCompatibilityError(robot_name="arm", data_config="foo", supported=["bar"])
        assert e.robot_name == "arm"
        assert e.data_config == "foo"
        assert e.supported == ["bar"]
        # Subclasses ValueError so broad except ValueError still works.
        assert isinstance(e, ValueError)
