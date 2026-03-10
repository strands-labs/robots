# Copyright Strands Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SimulationBackend ABC.

Verifies:
- ABC contract enforcement (cannot instantiate with missing methods)
- Complete implementations can be instantiated
- Polymorphism and factory patterns
- Zero heavy-dependency imports
- Signature compatibility with MuJoCo, Isaac, Newton patterns
"""

import inspect

import pytest

from strands_robots.simulation_backend import SimulationBackend

# ── Helpers ─────────────────────────────────────────────────────────


class StubBackend(SimulationBackend):
    """Minimal concrete implementation for testing."""

    def create_world(self, gravity=None, ground_plane=True, **kw):
        return {"status": "success", "content": [{"text": "World created"}]}

    def destroy(self):
        return {"status": "success", "content": [{"text": "Destroyed"}]}

    def reset(self, **kw):
        return {"status": "success", "content": [{"text": "Reset"}]}

    def get_state(self):
        return {"status": "success", "content": [{"text": "State"}]}

    def add_robot(self, name, **kw):
        return {"status": "success", "content": [{"text": f"Added {name}"}]}

    def step(self, **kw):
        return {"status": "success", "content": [{"text": "Stepped"}]}

    def get_observation(self, robot_name=None, **kw):
        return {"joint_0": 0.0, "joint_1": 0.5}

    def render(self, camera_name=None, width=None, height=None, **kw):
        return {"status": "success", "content": [{"text": "Rendered"}]}

    def run_policy(self, robot_name, policy_provider="mock", instruction="", duration=10.0, **kw):
        return {"status": "success", "content": [{"text": "Policy done"}]}

    def record_video(self, **kw):
        return {"status": "success", "content": [{"text": "Recorded"}]}


# ── ABC Contract Tests ──────────────────────────────────────────────


class TestABCContract:
    """Verify the ABC enforces the contract."""

    def test_cannot_instantiate_abc_directly(self):
        """SimulationBackend itself cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract method"):
            SimulationBackend()

    def test_exactly_10_abstract_methods(self):
        """The ABC defines exactly 10 abstract methods."""
        abstract_methods = set()
        for name, method in inspect.getmembers(SimulationBackend):
            if getattr(method, "__isabstractmethod__", False):
                abstract_methods.add(name)

        assert abstract_methods == {
            "create_world",
            "destroy",
            "reset",
            "get_state",
            "add_robot",
            "step",
            "get_observation",
            "render",
            "run_policy",
            "record_video",
        }

    def test_no_heavy_imports(self):
        """The ABC module has zero heavy dependencies."""
        import strands_robots.simulation_backend as mod

        source = inspect.getsource(mod)
        forbidden = ["numpy", "mujoco", "torch", "warp", "isaaclab", "isaacsim"]
        for dep in forbidden:
            assert f"import {dep}" not in source, f"Found forbidden import: {dep}"


# ── Partial Implementation Tests ────────────────────────────────────


class TestPartialImplementation:
    """Verify incomplete implementations fail at instantiation."""

    def test_missing_one_method_fails(self):
        """Missing a single abstract method prevents instantiation."""

        class IncompleteBackend(SimulationBackend):
            def create_world(self, gravity=None, ground_plane=True, **kw):
                return {}

            def destroy(self):
                return {}

            def reset(self, **kw):
                return {}

            def get_state(self):
                return {}

            def add_robot(self, name, **kw):
                return {}

            def step(self, **kw):
                return {}

            def get_observation(self, robot_name=None, **kw):
                return {}

            def render(self, camera_name=None, width=None, height=None, **kw):
                return {}

            def run_policy(self, robot_name, policy_provider="mock", instruction="", duration=10.0, **kw):
                return {}

            # record_video is missing

        with pytest.raises(TypeError, match="record_video"):
            IncompleteBackend()

    def test_missing_multiple_methods_fails(self):
        """Missing multiple abstract methods lists them all."""

        class EmptyBackend(SimulationBackend):
            pass

        with pytest.raises(TypeError):
            EmptyBackend()


# ── Complete Implementation Tests ───────────────────────────────────


class TestCompleteImplementation:
    """Verify a complete implementation works correctly."""

    def test_can_instantiate(self):
        """A complete implementation can be instantiated."""
        backend = StubBackend()
        assert backend is not None

    def test_isinstance_check(self):
        """isinstance works for type checking."""
        backend = StubBackend()
        assert isinstance(backend, SimulationBackend)

    def test_return_format(self):
        """Methods return the documented format."""
        backend = StubBackend()

        result = backend.create_world()
        assert result["status"] == "success"
        assert isinstance(result["content"], list)
        assert "text" in result["content"][0]

        result = backend.add_robot("panda")
        assert result["status"] == "success"
        assert "panda" in result["content"][0]["text"].lower() or "Added" in result["content"][0]["text"]


# ── Polymorphism Tests ──────────────────────────────────────────────


class TestPolymorphism:
    """Verify backends can be used polymorphically."""

    def test_factory_pattern(self):
        """Backends can be created via a factory function."""

        def create_backend(name: str) -> SimulationBackend:
            if name == "stub":
                return StubBackend()
            raise ValueError(f"Unknown backend: {name}")

        backend = create_backend("stub")
        assert isinstance(backend, SimulationBackend)
        result = backend.create_world()
        assert result["status"] == "success"

    def test_multiple_backends(self):
        """Multiple backend instances work independently."""
        b1 = StubBackend()
        b2 = StubBackend()

        r1 = b1.add_robot("robot_a")
        r2 = b2.add_robot("robot_b")

        assert "robot_a" in r1["content"][0]["text"].lower()
        assert "robot_b" in r2["content"][0]["text"].lower()

    def test_backend_with_extensions(self):
        """Backends can add methods beyond the ABC contract."""

        class ExtendedBackend(StubBackend):
            def add_cloth(self, name, **kw):
                return {"status": "success", "content": [{"text": f"Cloth {name}"}]}

        backend = ExtendedBackend()
        assert isinstance(backend, SimulationBackend)
        assert backend.add_cloth("towel")["status"] == "success"


# ── Signature Compatibility Tests ───────────────────────────────────


class TestSignatureCompatibility:
    """Verify the ABC supports all backend calling patterns."""

    def test_kwargs_for_create_world(self):
        """create_world accepts backend-specific kwargs."""
        backend = StubBackend()
        # MuJoCo pattern
        backend.create_world(gravity=[0, 0, -9.81], ground_plane=True)
        # Newton pattern
        backend.create_world(gravity=(0, -9.81, 0), up_axis="y")
        # Isaac pattern
        backend.create_world(gravity=None, ground_plane=True)

    def test_kwargs_for_add_robot(self):
        """add_robot supports different model format kwargs."""
        backend = StubBackend()
        # MuJoCo: URDF/MJCF path
        backend.add_robot("panda", urdf_path="/path/to/panda.xml")
        # Isaac: USD path
        backend.add_robot("go2", usd_path="/path/to/go2.usd")
        # Newton: data_config with scale
        backend.add_robot("so100", data_config="so100", scale=1.5)
        # Alias-only (factory resolves)
        backend.add_robot("g1")

    def test_kwargs_for_step(self):
        """step supports count-based and action-based patterns."""
        backend = StubBackend()
        # MuJoCo pattern
        backend.step(n_steps=10)
        # GPU backend pattern
        backend.step(actions="tensor_placeholder")
