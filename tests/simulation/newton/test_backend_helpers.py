"""Unit tests for the Newton backend helpers (no GPU / no Newton required)."""

from __future__ import annotations

import importlib.util
import math

import pytest

from strands_robots.simulation.newton import backend
from strands_robots.simulation.newton.simulation import (
    _quat_rotate_inverse_wxyz,
    _short_joint_name,
)
from strands_robots.simulation.predicates import (
    _quat_rotate_inverse_wxyz as _predicates_quat_rotate_inverse_wxyz,
)

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None


class TestSolverRegistry:
    def test_registry_lists_rigid_body_solvers(self):
        reg = backend.solver_registry()
        assert reg["mujoco"] == "SolverMuJoCo"
        assert reg["featherstone"] == "SolverFeatherstone"
        assert reg["xpbd"] == "SolverXPBD"

    def test_registry_keys_are_lowercase(self):
        assert all(k == k.lower() for k in backend.solver_registry())


class TestShortJointName:
    def test_strips_hierarchical_path(self):
        label = "so_arm100/worldbody/Base/Rotation_Pitch/Rotation"
        assert _short_joint_name(label) == "Rotation"

    def test_plain_name_unchanged(self):
        assert _short_joint_name("Jaw") == "Jaw"


def _reference_world_to_body(quat_wxyz: list[float], vec: list[float]) -> list[float]:
    """Reference world->body rotation: ``R(q)^T @ vec`` from first principles.

    Builds the rotation matrix from a normalised (w, x, y, z) quaternion and
    applies its transpose, independent of the implementation under test.
    """
    w, x, y, z = quat_wxyz
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    # Body<-world rotation is R(q) transposed (== R(q^-1)).
    r = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    # transpose(R) @ vec
    return [sum(r[k][i] * vec[k] for k in range(3)) for i in range(3)]


class TestQuatRotateInverseWxyz:
    """Body-frame the base angular velocity: ``R(q)^T @ vec`` for a (w,x,y,z) quat.

    The Newton backend rotates the free-joint world-frame angular velocity into
    the body frame so ``base_ang_vel`` matches the MuJoCo backend and the
    IMU-gyro convention locomotion / WBC controllers consume. A wrong transform
    silently corrupts the observation those controllers close the loop on, so
    the numeric contract is pinned directly (it is otherwise unreachable without
    a GPU + Newton install).
    """

    def test_identity_quaternion_returns_vector_unchanged(self):
        assert _quat_rotate_inverse_wxyz([1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == pytest.approx(
            [1.0, 2.0, 3.0], abs=1e-12
        )

    def test_yaw_90_matches_first_principles_reference(self):
        # +90 deg about world +Z. A world +X vector reads as body -Y.
        q = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
        got = _quat_rotate_inverse_wxyz(q, [1.0, 0.0, 0.0])
        assert got == pytest.approx(_reference_world_to_body(q, [1.0, 0.0, 0.0]), abs=1e-9)
        assert got == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)

    def test_arbitrary_rotation_matches_first_principles_reference(self):
        q = [0.5, 0.5, -0.5, 0.5]  # a valid unit quaternion (90 deg composite)
        vec = [0.3, -1.2, 4.5]
        assert _quat_rotate_inverse_wxyz(q, vec) == pytest.approx(_reference_world_to_body(q, vec), abs=1e-9)

    def test_unnormalized_quaternion_is_normalized_internally(self):
        # Scaling a quaternion does not change the rotation it encodes; the
        # helper must normalise so callers can pass a raw free-joint quat.
        q_unit = [0.5, 0.5, -0.5, 0.5]
        q_scaled = [4.0 * c for c in q_unit]
        vec = [1.0, -2.0, 0.5]
        assert _quat_rotate_inverse_wxyz(q_scaled, vec) == pytest.approx(
            _quat_rotate_inverse_wxyz(q_unit, vec), abs=1e-12
        )

    def test_zero_norm_quaternion_returns_vector_unchanged(self):
        # A degenerate (~zero-norm) quaternion cannot define a rotation; the
        # documented contract returns the input vector rather than dividing by ~0.
        assert _quat_rotate_inverse_wxyz([0.0, 0.0, 0.0, 0.0], [5.0, 6.0, 7.0]) == pytest.approx(
            [5.0, 6.0, 7.0], abs=1e-12
        )
        assert _quat_rotate_inverse_wxyz([1e-12, 0.0, 0.0, 0.0], [5.0, 6.0, 7.0]) == pytest.approx(
            [5.0, 6.0, 7.0], abs=1e-12
        )

    def test_parity_with_canonical_predicates_implementation(self):
        # The predicates module keeps an intentional numpy-free mirror of this
        # helper (predicates stay dependency-free). Two copies can silently
        # drift; assert bit-for-bit agreement across a spread of quats/vectors
        # so a fix or regression in one copy cannot diverge unnoticed.
        cases = [
            ([1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),
            ([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)], [1.0, 0.0, 0.0]),
            ([0.5, 0.5, -0.5, 0.5], [0.3, -1.2, 4.5]),
            ([2.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),  # unnormalized
            ([0.0, 0.0, 0.0, 0.0], [5.0, 6.0, 7.0]),  # degenerate
        ]
        for quat, vec in cases:
            assert _quat_rotate_inverse_wxyz(quat, vec) == pytest.approx(
                _predicates_quat_rotate_inverse_wxyz(quat, vec), abs=1e-9
            )


@pytest.mark.skipif(_HAS_NEWTON, reason="newton installed; missing-dep path not exercised")
class TestMissingDependency:
    def test_resolve_solver_class_raises_without_newton(self):
        with pytest.raises(ImportError, match="sim-newton"):
            backend.resolve_solver_class("mujoco")


@pytest.mark.skipif(not _HAS_NEWTON, reason="newton not installed")
class TestSolverResolution:
    def test_resolve_known_solver(self):
        cls = backend.resolve_solver_class("featherstone")
        assert cls.__name__ == "SolverFeatherstone"

    def test_resolve_unknown_solver_raises(self):
        with pytest.raises(ValueError, match="Unknown Newton solver"):
            backend.resolve_solver_class("not_a_solver")


class _FakeSolvers:
    """Stand-in for ``newton.solvers`` exposing the registry class names."""

    class SolverMuJoCo:  # noqa: N801 - mirrors the real newton class name
        pass

    class SolverFeatherstone:  # noqa: N801
        pass


class _FakeNewton:
    """Minimal stub of the ``newton`` module used to drive resolution paths."""

    solvers = _FakeSolvers


class TestLazyImportCache:
    """The ``_modules`` cache is the documented single source of import state."""

    def test_ensure_newton_returns_cached_modules(self, monkeypatch):
        fake_nt, fake_wp = _FakeNewton(), object()
        monkeypatch.setattr(backend, "_modules", {"newton": fake_nt, "warp": fake_wp})
        # require_optional must NOT be consulted on a cache hit; if it is, fail loud.
        monkeypatch.setattr(
            backend,
            "require_optional",
            lambda *a, **k: pytest.fail("cache hit should not import"),
        )
        assert backend.ensure_newton() == (fake_nt, fake_wp)

    def test_ensure_newton_imports_and_populates_cache(self, monkeypatch):
        fake_nt, fake_wp = _FakeNewton(), object()

        def fake_require(name, **kwargs):
            return {"warp": fake_wp, "newton": fake_nt}[name]

        monkeypatch.setattr(backend, "_modules", {})
        monkeypatch.setattr(backend, "require_optional", fake_require)

        assert backend.ensure_newton() == (fake_nt, fake_wp)
        # Second call is served from the now-populated cache.
        assert backend._modules == {"newton": fake_nt, "warp": fake_wp}


class TestSolverResolutionWithStub:
    """resolve_solver_class maps friendly names to newton.solvers classes."""

    def test_resolve_known_solver_returns_class(self, monkeypatch):
        monkeypatch.setattr(backend, "_modules", {"newton": _FakeNewton(), "warp": object()})
        assert backend.resolve_solver_class("MuJoCo") is _FakeSolvers.SolverMuJoCo

    def test_resolve_unknown_solver_raises_value_error(self, monkeypatch):
        monkeypatch.setattr(backend, "_modules", {"newton": _FakeNewton(), "warp": object()})
        with pytest.raises(ValueError, match="Unknown Newton solver 'nope'"):
            backend.resolve_solver_class("nope")
