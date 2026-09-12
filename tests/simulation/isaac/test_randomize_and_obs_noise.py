"""Isaac's ``randomize`` / ``set_obs_noise``: the two stubs with the most callers.

Until this module's subject existed, both were the ``SimEngine`` raising stubs -
while ``docs/simulation/domain-randomization.md`` and ~30 example call sites
drive ``randomize()`` and ~10 drive ``set_obs_noise()`` through the
backend-agnostic surface, so the identical script randomized on MuJoCo and
Newton and raised ``NotImplementedError`` on Isaac.

What is pinned here, without a GPU (the USD/physics leaves are stood in; the
axes' real writes - displayColor reaching the RTX frame, ``set_mass``,
``PhysicsMaterial`` friction, the ground plane's ``SphereLight`` - are verified
live):

* the validation order and the shared domains - unknown kwargs refused by name,
  posture flags on ``boolean_flag_error`` (a truthy ``"false"`` must not turn an
  axis ON), ranges on ``randomization_range_error``, the seed on
  ``randomization_seed_error``, all before anything is written;
* seed determinism - two calls with one seed record identical samples in the
  report, which is what makes a run reproducible from its own json;
* anti-compounding - every scale is measured from the FIRST-TOUCH base, so two
  randomize calls do not walk a mass off; the second call's report scales the
  same base;
* the sensor-noise pass - suffix-keyed exactly as MuJoCo's (``joint_pos_std``
  to position floats, ``joint_vel_std`` to ``.vel`` floats, an integer-pixel
  roll to frames, list values untouched), applied by ``get_observation``;
* the signatures match the sibling backends' shared order (also held by the
  derived-inventory grader in ``test_backend_shared_parameter_order.py``).
"""

from __future__ import annotations

import threading
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac import randomization as rnd_module  # noqa: E402
from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    IsaacConfig,
    IsaacSimulation,
    _ObjectState,
    _RobotState,
)


class _Handle:
    """A dynamic-object handle covering the physics + position surface."""

    def __init__(self, mass: float = 0.1, pos: tuple[float, float, float] = (0.3, 0.0, 0.2)) -> None:
        self._mass = mass
        self._pos = np.asarray(pos, dtype=float)
        self.material = types.SimpleNamespace(
            prim_path="/World/Physics_Materials/m1",
            _static=0.2,
            _dynamic=1.0,
        )
        self.material.get_static_friction = lambda m=self.material: m._static
        self.material.get_dynamic_friction = lambda m=self.material: m._dynamic
        self.material.set_static_friction = lambda v, m=self.material: setattr(m, "_static", float(v))
        self.material.set_dynamic_friction = lambda v, m=self.material: setattr(m, "_dynamic", float(v))

    def get_mass(self) -> float:
        return self._mass

    def set_mass(self, value: float) -> None:
        self._mass = float(value)

    def get_applied_physics_material(self) -> Any:
        return self.material

    def get_world_pose(self) -> Any:
        return self._pos.copy(), np.array([1.0, 0.0, 0.0, 0.0])

    def set_world_pose(self, position: Any = None, orientation: Any = None) -> None:
        if position is not None:
            self._pos = np.asarray(position, dtype=float)


def _engine(with_object: bool = True) -> Any:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(render_mode="headless")
    engine._world = types.SimpleNamespace()
    engine._world_created = True
    engine._robots = {}
    engine._cameras = {}
    engine._objects = {}
    if with_object:
        state = _ObjectState(name="cube", prim_path="/World/Objects/cube", shape="box", is_static=False)
        state.handle = _Handle()
        engine._objects["cube"] = state
    return engine


@pytest.fixture(autouse=True)
def _stage_free(monkeypatch):
    """The color/lighting axes walk the USD stage; keep them inert here so the
    physics/positions axes - the ones the stand-ins cover - are the measurement.
    The real stage writes are the GPU probe's business."""
    monkeypatch.setattr(
        rnd_module.IsaacRandomizationMixin,
        "_randomize_colors",
        lambda self, rng, cr, applied: applied.setdefault("colors", {}) or 0,
    )
    monkeypatch.setattr(
        rnd_module.IsaacRandomizationMixin,
        "_randomize_lighting",
        lambda self, rng, cr, applied: applied.setdefault("lights", {}) or 0,
    )


class TestTheValidationRunsBeforeAnyWrite:
    def test_an_unknown_kwarg_is_refused_by_name(self) -> None:
        result = _engine().randomize(randomize_colours=True)
        assert result["status"] == "error"
        assert "randomize_colours" in result["content"][0]["text"]

    def test_no_world_is_the_first_scene_check(self) -> None:
        engine = _engine()
        engine._world_created = False
        assert "No world" in _engine_text(engine.randomize())

    @pytest.mark.parametrize("bad", ["false", "no", "off", "0", "", None, 0, 1])
    @pytest.mark.parametrize(
        "flag", ["randomize_colors", "randomize_lighting", "randomize_physics", "randomize_positions"]
    )
    def test_every_axis_flag_is_checked_not_read_by_truthiness(self, flag: str, bad: Any) -> None:
        """``randomize_physics="false"`` is a truthy string: read by truthiness
        it would scale every mass for a caller who spelled out that they wanted
        physics untouched."""
        result = _engine().randomize(**{flag: bad})
        assert result["status"] == "error", (flag, bad)
        assert flag in result["content"][0]["text"]

    @pytest.mark.parametrize("param", ["color_range", "friction_range", "mass_range"])
    def test_a_malformed_range_is_refused(self, param: str) -> None:
        result = _engine().randomize(**{param: (1.5, 0.5)})
        assert result["status"] == "error", param
        assert param in result["content"][0]["text"]

    def test_a_negative_position_noise_is_refused(self) -> None:
        result = _engine().randomize(randomize_positions=True, position_noise=-0.1)
        assert result["status"] == "error"
        assert "position_noise" in result["content"][0]["text"]

    def test_a_bad_seed_is_refused(self) -> None:
        assert _engine().randomize(seed="7")["status"] == "error"

    def test_a_refused_call_wrote_nothing(self) -> None:
        engine = _engine()
        handle = engine._objects["cube"].handle
        engine.randomize(randomize_physics=True, mass_range=(1.5, 0.5))
        assert handle.get_mass() == pytest.approx(0.1), "a refused range still scaled the mass"


class TestTheAxesWriteWhatTheReportSays:
    def test_no_axes_is_a_no_op(self) -> None:
        result = _engine().randomize(
            randomize_colors=False, randomize_lighting=False, randomize_physics=False, randomize_positions=False
        )
        assert result["status"] == "success"
        assert "nothing randomized" in _text(result).lower()

    def test_physics_scales_mass_and_friction_from_base(self) -> None:
        engine = _engine()
        handle = engine._objects["cube"].handle
        result = engine.randomize(
            randomize_colors=False, randomize_lighting=False, randomize_physics=True, seed=7, mass_range=(2.0, 2.0)
        )
        assert result["status"] == "success", result
        assert handle.get_mass() == pytest.approx(0.2), "mass = base 0.1 x the pinned scale 2.0"
        report = _json(result)
        assert report["masses"]["cube"]["base"] == pytest.approx(0.1)
        assert report["frictions"]["/World/Physics_Materials/m1"]["scale"] > 0

    def test_repeated_calls_scale_the_base_not_the_last_value(self) -> None:
        """The anti-compounding anchor: mass_range=(2,2) applied twice must
        yield base x 2, not base x 4."""
        engine = _engine()
        self._assert_no_compounding(engine)

    def test_no_compounding_on_a_constructor_built_engine_either(self) -> None:
        """The GPU run caught what the ``__new__`` skeleton could not: the base
        registry used to be read with ``getattr(...) or {}``, and ``__init__``
        pre-creates it EMPTY - falsy - so a real engine rebuilt the base every
        call and compounded (0.105 kg -> 0.421 kg over two x2 calls, measured
        live), while the skeleton, having no ``_dr_base`` at all, took the
        branch that stored it and passed. This engine runs the real
        ``__init__``, so the falsy-empty-dict spelling cannot come back
        unnoticed."""
        engine = IsaacSimulation()
        engine._world = types.SimpleNamespace()
        engine._world_created = True
        state = _ObjectState(name="cube", prim_path="/World/Objects/cube", shape="box", is_static=False)
        state.handle = _Handle()
        engine._objects["cube"] = state
        self._assert_no_compounding(engine)

    @staticmethod
    def _assert_no_compounding(engine: Any) -> None:
        handle = engine._objects["cube"].handle
        for _ in range(2):
            assert (
                engine.randomize(
                    randomize_colors=False, randomize_lighting=False, randomize_physics=True, mass_range=(2.0, 2.0)
                )["status"]
                == "success"
            )
        assert handle.get_mass() == pytest.approx(0.2), f"compounded to {handle.get_mass()}"

    def test_positions_offset_xy_from_base_and_preserve_z(self) -> None:
        engine = _engine()
        handle = engine._objects["cube"].handle
        result = engine.randomize(
            randomize_colors=False,
            randomize_lighting=False,
            randomize_positions=True,
            position_noise=0.05,
            seed=3,
        )
        assert result["status"] == "success", result
        pos = handle._pos
        assert pos[2] == pytest.approx(0.2), "z must be preserved - a z offset buries or drops the object"
        assert abs(pos[0] - 0.3) <= 0.05 and abs(pos[1] - 0.0) <= 0.05
        assert (pos[0], pos[1]) != (0.3, 0.0)

    def test_a_static_object_is_not_mass_scaled_or_moved(self) -> None:
        engine = _engine()
        engine._objects["cube"].is_static = True
        result = engine.randomize(
            randomize_colors=False, randomize_lighting=False, randomize_physics=True, randomize_positions=True
        )
        assert result["status"] == "success"
        report = _json(result)
        assert report["masses"] == {} and report["positions"] == {}

    def test_one_seed_records_one_sample_set(self) -> None:
        """Reproducibility from the report + seed - two engines, one seed,
        identical sampled scales."""
        reports = []
        for _ in range(2):
            result = _engine().randomize(
                randomize_colors=False, randomize_lighting=False, randomize_physics=True, seed=42
            )
            reports.append(_json(result))
        assert reports[0] == reports[1]


class TestObsNoise:
    def _noisy_engine(self, **noise: float) -> Any:
        engine = _engine(with_object=False)
        engine._robots["arm"] = _RobotState(
            name="arm",
            prim_path="/World/Robots/arm",
            joint_names=["j0"],
            articulation=types.SimpleNamespace(get_joint_positions=lambda: np.array([0.5], dtype=np.float32)),
        )
        assert engine.set_obs_noise(seed=5, **noise)["status"] == "success"
        return engine

    def test_position_noise_lands_on_the_joint_floats(self) -> None:
        obs = self._noisy_engine(joint_pos_std=0.3).get_observation()
        assert obs["j0"] != pytest.approx(0.5)

    def test_zero_stds_clear_the_noise(self) -> None:
        engine = self._noisy_engine(joint_pos_std=0.3)
        assert "cleared" in _text(engine.set_obs_noise())
        assert engine.get_observation()["j0"] == pytest.approx(0.5)

    def test_the_pass_is_suffix_keyed_and_leaves_lists_alone(self) -> None:
        """Graded on the helper directly with every value shape the schema
        carries: position float, ``.vel`` float, ndarray frame, base_* list."""
        engine = _engine(with_object=False)
        engine.set_obs_noise(joint_pos_std=0.5, joint_vel_std=0.0, camera_jitter_px=0.0, seed=1)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        obs = {"j0": 0.5, "j0.vel": 1.5, "cam": frame, "base_quat": [1.0, 0.0, 0.0, 0.0]}
        out = engine._apply_obs_noise(obs)
        assert out["j0"] != pytest.approx(0.5), "joint_pos_std configured but not applied"
        assert out["j0.vel"] == pytest.approx(1.5), "joint_pos_std leaked onto a .vel key"
        assert out["cam"] is frame, "no jitter configured, the frame must pass through"
        assert out["base_quat"] == [1.0, 0.0, 0.0, 0.0]

    def test_camera_jitter_rolls_the_frame(self) -> None:
        engine = _engine(with_object=False)
        engine.set_obs_noise(camera_jitter_px=2, seed=2)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[0, 0] = 255
        out = engine._apply_obs_noise({"cam": frame})
        assert out["cam"].shape == frame.shape
        assert not np.array_equal(out["cam"], frame), "jitter configured but the frame is unmoved"
        assert out["cam"].sum() == frame.sum(), "a roll relocates pixels, never invents or drops them"

    def test_an_unknown_kwarg_is_refused_by_name(self) -> None:
        result = _engine().set_obs_noise(joint_pos_st=0.1)
        assert result["status"] == "error"
        assert "joint_pos_st" in result["content"][0]["text"]

    @pytest.mark.parametrize("param", ["joint_pos_std", "joint_vel_std", "camera_jitter_px"])
    def test_a_negative_std_is_refused(self, param: str) -> None:
        result = _engine().set_obs_noise(**{param: -0.1})
        assert result["status"] == "error", param
        assert param in result["content"][0]["text"]


class TestTheSharedSignatureOrder:
    """Held structurally by test_backend_shared_parameter_order.py's derived
    inventory; asserted here too so a failure names this feature directly."""

    def test_randomize_matches_the_mujoco_order(self) -> None:
        import inspect

        from strands_robots.simulation.mujoco.randomization import RandomizationMixin

        ours = [p for p in inspect.signature(IsaacSimulation.randomize).parameters if p not in ("self", "kwargs")]
        theirs = [p for p in inspect.signature(RandomizationMixin.randomize).parameters if p not in ("self", "kwargs")]
        assert ours == theirs

    def test_set_obs_noise_matches_the_mujoco_order(self) -> None:
        import inspect

        from strands_robots.simulation.mujoco.randomization import RandomizationMixin

        ours = [p for p in inspect.signature(IsaacSimulation.set_obs_noise).parameters if p not in ("self", "kwargs")]
        theirs = [
            p for p in inspect.signature(RandomizationMixin.set_obs_noise).parameters if p not in ("self", "kwargs")
        ]
        assert ours == theirs


def _text(result: dict[str, Any]) -> str:
    return " ".join(str(b.get("text", "")) for b in result["content"])


def _engine_text(result: dict[str, Any]) -> str:
    return _text(result)


def _json(result: dict[str, Any]) -> dict[str, Any]:
    return next(b["json"] for b in result["content"] if "json" in b)
