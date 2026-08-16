"""Backend-agnostic motion-primitives core, exercised without an engine.

``MotionPrimitivesCore`` (:mod:`strands_robots.simulation.motion_primitives_base`,
extracted from the MuJoCo mixin per GH #2153) owns the parameter domains, the
registry gripper-metadata contract, and the shared result envelopes for
``move_to`` / ``set_gripper`` / ``rotate_wrist``. The full agent-facing
behaviour on a live world is pinned by the MuJoCo suites
(``tests/simulation/mujoco/test_motion_primitives.py`` and siblings); this file
pins what those suites cannot: that the core answers on plain data with no
backend at all, and that the module imports without MuJoCo installed - the
property the Isaac adapter (GH #2123) builds on.
"""

import inspect
import math
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from strands_robots.simulation.motion_primitives_base import (
    _DEFAULT_ORIENTATION_TOL_RAD,
    MotionPrimitivesCore,
    _quat_angle_error,
)


def _error_text(result: dict | None) -> str:
    assert result is not None and result["status"] == "error", result
    return result["content"][0]["text"]


@pytest.fixture()
def core() -> MotionPrimitivesCore:
    return MotionPrimitivesCore()


class TestModuleIsBackendFree:
    def test_imports_without_mujoco(self) -> None:
        """The core module must import with no MuJoCo anywhere in the process.

        Fresh-interpreter check (immune to ``sys.modules`` pollution from
        other tests): the Isaac adapter consumes this module on machines
        without the ``[sim-mujoco]`` extra.
        """
        code = (
            "import sys\n"
            "sys.modules['mujoco'] = None  # any 'import mujoco' now explodes\n"
            "import strands_robots.simulation.motion_primitives_base as m\n"
            "leaked = [k for k, v in sys.modules.items() if k.startswith('mujoco') and v is not None]\n"
            "assert not leaked, leaked\n"
            "print('ok')\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        assert out.stdout.strip() == "ok"


class TestMoveToArgValidation:
    def test_missing_position_is_refused(self, core: MotionPrimitivesCore) -> None:
        target, quat, steps, _, err = core._validate_move_to_args(None, None, 0.01, 200)
        assert (target, quat, steps) == (None, None, 0)
        assert "requires 'position'" in _error_text(err)

    @pytest.mark.parametrize("position", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], "abc", [1.0, float("nan"), 3.0]])
    def test_malformed_position_is_refused(self, core: MotionPrimitivesCore, position) -> None:
        _, _, _, _, err = core._validate_move_to_args(position, None, 0.01, 200)
        assert err is not None
        assert "position" in _error_text(err)

    def test_zero_norm_quaternion_is_refused(self, core: MotionPrimitivesCore) -> None:
        _, _, _, _, err = core._validate_move_to_args([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 0.0], 0.01, 200)
        assert "~zero norm" in _error_text(err)

    @pytest.mark.parametrize("tol", [0.0, -0.01, float("nan"), float("inf"), "0.01", True])
    def test_off_domain_tol_is_refused(self, core: MotionPrimitivesCore, tol) -> None:
        _, _, _, _, err = core._validate_move_to_args([0.1, 0.2, 0.3], None, tol, 200)
        assert "'tol' must be a positive number" in _error_text(err)

    def test_valid_args_coerce_to_arrays(self, core: MotionPrimitivesCore) -> None:
        target, quat, max_steps, _, err = core._validate_move_to_args([0.1, 0.2, 0.3], [1, 0, 0, 0], 0.01, 200)
        assert err is None
        assert target is not None and quat is not None
        np.testing.assert_allclose(target, np.array([0.1, 0.2, 0.3]))
        np.testing.assert_allclose(quat, np.array([1.0, 0.0, 0.0, 0.0]))
        assert target.dtype == np.float64 and quat.dtype == np.float64
        assert max_steps == 200

    def test_orientation_is_optional(self, core: MotionPrimitivesCore) -> None:
        target, quat, _, _, err = core._validate_move_to_args([0.1, 0.2, 0.3], None, 0.01, 200)
        assert err is None and target is not None and quat is None


class TestOrientationToleranceResolution:
    """The rotational tolerance exists only where there is a rotation to bound."""

    def test_a_position_only_call_resolves_no_orientation_tolerance(self, core: MotionPrimitivesCore) -> None:
        """``None`` is what tells the backends to report no rotational error."""
        _, quat, _, orientation_tol, err = core._validate_move_to_args([0.1, 0.2, 0.3], None, 0.01, 200)
        assert err is None
        assert quat is None
        assert orientation_tol is None

    def test_an_orientation_without_a_tolerance_takes_the_documented_default(self, core: MotionPrimitivesCore) -> None:
        _, _, _, orientation_tol, err = core._validate_move_to_args([0.1, 0.2, 0.3], [1, 0, 0, 0], 0.01, 200)
        assert err is None
        assert orientation_tol == pytest.approx(_DEFAULT_ORIENTATION_TOL_RAD)

    def test_an_explicit_tolerance_wins_over_the_default(self, core: MotionPrimitivesCore) -> None:
        _, _, _, orientation_tol, err = core._validate_move_to_args([0.1, 0.2, 0.3], [1, 0, 0, 0], 0.01, 200, 0.02)
        assert err is None
        assert orientation_tol == pytest.approx(0.02)

    def test_a_tolerance_with_nothing_to_bound_is_refused_not_dropped(self, core: MotionPrimitivesCore) -> None:
        """Silently ignoring a tolerance the caller set is how an unmet request hides."""
        _, _, _, orientation_tol, err = core._validate_move_to_args([0.1, 0.2, 0.3], None, 0.01, 200, 0.02)
        assert orientation_tol is None
        assert "'orientation_tol' only bounds an 'orientation' target" in _error_text(err)

    @pytest.mark.parametrize("bad", [0.0, -0.01, float("nan"), float("inf"), "0.1", True])
    def test_off_domain_orientation_tol_is_refused(self, core: MotionPrimitivesCore, bad) -> None:
        _, _, _, _, err = core._validate_move_to_args([0.1, 0.2, 0.3], [1, 0, 0, 0], 0.01, 200, bad)
        assert "'orientation_tol' must be a positive number of radians" in _error_text(err)


class TestQuaternionAngleError:
    """The rotational error metric both backends measure convergence with."""

    def test_identical_orientations_have_no_error(self) -> None:
        assert _quat_angle_error([1, 0, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.0, abs=1e-12)

    def test_a_negated_quaternion_is_the_same_rotation(self) -> None:
        """The double cover: ``q`` and ``-q`` denote one orientation.

        Without the sign fold an exactly-met orientation measures ``pi`` - the
        maximum possible error - so every such request would be refused.
        """
        assert _quat_angle_error([1, 0, 0, 0], [-1, 0, 0, 0]) == pytest.approx(0.0, abs=1e-12)
        assert _quat_angle_error([0.5, 0.5, 0.5, 0.5], [-0.5, -0.5, -0.5, -0.5]) == pytest.approx(0.0, abs=1e-9)

    def test_a_quarter_turn_measures_a_quarter_turn(self) -> None:
        half = math.sqrt(0.5)
        assert _quat_angle_error([1, 0, 0, 0], [half, half, 0, 0]) == pytest.approx(math.pi / 2, abs=1e-9)

    def test_an_unnormalized_input_is_normalized_first(self) -> None:
        assert _quat_angle_error([2, 0, 0, 0], [0.5, 0, 0, 0]) == pytest.approx(0.0, abs=1e-12)

    def test_the_metric_is_bounded_by_pi(self) -> None:
        """A rotation and its opposite are half a turn apart, never more."""
        assert _quat_angle_error([1, 0, 0, 0], [0, 1, 0, 0]) == pytest.approx(math.pi, abs=1e-9)


class TestPoseViolation:
    """One scalar over two quantities: ``<= 1`` iff every requested component fits."""

    def test_a_position_only_call_ranks_by_the_position_residual(self, core: MotionPrimitivesCore) -> None:
        """The historical ordering, preserved exactly so position-only behaviour is unchanged."""
        assert core._pose_violation(0.005, 0.01) == pytest.approx(0.5)
        assert core._pose_violation(0.02, 0.01) == pytest.approx(2.0)

    def test_the_worse_normalized_component_decides(self, core: MotionPrimitivesCore) -> None:
        # Position comfortably inside, orientation twice its bound: not converged.
        assert core._pose_violation(0.001, 0.01, 0.2, 0.1) == pytest.approx(2.0)
        # Orientation comfortably inside, position twice its bound: not converged.
        assert core._pose_violation(0.02, 0.01, 0.01, 0.1) == pytest.approx(2.0)

    def test_both_components_inside_their_own_bound_is_converged(self, core: MotionPrimitivesCore) -> None:
        assert core._pose_violation(0.009, 0.01, 0.09, 0.1) <= 1.0

    def test_metres_are_not_compared_against_radians(self, core: MotionPrimitivesCore) -> None:
        """The normalization is the point: 0.05 rad is fine, 0.05 m is not.

        Raw magnitudes would call these the same error. Each component is
        divided by its OWN tolerance, so the same number in different units
        lands on opposite sides of the gate.
        """
        assert core._pose_violation(0.001, 0.01, 0.05, 0.1) <= 1.0
        assert core._pose_violation(0.05, 0.01, 0.001, 0.1) > 1.0


class TestStepBudgetCap:
    @pytest.mark.parametrize("value", [0, -1, 10_001, 1.5, "200", None, True])
    def test_off_domain_budget_is_refused(self, core: MotionPrimitivesCore, value) -> None:
        err = core._validate_step_budget("move_to", "max_steps", value)
        assert err is not None
        text = _error_text(err)
        assert "max_steps" in text and "move_to" in text

    @pytest.mark.parametrize("value", [1, 200, 10_000])
    def test_in_domain_budget_passes(self, core: MotionPrimitivesCore, value) -> None:
        assert core._validate_step_budget("move_to", "max_steps", value) is None

    def test_cap_is_named_in_the_refusal(self, core: MotionPrimitivesCore) -> None:
        err = core._validate_step_budget("set_gripper", "steps", 10_001)
        assert "between 1 and 10000" in _error_text(err)


class TestSetGripperArgValidation:
    @pytest.mark.parametrize("state", [None, "opened", "CLOSE", 1])
    def test_off_domain_state_is_refused(self, core: MotionPrimitivesCore, state) -> None:
        steps, err = core._validate_set_gripper_args(state, 12)
        assert steps == 0
        assert '"open" or "close"' in _error_text(err)

    @pytest.mark.parametrize("state", ["open", "close"])
    def test_valid_state_passes(self, core: MotionPrimitivesCore, state) -> None:
        steps, err = core._validate_set_gripper_args(state, 12)
        assert (steps, err) == (12, None)


class TestRotateWristArgValidation:
    def test_missing_target_yaw_is_refused(self, core: MotionPrimitivesCore) -> None:
        _, _, err = core._validate_rotate_wrist_args(None, 0.02, 200)
        assert "requires 'target_yaw'" in _error_text(err)

    @pytest.mark.parametrize("target_yaw", [float("nan"), float("inf"), "0.5", True])
    def test_non_finite_target_yaw_is_refused(self, core: MotionPrimitivesCore, target_yaw) -> None:
        _, _, err = core._validate_rotate_wrist_args(target_yaw, 0.02, 200)
        assert "'target_yaw' must be a finite number" in _error_text(err)

    def test_valid_args_pass(self, core: MotionPrimitivesCore) -> None:
        target_yaw, max_steps, err = core._validate_rotate_wrist_args(0.5, 0.02, 200)
        assert (target_yaw, max_steps, err) == (0.5, 200, None)


class TestWorkspaceSanity:
    def test_far_target_is_refused_naming_the_distance(self, core: MotionPrimitivesCore) -> None:
        err = core._workspace_sanity_error("arm", np.array([10.0, 0.0, 0.0]), np.zeros(3))
        text = _error_text(err)
        assert "10.00 m" in text and "workspace" in text and "arm" in text

    def test_near_target_passes(self, core: MotionPrimitivesCore) -> None:
        assert core._workspace_sanity_error("arm", np.array([0.3, 0.2, 0.1]), np.zeros(3)) is None

    def test_distance_is_measured_from_the_base(self, core: MotionPrimitivesCore) -> None:
        base = np.array([10.0, 0.0, 0.0])
        assert core._workspace_sanity_error("arm", np.array([10.3, 0.0, 0.2]), base) is None


class _RegistryCore(MotionPrimitivesCore):
    """Core wired to a registry-shaped dict instead of the shipped robots.json."""

    def __init__(self, registry: dict) -> None:
        self._registry = registry

    def _get_registry_robot(self, data_config: str) -> dict | None:  # type: ignore[override]
        return self._registry.get(data_config)


class TestRegistryGripperMetadata:
    def _robot(self, data_config: str | None = "so101") -> SimpleNamespace:
        return SimpleNamespace(name="arm", data_config=data_config)

    def test_well_formed_block_resolves(self) -> None:
        core = _RegistryCore({"so101": {"gripper": {"actuators": ["Jaw"], "closed": "low", "open": "high"}}})
        meta, reason = core._registry_gripper_metadata(self._robot())
        assert reason is None
        assert meta == {"actuators": ["Jaw"], "closed": "low", "open": "high"}

    def test_no_data_config_means_heuristic_fallback(self) -> None:
        core = _RegistryCore({})
        assert core._registry_gripper_metadata(self._robot(data_config=None)) == (None, None)

    def test_unknown_robot_means_heuristic_fallback(self) -> None:
        core = _RegistryCore({})
        assert core._registry_gripper_metadata(self._robot()) == (None, None)

    def test_entry_without_gripper_block_means_heuristic_fallback(self) -> None:
        core = _RegistryCore({"so101": {"hardware": {}}})
        assert core._registry_gripper_metadata(self._robot()) == (None, None)

    @pytest.mark.parametrize(
        "block",
        [
            {"actuators": [], "closed": "low", "open": "high"},  # empty actuator list
            {"actuators": ["Jaw", 3], "closed": "low", "open": "high"},  # non-str actuator
            {"actuators": ["Jaw"], "closed": "low", "open": "low"},  # closed == open
            {"actuators": ["Jaw"], "closed": "mid", "open": "high"},  # off-domain end
            "jaw",  # not a dict at all
        ],
    )
    def test_malformed_block_is_a_loud_reason_not_a_fallback(self, block) -> None:
        core = _RegistryCore({"so101": {"gripper": block}})
        meta, reason = core._registry_gripper_metadata(self._robot())
        assert meta is None
        assert reason is not None and "malformed" in reason

    def test_state_end_mapping_defaults_and_metadata_override(self) -> None:
        core = MotionPrimitivesCore()
        # Convention with no metadata: open=HIGH / close=LOW.
        assert core._gripper_state_end("open", None) == "high"
        assert core._gripper_state_end("close", None) == "low"
        # Metadata overrides the convention (the inverted-gripper sign trap).
        meta = {"actuators": ["Jaw"], "closed": "high", "open": "low"}
        assert core._gripper_state_end("open", meta) == "low"
        assert core._gripper_state_end("close", meta) == "high"


def _move_to_envelope(**overrides: object) -> dict:
    """``_move_to_result`` on plain values; each override patches one field.

    Defaults are a real converging run measured on the MuJoCo suite's inline
    arm (43 ticks, 19.8 mm final error, 1.7 mm IK residual), so the not-reached
    variants below differ from a genuine success in exactly the field under
    test.
    """
    kwargs: dict = {
        "reached": True,
        "steps_used": 43,
        "position_error": 0.0198,
        "ik_residual": 0.0017,
        "ee_pos": [0.19, 0.10, 0.19],
        "ee_quat": [1.0, 0.0, 0.0, 0.0],
        "frame_name": "arm/ee_site",
        "frame_type": "site",
    }
    kwargs.update(overrides)
    return MotionPrimitivesCore._move_to_result("arm", np.array([0.2, 0.1, 0.2]), 0.02, 400, **kwargs)


def _rotate_wrist_envelope(**overrides: object) -> dict:
    """``_rotate_wrist_result`` on plain values; each override patches one field."""
    kwargs: dict = {
        "reached": True,
        "steps_used": 76,
        "wrist_name": "wrist_roll",
        "target_yaw": 0.3,
        "final_yaw": 0.2807,
        "yaw_error": 0.0193,
    }
    kwargs.update(overrides)
    return MotionPrimitivesCore._rotate_wrist_result("arm", 0.02, 300, **kwargs)


def _payload(result: dict) -> dict:
    """The json details block every envelope carries."""
    blocks = [b["json"] for b in result["content"] if "json" in b]
    assert len(blocks) == 1, f"expected exactly one json block, got {result['content']}"
    return blocks[0]


class TestMoveToResultEnvelope:
    """``_move_to_result`` - both halves of the envelope the backends hand back.

    The module docstring names the result envelopes as part of what the core
    owns. The reached half is exercised through the MuJoCo mixin by the
    live-world suites; the not-reached half was exercised nowhere, and it is
    the half an agent has to act on - it is what decides whether to retry with
    a larger budget, and that decision reads the json block rather than the
    sentence.
    """

    def test_reached_is_a_success_envelope_carrying_the_payload(self) -> None:
        result = _move_to_envelope()
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "reached" in text and "[0.2, 0.1, 0.2]" in text
        payload = _payload(result)
        assert payload["reached"] is True
        assert payload["steps"] == 43
        assert payload["frame"] == "arm/ee_site"
        assert payload["frame_type"] == "site"

    def test_not_reached_is_an_error_naming_the_budget_it_spent(self) -> None:
        result = _move_to_envelope(reached=False, steps_used=2, position_error=0.1815)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "did not reach" in text
        assert "tol=0.02" in text and "max_steps=400" in text
        assert "0.1815" in text and "0.0017" in text

    def test_not_reached_still_carries_the_json_details(self) -> None:
        """The refusal is what an agent retries from, so it keeps the payload.

        An error whose only content were the sentence would force the caller to
        parse prose to find out how far off it was.
        """
        payload = _payload(_move_to_envelope(reached=False, steps_used=2, position_error=0.1815))
        assert payload["reached"] is False
        assert payload["steps"] == 2
        assert payload["position_error_m"] == 0.1815

    def test_the_two_residuals_stay_separate_fields(self) -> None:
        """``position_error_m`` and ``ik_residual_m`` answer different questions.

        A small IK residual beside a large position error says the pose is
        solvable and the servo merely ran out of steps; a large IK residual says
        the pose is out of reach. Collapsing them would erase that distinction.
        """
        payload = _payload(_move_to_envelope(reached=False, position_error=0.1815, ik_residual=0.0017))
        assert payload["position_error_m"] == 0.1815
        assert payload["ik_residual_m"] == 0.0017

    def test_both_halves_carry_the_same_payload_keys(self) -> None:
        """One reader shape for success and failure, so the caller need not branch."""
        assert sorted(_payload(_move_to_envelope())) == sorted(
            _payload(_move_to_envelope(reached=False, position_error=0.1815))
        )


class TestRotateWristResultEnvelope:
    """``_rotate_wrist_result`` - the second converging primitive's envelope."""

    def test_reached_is_a_success_envelope_carrying_the_payload(self) -> None:
        result = _rotate_wrist_envelope()
        assert result["status"] == "success"
        assert "reached" in result["content"][0]["text"]
        payload = _payload(result)
        assert payload["reached"] is True
        assert payload["wrist_joint"] == "wrist_roll"
        assert payload["steps"] == 76

    def test_not_reached_is_an_error_naming_the_budget_it_spent(self) -> None:
        result = _rotate_wrist_envelope(reached=False, steps_used=1, final_yaw=0.0016, yaw_error=0.2984)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "did not reach" in text
        assert "tol=0.02" in text and "max_steps=300" in text
        assert "0.2984" in text
        assert "wrist_roll" in text

    def test_not_reached_still_carries_the_json_details(self) -> None:
        payload = _payload(_rotate_wrist_envelope(reached=False, steps_used=1, yaw_error=0.2984))
        assert payload["reached"] is False
        assert payload["steps"] == 1
        assert payload["yaw_error_rad"] == 0.2984

    def test_both_halves_carry_the_same_payload_keys(self) -> None:
        assert sorted(_payload(_rotate_wrist_envelope())) == sorted(
            _payload(_rotate_wrist_envelope(reached=False, yaw_error=0.2984))
        )


class TestSetGripperResultEnvelope:
    """``_set_gripper_result`` - the third envelope, and the one with no failure half.

    ``set_gripper`` drives the gripper open-loop and measures no convergence,
    so success is its only outcome. Pinned here so that asymmetry with
    ``move_to`` / ``rotate_wrist`` is recorded rather than inferred from the
    absence of a branch.
    """

    def test_success_envelope_reports_state_actuators_and_ticks(self) -> None:
        result = MotionPrimitivesCore._set_gripper_result(
            "arm",
            "close",
            5,
            ["Jaw"],
            {"Jaw": 0.0},
            {"Jaw": "ctrlrange-low"},
            {"jaw": 0.0012},
        )
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "close" in text and "5 ticks" in text and "Jaw" in text
        payload = _payload(result)
        assert payload["state"] == "close"
        assert payload["actuators"] == ["Jaw"]
        assert payload["setpoint_sources"] == {"Jaw": "ctrlrange-low"}
        assert payload["gripper_joint_positions"] == {"jaw": 0.0012}

    def test_there_is_no_not_reached_half(self) -> None:
        source = inspect.getsource(MotionPrimitivesCore._set_gripper_result)
        assert "reached" not in source, source


class TestRegistryLookupSeamDefault:
    """``_get_registry_robot`` - the default body a non-overriding backend inherits.

    The seam's own docstring records that an adapter may override it to keep a
    historical patch point alive, and the MuJoCo mixin does; so does the
    ``_RegistryCore`` stand-in above, which is how every existing exercise of
    ``_registry_gripper_metadata`` bypasses the default. That left the default -
    the one a new backend adapter gets for free - resolving through this
    module's own ``get_robot`` with nothing checking that it does.
    """

    @staticmethod
    def _robot(data_config: str) -> SimpleNamespace:
        return SimpleNamespace(name="arm", data_config=data_config)

    def test_default_seam_resolves_a_shipped_registry_entry(self) -> None:
        """so100 ships a gripper block, so the default lookup must find it."""
        core = MotionPrimitivesCore()
        meta, reason = core._registry_gripper_metadata(self._robot("so100"))
        assert reason is None
        assert meta is not None
        assert meta["actuators"] == ["Jaw"]

    def test_default_seam_falls_back_for_an_unknown_robot(self) -> None:
        """An unregistered ``data_config`` is the heuristic's cue, not an error."""
        core = MotionPrimitivesCore()
        assert core._registry_gripper_metadata(self._robot("not-a-shipped-robot")) == (None, None)

    def test_the_mujoco_adapter_overrides_the_seam(self) -> None:
        """Why the default body needs its own test: no shipped backend runs it."""
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.motion_primitives import MotionPrimitivesMixin

        assert MotionPrimitivesMixin._get_registry_robot is not MotionPrimitivesCore._get_registry_robot
