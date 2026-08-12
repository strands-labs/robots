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

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from strands_robots.simulation.motion_primitives_base import (
    MotionPrimitivesCore,
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
        target, quat, steps, err = core._validate_move_to_args(None, None, 0.01, 200)
        assert (target, quat, steps) == (None, None, 0)
        assert "requires 'position'" in _error_text(err)

    @pytest.mark.parametrize("position", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], "abc", [1.0, float("nan"), 3.0]])
    def test_malformed_position_is_refused(self, core: MotionPrimitivesCore, position) -> None:
        _, _, _, err = core._validate_move_to_args(position, None, 0.01, 200)
        assert err is not None
        assert "position" in _error_text(err)

    def test_zero_norm_quaternion_is_refused(self, core: MotionPrimitivesCore) -> None:
        _, _, _, err = core._validate_move_to_args([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 0.0], 0.01, 200)
        assert "~zero norm" in _error_text(err)

    @pytest.mark.parametrize("tol", [0.0, -0.01, float("nan"), float("inf"), "0.01", True])
    def test_off_domain_tol_is_refused(self, core: MotionPrimitivesCore, tol) -> None:
        _, _, _, err = core._validate_move_to_args([0.1, 0.2, 0.3], None, tol, 200)
        assert "'tol' must be a positive number" in _error_text(err)

    def test_valid_args_coerce_to_arrays(self, core: MotionPrimitivesCore) -> None:
        target, quat, max_steps, err = core._validate_move_to_args([0.1, 0.2, 0.3], [1, 0, 0, 0], 0.01, 200)
        assert err is None
        assert target is not None and quat is not None
        np.testing.assert_allclose(target, np.array([0.1, 0.2, 0.3]))
        np.testing.assert_allclose(quat, np.array([1.0, 0.0, 0.0, 0.0]))
        assert target.dtype == np.float64 and quat.dtype == np.float64
        assert max_steps == 200

    def test_orientation_is_optional(self, core: MotionPrimitivesCore) -> None:
        target, quat, _, err = core._validate_move_to_args([0.1, 0.2, 0.3], None, 0.01, 200)
        assert err is None and target is not None and quat is None


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
