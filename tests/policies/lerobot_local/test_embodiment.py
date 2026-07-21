"""Tests for the declarative EmbodimentMap mapping (SOLUTION.md).

These tests exercise the REAL mapping path (not mocked) to close the gap that
let B7/B12 slip past the mock-heavy existing suite.
"""

import math

import numpy as np
import pytest

from strands_robots.policies.lerobot_local.embodiment import (
    EMBODIMENT_MAP,
    EmbodimentMap,
    load_embodiment,
    reconcile_dim,
    register_pack_state_step,
)


class _Feat:
    """Minimal stand-in for a LeRobot PolicyFeature (just needs .shape)."""

    def __init__(self, shape):
        self.shape = shape


# reconcile_dim


def test_reconcile_dim_exact():
    assert reconcile_dim([1.0, 2.0, 3.0], 3, "strict") == [1.0, 2.0, 3.0]


def test_reconcile_dim_pad():
    assert reconcile_dim([1.0, 2.0], 4, "pad") == [1.0, 2.0, 0.0, 0.0]


def test_reconcile_dim_truncate():
    assert reconcile_dim([1.0, 2.0, 3.0, 4.0], 2, "truncate") == [1.0, 2.0]


def test_reconcile_dim_strict_raises():
    with pytest.raises(ValueError, match="dim_policy"):
        reconcile_dim([1.0, 2.0], 4, "strict")


def test_reconcile_dim_pad_cannot_shrink():
    with pytest.raises(ValueError, match="cannot pad"):
        reconcile_dim([1.0, 2.0, 3.0], 2, "pad")


def test_reconcile_dim_unknown_policy():
    # Use a length != expected so the policy branch is actually reached
    # (an exact-length match short-circuits and returns before policy check).
    with pytest.raises(ValueError, match="Unknown dim_policy"):
        reconcile_dim([1.0, 2.0], 1, "bogus")


# Registry loading + _extends + aliases


def test_registry_loaded():
    assert "panda_libero" in EMBODIMENT_MAP
    assert "so100" in EMBODIMENT_MAP


def test_extends_inheritance():
    # wx250s _extends vx300s (identical Trossen arm joint topology).
    wx = load_embodiment("wx250s")
    vx = load_embodiment("vx300s")
    assert wx.state_keys == vx.state_keys
    assert wx.dim_policy == vx.dim_policy


def test_so100_so101_are_distinct():
    # Regression: so100 (trs_so_arm100 XML: Rotation/Pitch/...) and so101
    # (robotstudio_so101 XML: 1..6) have DIFFERENT sim joint names and must
    # NOT share a schema. The old config wrongly had so101 _extends so100.
    so100 = load_embodiment("so100")
    so101 = load_embodiment("so101")
    assert so100.state_keys == ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    assert so101.state_keys == ["1", "2", "3", "4", "5", "6"]
    assert so100.state_keys != so101.state_keys


def test_aliases():
    assert load_embodiment("franka_libero").name == "panda_libero"
    # panda (joint-space sim) is distinct from panda_libero (EEF task-space).
    assert load_embodiment("panda").name == "panda"
    assert load_embodiment("franka").name == "panda"
    assert load_embodiment("so100_real").name == "so_real"
    assert load_embodiment("kinova").name == "kinova_gen3"


def test_real_hardware_keys_have_pos_suffix():
    # *_real entries must use lerobot driver motor-feature names ('<motor>.pos').
    real = load_embodiment("so_real")
    assert all(k.endswith(".pos") for k in real.state_keys)
    assert "shoulder_pan.pos" in real.state_keys


def test_load_unknown_raises():
    with pytest.raises(ValueError, match="Unknown embodiment"):
        load_embodiment("does_not_exist")


def test_load_inline_dict():
    em = load_embodiment({"obs_rename": {"a": "observation.images.a"}, "state_keys": ["j1"]})
    assert em.obs_rename == {"a": "observation.images.a"}
    assert em.state_keys == ["j1"]


def test_load_passthrough_instance():
    em = EmbodimentMap(name="x")
    assert load_embodiment(em) is em


# validate() fail-fast


def _features():
    inp = {
        "observation.images.image": _Feat((3, 256, 256)),
        "observation.images.wrist_image": _Feat((3, 256, 256)),
        "observation.state": _Feat((7,)),
    }
    out = {"action": _Feat((7,))}
    return inp, out


def test_validate_ok():
    inp, out = _features()
    load_embodiment("panda_libero").validate(inp, out)  # 7 state, 7 action -> OK


def test_validate_unknown_rename_target():
    inp, out = _features()
    em = EmbodimentMap(name="bad", obs_rename={"cam": "observation.images.NOPE"})
    with pytest.raises(ValueError, match="doesn't declare"):
        em.validate(inp, out)


def test_validate_state_dim_mismatch_strict():
    inp, out = _features()
    em = EmbodimentMap(name="bad", state_keys=["a", "b", "c"], dim_policy="strict")
    with pytest.raises(ValueError, match="state_keys"):
        em.validate(inp, out)


def test_validate_state_dim_mismatch_pad_allowed():
    inp, out = _features()
    # pad opts in to adaptation -> no raise
    em = EmbodimentMap(name="ok", state_keys=["a", "b", "c"], dim_policy="pad")
    em.validate(inp, out)


def test_validate_action_dim_mismatch():
    inp, out = _features()
    em = EmbodimentMap(name="bad", action_keys=["a", "b"])  # 2 != 7
    with pytest.raises(ValueError, match="action_keys"):
        em.validate(inp, out)


# PackStateProcessorStep


def _require_pack_state():
    """Return the registered PackState step class, skipping if lerobot absent.

    ``register_pack_state_step`` returns ``None`` when lerobot's processor
    framework is not importable. Without this guard the tests below hard-fail
    (``None`` is not callable) on a minimal env instead of skipping cleanly.
    """
    Step = register_pack_state_step()
    if Step is None:
        pytest.skip("lerobot processor framework unavailable")
    return Step


def test_pack_state_composes_in_order():
    Step = _require_pack_state()
    s = Step(state_keys=["x", "y", "z"], expected_dim=3, dim_policy="strict")
    obs = {"x": 1.0, "y": 2.0, "z": 3.0, "observation.images.image": np.zeros((3, 4, 4))}
    out = s.observation(dict(obs))
    assert list(out["observation.state"]) == [1.0, 2.0, 3.0]
    assert "x" not in out and "y" not in out and "z" not in out
    assert "observation.images.image" in out  # non-state keys preserved


def test_pack_state_idempotent_when_already_packed():
    Step = _require_pack_state()
    s = Step(state_keys=["x"], expected_dim=1, dim_policy="strict")
    pre = {"observation.state": np.array([9.0, 9.0])}
    out = s.observation(dict(pre))
    assert list(out["observation.state"]) == [9.0, 9.0]


def test_pack_state_pads():
    Step = _require_pack_state()
    s = Step(state_keys=["a", "b"], expected_dim=4, dim_policy="pad")
    out = s.observation({"a": 1.0, "b": 2.0})
    assert list(out["observation.state"]) == [1.0, 2.0, 0.0, 0.0]


def test_pack_state_get_config_roundtrips():
    Step = _require_pack_state()
    s = Step(state_keys=["a", "b"], expected_dim=2, dim_policy="pad")
    cfg = s.get_config()
    assert cfg == {"state_keys": ["a", "b"], "expected_dim": 2, "dim_policy": "pad"}


# Full lerobot driver coverage guard


def test_all_lerobot_drivers_have_embodiment():
    """Every robot subclass registered in lerobot.robots must resolve to an
    embodiment (directly or via alias). Guards against a new lerobot driver
    silently lacking a key-mapping. Ground truth: the @RobotConfig
    .register_subclass names in lerobot-src/src/lerobot/robots/*.
    """
    lerobot_drivers = [
        "so100_follower",
        "so101_follower",
        "koch_follower",
        "omx_follower",
        "openarm_follower",
        "bi_openarm_follower",
        "bi_so_follower",
        "rebot_b601_follower",
        "bi_rebot_b601_follower",
        "lekiwi",
        "lekiwi_client",
        "reachy2",
        "hope_jr_hand",
        "hope_jr_arm",
        "earthrover_mini_plus",
        "unitree_g1",
    ]
    missing = []
    for name in lerobot_drivers:
        try:
            load_embodiment(name)
        except ValueError:
            missing.append(name)
    assert not missing, f"lerobot drivers without embodiment mapping: {missing}"


def test_real_hardware_entries_use_pos_or_velocity_keys():
    """All *_real arm/hand entries use '<motor>.pos' driver feature keys
    (the lerobot _motors_ft convention). The mobile rover uses velocity cmds.
    """
    pos_robots = [
        "omx_real",
        "bi_so_real",
        "openarm_real",
        "bi_openarm_real",
        "rebot_b601_real",
        "bi_rebot_b601_real",
        "reachy2_real",
        "hope_jr_arm_real",
        "hope_jr_hand_real",
    ]
    for name in pos_robots:
        em = load_embodiment(name)
        assert all(k.endswith(".pos") for k in em.state_keys), f"{name} non-.pos keys"
    rover = load_embodiment("earthrover_real")
    assert rover.state_keys == ["linear_velocity", "angular_velocity"]


# reconcile_dim truncate guard


def test_reconcile_dim_truncate_cannot_grow():
    # truncate cannot lengthen a too-short vector -> directs caller to 'pad'
    with pytest.raises(ValueError, match="cannot truncate"):
        reconcile_dim([1.0, 2.0], 4, "truncate")


# PackStateProcessorStep value-coercion branches


def test_pack_state_coerces_array_and_list_values():
    Step = _require_pack_state()
    s = Step(state_keys=["scalar", "vec", "lst"], expected_dim=5, dim_policy="pad")
    obs = {
        "scalar": np.array(1.0),  # 0-dim ndarray -> single float
        "vec": np.array([2.0, 3.0]),  # multi-dim ndarray -> ravel/extend
        "lst": [4.0, 5.0],  # list -> extend
    }
    out = s.observation(obs)
    # 1 + 2 + 2 = 5 collected values, exact target -> no padding
    assert list(out["observation.state"]) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_pack_state_passthrough_when_no_state_keys_present():
    # None of the declared state keys are in the obs -> leave obs untouched
    # so a clearer downstream error can surface (never fabricate state).
    Step = _require_pack_state()
    s = Step(state_keys=["a", "b"], expected_dim=2, dim_policy="strict")
    obs = {"observation.images.image": np.zeros((3, 4, 4))}
    out = s.observation(dict(obs))
    assert "observation.state" not in out
    assert "observation.images.image" in out


# SO-arm degrees units on the declarative packing step. The direct
# EmbodimentMap.sim_state_to_model conversion is covered in test_so_arm_units;
# these pin the SAME rad->deg / gripper-0..100 conversion where it actually runs
# at inference time: inside PackStateProcessorStep.observation when the
# embodiment declares state_units="degrees". A native step must leave the raw
# radian/joint values untouched, so the two together lock the boundary.

# so101 sim gripper joint range (matches test_so_arm_units.GRIPPER_RANGE).
_SO_GRIPPER_RANGE = [-0.175, 1.745]


def test_pack_state_degrees_units_converts_radians_to_model_units():
    """A degrees embodiment must convert the packed observation.state from sim
    radians to the model's training units: arm joints in degrees, gripper in
    RANGE_0_100. pi/2 rad arm joints -> 90 deg; a mid-range gripper -> 50."""
    Step = _require_pack_state()
    s = Step(
        state_keys=["1", "2", "3", "4", "5", "6"],
        expected_dim=6,
        dim_policy="strict",
        state_units="degrees",
        gripper_index=5,
        gripper_joint_range=_SO_GRIPPER_RANGE,
    )
    lo, hi = _SO_GRIPPER_RANGE
    obs = {"1": math.pi / 2, "2": math.pi / 2, "3": math.pi / 2, "4": math.pi / 2, "5": math.pi / 2, "6": (lo + hi) / 2}
    out = list(s.observation(obs)["observation.state"])
    for i in range(5):
        assert math.isclose(out[i], 90.0, abs_tol=1e-4), (i, out)
    assert math.isclose(out[5], 50.0, abs_tol=1e-4), out


def test_pack_state_degrees_units_applies_joint_mids():
    """When the embodiment supplies per-joint calibration mids (degrees), the
    degrees packing path must subtract them (mid-centered degrees, matching
    lerobot motors_bus DEGREES mode), not emit the absolute joint angle."""
    Step = _require_pack_state()
    mids = [10.0, -20.0, 30.0, -5.0, 15.0, 0.0]
    s = Step(
        state_keys=["1", "2", "3", "4", "5", "6"],
        expected_dim=6,
        dim_policy="strict",
        state_units="degrees",
        gripper_index=5,
        gripper_joint_range=_SO_GRIPPER_RANGE,
        joint_mids=mids,
    )
    lo, hi = _SO_GRIPPER_RANGE
    obs = {k: math.pi / 2 for k in ("1", "2", "3", "4", "5")}
    obs["6"] = (lo + hi) / 2
    out = list(s.observation(obs)["observation.state"])
    for i in range(5):
        assert math.isclose(out[i], 90.0 - mids[i], abs_tol=1e-4), (i, out)
    assert math.isclose(out[5], 50.0, abs_tol=1e-4), out


def test_pack_state_native_units_leaves_values_unconverted():
    """The default native step is the boundary case: raw radian arm values pass
    through unchanged (no rad->deg), so observation.state carries sim units."""
    Step = _require_pack_state()
    s = Step(state_keys=["1", "2", "3"], expected_dim=3, dim_policy="strict")
    out = list(s.observation({"1": math.pi / 2, "2": 0.5, "3": -1.0})["observation.state"])
    assert math.isclose(out[0], math.pi / 2, abs_tol=1e-6), out
    assert math.isclose(out[1], 0.5, abs_tol=1e-6)
    assert math.isclose(out[2], -1.0, abs_tol=1e-6)


def test_pack_state_transform_features_is_passthrough():
    # State composition reshapes runtime obs only; the model's declared feature
    # set is unchanged, so transform_features returns its input unchanged.
    Step = _require_pack_state()
    s = Step(state_keys=["a"], expected_dim=1, dim_policy="strict")
    sentinel = {"observation.state": object()}
    assert s.transform_features(sentinel) is sentinel


# expected_state_dim


def test_expected_state_dim_prefers_model_feature():
    em = EmbodimentMap(name="x", state_keys=["a", "b"])
    assert em.expected_state_dim({"observation.state": _Feat((7,))}) == 7


def test_expected_state_dim_falls_back_to_state_keys():
    # No declared observation.state feature -> use len(state_keys).
    em = EmbodimentMap(name="x", state_keys=["a", "b", "c"])
    assert em.expected_state_dim({}) == 3


# JSON loader internals (_extends inheritance + missing config file)


def test_resolve_extends_merges_child_overrides():
    from strands_robots.policies.lerobot_local.embodiment import _resolve

    definitions = {
        "base": {
            "obs_rename": {"image": "observation.images.image"},
            "state_keys": ["a", "b"],
            "action_keys": ["a", "b"],
            "dim_policy": "strict",
        },
        "child": {
            "_extends": "base",
            "__note__": "doc metadata is stripped",
            "dim_policy": "pad",  # child override wins over inherited value
        },
    }
    child = _resolve("child", definitions)
    assert child.name == "child"
    assert child.dim_policy == "pad"  # overridden
    assert child.state_keys == ["a", "b"]  # inherited
    assert child.obs_rename == {"image": "observation.images.image"}  # inherited


def test_load_defs_returns_empty_when_config_missing(monkeypatch, tmp_path):
    from strands_robots.policies.lerobot_local import embodiment as emb

    monkeypatch.setattr(emb, "_CONFIG_FILE", tmp_path / "does_not_exist.json")
    assert emb._load_defs() == ({}, {})


# load_embodiment type rejection


def test_load_embodiment_rejects_unsupported_type():
    with pytest.raises(ValueError, match="must be str"):
        load_embodiment(42)  # type: ignore[arg-type]
